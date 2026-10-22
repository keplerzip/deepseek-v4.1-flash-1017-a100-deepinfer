"""Official, byte-identical recipe example server with a local EngineClient.

Only inference and checkpoint tokenizer bindings live here. Tool conversion,
history, prompt syntax, stream parsing, accumulation and SSE are upstream code.
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
from pathlib import Path
import sys
import uuid

import deepseek_recipe as recipe
from fastapi import Request
from adapter.acceptance_trace import CURRENT_TRACE, TraceStore, TraceMiddleware, active, record_output

ROOT = Path(__file__).resolve().parents[1]
SPECIAL = recipe.SpecialTokens(image="<｜deepseek_image｜>")
ALIASES = ("DeepSeek-Flash", "deepseek-flash", "DeepSeek-V4.1-Flash",
           "DeepSeek-V4.1-Flash-1017", "deepseek-v4.1-flash-1017")


def load_official():
    path = ROOT / "vendor/official-recipe/source/server-py/main.py"
    spec = importlib.util.spec_from_file_location("deepinfer_official_recipe_server", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # This checkpoint's image ID is 129264 with a different spelling. Keep
    # conversion/parsing and encoding on the same published SpecialTokens API.
    module.PROMPT_ENCODINGS = {
        recipe.ModelVersion.DeepseekV41Flash:
            lambda: recipe.DeepseekV41Encoding().with_special_tokens(SPECIAL),
    }
    return module


def options():
    return recipe.ConversionOptions().with_special_tokens(SPECIAL).with_model_versions_by_name({
        name: recipe.ModelVersion.DeepseekV41Flash for name in ALIASES
    })


class EngineInference:
    def __init__(self, engine, tokenizer):
        self.engine, self.tokenizer = engine, tokenizer

    async def prepare(self, request):
        from vllm.sampling_params import RequestOutputKind, SamplingParams, StructuredOutputsParams
        from PIL import Image

        converted = request.conversation_request
        encoding = recipe.DeepseekV41Encoding().with_special_tokens(SPECIAL).with_tokenizer(self.tokenizer)
        ids = encoding.encode(converted.conversation)
        prompt = {"prompt_token_ids": ids}
        if request.images:
            def decode():
                images = []
                for info in request.images:
                    with Image.open(io.BytesIO(info.data)) as image:
                        images.append(image.convert("RGB"))
                return images
            prompt["multi_modal_data"] = {"image": await asyncio.to_thread(decode)}
        (engine_input,) = await self.engine.renderer.render_cmpl_async([prompt])
        actual = engine_input["prompt_token_ids"]
        if not request.images and list(actual) != ids:
            raise RuntimeError("Engine renderer changed the official token prompt")
        limit = self.engine.model_config.max_model_len
        max_tokens = converted.resolve_max_tokens(len(actual), limit)
        inference = request.inference_options
        constraint = inference.output_constraint
        structured = None
        if constraint is not None and constraint.kind == "json":
            structured = (StructuredOutputsParams(json=json.loads(constraint.schema))
                          if constraint.schema else StructuredOutputsParams(json_object=True))
        elif constraint is not None and constraint.kind == "regex":
            structured = StructuredOutputsParams(regex=constraint.schema)
        # Upstream exposes tool constraints as data, without a DSML sampler.
        # Follow its example: let the official prompt/parser handle all tool
        # kinds, schemas and order. Do not impose our former DSML grammar.
        sampling = SamplingParams(
            n=1, max_tokens=max_tokens, temperature=inference.temperature,
            min_tokens=max_tokens if (CURRENT_TRACE.get() or {}).get("fixed_budget") else 0,
            top_p=inference.top_p, seed=inference.seed,
            logprobs=(inference.top_logprobs or 0) if inference.logprobs else None,
            output_kind=RequestOutputKind.DELTA, skip_special_tokens=False,
            structured_outputs=structured,
        )
        self.engine.check_admission(n=1)
        return engine_input, sampling

    async def __call__(self, request):
        engine_input, sampling = await self.prepare(request)
        request_id = "official-beta-" + uuid.uuid4().hex
        iterator = self.engine.generate(
            engine_input, sampling, request_id,
            reasoning_ended=not request.conversation_request.conversation.thinking_mode,
            reasoning_parser_kwargs={"chat_template_kwargs": {
                "thinking": request.conversation_request.conversation.thinking_mode}},
        )
        ready, finished = False, False
        try:
            async for output in iterator:
                if not ready:
                    if output.prompt_token_ids is None or output.num_cached_tokens is None:
                        raise RuntimeError("Engine omitted prompt/cache accounting")
                    yield recipe.InferenceChunk.ready(prompt_usage=recipe.PromptUsage(
                        prompt_tokens=len(output.prompt_token_ids),
                        prompt_cache_hit_tokens=output.num_cached_tokens))
                    ready = True
                if len(output.outputs) != 1:
                    raise RuntimeError("Expected one engine completion")
                completion = output.outputs[0]
                record_output(output, completion)
                ids = list(completion.token_ids)
                eos_count = 0
                if completion.finish_reason == "stop" and ids:
                    eos = getattr(self.engine.model_config.hf_config, "eos_token_id", None)
                    stop_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
                    stop_ids.update(self.tokenizer.encode(recipe.SpecialToken.Eos.default_str()))
                    if ids[-1] in stop_ids:
                        ids.pop()
                        eos_count = 1
                if ids:
                    logprobs = None
                    if request.inference_options.logprobs:
                        if completion.logprobs is None:
                            raise RuntimeError("Engine omitted requested log probabilities")
                        logprobs = []
                        for token_id, probs in zip(ids, completion.logprobs):
                            logprobs.append((probs[token_id].logprob,
                                             [(k, v.logprob) for k, v in sorted(
                                                 probs.items(), key=lambda pair: pair[1].rank or 10**9)]))
                    yield recipe.InferenceChunk.tokens(ids, logprobs)
                if eos_count:
                    yield recipe.InferenceChunk.text("", token_count=eos_count)
                if completion.finish_reason is not None:
                    reasons = {"stop": recipe.InferenceFinishReason.Stop,
                               "length": recipe.InferenceFinishReason.Length,
                               "content_filter": recipe.InferenceFinishReason.ContentFilter}
                    if completion.finish_reason not in reasons or not output.finished:
                        raise RuntimeError("Engine failed: " + str(completion.finish_reason))
                    finished = True
                    yield recipe.InferenceChunk.finish(reasons[completion.finish_reason])
                    return
            if not finished:
                raise RuntimeError("Engine ended without a finish reason")
        finally:
            try:
                await iterator.aclose()
            finally:
                if not finished:
                    await self.engine.abort(request_id)


def install_routes(app, *, engine=None, tokenizer=None):
    # Called only by entry.build_app before serving. The pinned SageMaker
    # bootstrap eagerly builds a stack; discard that cached instance so native
    # user middleware and our final routes/exception handlers rebuild together.
    app.middleware_stack = None
    official = load_official()
    holder = {}
    if not hasattr(app.state, "acceptance_traces"):
        app.state.acceptance_traces = TraceStore()
        app.add_middleware(TraceMiddleware, store=app.state.acceptance_traces)

    def backend():
        if "backend" not in holder:
            client = engine if engine is not None else app.state.engine_client
            tok = tokenizer or recipe.Tokenizer.from_file(str(Path(client.model_config.model) / "tokenizer.json"))
            holder["backend"] = EngineInference(client, tok)
        return holder["backend"]

    async def infer(request):
        iterator = backend()(request)
        try:
            async for chunk in iterator:
                yield chunk
        finally:
            await iterator.aclose()

    # The official example uses text chunks by default. Its documented native
    # StreamProcessor supports a tokenizer; bind it to preserve incremental
    # token decoding, reasoning usage and token logprobs from our backend.
    official.StreamProcessor = lambda generator, parsing: recipe.StreamProcessor(
        generator, parsing, backend().tokenizer)
    upstream_app = official.create_app(infer=infer, options=options())
    replacements = {"/v1/responses", "/v1/chat/completions", "/v1/messages",
                    "/messages", "/v1/models", "/tokenize", "/v1/messages/count_tokens",
                    "/messages/count_tokens", "/deepinfer/compatibility", "/deepinfer/cache/reset-local",
                    "/deepinfer/acceptance/tokenize", "/deepinfer/acceptance/trace/{identity}"}
    app.router.routes[:] = [route for route in app.router.routes
                           if getattr(route, "path", None) not in replacements
                           and not (getattr(route, "path", "") or "").startswith("/v1/responses/")]
    app.router.routes.extend(route for route in upstream_app.router.routes
                             if getattr(route, "path", "").startswith("/v1/"))
    app.exception_handlers.update(upstream_app.exception_handlers)

    async def models():
        return {"object": "list", "data": [{"id": "DeepSeek-Flash", "object": "model", "owned_by": "local"}]}

    async def compatibility():
        return {"release": "20261022-r3-cpukv", "protocol": "official-recipe-server-py",
                "recipe_version": recipe.__version__, "official_server_modified": False,
                "tool_constraint_enforcement": "official-prompt-only",
                "tool_execution": "client", "native_deepinfer_runtime": False,
                "compute_profile": "aggressive-sm80", "gpu_acceptance": "NOT_RUN_AT_BUILD"}

    app.add_api_route("/v1/models", models, methods=["GET"])
    app.add_api_route("/deepinfer/compatibility", compatibility, methods=["GET"])
    async def reset_local_cache():
        from fastapi import HTTPException
        if not Path('/state/cache-acceptance-active').is_file():
            raise HTTPException(status_code=404)
        client = engine if engine is not None else app.state.engine_client
        return {'success': bool(await client.reset_prefix_cache(False, False))}

    app.add_api_route('/deepinfer/cache/reset-local', reset_local_cache, methods=['POST'])
    async def acceptance_trace(identity: str):
        from fastapi import HTTPException
        if not active():
            raise HTTPException(status_code=404)
        row = app.state.acceptance_traces.take(identity)
        if row is None:
            raise HTTPException(status_code=409, detail='Trace not complete or missing')
        return row

    async def acceptance_tokenize(request: Request):
        from fastapi import HTTPException
        if not active():
            raise HTTPException(status_code=404)
        prepared = await asyncio.to_thread(official.prepare_request, 'chat_completions',
                                          await request.body(), options=options())
        engine_input, _ = await backend().prepare(prepared)
        return {'count': len(engine_input['prompt_token_ids']),
                'source': 'official recipe plus same engine renderer'}

    app.add_api_route('/deepinfer/acceptance/trace/{identity}', acceptance_trace, methods=['GET'])
    app.add_api_route('/deepinfer/acceptance/tokenize', acceptance_tokenize, methods=['POST'])
    app.openapi_schema = None
    return app


def main():
    from vllm.entrypoints.launchers.api_server import entry
    native = entry.build_app
    entry.build_app = lambda *a, **kw: install_routes(native(*a, **kw))
    entry.main()


if __name__ == "__main__":
    main()
