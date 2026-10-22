"""Raw token-ID adapter to the pinned vLLM SM80 EngineClient.

The existing renderer handles native vision expansion; no chat template is
applied a second time. Kernel code, weight loading and DSpark stay in vLLM.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path

import deepseek_recipe as recipe

from .constraints import structured_outputs
from .protocol import PreparedRequest, ProtocolError, prepare


MAX_IMAGE_BYTES = 32 * 1024**2
MAX_IMAGES_BYTES = 64 * 1024**2


def load_images(sources):
    """Decode only inline images, in recipe's final prompt order."""
    from PIL import Image

    images, total = [], 0
    if len(sources) > 999:
        raise ProtocolError("At most 999 images are supported")
    for source in sources:
        if source.url is not None:
            raise ProtocolError("Offline runtime accepts inline images only")
        data = source.data
        if data is None:
            url = source.data_url or ""
            if not url.startswith("data:") or ";base64," not in url:
                raise ProtocolError("Image must be a base64 data URL")
            encoded = url.split(",", 1)[1]
            if len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
                raise ProtocolError("Inline image exceeds 32 MiB")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ProtocolError("Invalid image base64") from exc
        total += len(data)
        if len(data) > MAX_IMAGE_BYTES or total > MAX_IMAGES_BYTES:
            raise ProtocolError("Image bytes exceed the request limit")
        try:
            with Image.open(io.BytesIO(data)) as opened:
                if opened.format not in {"PNG", "JPEG", "GIF", "WEBP"}:
                    raise ProtocolError("Unsupported image format")
                opened.load()
                image = opened.convert("RGB")
            if source.detail == "low":
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        except (OSError, Image.DecompressionBombError) as exc:
            raise ProtocolError("Invalid or oversized image") from exc
        images.append(image)
    return images


@dataclass
class Generation:
    prepared: PreparedRequest
    engine_input: dict
    sampling_params: object


class VllmBackend:
    def __init__(self, engine, *, tokenizer_path=None):
        self.engine = engine
        model = Path(engine.model_config.model)
        self.tokenizer = recipe.Tokenizer.from_file(str(tokenizer_path or model / "tokenizer.json"))
        self.context_limit = int(engine.model_config.max_model_len)

    async def prepare(self, protocol, body):
        prepared = await asyncio.to_thread(prepare, protocol, body, self.tokenizer, context_limit=self.context_limit)
        prompt = {"prompt_token_ids": prepared.prompt_token_ids}
        if prepared.image_sources:
            prompt["multi_modal_data"] = {"image": await asyncio.to_thread(load_images, prepared.image_sources)}
        # render_cmpl accepts token prompts and only runs multimodal expansion.
        (engine_input,) = await self.engine.renderer.render_cmpl_async([prompt])
        actual_ids = engine_input["prompt_token_ids"]
        adjustment = len(actual_ids) - len(prepared.prompt_token_ids)
        if adjustment < 0:
            raise ProtocolError("Renderer unexpectedly truncated the encoded prompt")
        if not prepared.image_sources and list(actual_ids) != prepared.prompt_token_ids:
            raise ProtocolError("Renderer changed the recipe token prompt")
        # Budget against the expanded native vision prompt, not placeholders.
        if len(actual_ids) >= self.context_limit:
            raise ProtocolError("Expanded image prompt leaves no room for output in the context window")
        max_tokens = prepared.converted.resolve_max_tokens(len(actual_ids), self.context_limit)
        prepared = replace(prepared, max_tokens=max_tokens, prompt_tokens=len(actual_ids))
        options = prepared.inference_options
        from vllm.sampling_params import RequestOutputKind, SamplingParams
        sampling = SamplingParams(
            n=1, max_tokens=max_tokens, temperature=options.temperature,
            top_p=options.top_p, seed=options.seed,
            logprobs=options.top_logprobs if options.logprobs else None,
            output_kind=RequestOutputKind.DELTA, detokenize=True, skip_special_tokens=False,
            structured_outputs=structured_outputs(prepared),
        )
        self.engine.check_admission(n=1)
        return Generation(prepared, engine_input, sampling)

    async def chunks(self, generation, request_id):
        prepared = generation.prepared
        ready, finished = False, False
        iterator = self.engine.generate(
            generation.engine_input, generation.sampling_params, request_id,
            reasoning_ended=not prepared.converted.conversation.thinking_mode,
            reasoning_parser_kwargs={"chat_template_kwargs": {
                "thinking": prepared.converted.conversation.thinking_mode,
            }},
        )
        try:
            async for output in iterator:
                if not ready:
                    ids = output.prompt_token_ids
                    if ids is None or len(ids) != prepared.prompt_tokens:
                        raise ProtocolError("Engine did not report the actual expanded prompt")
                    cache = output.num_cached_tokens
                    # vLLM uses None before cache accounting; fail instead of fabricating zero.
                    if cache is None:
                        raise ProtocolError("Engine did not report prefix-cache usage")
                    yield recipe.InferenceChunk.ready(prompt_usage=recipe.PromptUsage(
                        prompt_tokens=len(ids), prompt_cache_hit_tokens=cache,
                    ))
                    ready = True
                if len(output.outputs) != 1 or output.outputs[0].index != 0:
                    raise ProtocolError("Expected exactly one output sequence")
                completion = output.outputs[0]
                ids = list(completion.token_ids)
                # R2's DELTA output retains the EOS ID even though its
                # detokenized text hides it. Recipe would display that control
                # token verbatim. Strip only the actual terminating ID, keeping
                # any valid content delivered in the same final delta.
                stopped_tokens = 0
                if completion.finish_reason == "stop" and ids:
                    hf_config = getattr(self.engine.model_config, "hf_config", None)
                    eos = getattr(hf_config, "eos_token_id", None)
                    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
                    eos_ids.update(self.tokenizer.encode(recipe.SpecialToken.Eos.default_str()))
                    stop_id = completion.stop_reason
                    if isinstance(stop_id, int):
                        eos_ids.add(stop_id)
                    if ids[-1] in eos_ids:
                        ids.pop()
                        stopped_tokens = 1
                if ids:
                    logprobs = None
                    if prepared.inference_options.logprobs:
                        if completion.logprobs is None or len(completion.logprobs) != len(completion.token_ids):
                            raise ProtocolError("Engine omitted requested token log probabilities")
                        logprobs = []
                        for token_id, probabilities in zip(ids, completion.logprobs[:len(ids)]):
                            sampled = probabilities.get(token_id)
                            if sampled is None:
                                raise ProtocolError("Sampled token log probability is missing")
                            top = sorted(probabilities.items(), key=lambda pair: pair[1].rank or 10**9)
                            logprobs.append((sampled.logprob, [(k, v.logprob) for k, v in top]))
                    yield recipe.InferenceChunk.tokens(ids, logprobs)
                if stopped_tokens:
                    # Count the sampled termination token without rendering it.
                    # An empty text chunk changes usage, not visible content.
                    yield recipe.InferenceChunk.text("", token_count=stopped_tokens)
                if completion.finish_reason is not None:
                    reasons = {"stop": recipe.InferenceFinishReason.Stop,
                               "length": recipe.InferenceFinishReason.Length,
                               "content_filter": recipe.InferenceFinishReason.ContentFilter}
                    if completion.finish_reason not in reasons:
                        raise ProtocolError("Engine generation failed: " + completion.finish_reason)
                    if not output.finished:
                        raise ProtocolError("Finish reason arrived before engine completion")
                    finished = True
                    yield recipe.InferenceChunk.finish(reasons[completion.finish_reason])
                    return
                if output.finished:
                    raise ProtocolError("Engine completed without a finish reason")
            if not finished:
                raise ProtocolError("Engine stream ended before completion")
        finally:
            try:
                await iterator.aclose()
            finally:
                if not finished:
                    await self.engine.abort(request_id)

    async def abort(self, request_id):
        await self.engine.abort(request_id)
