"""Real R2 output/parameter types with a CPU-only scripted engine contract."""

import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import deepseek_recipe as recipe

from adapter.backend import VllmBackend, load_images
from adapter.protocol import ProtocolError, ProtocolStream
from test_protocol import tokenizer_path

try:
    from vllm.outputs import CompletionOutput, RequestOutput
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.logprobs import Logprob
except ImportError:
    CompletionOutput = None


class ScriptedEngine:
    def __init__(self, tokenizer, *, max_model_len=524288):
        self.tokenizer = tokenizer
        eos = tokenizer.encode(recipe.SpecialToken.Eos.default_str())[0]
        self.model_config = SimpleNamespace(model=str(Path(tokenizer_path()).parent),
                                            max_model_len=max_model_len,
                                            hf_config=SimpleNamespace(eos_token_id=eos))
        self.renderer = SimpleNamespace(render_cmpl_async=self.render)
        self.outputs = []
        self.aborted = []
        self.generated = []
        self.expansion = 0
        self.seen_prompt = None
        self.admitted = []

    def check_admission(self, **kwargs):
        self.admitted.append(kwargs)

    async def render(self, prompts):
        self.seen_prompt = prompts[0]
        prompt = dict(prompts[0])
        prompt["prompt_token_ids"] = list(prompt["prompt_token_ids"]) + [42] * self.expansion
        return [prompt]

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.generated.append((prompt, sampling_params, request_id, kwargs))
        for output in self.outputs:
            if isinstance(output, Exception):
                raise output
            yield output

    async def abort(self, request_id):
        self.aborted.append(request_id)


@unittest.skipIf(CompletionOutput is None, "Run backend contract tests in the pinned R2 image")
class BackendTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = recipe.Tokenizer.from_file(tokenizer_path())

    async def asyncSetUp(self):
        self.engine = ScriptedEngine(self.tokenizer)
        self.backend = VllmBackend(self.engine, tokenizer_path=tokenizer_path())

    async def generation(self, **updates):
        body = {"model": "DeepSeek-Flash", "input": "Hello", "reasoning": {"effort": "none"}}
        body.update(updates)
        return await self.backend.prepare("responses", body)

    def output(self, generation, ids, *, reason=None, finished=False, cache=0, logprobs=None):
        completion = CompletionOutput(index=0, text="ignored by adapter", token_ids=ids,
                                      cumulative_logprob=None, logprobs=logprobs,
                                      finish_reason=reason, stop_reason=None)
        return RequestOutput(request_id="req_test", prompt=None,
                             prompt_token_ids=generation.engine_input["prompt_token_ids"],
                             prompt_logprobs=None, outputs=[completion], finished=finished,
                             num_cached_tokens=cache)

    async def test_real_sampling_params_and_unmodified_token_prompt(self):
        generation = await self.generation()
        params = generation.sampling_params
        self.assertIsInstance(params, SamplingParams)
        self.assertEqual(params.output_kind, RequestOutputKind.DELTA)
        self.assertFalse(params.skip_special_tokens)
        self.assertEqual(params.max_tokens, generation.prepared.max_tokens)
        self.assertEqual(self.engine.seen_prompt["prompt_token_ids"], generation.prepared.prompt_token_ids)
        self.assertEqual(self.engine.admitted, [{"n": 1}])

    async def test_delta_stop_removes_only_eos_and_preserves_real_usage(self):
        generation = await self.generation()
        eos = self.engine.model_config.hf_config.eos_token_id
        first = self.tokenizer.encode("Hello")
        tail = self.tokenizer.encode(" world")
        self.engine.outputs = [self.output(generation, first, cache=3),
                               self.output(generation, tail + [eos], reason="stop", finished=True, cache=3)]
        stream = ProtocolStream(generation.prepared)
        events = []
        async for chunk in self.backend.chunks(generation, "req_test"):
            events += stream.push(chunk)
        self.assertEqual(stream.response["output"][0]["content"][0]["text"], "Hello world")
        self.assertEqual(stream.response["usage"]["output_tokens"], len(first) + len(tail) + 1)
        self.assertEqual(stream.response["usage"]["input_tokens_details"]["cached_tokens"], 3)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(self.engine.aborted, [])

    async def test_length_keeps_the_final_delta(self):
        generation = await self.generation(max_output_tokens=1)
        ids = self.tokenizer.encode("Hi")
        self.engine.outputs = [self.output(generation, ids, reason="length", finished=True)]
        stream = ProtocolStream(generation.prepared)
        async for chunk in self.backend.chunks(generation, "req_test"):
            stream.push(chunk)
        self.assertEqual(stream.response["output"][0]["content"][0]["text"], "Hi")
        self.assertEqual(stream.response["status"], "incomplete")

    async def test_delta_logprobs_align_after_eos_removal(self):
        generation = await self.generation(top_logprobs=1)
        ids = self.tokenizer.encode("Hi") + [self.engine.model_config.hf_config.eos_token_id]
        logprobs = [{token: Logprob(logprob=-0.3, rank=1)} for token in ids]
        self.engine.outputs = [self.output(generation, ids, reason="stop", finished=True, logprobs=logprobs)]
        chunks = [chunk async for chunk in self.backend.chunks(generation, "req_test")]
        stream = ProtocolStream(generation.prepared)
        for chunk in chunks:
            stream.push(chunk)
        self.assertEqual(stream.response["output"][0]["content"][0]["text"], "Hi")
        self.assertEqual(stream.response["usage"]["output_tokens"], len(ids))

    async def test_missing_cache_and_eof_abort(self):
        for outputs in ([], [RuntimeError("engine failed")]):
            self.engine.outputs = outputs
            with self.assertRaises((ProtocolError, RuntimeError)):
                [chunk async for chunk in self.backend.chunks(await self.generation(), "req_test")]
            self.assertIn("req_test", self.engine.aborted)
        generation = await self.generation()
        self.engine.outputs = [self.output(generation, [1], cache=None)]
        with self.assertRaises(ProtocolError):
            [chunk async for chunk in self.backend.chunks(generation, "missing-cache")]
        self.assertIn("missing-cache", self.engine.aborted)

    async def test_image_renderer_expansion_changes_budget(self):
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (3, 4), "red").save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        self.engine.expansion = 20
        generation = await self.generation(input=[{"role": "user", "content": [
            {"type": "input_image", "image_url": url}]}])
        self.assertEqual(generation.prepared.prompt_tokens,
                         len(generation.prepared.prompt_token_ids) + 20)
        self.assertEqual(self.engine.seen_prompt["multi_modal_data"]["image"][0].size, (3, 4))

    async def test_text_renderer_change_is_rejected(self):
        self.engine.expansion = 1
        with self.assertRaises(ProtocolError):
            await self.generation()


class ImageInputTests(unittest.TestCase):
    def test_offline_url_is_rejected_without_fetch(self):
        source = SimpleNamespace(url="https://example.com/private.png")
        with self.assertRaises(ProtocolError):
            load_images([source])

    def test_inline_invalid_base64_is_rejected(self):
        source = SimpleNamespace(url=None, data=None, data_url="data:image/png;base64,%%%", detail="low")
        with self.assertRaises(ProtocolError):
            load_images([source])


if __name__ == "__main__":
    unittest.main()
