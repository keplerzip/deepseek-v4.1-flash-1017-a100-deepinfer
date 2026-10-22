"""Real FastAPI/ASGI HTTP tests with official recipe and fake token-ID inference.

The real VllmBackend converts engine outputs; the fake engine never loads
weights, allocates CUDA memory, executes tools or contacts a network service.
"""
from __future__ import annotations

import asyncio
from collections import deque
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import deepseek_recipe as recipe
from fastapi import FastAPI
import httpx

from adapter.backend import VllmBackend
from adapter import server


ROOT = Path(__file__).resolve().parents[1]
MODEL = "DeepSeek-Flash"
PATHS = {"responses": "/v1/responses", "chat_completions": "/v1/chat/completions",
         "messages": "/v1/messages"}


def tokenizer_path():
    for candidate in (os.environ.get("DEEPINFER_TOKENIZER"), ROOT / "vendor/tokenizer/tokenizer.json",
                      "/models/DeepSeek-V4.1-Flash-1017/tokenizer.json"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise RuntimeError("Set DEEPINFER_TOKENIZER to the matching 1017 tokenizer")


def body(protocol, *, stream=False, **updates):
    result = {"model": MODEL, "stream": stream}
    if protocol == "responses":
        result.update(input="Hello", reasoning={"effort": "none"}, max_output_tokens=512)
    elif protocol == "chat_completions":
        result.update(messages=[{"role": "user", "content": "Hello"}],
                      reasoning_effort="none", max_tokens=512)
    else:
        result.update(messages=[{"role": "user", "content": "Hello"}],
                      thinking={"type": "disabled"}, max_tokens=512)
    result.update(updates)
    return result


def parse_sse(response):
    frames = []
    for frame in response.text.split("\n\n"):
        if not frame.strip() or frame.startswith(":"):
            continue
        lines = frame.splitlines()
        data = "\n".join(line[6:] for line in lines if line.startswith("data: "))
        name = next((line[7:] for line in lines if line.startswith("event: ")), None)
        frames.append((name, data if data == "[DONE]" else json.loads(data)))
    return frames


def response_text(protocol, response):
    if protocol == "responses":
        return "".join(part.get("text", "") for item in response["output"]
                       if item["type"] == "message" for part in item["content"])
    if protocol == "chat_completions":
        return response["choices"][0]["message"].get("content") or ""
    return "".join(item.get("text", "") for item in response["content"] if item["type"] == "text")


def tool_schema(protocol):
    function = {"name": "shell", "description": "Return an acknowledgement",
                "parameters": {"type": "object", "properties": {"task": {"type": "string"}},
                               "required": ["task"]}}
    if protocol == "responses":
        return [{"type": "function", **function}]
    if protocol == "chat_completions":
        return [{"type": "function", "function": function}]
    return [{"name": "shell", "description": function["description"],
             "input_schema": function["parameters"]}]


def dsml(task="run"):
    return ('<｜DSML｜function_calls><｜DSML｜invoke name="shell">'
            '<｜DSML｜parameter name="task" string="true">' + task +
            '</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>')


class FakeTokenEngine:
    """A deterministic raw vLLM EngineClient-shaped source of token IDs."""
    def __init__(self, tokenizer, path):
        self.tokenizer = tokenizer
        self.model_config = SimpleNamespace(model=str(path.parent), max_model_len=524288,
            hf_config=SimpleNamespace(eos_token_id=tokenizer.encode(recipe.SpecialToken.Eos.default_str())[0]))
        self.renderer = SimpleNamespace(render_cmpl_async=self.render)
        self.plans = deque()
        self.calls = []
        self.rendered = []
        self.aborted = []
        self.closed = []
        self.blocked = asyncio.Event()
        self.renderer_error = None

    async def render(self, prompts):
        if self.renderer_error:
            raise RuntimeError(self.renderer_error)
        self.rendered.extend(prompts)
        return prompts

    def check_admission(self, **kwargs):
        if kwargs != {"n": 1}:
            raise AssertionError("Unexpected backend admission options")

    def plan(self, output="你好 world", *, mode="complete", finish="stop"):
        self.plans.append((output, mode, finish))

    def generate(self, prompt, sampling, request_id, **kwargs):
        output, mode, finish = self.plans.popleft() if self.plans else ("你好 world", "complete", "stop")
        self.calls.append({"request_id": request_id, "prompt": prompt,
                           "sampling": sampling, "kwargs": kwargs})

        def frame(ids, finish_reason=None):
            return SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], num_cached_tokens=0,
                outputs=[SimpleNamespace(index=0, token_ids=ids, finish_reason=finish_reason,
                                         stop_reason=None, logprobs=None)],
                finished=finish_reason is not None)

        async def produce():
            try:
                token_ids = self.tokenizer.encode(output)
                for offset in range(0, len(token_ids), 3):
                    yield frame(token_ids[offset:offset + 3])
                if mode == "exception":
                    raise RuntimeError("private backend detail must not reach the client")
                if mode == "eof":
                    return
                if mode == "block":
                    self.blocked.set()
                    await asyncio.Event().wait()
                    return
                # vLLM DELTA retains the sampled EOS ID on normal stop.
                yield frame([self.model_config.hf_config.eos_token_id] if finish == "stop" else [], finish)
            finally:
                self.closed.append(request_id)
        return produce()

    async def abort(self, request_id):
        self.aborted.append(request_id)


class HttpTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer as HFTokenizer
        cls.tokenizer_file = tokenizer_path()
        cls.tokenizer = recipe.Tokenizer.from_file(str(cls.tokenizer_file))
        cls.hf_tokenizer = HFTokenizer.from_file(str(cls.tokenizer_file))

    async def asyncSetUp(self):
        self.engine = FakeTokenEngine(self.tokenizer, self.tokenizer_file)
        self.backend = VllmBackend(self.engine, tokenizer_path=self.tokenizer_file)
        self.app = FastAPI()

        @self.app.get("/health")
        async def health():
            return {"status": "native-health"}

        @self.app.post("/v1/responses")
        async def stale_responses():
            raise AssertionError("The old route must be replaced")

        @self.app.get("/v1/responses/{response_id}")
        async def stale_stateful_responses(response_id):
            raise AssertionError("The unsupported native stateful route must be removed")

        server.install_routes(self.app, self.backend)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app, raise_app_exceptions=False),
                                        base_url="http://adapter.test")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_models_compatibility_and_native_route(self):
        response = await self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()["data"]], [MODEL])
        self.assertEqual((await self.client.get("/health")).json(), {"status": "native-health"})
        info = (await self.client.get("/deepinfer/compatibility")).json()
        self.assertEqual(info["recipe_version"], recipe.__version__)
        self.assertFalse(info["responses_stateful"])
        self.assertEqual(info["tool_execution"], "client")
        self.assertEqual((await self.client.get("/v1/responses/old")).status_code, 404)

    async def test_three_protocol_json_and_truthful_usage(self):
        for protocol, path in PATHS.items():
            with self.subTest(protocol=protocol):
                response = await self.client.post(path, json=body(protocol))
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(response_text(protocol, data), "你好 world")
                self.assertNotIn(recipe.SpecialToken.Eos.default_str(), response.text)
                self.assertEqual(data["model"], MODEL)
                tokens = len(self.tokenizer.encode("你好 world")) + 1
                usage = data["usage"]
                output_key = "completion_tokens" if protocol == "chat_completions" else "output_tokens"
                input_key = "prompt_tokens" if protocol == "chat_completions" else "input_tokens"
                self.assertEqual(usage[output_key], tokens)
                self.assertEqual(usage[input_key], len(self.engine.calls[-1]["prompt"]["prompt_token_ids"]))
                self.assertEqual(self.engine.calls[-1]["sampling"].max_tokens, 512)
        self.assertEqual(self.engine.aborted, [])
        self.assertEqual(len(self.engine.closed), 3)

    async def test_three_protocol_sse_framing_and_terminal(self):
        for protocol, path in PATHS.items():
            with self.subTest(protocol=protocol):
                response = await self.client.post(path, json=body(protocol, stream=True))
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn("text/event-stream", response.headers["content-type"])
                frames = parse_sse(response)
                data = [payload for _, payload in frames if payload != "[DONE]"]
                if protocol == "responses":
                    self.assertTrue(all(name == payload["type"] for name, payload in frames))
                    self.assertEqual(data[-1]["type"], "response.completed")
                    self.assertEqual(response_text(protocol, data[-1]["response"]), "你好 world")
                    self.assertEqual([item["sequence_number"] for item in data], list(range(len(data))))
                    done = [item["item"] for item in data if item["type"] == "response.output_item.done"]
                    self.assertEqual(done, data[-1]["response"]["output"])
                elif protocol == "chat_completions":
                    self.assertEqual(sum(payload == "[DONE]" for _, payload in frames), 1)
                    self.assertTrue(all(name is None for name, _ in frames))
                    text = "".join(choice.get("delta", {}).get("content") or ""
                                   for item in data for choice in item.get("choices", []))
                    self.assertEqual(text, "你好 world")
                else:
                    self.assertEqual(data[-1]["type"], "message_stop")
                    text = "".join(item["delta"].get("text", "") for item in data
                                   if item["type"] == "content_block_delta")
                    self.assertEqual(text, "你好 world")
                    self.assertTrue(all(name == payload["type"] for name, payload in frames))

    async def test_three_protocol_tool_current_response_roundtrip(self):
        for protocol, path in PATHS.items():
            with self.subTest(protocol=protocol):
                tools = tool_schema(protocol)
                self.engine.plan(dsml())
                first = await self.client.post(path, json=body(protocol, tools=tools))
                self.assertEqual(first.status_code, 200, first.text)
                current = first.json()
                if protocol == "responses":
                    call = next(item for item in current["output"] if item["type"] == "function_call")
                    self.assertEqual(json.loads(call["arguments"]), {"task": "run"})
                    continuation = body(protocol, tools=tools, input=[{"role": "user", "content": "Hello"},
                        *current["output"], {"type": "function_call_output", "call_id": call["call_id"], "output": "ACK-SHELL"}])
                elif protocol == "chat_completions":
                    message = current["choices"][0]["message"]
                    call = message["tool_calls"][0]
                    self.assertEqual(json.loads(call["function"]["arguments"]), {"task": "run"})
                    continuation = body(protocol, tools=tools, messages=[{"role": "user", "content": "Hello"}, message,
                        {"role": "tool", "tool_call_id": call["id"], "content": "ACK-SHELL"}])
                else:
                    call = next(item for item in current["content"] if item["type"] == "tool_use")
                    self.assertEqual(call["input"], {"task": "run"})
                    continuation = body(protocol, tools=tools, messages=[{"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": current["content"]}, {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": call["id"], "content": "ACK-SHELL"}]}])
                self.engine.plan("Acknowledged")
                second = await self.client.post(path, json=continuation)
                self.assertEqual(second.status_code, 200, second.text)
                self.assertEqual(response_text(protocol, second.json()), "Acknowledged")
                prompt_text = self.hf_tokenizer.decode(self.engine.calls[-1]["prompt"]["prompt_token_ids"], skip_special_tokens=False)
                self.assertIn("ACK-SHELL", prompt_text)
                self.assertIn("shell", prompt_text)
        self.assertEqual(len(self.engine.calls), 6)
        self.assertEqual(self.engine.aborted, [])

    async def test_responses_tool_sse_ids_and_argument_deltas(self):
        self.engine.plan(dsml("run UTF-8 文件"))
        response = await self.client.post(PATHS["responses"], json=body("responses", stream=True,
                                         tools=tool_schema("responses")))
        events = [payload for _, payload in parse_sse(response)]
        self.assertEqual(events[-1]["type"], "response.completed")
        call = events[-1]["response"]["output"][0]
        done = next(event for event in events if event["type"] == "response.function_call_arguments.done")
        self.assertEqual(done["item_id"], call["id"])
        self.assertEqual(done["arguments"], call["arguments"])
        self.assertEqual("".join(event["delta"] for event in events
                                if event["type"] == "response.function_call_arguments.delta"), call["arguments"])
        self.assertEqual(json.loads(call["arguments"]), {"task": "run UTF-8 文件"})
        opened = next(event["item"] for event in events if event["type"] == "response.output_item.added")
        self.assertEqual(opened["call_id"], call["call_id"])

    async def test_backend_exception_and_eof_fail_json_and_sse(self):
        for protocol, path in PATHS.items():
            for mode in ("exception", "eof"):
                for streaming in (False, True):
                    with self.subTest(protocol=protocol, mode=mode, streaming=streaming):
                        self.engine.plan("partial", mode=mode)
                        response = await self.client.post(path, json=body(protocol, stream=streaming))
                        self.assertEqual(response.status_code, 200 if streaming else 502, response.text)
                        self.assertNotIn("private backend detail", response.text)
                        if streaming:
                            events = [payload for _, payload in parse_sse(response)]
                            self.assertNotIn("[DONE]", events)
                            terminal = events[-1]
                        else:
                            terminal = response.json()
                        if protocol == "responses":
                            if streaming:
                                self.assertEqual(terminal["type"], "response.failed")
                                terminal = terminal["response"]
                            self.assertEqual(terminal["status"], "failed")
                            self.assertIsNone(terminal["usage"])
                        else:
                            self.assertIn("error", terminal)
                        self.assertIn(self.engine.calls[-1]["request_id"], self.engine.aborted)

    async def test_cancellation_aborts_raw_engine_and_closes_iterator(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.engine.blocked = asyncio.Event()
                self.engine.plan("partial", mode="block")
                task = asyncio.create_task(self.client.post(PATHS["responses"], json=body("responses", stream=streaming)))
                await asyncio.wait_for(self.engine.blocked.wait(), timeout=10)
                request_id = self.engine.calls[-1]["request_id"]
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertIn(request_id, self.engine.aborted)
                self.assertIn(request_id, self.engine.closed)

    async def test_nonstream_http_disconnect_aborts_without_external_task_cancel(self):
        self.engine.plan("partial", mode="block")
        request_body_finished = False

        async def disconnecting_app(scope, receive, send):
            async def disconnected_receive():
                nonlocal request_body_finished
                # Preserve real ASGI request-body delivery. Only subsequent
                # receives expose a peer disconnect after inference is live.
                if request_body_finished and self.engine.blocked.is_set():
                    return {"type": "http.disconnect"}
                event = await receive()
                if event["type"] == "http.request" and not event.get("more_body", False):
                    request_body_finished = True
                return event
            await self.app(scope, disconnected_receive, send)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=disconnecting_app),
                                     base_url="http://adapter.test") as client:
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(client.post(PATHS["responses"], json=body("responses")), timeout=10)
        self.assertTrue(self.engine.blocked.is_set())
        request_id = self.engine.calls[-1]["request_id"]
        self.assertEqual(self.engine.aborted.count(request_id), 1)
        self.assertIn(request_id, self.engine.closed)

    async def test_length_finish_is_incomplete_and_keeps_real_usage(self):
        for streaming in (False, True):
            self.engine.plan("partial", finish="length")
            response = await self.client.post(PATHS["responses"], json=body("responses", stream=streaming))
            self.assertEqual(response.status_code, 200, response.text)
            if streaming:
                terminal = parse_sse(response)[-1][1]
                self.assertEqual(terminal["type"], "response.incomplete")
                result = terminal["response"]
            else:
                result = response.json()
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["incomplete_details"], {"reason": "max_output_tokens"})
            self.assertEqual(result["usage"]["output_tokens"], len(self.tokenizer.encode("partial")))
        self.assertEqual(self.engine.aborted, [])

    async def test_body_limit_covers_streaming_request_without_content_length(self):
        async def content():
            yield b'{"model":"DeepSeek-Flash","input":"'
            yield b"x" * 128
            yield b'"}'
        with patch.object(server, "MAX_BODY_BYTES", 64):
            response = await self.client.post(PATHS["responses"], content=content(),
                                              headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 413, response.text)
        self.assertIn("error", response.json())
        self.assertEqual(self.engine.calls, [])

    async def test_tokenize_and_messages_count_use_prepared_prompt_without_generation(self):
        tokenized = await self.client.post("/tokenize", json={"model": MODEL, "prompt": "Count me"})
        self.assertEqual(tokenized.status_code, 200, tokenized.text)
        data = tokenized.json()
        self.assertEqual(data["count"], len(data["tokens"]))
        self.assertEqual(data["max_model_len"], 524288)
        for path in ("/v1/messages/count_tokens", "/messages/count_tokens"):
            counted = await self.client.post(path, json=body("messages"))
            self.assertEqual(counted.status_code, 200, counted.text)
            self.assertGreater(counted.json()["input_tokens"], 0)
        self.assertEqual(self.engine.calls, [])

    async def test_request_validation_and_stateless_rejection(self):
        cases = [("/v1/responses", body("responses", previous_response_id="old-response")),
                 ("/v1/responses", body("responses", store=True)),
                 ("/v1/responses", body("responses", n=2)),
                 ("/v1/responses", body("responses", model="not-a-model"))]
        for path, payload in cases:
            with self.subTest(payload=payload):
                response = await self.client.post(path, json=payload)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("error", response.json())
        for raw in ("[1]", "{invalid-json"):
            response = await self.client.post("/v1/responses", content=raw)
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.engine.calls, [])

    async def test_prepare_exception_returns_safe_service_error_on_all_post_routes(self):
        self.engine.renderer_error = "private renderer detail"
        cases = [(PATHS[protocol], body(protocol)) for protocol in PATHS]
        cases.extend([("/tokenize", {"model": MODEL, "prompt": "x"}),
                      ("/v1/messages/count_tokens", body("messages"))])
        for path, payload in cases:
            with self.subTest(path=path):
                response = await self.client.post(path, json=payload)
                self.assertEqual(response.status_code, 503, response.text)
                self.assertIn("error", response.json())
                self.assertNotIn("private renderer detail", response.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
