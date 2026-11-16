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


