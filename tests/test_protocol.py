"""CPU-only contract tests against the pinned official recipe wheel."""

import json
import os
from pathlib import Path
import unittest

import deepseek_recipe as recipe

from adapter.protocol import FLASH_ALIASES, SPECIAL_TOKENS, ProtocolError, ProtocolStream, prepare


ROOT = Path(__file__).resolve().parents[1]


def tokenizer_path():
    candidates = [
        os.environ.get("DEEPINFER_TOKENIZER", ""),
        ROOT / "research/model-metadata/tokenizer.json",
        ROOT / "vendor/tokenizer/tokenizer.json",
        "/models/DeepSeek-V4.1-Flash-1017/tokenizer.json",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("Set DEEPINFER_TOKENIZER to the matching tokenizer.json")


def function(name="shell"):
    return {"type": "function", "name": name, "parameters": {
        "type": "object", "properties": {"task": {"type": "string"}},
        "required": ["task"],
    }}


def dsml(name, key, value, *, string=True):
    literal = value if string else json.dumps(value)
    return (f'<｜DSML｜function_calls><｜DSML｜invoke name="{name}">'
            f'<｜DSML｜parameter name="{key}" string="{str(string).lower()}">{literal}'
            '</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>')


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = recipe.Tokenizer.from_file(tokenizer_path())

    def prepared(self, protocol="responses", **updates):
        body = {"model": "DeepSeek-Flash"}
        if protocol == "responses":
            body.update(input="Hello", reasoning={"effort": "none"})
        else:
            body.update(messages=[{"role": "user", "content": "Hello"}],
                        thinking={"type": "disabled"})
        if protocol == "messages":
            body["max_tokens"] = 1024
        body.update(updates)
        return prepare(protocol, body, self.tokenizer)

    def ready(self, stream):
        return stream.push(recipe.InferenceChunk.ready(prompt_usage=recipe.PromptUsage(
            prompt_tokens=stream.prepared.prompt_tokens,
        )))

    def generate(self, prepared, output):
        stream = ProtocolStream(prepared, response_id="resp_test")
        events = self.ready(stream)
        # Split inside DSML tokens and UTF-8 characters through token-ID chunks.
        ids = self.tokenizer.encode(output)
        for offset in range(0, len(ids), 3):
            events.extend(stream.push(recipe.InferenceChunk.tokens(ids[offset:offset + 3])))
        events.extend(stream.push(recipe.InferenceChunk.finish(recipe.InferenceFinishReason.Stop)))
        return stream, events

    def test_all_aliases_and_native_token_encoding(self):
        for alias in FLASH_ALIASES:
            with self.subTest(alias=alias):
                prepared = self.prepared(model=alias)
                self.assertEqual(prepared.model, alias)
                expected = recipe.DeepseekV41Encoding().with_special_tokens(SPECIAL_TOKENS).with_tokenizer(self.tokenizer).encode(
                    prepared.converted.conversation)
                self.assertEqual(prepared.prompt_token_ids, expected)
                self.assertEqual(prepared.max_tokens, 8192)

    def test_user_control_token_text_is_not_a_control_token(self):
        prepared = self.prepared(input="literal <｜User｜> </think>")
        user_id = self.tokenizer.encode("<｜User｜>")[0]
        self.assertEqual(prepared.prompt_token_ids.count(user_id), 1)

    def test_1017_image_special_token_and_accounting(self):
        body = {"model": "DeepSeek-Flash", "reasoning": {"effort": "none"},
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": "literal <｜deepseek_image｜>"},
                    {"type": "input_image", "image_url": "https://example.com/image.png"},
                ]}]}
        prepared = prepare("responses", body, self.tokenizer, image_token_adjustment=10)
        self.assertEqual(prepared.prompt_token_ids.count(recipe.IMAGE_SPECIAL_TOKEN_ID), 1)
        self.assertEqual(len(prepared.image_sources), 1)
        self.assertEqual(prepared.prompt_tokens, len(prepared.prompt_token_ids) + 10)

    def test_three_protocols_finish_and_accumulate(self):
        for protocol in ("responses", "chat_completions", "messages"):
            with self.subTest(protocol=protocol):
                stream, events = self.generate(self.prepared(protocol), "你好 world")
                self.assertTrue(stream.finished)
                self.assertTrue(stream.successful)
                self.assertEqual(stream.finish(), [])
                response = stream.response
                if protocol == "responses":
                    self.assertEqual(events[-1]["type"], "response.completed")
                    self.assertEqual(events[-1]["response"], response)
                    self.assertEqual(response["output"][0]["content"][0]["text"], "你好 world")
                    self.assertEqual(response["usage"]["input_tokens"], stream.prepared.prompt_tokens)
                    self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
                elif protocol == "chat_completions":
                    self.assertEqual(response["choices"][0]["message"]["content"], "你好 world")
                else:
                    self.assertEqual(response["content"][0]["text"], "你好 world")
                frames = stream.sse(events)
                self.assertEqual(any("[DONE]" in frame for frame in frames), protocol == "chat_completions")
                self.assertEqual(stream.sse([]), [])

    def test_namespace_and_call_id_pairing_roundtrip(self):
        prepared = self.prepared(tools=[{"type": "namespace", "name": "functions",
                                        "tools": [function()]}])
        stream, events = self.generate(prepared, dsml("functions::shell", "task", "run"))
        call = stream.response["output"][0]
        self.assertEqual((call["namespace"], call["name"]), ("functions", "shell"))
        self.assertEqual(json.loads(call["arguments"]), {"task": "run"})
        self.assertTrue(call["call_id"].startswith("call_00_"))
        history = [{"role": "user", "content": "Hello"}, call,
                   {"type": "function_call_output", "call_id": call["call_id"], "output": "ok"}]
        next_prepared = self.prepared(input=history, tools=[{"type": "namespace", "name": "functions", "tools": [function()]}])
        self.assertIn("run", next_prepared.rendered.prompt)

    def test_apply_patch_custom_stream_and_roundtrip(self):
        patch = "*** Begin Patch\n*** Add File: 文件.txt\n+hello\n*** End Patch"
        stream, events = self.generate(self.prepared(tools=[{"type": "custom", "name": "apply_patch"}]),
                                       dsml("apply_patch", "input", patch))
        call = stream.response["output"][0]
        self.assertEqual(call["type"], "custom_tool_call")
        self.assertEqual(call["input"], patch)
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "response.custom_tool_call_input.delta"), patch)
        self.prepared(input=[{"role": "user", "content": "edit"}, call,
                             {"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "ok"}],
                      tools=[{"type": "custom", "name": "apply_patch"}])

    def test_client_tool_search_and_discovered_tool_roundtrip(self):
        tools = [{"type": "tool_search", "execution": "client", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]
        stream, _ = self.generate(self.prepared(tools=tools), dsml("tool_search", "query", "shell"))
        call = stream.response["output"][0]
        self.assertEqual(call["type"], "tool_search_call")
        self.assertEqual(call["arguments"], {"query": "shell"})
        prepared = self.prepared(tools=tools, input=[{"role": "user", "content": "Find shell"}, call,
            {"type": "tool_search_output", "call_id": call["call_id"], "tools": [function()]}])
        self.assertIn("shell", prepared.rendered.prompt)

    def test_thinking_roundtrip_is_preserved_and_missing_reasoning_rejected(self):
        stream, _ = self.generate(self.prepared(tools=[function()], reasoning={"effort": "high"}),
                                   "考虑</think>" + dsml("shell", "task", "run"))
        output = stream.response["output"]
        self.assertEqual(output[0]["type"], "reasoning")
        self.assertGreater(stream.response["usage"]["output_tokens_details"]["reasoning_tokens"], 0)
        call = output[-1]
        tail = {"type": "function_call_output", "call_id": call["call_id"], "output": "ok"}
        self.prepared(input=[{"role": "user", "content": "run"}, *output, tail],
                      tools=[function()], reasoning={"effort": "high"})
        with self.assertRaises(recipe.ConversionError):
            self.prepared(input=[{"role": "user", "content": "run"}, call, tail],
                          tools=[function()], reasoning={"effort": "high"})

    def test_wrapper_arguments_are_not_silently_rewritten(self):
        stream, _ = self.generate(self.prepared(tools=[function()]),
                                   dsml("shell", "arguments", '{"task":"run"}'))
        self.assertEqual(json.loads(stream.response["output"][0]["arguments"]),
                         {"arguments": '{"task":"run"}'})

    def test_constraints_are_exposed_without_claiming_enforcement(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}},
                  "required": ["answer"], "additionalProperties": False}
        prepared = self.prepared(text={"format": {"type": "json_schema", "name": "answer", "schema": schema, "strict": True}})
        self.assertEqual(prepared.inference_options.output_constraint.kind, "json")
        self.assertEqual(json.loads(prepared.inference_options.output_constraint.schema), schema)
        strict_tool = function()
        strict_tool["strict"] = True
        prepared = self.prepared(tools=[strict_tool])
        self.assertEqual(prepared.inference_options.output_constraint.kind, "tool_call")

    def test_eof_and_backend_exception_never_complete(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                stream = ProtocolStream(self.prepared())
                events = self.ready(stream)
                if partial:
                    events += stream.push(recipe.InferenceChunk.tokens(self.tokenizer.encode("partial")))
                events += stream.finish()
                self.assertEqual(events[-1]["type"], "response.failed")
                self.assertFalse(any(e["type"] == "response.completed" for e in events))
                self.assertEqual(stream.response, events[-1]["response"])
                self.assertIsNone(stream.response["usage"])
                self.assertFalse(stream.successful)
                self.assertNotIn("[DONE]", "".join(stream.sse(events)))
        stream = ProtocolStream(self.prepared())
        events = stream.fail("backend unavailable")
        self.assertEqual(events[0]["response"]["error"]["code"], "backend_error")

    def test_length_limit_is_incomplete(self):
        stream = ProtocolStream(self.prepared(max_output_tokens=1))
        self.ready(stream)
        stream.push(recipe.InferenceChunk.tokens(self.tokenizer.encode("Hi")))
        events = stream.push(recipe.InferenceChunk.finish(recipe.InferenceFinishReason.Length))
        self.assertEqual(events[-1]["type"], "response.incomplete")
        self.assertEqual(stream.response["incomplete_details"], {"reason": "max_output_tokens"})

    def test_partial_tool_args_cannot_be_completed_on_stop(self):
        prepared = self.prepared(tools=[function()])
        stream = ProtocolStream(prepared)
        events = self.ready(stream)
        partial = '<｜DSML｜function_calls><｜DSML｜invoke name="shell"><｜DSML｜parameter name="task" string="true">partial'
        events += stream.push(recipe.InferenceChunk.tokens(self.tokenizer.encode(partial)))
        events += stream.push(recipe.InferenceChunk.finish(recipe.InferenceFinishReason.Stop))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertFalse(any(e["type"] == "response.completed" for e in events))
        self.assertEqual(stream.response["output"][0]["status"], "incomplete")

    def test_backend_budget_and_usage_contract(self):
        stream = ProtocolStream(self.prepared(max_output_tokens=1))
        self.ready(stream)
        events = stream.push(recipe.InferenceChunk.tokens(self.tokenizer.encode("many many many tokens")))
        self.assertEqual(events[-1]["type"], "response.failed")
        stream = ProtocolStream(self.prepared())
        events = stream.push(recipe.InferenceChunk.ready())
        self.assertEqual(events[-1]["type"], "response.failed")
        stream = ProtocolStream(self.prepared())
        events = stream.push(recipe.InferenceChunk.text("unaccounted"))
        self.assertEqual(events[-1]["type"], "response.failed")

    def test_native_processor_error_is_a_failed_terminal(self):
        stream = ProtocolStream(self.prepared())
        self.ready(stream)
        events = stream.push(recipe.InferenceChunk.tokens([1], logprobs=[]))
        self.assertEqual(events[-1]["type"], "response.failed")

    def test_official_stop_sequence_is_an_authoritative_termination(self):
        stream = ProtocolStream(self.prepared("chat_completions", stop=["STOP"]))
        self.ready(stream)
        events = stream.push(recipe.InferenceChunk.tokens(self.tokenizer.encode("helloSTOPafter")))
        self.assertTrue(stream.finished)
        self.assertTrue(stream.successful)
        self.assertEqual(stream.response["choices"][0]["message"]["content"], "hello")
        self.assertEqual(stream.response["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("error", events[-1])
        self.assertEqual(stream.finish(), [])

    def test_request_budget_and_stateless_rejections(self):
        for field, value in (("previous_response_id", "old"), ("conversation", "old"), ("store", True)):
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                self.prepared(**{field: value})
        with self.assertRaises(ProtocolError):
            prepare("responses", {"model": "DeepSeek-Flash", "input": "x"}, self.tokenizer, context_limit=1)
        with self.assertRaises(recipe.ConversionError):
            self.prepared(max_output_tokens=999999)


if __name__ == "__main__":
    unittest.main()
