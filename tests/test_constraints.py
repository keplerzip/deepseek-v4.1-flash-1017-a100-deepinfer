"""Compile actual xgrammar grammars and verify schema and continuation behavior."""

import json
import unittest

import deepseek_recipe as recipe

from adapter.constraints import structured_outputs
from adapter.protocol import ProtocolStream, prepare
from test_protocol import tokenizer_path

try:
    import xgrammar as xgr
    from transformers import PreTrainedTokenizerFast
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams
except ImportError:
    xgr = None


@unittest.skipIf(xgr is None, "Run constrained-decoding tests in the pinned R2 image")
class ConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = recipe.Tokenizer.from_file(tokenizer_path())
        cls.eos_id = cls.tokenizer.encode(recipe.SpecialToken.Eos.default_str())[0]
        hf = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path())
        info = xgr.TokenizerInfo.from_huggingface(hf, stop_token_ids=[cls.eos_id])
        cls.compiler = xgr.GrammarCompiler(info, max_threads=2)

    def prepared(self, *, strict=True, choice="auto", schema=None):
        schema = schema or {"type": "object", "properties": {
            "task": {"type": "string"}, "count": {"type": "integer", "minimum": 1}},
            "required": ["task", "count"], "additionalProperties": False}
        return prepare("responses", {"model": "DeepSeek-Flash", "input": "Do work",
            "reasoning": {"effort": "none"}, "tool_choice": choice,
            "tools": [{"type": "function", "name": "shell", "strict": strict, "parameters": schema}]},
            self.tokenizer)

    def grammar(self, prepared):
        params = structured_outputs(prepared)
        self.assertIsInstance(params, StructuredOutputsParams)
        # Exercise actual R2 SamplingParams validation too.
        SamplingParams(max_tokens=20, structured_outputs=params)
        return self.compiler.compile_structural_tag(params.structural_tag)

    @staticmethod
    def continuation(task='"hello"', count="2", *, name="shell"):
        return (f'<｜DSML｜ invoke name="{name}">\n'
                f'<｜DSML｜ parameter name="task" string="false">{task}</｜DSML｜ parameter>\n'
                f'<｜DSML｜ parameter name="count" string="false">{count}</｜DSML｜ parameter>\n'
                '</｜DSML｜ invoke>\n</｜DSML｜ calls>')

    def assert_accepts(self, compiled, sample):
        matcher = xgr.GrammarMatcher(compiled)
        self.assertTrue(matcher.accept_string(sample), sample)
        self.assertTrue(matcher.accept_token(self.eos_id), "EOS must be enabled after a complete output")
        self.assertTrue(matcher.is_terminated())

    def assert_rejects(self, compiled, sample):
        matcher = xgr.GrammarMatcher(compiled)
        accepted = matcher.accept_string(sample)
        if accepted:
            self.assertFalse(matcher.accept_token(self.eos_id), "Incomplete output unexpectedly permits EOS")

    def test_strict_auto_compiles_and_enforces_parameter_schema(self):
        compiled = self.grammar(self.prepared())
        prefix = "<｜DSML｜ calls>\n"
        self.assert_accepts(compiled, prefix + self.continuation())
        self.assert_rejects(compiled, prefix + self.continuation(task="123"))
        self.assert_rejects(compiled, prefix + self.continuation(count='"wrong"'))
        self.assert_rejects(compiled, prefix + self.continuation(count="0"))
        self.assert_rejects(compiled, prefix + self.continuation(name="unknown"))
        self.assert_rejects(compiled, prefix + self.continuation().replace('name="count"', 'name="extra"'))
        self.assert_rejects(compiled, prefix + self.continuation().split('<｜DSML｜ parameter name="count"')[0])

    def test_strict_required_continues_existing_prompt_prefix(self):
        prepared = self.prepared(choice="required")
        self.assertTrue(prepared.rendered.prompt.endswith("\n\n<｜DSML｜ calls>\n"))
        compiled = self.grammar(prepared)
        self.assert_accepts(compiled, self.continuation())
        self.assert_rejects(compiled, "\n\n<｜DSML｜ calls>\n" + self.continuation())
        self.assert_rejects(compiled, self.continuation(count='"wrong"'))

    def test_named_tool_uses_required_continuation(self):
        prepared = self.prepared(choice={"type": "function", "name": "shell"})
        self.assert_accepts(self.grammar(prepared), self.continuation())

    def test_non_strict_required_uses_pinned_registry_continuation(self):
        prepared = self.prepared(strict=False, choice="required")
        self.assert_accepts(self.grammar(prepared), self.continuation())

    def test_false_string_json_quoting_roundtrips_through_recipe(self):
        prepared = self.prepared(choice="required")
        sample = self.continuation(task=json.dumps('hello "世界"\nnext', ensure_ascii=False))
        self.assert_accepts(self.grammar(prepared), sample)
        stream = ProtocolStream(prepared)
        stream.push(recipe.InferenceChunk.ready(prompt_usage=recipe.PromptUsage(prompt_tokens=prepared.prompt_tokens)))
        stream.push(recipe.InferenceChunk.tokens(self.tokenizer.encode(sample)))
        stream.push(recipe.InferenceChunk.finish(recipe.InferenceFinishReason.Stop))
        call = stream.response["output"][0]
        self.assertEqual(json.loads(call["arguments"]), {"task": 'hello "世界"\nnext', "count": 2})

    def test_nested_schema_is_enforced(self):
        schema = {"type": "object", "properties": {"task": {
            "type": "object", "properties": {"mode": {"type": "string", "enum": ["safe"]}},
            "required": ["mode"], "additionalProperties": False}},
            "required": ["task"], "additionalProperties": False}
        compiled = self.grammar(self.prepared(choice="required", schema=schema))
        def sample(value):
            return ('<｜DSML｜ invoke name="shell">\n<｜DSML｜ parameter name="task" string="false">'
                    + value + '</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        self.assert_accepts(compiled, sample('{"mode":"safe"}'))
        self.assert_rejects(compiled, sample('{"mode":"unsafe"}'))
        self.assert_rejects(compiled, sample('{"mode":"safe","extra":1}'))

    def test_json_schema_response_compiles_and_rejects_wrong_types(self):
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}},
                  "required": ["answer"], "additionalProperties": False}
        prepared = prepare("responses", {"model": "DeepSeek-Flash", "input": "Return JSON",
            "reasoning": {"effort": "none"}, "text": {"format": {
                "type": "json_schema", "name": "answer", "schema": schema, "strict": True}}}, self.tokenizer)
        params = structured_outputs(prepared)
        self.assertIsInstance(params, StructuredOutputsParams)
        compiled = self.compiler.compile_json_schema(params.json)
        self.assert_accepts(compiled, '{"answer":1}')
        self.assert_rejects(compiled, '{"answer":"wrong"}')
        self.assert_rejects(compiled, '{"answer":1,"extra":2}')


if __name__ == "__main__":
    unittest.main()
