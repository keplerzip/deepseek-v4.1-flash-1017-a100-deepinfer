"""CPU contracts for per-request raw native-ID accounting; no GPU inference."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapter.native_token_usage import NativeTokenUsage


class Tokenizer:
    def encode(self, text):
        return {'</think>':[128822], '｜DSML｜':[128840]}.get(text,[99])


def options(stage='ReasoningStart', required=False, **kwargs):
    return SimpleNamespace(thinking_initial_stage=stage, starts_in_tool_calls=required, **kwargs)


class NativeUsageTests(unittest.TestCase):
    def counter(self, stage='ReasoningStart', required=False, tokenizer=None):
        return NativeTokenUsage().configure(tokenizer or Tokenizer(), options(stage,required))

    def test_defaults_and_configuration_required(self):
        u=NativeTokenUsage()
        self.assertFalse(u.configured);self.assertFalse(u.engine_finished)
        self.assertEqual(u.total,0);self.assertIsNone(u.reasoning)
        with self.assertRaises(RuntimeError):u.ingest_tokens([1])

    def test_original_invalid_unicode_ids_count_as_two_native_tokens(self):
        u=self.counter();u.ingest_tokens([9986,35])
        self.assertEqual((u.total,u.reasoning),(2,2))

    def test_end_across_yields_and_repeated_markers_do_not_restart(self):
        u=self.counter();u.ingest_tokens([1,2]);u.ingest_tokens([128822]);u.ingest_tokens([128822,3,4])
        self.assertEqual((u.total,u.reasoning),(6,2));self.assertTrue(u.finished_phase)

    def test_thinking_start_plain_spelling_and_leading_newlines_are_raw_ids(self):
        u=self.counter();u.ingest_tokens([7,8,9,10,11])
        self.assertEqual((u.total,u.reasoning),(5,5))
        u.ingest_tokens([128822,9]);self.assertEqual(u.reasoning,5)

    def test_off_content_continuation_and_required_take_precedence(self):
        for stage,required in [(None,False),('ContentContinuation',False),('ReasoningStart',True),('ReasoningContinuation',True)]:
            with self.subTest(stage=stage,required=required):
                u=self.counter(stage,required);u.ingest_tokens([9986,35,128840])
                self.assertEqual((u.total,u.reasoning),(3,0))

    def test_reasoning_continuation_and_enum_name_or_string(self):
        for stage in [SimpleNamespace(name='ReasoningContinuation'),'ThinkingInitialStage.ReasoningStart']:
            u=self.counter(stage);u.ingest_tokens([1,128822,2])
            self.assertEqual((u.total,u.reasoning),(3,1))

    def test_dsml_before_end_is_unknown_and_stays_unknown(self):
        u=self.counter();u.ingest_tokens([1,128840,2]);u.ingest_tokens([128822,3])
        self.assertEqual(u.total,5);self.assertIsNone(u.reasoning)
        self.assertEqual(u.unknown_reason,'dsml_before_thinking_end')

    def test_eos_is_total_but_not_reasoning(self):
        u=self.counter();u.ingest_tokens([1,2,77,3],eos_ids=[77])
        self.assertEqual((u.total,u.reasoning),(4,2));self.assertEqual(u.phase,'eos')

    def test_end_or_dsml_mapping_failure_does_not_fabricate_reasoning(self):
        class Broken(Tokenizer):
            def encode(self,text):return [1,2] if text=='</think>' else super().encode(text)
        u=self.counter(tokenizer=Broken());u.ingest_tokens([1,2])
        self.assertEqual(u.total,2);self.assertIsNone(u.reasoning)
        u=self.counter(stage=None,tokenizer=Broken());u.ingest_tokens([1,2])
        self.assertEqual(u.reasoning,0)

    def test_unknown_stage_is_not_assumed_to_be_content(self):
        u=self.counter('FutureStage');u.ingest_tokens([1,2])
        self.assertIsNone(u.reasoning);self.assertEqual(u.total,2)

    def test_instances_are_isolated_and_finished_counter_cannot_reuse(self):
        a,b=self.counter(),self.counter();a.ingest_tokens([1]);b.ingest_tokens([2,3]);a.ingest_tokens([128822])
        self.assertEqual((a.total,a.reasoning,b.total,b.reasoning),(2,1,2,2))
        a.engine_finished=True
        with self.assertRaises(RuntimeError):a.ingest_tokens([4])
        with self.assertRaises(RuntimeError):a.configure(Tokenizer(),options())

    def test_invalid_id_input_is_transactional(self):
        u=self.counter()
        for ids in ([1,-1],[True],[1,'2']):
            with self.assertRaises(ValueError):u.ingest_tokens(ids)
            self.assertEqual((u.total,u.reasoning),(0,0))

    def test_custom_markers_are_resolved_from_options(self):
        class Custom(Tokenizer):
            def encode(self,text):return {'END!':[51],'DSML!':[52]}.get(text,[])
        p=options(special_tokens=SimpleNamespace(thinking_end='END!',dsml='DSML!'))
        u=NativeTokenUsage().configure(Custom(),p);u.ingest_tokens([1,51,2])
        self.assertEqual((u.total,u.reasoning),(3,1))

    def test_exact_bundled_native_enum_and_tokenizer(self):
        # Read the delivered wheel into a temporary directory; no install/network.
        wheel=next((ROOT/'vendor/wheels').glob('deepseek_recipe-0.2.0-*.whl'),None)
        if wheel is None:self.skipTest('Exact bundled recipe wheel is unavailable')
        with tempfile.TemporaryDirectory(prefix='native-usage-cpu-') as directory:
            zipfile.ZipFile(wheel).extractall(directory);sys.path.insert(0,directory)
            try:
                import deepseek_recipe as recipe
                tok=recipe.Tokenizer.from_file(str(ROOT/'vendor/tokenizer/tokenizer.json'))
                self.assertEqual(tok.encode('</think>'),[128822])
                self.assertEqual(len(tok.encode('｜DSML｜')),1)
                self.assertGreater(len(tok.encode('<｜DSML｜>')),1)
                for stage in (recipe.ThinkingInitialStage.ReasoningStart,recipe.ThinkingInitialStage.ReasoningContinuation):
                    p=recipe.ParsingOptions(thinking_initial_stage=stage)
                    u=NativeTokenUsage().configure(tok,p);u.ingest_tokens([9986,35]);u.ingest_tokens(tok.encode('</think>'))
                    self.assertEqual((u.total,u.reasoning),(3,2))
                p=recipe.ParsingOptions(thinking_initial_stage=recipe.ThinkingInitialStage.ContentContinuation)
                u=NativeTokenUsage().configure(tok,p);u.ingest_tokens([9986,35])
                self.assertEqual(u.reasoning,0)
            finally:sys.path.remove(directory)


if __name__=='__main__':unittest.main()
