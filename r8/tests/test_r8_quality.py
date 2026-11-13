"""Exercise quality scoring and paired-report failure handling without inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r8_quality',ROOT/'scripts/quality_suite.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

class QualityTests(unittest.TestCase):
    def test_gsm_reference_answers_are_all_parseable(self):
        cases=list(q.gsm_cases(1319))
        self.assertEqual(len(cases),2638)
        self.assertTrue(all(c['answer'] is not None for c in cases))
        self.assertEqual(q.number('reasoning\n#### 1,234.00'),'1234')
        self.assertIsNone(q.number('I think it is 1234'))

    def test_isolation_cases_have_unique_nonces_and_requests(self):
        cases=list(q.isolation_cases())
        self.assertEqual(len(cases),64)
        self.assertEqual(len({c['answer'] for c in cases}),64)
        self.assertEqual(cases,list(q.isolation_cases()))

    def test_corruption_and_budget_truncation_are_not_scored_as_correct(self):
        c=next(q.isolation_cases())
        value={'choices':[{'message':{'content':c['answer']},'finish_reason':'length'}],
               'usage':{'prompt_tokens':20,'completion_tokens':64}}
        with patch.object(q,'request',return_value=value):
            r=q.evaluate('http://example.invalid/v1',c)
        self.assertEqual(r['status'],'FAIL')
        self.assertTrue(r['correct'])
        value['choices'][0].update(finish_reason='stop',message={'content':c['answer']+'</parameter>'})
        with patch.object(q,'request',return_value=value):
            r=q.evaluate('http://example.invalid/v1',c)
        self.assertFalse(r['protocol_clean'])

    def test_long_context_rejects_fabricated_token_count(self):
        c=dict(id='long/off/523776',kind='long',mode='off',body={},answer={'a':'b'},
               prompt_tokens=523770,target_tokens=523776)
        value={'choices':[{'message':{'content':'{"a":"b"}'},'finish_reason':'stop'}],
               'usage':{'prompt_tokens':2000,'completion_tokens':8}}
        with patch.object(q,'request',return_value=value):r=q.evaluate('http://example.invalid',c)
        self.assertEqual(r['status'],'ERROR')

    def test_paired_compare_refuses_missing_or_different_requests(self):
        r=dict(schema=q.SCHEMA,dataset_sha256='a',gsm_limit=1,sampling={},status='MEASURED',
               long_context_512k='PASS',cases=[dict(id='gsm8k/off/0',kind='gsm8k',mode='off',request_sha256='x',status='PASS')])
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.json',Path(d)/'b.json'
            a.write_text(json.dumps(r));b.write_text(json.dumps(r))
            self.assertEqual(q.compare(a,b)['status'],'REVIEW_REQUIRED')
            r['cases'][0]['request_sha256']='different';b.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'requests differ'):q.compare(a,b)

    def test_official_encoder_accepts_both_modes_and_near_512k_prompt(self):
        count=q.prompt_counter()
        for mode in ('off','high'):
            self.assertGreater(count(q.body_for('Solve 17+25.',mode,4096)),0)
        case=q.long_case(523776,count)
        self.assertTrue(523648<=case['prompt_tokens']<=523776)
        self.assertLessEqual(case['prompt_tokens']+case['body']['max_tokens'],524288)

    def test_pr_snapshot_and_stable_protocol_boundary(self):
        m=json.loads((ROOT/'source/r8-pr66497/manifest.json').read_text())
        self.assertEqual(m['head'],'d423120cdc36d4cf8456f43ddfcc654b0c69c5ea')
        self.assertTrue(all(r['path'].endswith('.py') for r in m['files']))
        for r in m['files']:
            self.assertEqual(hashlib.sha256((ROOT/'source/r8-pr66497/files'/r['path']).read_bytes()).hexdigest(),r['sha256'])
        c=json.loads((ROOT/'configs/runtime.json').read_text())
        self.assertEqual(c['max_model_len'],524288)
        self.assertEqual(c['gpu_memory_utilization'],.94)
        self.assertFalse(c['enable_expert_parallel'])
        self.assertFalse(c['swa_bounded_replay'])
        self.assertFalse(c['kv_sharing_fast_prefill'])
        self.assertFalse(c['compilation_config']['cudagraph_decoder_replay'])
        self.assertEqual(c['speculative_config']['rejection_sample_method'],'standard')
        self.assertFalse(c['speculative_config']['use_local_argmax_reduction'])

    def test_dspark_uses_supported_draft_validation(self):
        from types import SimpleNamespace
        from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
        from vllm.models.deepseek_v41.nvidia.dspark import DSparkDeepseekV4ForCausalLM
        self.assertFalse(hasattr(DSparkDeepseekV4ForCausalLM,'get_top_tokens'))
        model=type('DSparkDeepseekV4ForCausalLM',(),{})()
        draft=SimpleNamespace(use_local_argmax_reduction=True,model=model,
                              speculative_config=SimpleNamespace(draft_sample_method='greedy'))
        validate=DraftModelSpeculator._validate_local_argmax_reduction
        with self.assertRaisesRegex(ValueError,'does not implement get_top_tokens'):
            validate(draft)
        draft.use_local_argmax_reduction=False
        validate(draft)

if __name__=='__main__':unittest.main()
