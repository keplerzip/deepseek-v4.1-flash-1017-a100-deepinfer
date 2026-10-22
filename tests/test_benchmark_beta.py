"""CPU regression for the actual recipe/tokenizer and benchmark HTTP client."""
import asyncio
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import deepseek_recipe as recipe
import httpx

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('benchmark_beta_under_test',ROOT/'scripts/benchmark_beta.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def test_real_checkpoint_filler_counts_include_512k_target(self):
        path=ROOT/'vendor/tokenizer/tokenizer.json'
        tokenizer=recipe.Tokenizer.from_file(str(path))
        for count in (1,8192,520192):
            with self.subTest(count=count):
                filler=bench.build_filler(path,count)
                self.assertEqual(len(tokenizer.encode(filler)),count)

    async def exercise(self,usage_tokens=64,status_code=200,terminal=True):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(tokenizer=ROOT/'vendor/tokenizer/tokenizer.json',prompt_tokens=16,
                concurrency=2,requests=2,uds=None,url='http://test',timeout=10,
                output_tokens=128,report=Path(tmp)/'report.json')
            sent=[]
            def handler(request):
                if request.url.path=='/deepinfer/compatibility':
                    return httpx.Response(200,json={'protocol':'official-recipe-server-py'})
                self.assertEqual(request.url.path,'/v1/responses')
                payload=json.loads(request.content);sent.append(payload)
                marker=re.search(r'BETA_NEEDLE_\d+_9f3a',payload['input']).group()
                events=[{'type':'response.output_text.delta','delta':marker}]
                if terminal:
                    events.append({'type':'response.completed','response':{'status':'completed',
                        'usage':{'input_tokens':usage_tokens,'output_tokens':8}}})
                data=''.join('data: '+json.dumps(event)+'\n\n' for event in events)
                return httpx.Response(status_code,text=data,headers={'Content-Type':'text/event-stream'})
            client_class=httpx.AsyncClient
            def client(**kwargs):
                kwargs['transport']=httpx.MockTransport(handler)
                return client_class(**kwargs)
            with patch.object(bench.httpx,'AsyncClient',side_effect=client),contextlib.redirect_stdout(io.StringIO()):
                result=await bench.run(args)
            return result,json.loads(args.report.read_text()),sent

    async def test_real_benchmark_client_sse_and_usage_report(self):
        code,report,sent=await self.exercise()
        self.assertEqual(code,0)
        self.assertEqual(len(sent),2)
        self.assertTrue(report['all_requests_finished'])
        self.assertTrue(report['all_needles_found'])
        self.assertEqual(report['actual_filler_tokens'],16)
        self.assertTrue(all(row['usage']['input_tokens']==64 for row in report['requests']))
        self.assertTrue(all(row['ttft_seconds'] is not None for row in report['requests']))

    async def test_short_actual_input_usage_fails_acceptance(self):
        code,report,_=await self.exercise(usage_tokens=15)
        self.assertEqual(code,1)
        self.assertFalse(report['all_requests_finished'])
        self.assertTrue(all(row['status']=='ERROR' for row in report['requests']))

    async def test_http_failure_and_missing_terminal_fail_acceptance(self):
        for options in ({'status_code':500},{'terminal':False}):
            with self.subTest(options=options):
                code,report,_=await self.exercise(**options)
                self.assertEqual(code,1)
                self.assertFalse(report['all_requests_finished'])
