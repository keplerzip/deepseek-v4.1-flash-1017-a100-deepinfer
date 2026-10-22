import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import kv_acceptance as kv

class KVAcceptanceTests(unittest.TestCase):
    def test_metric_parser_ignores_histograms_and_other_counters(self):
        text='''# HELP x
vllm:kv_offload_load_bytes_total{engine="0"} 100
vllm:kv_offload_load_bytes_total{engine="1"} 200
vllm:kv_offload_load_bytes_sum 9000
vllm:other_load_bytes_total 500
'''
        self.assertEqual(kv.metric_total(text,'vllm:kv_offload_load_bytes'),300)

    def exercise(self,reload_bytes,use_tool=False):
        calls=[];metric_reads=0
        def request(sock,method,path,body=None):
            nonlocal metric_reads
            calls.append((method,path))
            if path=='/metrics':
                metric_reads+=1
                store=0 if metric_reads==1 else 1024
                load=reload_bytes if metric_reads>=3 else 0
                return f'vllm:kv_offload_store_bytes_total {store}\nvllm:kv_offload_load_bytes_total {load}\n'.encode()
            if path.endswith('reset-local'):return b'{"success":true}'
            marker=body['input'].split('The exact secret is ',1)[1].split('.',1)[0]
            output=([{'type':'function_call','name':'echo_probe','arguments':json.dumps({'marker':marker})}] if body.get('tools') else [{'type':'message','content':[{'type':'output_text','text':marker}]}])
            return json.dumps({'status':'completed','output':output,
                               'usage':{'input_tokens_details':{'cached_tokens':128}}}).encode()
        with patch.object(kv,'request',side_effect=request),patch.object(kv.time,'sleep'):result=kv.run('/fake.sock',use_tool=use_tool)
        self.assertIn(('POST','/deepinfer/cache/reset-local'),calls)
        return result

    def test_requires_actual_cpu_reload_after_gpu_reset(self):
        r=self.exercise(1024)
        self.assertEqual(r['status'],'PASS')
        self.assertTrue(r['greedy_output_identical'])
        self.assertGreater(r['cpu_reload_bytes'],0)

    def test_cache_accounting_without_cpu_reload_does_not_pass(self):
        with self.assertRaisesRegex(ValueError,'No bytes actually reloaded'):
            self.exercise(0)

    def test_tool_arguments_are_compared_after_actual_restore(self):
        r=self.exercise(1024,use_tool=True)
        self.assertEqual(r['mode'],'tools')
        self.assertTrue(r['greedy_output_identical'])
