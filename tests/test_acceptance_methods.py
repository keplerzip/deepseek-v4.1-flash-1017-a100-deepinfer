"""CPU evidence for the reused test methods; never model/GPU acceptance."""
import copy
import hashlib
import http.server
import json
import threading
import types
import unittest
from unittest.mock import patch

import acceptance as suite
import diagnose_speed as diagnosis
from adapter.acceptance_trace import TraceStore
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('formal_acceptance_seal', Path(__file__).resolve().parents[1]/'scripts/finalize_acceptance.py')
seal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seal)


class AcceptanceMethodsTests(unittest.TestCase):
    def test_official_thinking_controls_replace_legacy_template_options(self):
        client = suite.Client('http://localhost', suite.MODEL, 2)
        self.assertEqual(client.body('x', 'off')['thinking'], {'type': 'disabled'})
        body = client.body('x', 'high')
        self.assertEqual(body['thinking'], {'type': 'enabled'})
        self.assertEqual(body['reasoning_effort'], 'high')
        self.assertNotIn('chat_template_kwargs', body)
        template = {'messages': [{'role': 'user', 'content': 'private'}], 'tools': []}
        original = copy.deepcopy(template)
        body = diagnosis.make_body(client, 'request', 'high', 'sample', 0, 'seed', 1024, template)
        self.assertEqual(template, original)
        self.assertEqual(body['messages'], template['messages'])
        self.assertEqual(body['thinking'], {'type': 'enabled'})

    def test_batch_timing_includes_reasoning_and_excludes_first_batch(self):
        result = {'token_batches': [{'seconds': 2., 'count': 4}, {'seconds': 3., 'count': 60}, {'seconds': 4., 'count': 64}],
                  'usage': {'prompt_tokens': 50, 'completion_tokens': 128, 'total_tokens': 178,
                            'completion_tokens_details': {'reasoning_tokens': 80}},
                  'first_visible_seconds': 2.1, 'elapsed_seconds': 4.2, 'finish_reason': 'length',
                  'text': 'fixture', 'reasoning': 'fixture',
                  'engine_trace': {'trace_id': 'fixture', 'timing_source': 'EngineClient yield'}}
        stats = diagnosis.timing(result, 128)
        self.assertEqual(stats['decode_measured_tokens'], 124)
        self.assertEqual(stats['decode_tokens_per_second'], 62)
        self.assertEqual(stats['reasoning_tokens'], 80)
        result['token_batches'][-1]['count'] = 63
        with self.assertRaises(ValueError):
            diagnosis.timing(result, 128)

    def test_generation_and_k5_accounting_rejects_contamination(self):
        def state(tokens, drafts=0, proposed=0, accepted=0):
            return {'settled': True, 'metrics': {'vllm:generation_tokens_total': tokens,
                'vllm:spec_decode_num_drafts_total': drafts,
                'vllm:spec_decode_num_draft_tokens_total': proposed,
                'vllm:spec_decode_num_accepted_tokens_total': accepted}}
        rows = [{'usage': {'completion_tokens': 128}}]
        self.assertFalse(suite.accounting(state(0), state(128, 30, 150, 98), rows, True)['issues'])
        for after in [state(129, 30, 150, 98), state(128), state(128, 30, 151, 98), state(128, 30, 150, 151)]:
            self.assertTrue(suite.accounting(state(0), after, rows, True)['issues'])

    def test_full_suite_retains_original_functional_and_benchmark_cases(self):
        obj = object.__new__(suite.Suite)
        obj.args = types.SimpleNamespace(suite='full', output='CPU fixture')
        obj.client = types.SimpleNamespace(ledger=[])
        obj.report = {'tests': [], 'contamination': []}
        names = []
        obj.save = lambda: None
        obj.test = lambda name, fn: (names.append(name), obj.report['tests'].append({'status': 'PASS'}))
        snapshot = {'settled': True, 'metrics': {'vllm:generation_tokens_total': 0}}
        with patch.object(suite, 'settle', return_value=snapshot):
            self.assertEqual(obj.run(), 0)
        self.assertEqual(names, ['model-identity', 'text-SSE-off', 'text-SSE-high', 'single-image-OCR',
            'multi-image-OCR', 'reverse-tool-result-image-association', 'prefix-cache-cold-hot-details',
            'C32-auto-tool-calls', 'DSpark-native-k5-counters', 'benchmark-off', 'benchmark-high',
            'independent-near-524288-token-retrieval'])

    def test_five_position_prompt_targets_real_tokenizer_count(self):
        obj = object.__new__(suite.Suite)
        obj.args = types.SimpleNamespace(model=suite.MODEL)
        obj.report, obj.save = {}, lambda: None
        def count(path, body):
            self.assertEqual(path, '/tokenize')
            self.assertEqual(body['thinking'], {'type': 'disabled'})
            return {'count': body['messages'][0]['content'].count('Archive record:') * 2 + 100}
        obj.client = types.SimpleNamespace(request=count)
        prompt, actual, values, probes = obj.near_full_prompt('nonce', nonce_first=True)
        self.assertTrue(prompt.startswith('nonce\n'))
        self.assertTrue(suite.CONTEXT-512-1024 <= actual <= suite.CONTEXT-512)
        self.assertEqual(len(values), 5)
        self.assertTrue(all('POSITION_%d=%s' % (i, value) in prompt for i, value in enumerate(values)))

    def test_trace_store_limits_do_not_evict_live_requests(self):
        store = TraceStore()
        for index in range(256):
            store.begin('%032x' % index)
        with self.assertRaises(ValueError):
            store.begin('f' * 32)
        store.records['%032x' % 0]['http_complete'] = True
        store.begin('f' * 32)
        self.assertNotIn('%032x' % 0, store.records)
        self.assertEqual(len(store.records), 256)
        self.assertIsNone(store.take('f' * 32))

    def test_client_arrival_mapping_retains_real_counts_under_split_and_coalesced_writes(self):
        trace = {'token_count': 6, 'response_body_bytes': 60, 'stream_boundaries': [
            {'end_offset': 5, 'token_count': 0}, {'end_offset': 10, 'token_count': 2},
            {'end_offset': 15, 'token_count': 4}, {'end_offset': 60, 'token_count': 6}]}
        arrivals = [{'end_offset': 20, 'seconds': 1.}, {'end_offset': 40, 'seconds': 2.},
                    {'end_offset': 60, 'seconds': 3.}]
        batches = suite.client_token_batches(trace, arrivals)
        self.assertEqual([(b['seconds'], b['count']) for b in batches], [(1., 4), (3., 2)])
        self.assertEqual(sum(b['count'] for b in batches), 6)

    def test_client_arrival_mapping_cannot_accept_unbound_tokens_or_byte_mismatch(self):
        base = {'token_count': 6, 'response_body_bytes': 60,
                'stream_boundaries': [{'end_offset': 60, 'token_count': 6}]}
        arrivals = [{'end_offset': 60, 'seconds': 1.}]
        for field, value in [('response_body_bytes', 59), ('token_count', 7),
                             ('stream_boundaries', [{'end_offset': 61, 'token_count': 6}])]:
            trace = dict(base); trace[field] = value
            with self.assertRaises(ValueError): suite.client_token_batches(trace, arrivals)

    def extended_fixture(self):
        full = {'status': 'PASS', 'suite': 'full', 'context': 524288, 'concurrency': 32,
                'contamination': [], 'tests': [{'name': name, 'status': 'PASS'} for name in seal.FULL_CASES],
                'benchmarks': {}}
        for mode in ['off', 'high']:
            full['benchmarks'][mode] = {'accounting': {'issues': []}, 'phases': {}, 'samples': []}
            for phase, count in [('warmup', 2), ('single', 10), ('c32', 128)]:
                full['benchmarks'][mode]['phases'][phase] = {'requests': count}
                full['benchmarks'][mode]['samples'].extend(
                    {'phase': phase, 'completion_tokens': 1024, 'status': 'PASS'} for _ in range(count))
        diag = {'status': 'PASS', 'warmups_per_case': 2, 'samples_per_case': 10, 'output_tokens': 1024,
                'summary': {name: {'samples': 10, 'measurement_status': 'CLEAN'}
                            for name in ['text/off', 'text/high', 'code/off', 'code/high']}}
        return full, diag

    def test_extended_gate_accepts_complete_method_evidence(self):
        seal.validate_extended(*self.extended_fixture())

    def test_extended_gate_rejects_missing_cases_or_shortened_benchmark(self):
        for mutate in [lambda f: f['tests'].pop(),
                       lambda f: f['benchmarks']['high']['samples'].pop(),
                       lambda f: f['contamination'].append('extra traffic')]:
            full, diag = self.extended_fixture(); mutate(full)
            with self.assertRaises(AssertionError): seal.validate_extended(full, diag)

    def test_extended_gate_rejects_incomplete_or_contaminated_diagnostics(self):
        full, diag = self.extended_fixture()
        diag['summary']['code/high']['measurement_status'] = 'NOT_SIGNOFF'
        with self.assertRaises(AssertionError): seal.validate_extended(full, diag)


class TraceClientFixtureTests(unittest.TestCase):
    def run_fixture(self, corrupt=False):
        traces = {}
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                identity = self.headers['X-Deepinfer-Acceptance-Id']
                traces[identity] = {'engine_finished': True, 'http_complete': True, 'http_status': 200,
                    'request_body_sha256': ('0'*64 if corrupt else hashlib.sha256(raw).hexdigest()),
                    'trace_id': identity, 'overflow': False, 'prompt_tokens': 10, 'token_count': 2,
                    'timing_source': 'CPU fixture only', 'batches': [
                        {'seconds': .1, 'count': 1, 'token_ids': [1]}, {'seconds': .2, 'count': 1, 'token_ids': [2]}]}
                frames = [{'id': 'fixture', 'model': suite.MODEL, 'choices': [{'index': 0, 'delta': {'content': 'OK'}, 'finish_reason': None}]},
                          {'id': 'fixture', 'model': suite.MODEL, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                           'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}}]
                parts = ['data: '+json.dumps(frame)+'\n\n' for frame in frames]+['data: [DONE]\n\n']
                payload = ''.join(parts)
                offset = 0
                traces[identity]['stream_boundaries'] = []
                for index, part in enumerate(parts):
                    offset += len(part.encode())
                    traces[identity]['stream_boundaries'].append({'end_offset': offset, 'token_count': min(index+1, 2)})
                traces[identity]['response_body_bytes'] = offset
                self.send_response(200); self.end_headers(); self.wfile.write(payload.encode())
            def do_GET(self):
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps(traces.pop(self.path.rsplit('/', 1)[1])).encode())
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            return suite.Client('http://127.0.0.1:%d' % server.server_port, suite.MODEL, 2).chat('fixture')
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_official_sse_without_token_ids_uses_bound_native_trace(self):
        result = self.run_fixture()
        self.assertEqual(result['text'], 'OK')
        self.assertEqual(sum(b['count'] for b in result['token_batches']), 2)

    def test_wrong_request_trace_cannot_sign_off(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_fixture(corrupt=True)
