"""Real local HTTP/SSE fixtures for the optional public-API tool probe.

No model/GPU, command execution, service stop or synthetic performance PASS.
"""
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r7_probe_tools_tested', ROOT/'scripts/probe_tools.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def frame(value, event=None):
    return ((('event: '+event+'\n') if event else '') + 'data: ' +
            (value if isinstance(value, str) else json.dumps(value)) + '\n\n').encode()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        responses = self.path == '/v1/responses'
        protocol = 'responses' if responses else 'chat_completions'
        phase = self.headers['X-R7-Probe-Phase']
        function = request['tools'][0] if responses else request['tools'][0]['function']
        expected = function['parameters']['properties']['cmd']['enum'][0]
        with self.server.guard:
            self.server.active += 1
            self.server.peak = max(self.server.peak, self.server.active)
            self.server.history.append((protocol, phase))
        try:
            if phase == 'concurrent':
                time.sleep(.025)
            identity = uuid.uuid4().hex
            call_id = 'call_' + identity
            actual = expected
            mode = self.server.mode
            if mode == 'bad_arguments':
                actual += ' WRONG_NONCE'
            if mode == 'identity' and phase == 'concurrent':
                call_id = 'call_reused_between_requests'
            arguments = json.dumps({'cmd': actual})
            if mode == 'bad_arguments' and responses:
                arguments = '{"cmd":'
            if responses:
                item_id = 'fc_' + identity
                item = {'type': 'function_call', 'id': item_id, 'call_id': call_id,
                        'name': 'r7_probe', 'arguments': arguments, 'status': 'completed'}
                response = {'id': 'resp_' + identity, 'object': 'response',
                            'status': 'completed', 'output': [item]}
                parts = [frame({'type': 'response.created', 'response': dict(response, status='in_progress', output=[])}),
                         frame({'type': 'response.function_call_arguments.done', 'item_id': item_id, 'arguments': arguments}),
                         frame({'type': 'response.output_item.done', 'item': item}),
                         frame({'type': 'response.completed', 'response': response})]
                if mode == 'bad_terminal':
                    parts = [parts[0], frame({'type': 'response.incomplete', 'response': dict(response, status='incomplete', incomplete_details={'reason':'max_output_tokens'})})]
                if mode == 'error_event':
                    parts = [parts[0], frame({'type':'error', 'message':'Synthetic failure'})]
            else:
                split = len(arguments)//2
                parts = [frame({'id': 'chat_' + identity, 'choices': [{'index':0, 'delta': {'tool_calls': [{'index':0, 'id':call_id, 'type':'function', 'function': {'name':'r7_probe', 'arguments':arguments[:split]}}]}, 'finish_reason':None}]}),
                         frame({'id': 'chat_' + identity, 'choices': [{'index':0, 'delta': {'tool_calls': [{'index':0, 'function': {'arguments':arguments[split:]}}]}, 'finish_reason':None}]}),
                         frame({'id': 'chat_' + identity, 'choices': [{'index':0, 'delta': {}, 'finish_reason':'tool_calls'}]}),
                         frame('[DONE]')]
                if mode == 'bad_terminal':
                    parts.pop()  # Clean HTTP EOF without the required SSE terminal.
                if mode == 'error_event':
                    parts = [parts[0], frame({'error': {'message':'Synthetic failure', 'code':'invalid_model_output'}}), frame('[DONE]')]
            body = b''.join(parts)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            # Split transport writes inside lines/JSON, not only at SSE frames.
            for piece in (body[:11], body[11:43], body[43:]):
                self.wfile.write(piece)
                self.wfile.flush()
        finally:
            with self.server.guard:
                self.server.active -= 1


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='r7-tool-probe-http-')
        self.addCleanup(self.tmp.cleanup)
        self.server = Server(('127.0.0.1', 0), Handler)
        self.server.mode = 'normal'
        self.server.guard = threading.Lock()
        self.server.active = self.server.peak = 0
        self.server.history = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def run_probe(self, mode='normal', count=2, concurrency=2):
        self.server.mode = mode
        path = Path(self.tmp.name)/(mode+'.json')
        with redirect_stdout(io.StringIO()):
            rc = probe.main(['--base-url', 'http://127.0.0.1:'+str(self.server.server_port)+'/v1',
                             '--requests', str(count), '--concurrency', str(concurrency),
                             '--timeout', '10', '--output', str(path)])
        return rc, json.loads(path.read_text())

    def test_valid_fragmented_sse_and_32_to_1_preserve_nonce_identity(self):
        rc, report = self.run_probe(count=32, concurrency=32)
        self.assertEqual(rc, 0, report)
        self.assertEqual((report['status'], report['passed'], report['failed']), ('PASS', 72, 0))
        self.assertEqual([p['requests'] for p in report['phases']], [1,32,3,1,32,3])
        self.assertGreater(self.server.peak, 1)
        ids = [row['call_id_sha256'] for row in report['cases']]
        self.assertEqual(len(ids), len(set(ids)))
        for protocol in ('responses', 'chat_completions'):
            phases = [phase for p,phase in self.server.history if p==protocol]
            self.assertEqual(phases, ['single_before']+['concurrent']*32+['single_after']*3)
        self.assertFalse(report['command_execution'])
        self.assertFalse(report['api_stop_requested'])
        self.assertFalse(report['acceptance_marker_used'])
        self.assertTrue(all('arguments' not in row and 'cmd' not in row for row in report['cases']))

    def test_invalid_json_and_valid_json_with_wrong_nonce_fail(self):
        rc, report = self.run_probe('bad_arguments')
        self.assertEqual(rc, 1)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual({r['error_code'] for r in report['cases']}, {'invalid_tool_json', 'wrong_tool_arguments'})

    def test_http_200_with_incomplete_or_missing_sse_terminal_fails(self):
        rc, report = self.run_probe('bad_terminal')
        self.assertEqual(rc, 1)
        self.assertEqual({r['error_code'] for r in report['cases']}, {'incomplete_response', 'stream_disconnected'})
        self.assertTrue(all(r['http_status']==200 for r in report['cases']))

    def test_reused_call_ids_and_explicit_error_events_fail(self):
        rc, report = self.run_probe('identity', count=8, concurrency=8)
        self.assertEqual(rc, 1)
        self.assertTrue(any(r.get('error_code')=='duplicate_call_id' for r in report['cases']))
        rc, report = self.run_probe('error_event')
        self.assertEqual(rc, 1)
        self.assertEqual({r['error_code'] for r in report['cases']}, {'server_error_event'})


if __name__ == '__main__':
    unittest.main()
