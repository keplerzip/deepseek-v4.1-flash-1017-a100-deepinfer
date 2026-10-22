"""Direct-UDS real-model smoke gate; does not imply 512K/C32 performance PASS."""
from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
from pathlib import Path
import socket
import time
import uuid


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout=600):
        super().__init__('localhost', timeout=timeout)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def request(path, method, endpoint, body=None):
    connection = UnixHTTPConnection(path)
    try:
        payload = None if body is None else json.dumps(body).encode()
        connection.request(method, endpoint, payload, {'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise ValueError(f'{endpoint} returned HTTP {response.status}: {raw[:512]!r}')
        return raw
    finally:
        connection.close()


def output_text(response):
    return ''.join(part.get('text', '') for item in response['output'] if item['type'] == 'message' for part in item['content'])


def function_schema():
    return {'type': 'function', 'name': 'echo_probe', 'strict': True, 'parameters': {
        'type': 'object', 'properties': {'marker': {'type': 'string'}},
        'required': ['marker'], 'additionalProperties': False}}


def run(path):
    checks = []
    models = json.loads(request(path, 'GET', '/v1/models'))
    assert [m['id'] for m in models['data']] == ['DeepSeek-Flash']
    checks.append('model_identity')
    nonce = 'probe_' + uuid.uuid4().hex[:12]
    first_input = [{'role': 'user', 'content': f'Call echo_probe once with marker exactly {nonce}.'}]
    body = {'model': 'DeepSeek-Flash', 'input': first_input, 'reasoning': {'effort': 'none'},
            'tools': [function_schema()], 'tool_choice': 'required', 'max_output_tokens': 512}
    first = json.loads(request(path, 'POST', '/v1/responses', body))
    calls = [item for item in first['output'] if item['type'] == 'function_call']
    assert first['status'] == 'completed' and len(calls) == 1
    assert calls[0]['name'] == 'echo_probe' and json.loads(calls[0]['arguments']) == {'marker': nonce}
    checks.append('responses_required_strict_actual_generation')
    result = 'result_' + uuid.uuid4().hex[:12]
    history = first_input + first['output'] + [{'type': 'function_call_output', 'call_id': calls[0]['call_id'], 'output': result},
                                             {'role': 'user', 'content': 'Reply with exactly the tool result.'}]
    follow = json.loads(request(path, 'POST', '/v1/responses', {**body, 'input': history, 'tool_choice': 'none'}))
    assert follow['status'] == 'completed' and result in output_text(follow)
    checks.append('responses_tool_result_roundtrip')
    stream = request(path, 'POST', '/v1/responses', {
        'model': 'DeepSeek-Flash', 'input': 'Reply with OK.', 'reasoning': {'effort': 'none'},
        'max_output_tokens': 128, 'stream': True}).decode()
    events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith('data: ')]
    assert events[-1]['type'] == 'response.completed'
    assert [e['sequence_number'] for e in events] == sorted({e['sequence_number'] for e in events})
    final = events[-1]['response']
    done = [e['item'] for e in events if e['type'] == 'response.output_item.done']
    assert final['output'] == done and final['usage']['output_tokens'] > 0
    assert '[DONE]' not in stream
    checks.append('responses_sse_identity_and_usage')
    chat = json.loads(request(path, 'POST', '/v1/chat/completions', {
        'model': 'DeepSeek-Flash', 'messages': [{'role': 'user', 'content': 'Reply with OK.'}],
        'thinking': {'type': 'disabled'}, 'max_tokens': 128}))
    assert chat['choices'][0]['message']['content'] and chat['usage']['completion_tokens'] > 0
    checks.append('chat_completions')
    message = json.loads(request(path, 'POST', '/v1/messages', {
        'model': 'DeepSeek-Flash', 'messages': [{'role': 'user', 'content': 'Reply with OK.'}],
        'thinking': {'type': 'disabled'}, 'max_tokens': 128}))
    assert any(p['type'] == 'text' and p['text'] for p in message['content'])
    checks.append('anthropic_messages')
    thought = json.loads(request(path, 'POST', '/v1/responses', {
        'model': 'DeepSeek-Flash', 'input': 'What is 17+25? Reply briefly.',
        'reasoning': {'effort': 'low'}, 'max_output_tokens': 2048}))
    assert thought['status'] == 'completed' and '42' in output_text(thought)
    assert any(i['type'] == 'reasoning' for i in thought['output'])
    assert thought['usage']['output_tokens_details']['reasoning_tokens'] > 0
    checks.append('thinking_reasoning_token_accounting')
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (240, 100), 'white')
    draw = ImageDraw.Draw(image)
    digits = str(1000 + int(uuid.uuid4().hex[:4], 16) % 9000)
    font = ImageFont.load_default(size=56)
    draw.text((20, 10), digits, font=font, fill='black')
    encoded = io.BytesIO()
    image.save(encoded, format='PNG')
    data_url = 'data:image/png;base64,' + base64.b64encode(encoded.getvalue()).decode()
    visual = json.loads(request(path, 'POST', '/v1/responses', {
        'model': 'DeepSeek-Flash', 'input': [{'role': 'user', 'content': [
            {'type': 'input_text', 'text': 'Read the four digits in this image. Reply with only the digits.'},
            {'type': 'input_image', 'image_url': data_url}]}],
        'reasoning': {'effort': 'none'}, 'max_output_tokens': 128}))
    assert visual['status'] == 'completed' and digits in output_text(visual)
    checks.append('native_vision_ocr')
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', default='/state/api.sock')
    parser.add_argument('--health', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.health:
        request(args.socket, 'GET', '/health')
        return 0
    report = {'status': 'RUNNING', 'scope': 'real-model direct-UDS protocol smoke',
              'real_model_inference': True, 'actual_app_gateway': 'NOT_RUN', 'c32': 'NOT_RUN',
              'long_context_512k': 'NOT_RUN', 'started_epoch': time.time()}
    try:
        report['checks'] = run(args.socket)
        report['status'] = 'PASS'
        code = 0
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        code = 1
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
