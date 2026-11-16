#!/usr/bin/env python3
"""Optional live tool-call quality probe. Never executes returned commands.

Uses the existing public API only: no GPU process, acceptance marker, cache
reset, service restart or performance accounting. This is an explicit manual
test and can add ordinary inference load to an otherwise available service.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request
import uuid

TOOL = 'r7_probe'
ROOT = Path(__file__).resolve().parents[1]


class ProbeError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not condition:
        raise ProbeError(code, message)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def strict_json(value):
    def reject(_):
        raise ValueError('Non-finite JSON constant')
    return json.loads(value, parse_constant=reject)


def command(nonce, variant):
    if variant == 0:
        return "printf '%s' 'R7_NONCE_" + nonce + "'"
    if variant == 1:
        return "printf '%s' '</parameter>' 'R7_NONCE_" + nonce + "'"
    if variant == 2:
        return "printf '%s' 'R7_NONCE_" + nonce + "' # </parameter>"
    return "cat <<'R7_EOF'\nR7_NONCE_" + nonce + "\n</parameter>\nR7_EOF"


def request_body(protocol, model, expected, effort="none", budget=1024, force=True):
    schema = {'type': 'object', 'properties': {'cmd': {'type': 'string', 'enum': [expected]}},
              'required': ['cmd'], 'additionalProperties': False}
    function = {'name': TOOL, 'description': 'Diagnostic string receiver. Copy the requested cmd exactly; nothing will be executed.',
                'parameters': schema, 'strict': True}
    prompt = ('Call r7_probe exactly once. Return the following object as its arguments, '
              'copying the cmd string exactly, including quotes, newlines and comment text. '
              'This is synthetic data; the client will only validate it and execute nothing.\n'
              + json.dumps({'cmd': expected}, ensure_ascii=False))
    body = {'model': model, 'stream': True, 'temperature': 0, 'parallel_tool_calls': False}
    if protocol == 'responses':
        body.update(input=prompt, reasoning={'effort': effort}, max_output_tokens=budget,
                    tools=[{'type': 'function', **function}],
                    tool_choice={'type': 'function', 'name': TOOL})
    else:
        body.update(messages=[{'role': 'user', 'content': prompt}], reasoning_effort=effort,
                    max_tokens=budget, tools=[{'type': 'function', 'function': function}],
                    tool_choice={'type': 'function', 'function': {'name': TOOL}})
    if not force:body['tool_choice']='auto'
    return body


def sse_events(response, digest, stats, timeout):
    event, data = None, []
    deadline = time.monotonic() + timeout
    for raw in response:
        digest.update(raw)
        stats['response_bytes'] += len(raw)
        require(stats['response_bytes'] <= 4 * 1024 * 1024, 'oversized_stream', 'SSE response exceeded 4 MiB')
        require(time.monotonic() <= deadline, 'request_timeout', 'SSE request exceeded its time budget')
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield event, '\n'.join(data)
            event, data = None, []
        elif line.startswith('data:'):
            data.append(line[5:].removeprefix(' '))
        elif line.startswith('event:'):
            event = line[6:].removeprefix(' ')
    if data:
        yield event, '\n'.join(data)


def parse_stream(protocol, events):
    calls, completed, ended = {}, None, False
    response_id, argument_done = None, {}
    for event, raw in events:
        require(not ended, 'data_after_terminal', 'SSE data followed the terminal event')
        if raw == '[DONE]':
            require(protocol == 'chat_completions' and completed == 'tool_calls',
                    'bad_terminal', 'DONE arrived without a completed tool call')
            ended = True
            continue
        try:
            value = strict_json(raw)
        except (TypeError, ValueError):
            raise ProbeError('invalid_sse_json', 'SSE data was not a valid JSON object') from None
        require(isinstance(value, dict), 'invalid_sse_json', 'SSE data was not an object')
        kind = value.get('type') or event
        if value.get('error') is not None or kind in ('error', 'response.failed'):
            raise ProbeError('server_error_event', 'The service returned an explicit SSE error')
        if kind == 'response.incomplete':
            raise ProbeError('incomplete_response', 'The service returned an incomplete response')
        if protocol == 'responses':
            response = value.get('response')
            if isinstance(response, dict) and isinstance(response.get('id'), str):
                current = response['id']
                require(response_id in (None, current), 'response_identity_changed', 'Response ID changed within one stream')
                response_id = current
            if kind == 'response.function_call_arguments.done':
                key = value.get('item_id')
                require(isinstance(key, str) and key not in argument_done, 'duplicate_tool_done', 'Tool arguments completion identity missing or duplicated')
                argument_done[key] = value.get('arguments')
            if kind == 'response.completed':
                require(isinstance(response, dict) and response.get('status') == 'completed',
                        'bad_terminal', 'Completed event did not contain a completed response')
                output = response.get('output')
                require(isinstance(output, list), 'missing_tool', 'Completed response omitted output')
                selected = [item for item in output if isinstance(item, dict) and item.get('type') == 'function_call']
                require(len(selected) == 1, 'tool_count', 'Expected exactly one function call')
                call = selected[0]
                require(call.get('id') in argument_done, 'missing_tool_done', 'Completed function omitted arguments.done')
                require(argument_done[call['id']] == call.get('arguments'), 'arguments_changed', 'Final arguments differ from arguments.done')
                calls[0] = {'id': call.get('call_id'), 'name': call.get('name'), 'arguments': call.get('arguments')}
                completed, ended = 'tool_calls', True
        else:
            choices = value.get('choices', [])
            require(isinstance(choices, list), 'invalid_choices', 'Chat choices must be a list')
            for choice in choices:
                require(choice.get('index', 0) == 0, 'unexpected_choice', 'Unexpected additional completion choice')
                delta = choice.get('delta') or {}
                for call in delta.get('tool_calls', []):
                    index = call.get('index', 0)
                    item = calls.setdefault(index, {'id': None, 'name': '', 'arguments': ''})
                    if call.get('id') is not None:
                        require(item['id'] in (None, call['id']), 'call_identity_changed', 'Call ID changed within one stream')
                        item['id'] = call['id']
                    function = call.get('function') or {}
                    item['name'] += function.get('name') or ''
                    item['arguments'] += function.get('arguments') or ''
                if choice.get('finish_reason') is not None:
                    require(completed is None and choice['finish_reason'] == 'tool_calls', 'bad_terminal', 'Chat did not end with tool_calls')
                    completed = 'tool_calls'
    require(ended, 'stream_disconnected', 'SSE ended without its expected terminal event')
    require(len(calls) == 1, 'tool_count', 'Expected exactly one function call')
    return next(iter(calls.values()))


def one_case(args, protocol, phase, index, variant):
    nonce = uuid.uuid4().hex
    expected = command(nonce, variant)
    body = request_body(protocol, args.model, expected, getattr(args,"reasoning_effort","none"), getattr(args,"max_tokens",1024), getattr(args,"tool_choice","forced")=="forced")
    payload = json.dumps(body, ensure_ascii=False).encode()
    digest = hashlib.sha256()
    row = {'protocol': protocol, 'phase': phase, 'index': index, 'variant': variant,
           'status': 'FAIL', 'nonce_sha256': sha(nonce), 'expected_cmd_sha256': sha(expected),
           'request_sha256': sha(payload), 'response_bytes': 0}
    started = time.monotonic()
    try:
        endpoint = args.base_url.rstrip('/') + ('/responses' if protocol == 'responses' else '/chat/completions')
        request = urllib.request.Request(endpoint, data=payload, headers={
            'Content-Type': 'application/json', 'Accept': 'text/event-stream',
            'X-R7-Probe-Phase': phase, 'X-R7-Probe-Protocol': protocol}, method='POST')
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            row['http_status'] = response.status
            require(response.status == 200, 'http_error', 'Unexpected HTTP status')
            require(response.headers.get_content_type() == 'text/event-stream', 'wrong_content_type', 'Expected text/event-stream')
            call = parse_stream(protocol, sse_events(response, digest, row, args.timeout))
        require(isinstance(call['id'], str) and bool(call['id']), 'missing_call_id', 'Tool call ID is missing')
        row['call_id_sha256'] = sha(call['id'])
        require(call['name'] == TOOL, 'wrong_tool_name', 'Unexpected tool name')
        try:
            arguments = strict_json(call['arguments'])
        except (TypeError, ValueError):
            raise ProbeError('invalid_tool_json', 'Tool arguments are not valid JSON') from None
        require(isinstance(arguments, dict) and set(arguments) == {'cmd'}, 'wrong_argument_shape', 'Expected only a cmd argument')
        require(arguments['cmd'] == expected, 'wrong_tool_arguments', 'Command differs from the exact synthetic value or nonce')
        row['status'] = 'PASS'
    except ProbeError as error:
        row.update(error_code=error.code, reason=str(error))
    except urllib.error.HTTPError as error:
        row.update(http_status=error.code, error_code='http_error', reason='HTTP request failed')
    except Exception as error:
        row.update(error_code='transport_or_protocol_error', reason=type(error).__name__)
    row.update(response_sha256=digest.hexdigest(), elapsed_seconds=time.monotonic() - started)
    return row


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.tool-probe-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(args):
    report = {'schema': 1, 'status': 'RUNNING', 'started_at': utc_now(),
              'scope': 'real-model public-API tool quality only; no command execution',
              'model': args.model, 'base_url': args.base_url, 'command_execution': False,
              'tool_choice':getattr(args,'tool_choice','forced'), 'reasoning_effort': getattr(args,'reasoning_effort','none'), 'max_tokens':getattr(args,'max_tokens',1024),
              'api_stop_requested': False, 'gpu_process_created': False,
              'acceptance_marker_used': False, 'performance_accounting_used': False,
              'concurrency': args.concurrency, 'concurrent_requests_per_protocol': args.requests,
              'phases': [], 'cases': []}
    identities = set()
    for protocol in ('responses', 'chat_completions'):
        for phase, count, concurrency in [('single_before', 1, 1), ('concurrent', args.requests, args.concurrency), ('single_after', 3, 1)]:
            rows = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(one_case, args, protocol, phase, i, i % 4 if phase == 'concurrent' else 0 if phase == 'single_before' else i + 1) for i in range(count)]
                for future in as_completed(futures):
                    row = future.result()
                    identity = row.get('call_id_sha256')
                    if identity:
                        if identity in identities:
                            row.update(status='FAIL', error_code='duplicate_call_id', reason='Call ID was reused across probe requests')
                        identities.add(identity)
                    rows.append(row)
            rows.sort(key=lambda row: row['index'])
            report['cases'].extend(rows)
            summary = {'protocol': protocol, 'phase': phase, 'requests': count,
                       'concurrency': concurrency, 'passed': sum(r['status'] == 'PASS' for r in rows),
                       'failed': sum(r['status'] != 'PASS' for r in rows)}
            report['phases'].append(summary)
            write_report(args.output, report)
            print(json.dumps(summary), flush=True)
    report.update(status='PASS' if all(r['status'] == 'PASS' for r in report['cases']) else 'FAIL', finished_at=utc_now())
    report['passed'] = sum(r['status'] == 'PASS' for r in report['cases'])
    report['failed'] = len(report['cases']) - report['passed']
    write_report(args.output, report)
    print(report['status'] + ': ' + str(args.output), flush=True)
    return 0 if report['status'] == 'PASS' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8005/v1')
    parser.add_argument('--model', default='DeepSeek-Flash')
    parser.add_argument('--tool-choice',choices=('forced','auto'),default='forced')
    parser.add_argument('--reasoning-effort', choices=('none','high'),default='none')
    parser.add_argument('--max-tokens',type=int,default=1024)
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--requests', type=int, default=32, help='Concurrent-phase requests per protocol')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.max_tokens <= 16384 or not 1 <= args.concurrency <= 256 or not 1 <= args.requests <= 10000 or args.timeout <= 0:
        parser.error('Positive timeout, concurrency 1..256 and requests 1..10000 required')
    if args.output is None:
        args.output = ROOT / 'runtime' / ('tool-probe-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid()) + '.json')
    args.output = args.output.resolve()
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
