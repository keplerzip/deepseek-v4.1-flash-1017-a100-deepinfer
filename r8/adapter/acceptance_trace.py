"""Bounded, opt-in engine token evidence for isolated acceptance only.

The official protocol response bytes and upstream parser are not modified.
Raw token timing is collected at EngineClient yields, not inferred from text
or SSE event counts. The operator must pause ingress and enable the marker.
"""
from collections import OrderedDict
from contextvars import ContextVar
import hashlib
from pathlib import Path
import re
import time

CURRENT_TRACE = ContextVar('deepinfer_acceptance_trace', default=None)


def active():
    return Path('/state/cache-acceptance-active').is_file()


class TraceStore:
    def __init__(self):
        self.records = OrderedDict()

    def begin(self, identity, fixed_budget=False):
        if not re.fullmatch(r'[0-9a-f]{32}', identity):
            raise ValueError('Invalid acceptance trace identity')
        if identity in self.records:
            raise ValueError('Acceptance trace identity already exists')
        while len(self.records) >= 256:
            completed = next((key for key, row in self.records.items() if row['http_complete']), None)
            if completed is None:
                raise ValueError('Too many active acceptance traces')
            del self.records[completed]
        row = {'trace_id': identity, 'started_monotonic': time.monotonic(),
               'fixed_budget': fixed_budget, 'batches': [], 'token_count': 0,
               'stream_boundaries': [], 'response_body_bytes': 0,
               'overflow': False, 'engine_finished': False, 'http_complete': False,
               'timing_source': 'EngineClient yield; excludes HTTP/SSE serialization'}
        self.records[identity] = row
        return row

    def take(self, identity):
        row = self.records.get(identity)
        if row is not None and row['http_complete']:
            return self.records.pop(identity)
        return None


def record_output(output, completion):
    row = CURRENT_TRACE.get()
    if row is None:
        return
    if output.prompt_token_ids is not None:
        row['prompt_tokens'] = len(output.prompt_token_ids)
        row['cached_tokens'] = output.num_cached_tokens
    ids = list(completion.token_ids)
    if ids:
        row['token_count'] += len(ids)
        if row['token_count'] <= 16384 and len(row['batches']) < 4096:
            row['batches'].append({'seconds': time.monotonic() - row['started_monotonic'],
                                   'count': len(ids), 'token_ids': ids})
        else:
            row['overflow'] = True
    if completion.finish_reason is not None:
        row['engine_finished'] = bool(output.finished)
        row['finish_reason'] = completion.finish_reason


class TraceMiddleware:
    def __init__(self, app, store):
        self.app, self.store = app, store

    async def __call__(self, scope, receive, send):
        headers = dict(scope.get('headers', []))
        identity = headers.get(b'x-deepinfer-acceptance-id', b'')
        if scope['type'] != 'http' or scope.get('path') != '/v1/chat/completions' or not identity or not active():
            return await self.app(scope, receive, send)
        from starlette.responses import JSONResponse
        try:
            row = self.store.begin(identity.decode('ascii'),
                                   headers.get(b'x-deepinfer-acceptance-fixed-budget') == b'1')
        except (ValueError, UnicodeError) as exc:
            return await JSONResponse({'error': str(exc)}, status_code=409)(scope, receive, send)
        digest = hashlib.sha256()
        token = CURRENT_TRACE.set(row)

        async def traced_receive():
            message = await receive()
            if message['type'] == 'http.request':
                digest.update(message.get('body', b''))
                if not message.get('more_body', False):
                    row['request_body_sha256'] = digest.hexdigest()
            return message

        async def traced_send(message):
            if message['type'] == 'http.response.start':
                row['http_status'] = message['status']
            if message['type'] == 'http.response.body' and not message.get('more_body', False):
                row['http_complete'] = True
            if message['type'] == 'http.response.body' and message.get('body'):
                row['response_body_bytes'] += len(message['body'])
                if len(row['stream_boundaries']) < 32768:
                    row['stream_boundaries'].append({'end_offset': row['response_body_bytes'],
                                                      'token_count': row['token_count']})
                else:
                    row['overflow'] = True
            await send(message)

        try:
            await self.app(scope, traced_receive, traced_send)
        finally:
            row['http_complete'] = True
            CURRENT_TRACE.reset(token)
