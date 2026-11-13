"""R6 integrity boundary around the unchanged official recipe response body.

The official converter/parser still owns all prompt and protocol semantics.
This layer counts native IDs, validates completed function arguments, and emits
explicit protocol errors instead of ending an HTTP-200 stream without a reason.
It never edits command contents or turns a failed generation into a success.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
import copy
import hashlib
import json
import logging
import os
import re
import time
import uuid

from adapter.native_token_usage import NativeTokenUsage

CURRENT_AUDIT = ContextVar('deepinfer_r6_native_usage', default=None)
_PROCESSOR_GUARD = ContextVar('deepinfer_r6_nonstream_processor_guard', default=None)
log = logging.getLogger(__name__)
_heartbeat_clock = time.monotonic
_SUFFIX = re.compile(r'(?:</(?:parameter|invoke|tool_calls|function_calls)>|</[｜|]DSML[｜|][^>\n]*>)\s*$')


class OutputIntegrityError(ValueError):
    def __init__(self, message, *, kind='invalid_output', diagnostic=None):
        super().__init__(message)
        self.kind = kind
        self.diagnostic = diagnostic or {}


def _integer(value):
    return type(value) is int and value >= 0


def _argument_diagnostic(raw):
    diagnostic = {'arguments_type': type(raw).__name__}
    text = raw if isinstance(raw, str) else None
    if isinstance(raw, (dict, list)):
        text = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        diagnostic['arguments_serialization'] = 'canonical_json'
    if text is not None:
        encoded = text.encode('utf-8', errors='surrogatepass')
        diagnostic.update(arguments_chars=len(text), arguments_bytes=len(encoded),
                          arguments_sha256=hashlib.sha256(encoded).hexdigest())
    return diagnostic


def _arguments(raw):
    diagnostic = _argument_diagnostic(raw)
    try:
        def reject_constant(_):
            raise ValueError('Non-finite JSON constant')
        value = json.loads(raw, parse_constant=reject_constant)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, json.JSONDecodeError):
            diagnostic.update(json_error_position=exc.pos,
                              json_error_line=exc.lineno, json_error_column=exc.colno)
        raise OutputIntegrityError('Tool call arguments are incomplete or invalid JSON',
                                   kind='invalid_tool_json', diagnostic=diagnostic) from None
    # Literal strings are not evidence of DSML token syntax. A valid shell
    # comment or heredoc can end in the same spelling. Default to audit-only;
    # preserve the old block behaviour solely as an explicit operator opt-in.
    if isinstance(value, dict):
        for key in ('cmd', 'command', 'script'):
            command = value.get(key)
            if isinstance(command, str) and _SUFFIX.search(command):
                if os.environ.get('DEEPINFER_R6_TOOL_SUFFIX_GUARD', '0') == '1':
                    raise OutputIntegrityError('Tool command ends in an unparsed protocol closing marker',
                                               kind='tool_suffix', diagnostic=diagnostic)
                audit = CURRENT_AUDIT.get()
                seen = audit.suffix_audits if audit is not None else set()
                identity = diagnostic.get('arguments_sha256')
                if identity not in seen:
                    seen.add(identity)
                    row = dict(diagnostic, classification='tool_suffix_observed', audit_only=True,
                               error_kind='literal_tool_suffix', command_field=key)
                    if audit is not None:
                        row['engine_request_id'] = audit.engine_request_id
                    log.info('R7 output diagnostic %s', json.dumps(row, separators=(',', ':')))
                break
    return value


class _ValidatedProcessor:
    """Validate native chunks before non-stream Messages loses bad JSON as {}."""
    def __init__(self, native, guard):
        self.native, self.guard = native, guard

    def _check(self, chunks):
        for chunk in chunks:
            self.guard.process(json.loads(chunk.to_json()))
        return chunks

    def push(self, chunk):
        return self._check(self.native.push(chunk))

    def finish(self):
        return self._check(self.native.finish())

    @property
    def finished(self):
        return self.native.finished

    def close(self):
        return self.native.close()


def wrap_processor(native):
    guard = _PROCESSOR_GUARD.get()
    return _ValidatedProcessor(native, guard) if guard is not None else native


def _completes_call_or_response(protocol, value, guard):
    if protocol == 'chat_completions':
        return any(c.get('delta', {}).get('tool_calls') or c.get('finish_reason') is not None
                   for c in value.get('choices', []))
    kind = value.get('type', '')
    if protocol == 'responses':
        return (kind in ('response.function_call_arguments.delta', 'response.function_call_arguments.done',
                         'response.custom_tool_call_input.delta', 'response.custom_tool_call_input.done',
                         'response.completed', 'response.incomplete', 'response.failed')
                or (kind in ('response.output_item.added', 'response.output_item.done')
                    and value.get('item', {}).get('type') in ('function_call', 'custom_tool_call', 'tool_search_call')))
    return (kind in ('message_delta', 'message_stop')
            or (kind == 'content_block_start' and value.get('content_block', {}).get('type') in ('tool_use', 'server_tool_use'))
            or (kind == 'content_block_delta' and value.get('delta', {}).get('type') == 'input_json_delta')
            or (kind == 'content_block_stop' and value.get('index') in guard.message_calls))


def _frame(event, data):
    prefix = '' if event is None else f'event: {event}\n'
    return prefix + 'data: ' + (data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(',', ':'))) + '\n\n'


class _Guard:
    def __init__(self, request, audit):
        self.protocol, self.audit = request.protocol, audit
        self.model = request.model or 'DeepSeek-Flash'
        self.response = None
        self.sequence = -1
        self.calls = {}
        self.message_calls = {}
        self.terminal = False
        self.call_sha256 = None
        self.has_tools = False

    def _call_identity(self, value):
        identity = value.get('call_id') or value.get('item_id') or value.get('id')
        if isinstance(identity, str):
            self.call_sha256 = hashlib.sha256(identity.encode()).hexdigest()

    def diagnostic(self, exc):
        result = {'protocol': self.protocol, 'error_type': type(exc).__name__,
                  'error_kind': getattr(exc, 'kind', 'upstream_error'),
                  'call_sha256': self.call_sha256,
                  'engine_request_id': self.audit.engine_request_id,
                  'backend_finish_reason': self.audit.backend_finish_reason,
                  'backend_finished': self.audit.backend_finished,
                  'prompt_tokens': self.audit.prompt_tokens,
                  'requested_max_tokens': self.audit.requested_max_tokens,
                  'effective_max_tokens': self.audit.effective_max_tokens,
                  'output_tokens': self.audit.total}
        result.update(getattr(exc, 'diagnostic', {}))
        return result

    def truncated_tool(self, exc):
        return (isinstance(exc, OutputIntegrityError)
                and exc.kind in ('invalid_tool_json', 'incomplete_tool_output')
                and self.audit.backend_finish_reason in ('length', 'content_filter')
                and self.audit.backend_finished is True)

    def incomplete_reason(self):
        return 'content_filter' if self.audit.backend_finish_reason == 'content_filter' else 'max_output_tokens'

    def _check_tool_end(self, raw=None):
        if (self.has_tools and self.audit.backend_finished is True
                and self.audit.backend_finish_reason in ('length', 'content_filter')):
            raise OutputIntegrityError('Tool output did not finish normally',
                kind='incomplete_tool_output', diagnostic=_argument_diagnostic(raw) if raw is not None else {})

    def _checked_arguments(self, raw):
        try:
            value = _arguments(raw)
        except OutputIntegrityError as exc:
            if (self.audit.backend_finished is True
                    and self.audit.backend_finish_reason in ('length', 'content_filter')):
                raise OutputIntegrityError('Tool output did not finish normally',
                    kind='incomplete_tool_output', diagnostic=exc.diagnostic) from None
            raise
        self._check_tool_end(raw)
        return value

    def _checked_tool_status(self, item):
        if item.get('status') in ('incomplete', 'failed', 'in_progress'):
            raw = item.get('arguments', item.get('input'))
            self._check_tool_end(raw)
            # The official tool_search generator marks bad raw JSON incomplete
            # and replaces its arguments with {}. Do not turn that into a
            # successful empty tool call merely because {} is valid JSON.
            raise OutputIntegrityError('Tool call did not complete',
                kind='incomplete_tool_status', diagnostic=_argument_diagnostic(raw))

    def _usage(self, usage, protocol):
        if not isinstance(usage, dict):
            return usage
        native = bool(getattr(self.audit, 'configured', False))
        if protocol == 'messages':
            if native:
                usage['output_tokens'] = self.audit.total
            if not _integer(usage.get('output_tokens')):
                raise OutputIntegrityError('Invalid output token usage')
            return usage
        output_key, input_key = ('output_tokens', 'input_tokens') if protocol == 'responses' else ('completion_tokens', 'prompt_tokens')
        if native:
            usage[output_key] = self.audit.total
            if _integer(usage.get(input_key)):
                usage['total_tokens'] = usage[input_key] + self.audit.total
        total = usage.get(output_key)
        if not _integer(total) or not _integer(usage.get(input_key)) or usage.get('total_tokens') != usage[input_key] + total:
            raise OutputIntegrityError('Invalid token usage totals')
        detail_key = 'output_tokens_details' if protocol == 'responses' else 'completion_tokens_details'
        detail = usage.get(detail_key)
        if native:
            reason = self.audit.reasoning
            if reason is None:
                if protocol == 'responses':
                    # Responses usage is nullable; unknown reasoning is not 0.
                    return None
                if isinstance(detail, dict):
                    detail.pop('reasoning_tokens', None)
                    if not detail:
                        usage.pop(detail_key, None)
            elif getattr(self.audit, 'thinking', False) or isinstance(detail, dict):
                usage.setdefault(detail_key, {})['reasoning_tokens'] = reason
        detail = usage.get(detail_key)
        if isinstance(detail, dict) and 'reasoning_tokens' in detail:
            reason = detail['reasoning_tokens']
            if not _integer(reason) or reason > total:
                raise OutputIntegrityError('Invalid reasoning token usage')
        return usage

    def _response_tools(self, response):
        for item in response.get('output', []):
            if item.get('type') in ('function_call', 'custom_tool_call', 'tool_search_call'):
                self.has_tools = True
                self._call_identity(item)
                self._checked_tool_status(item)
            if item.get('type') == 'function_call':
                self._checked_arguments(item.get('arguments'))
        self._check_tool_end()

    def process(self, value):
        if not isinstance(value, dict):
            raise OutputIntegrityError('Invalid protocol output object')
        if self.protocol == 'chat_completions':
            for choice in value.get('choices', []):
                for call in choice.get('delta', {}).get('tool_calls', []):
                    self.has_tools = True
                    self._call_identity(call)
                    index = call.get('index', 0)
                    state = self.calls.setdefault(index, '')
                    self.calls[index] = state + call.get('function', {}).get('arguments', '')
                if choice.get('finish_reason') is not None:
                    for arguments in self.calls.values():
                        self._checked_arguments(arguments)
                    self._check_tool_end()
                    self.terminal = True
                message = choice.get('message')
                if isinstance(message, dict):
                    for call in message.get('tool_calls', []):
                        if call.get('type') == 'function':
                            self.has_tools = True
                            self._checked_arguments(call.get('function', {}).get('arguments'))
            if value.get('usage') is not None:
                value['usage'] = self._usage(value['usage'], self.protocol)
            return value
        if self.protocol == 'responses':
            kind = value.get('type', '')
            if _integer(value.get('sequence_number')):
                self.sequence = max(self.sequence, value['sequence_number'])
            response = value.get('response')
            if isinstance(response, dict):
                self.response = response
            if kind == 'response.output_item.added' and value.get('item', {}).get('type') in ('function_call', 'custom_tool_call', 'tool_search_call'):
                self.has_tools = True
                self._call_identity(value['item'])
            if kind == 'response.function_call_arguments.done':
                self.has_tools = True
                self._call_identity(value)
                self._checked_arguments(value.get('arguments'))
            if kind == 'response.custom_tool_call_input.done':
                self.has_tools = True
                self._call_identity(value)
                self._check_tool_end(value.get('input'))
            if kind == 'response.output_item.done' and value.get('item', {}).get('type') in ('function_call', 'custom_tool_call', 'tool_search_call'):
                self.has_tools = True
                self._call_identity(value['item'])
                self._checked_tool_status(value['item'])
                if value['item']['type'] == 'function_call':
                    self._checked_arguments(value['item'].get('arguments'))
                self._check_tool_end()
            if kind in ('response.completed', 'response.incomplete'):
                self._response_tools(response or {})
                self.terminal = True
            if isinstance(response, dict) and response.get('usage') is not None:
                response['usage'] = self._usage(response['usage'], self.protocol)
            if value.get('object') == 'response':  # non-streaming response
                self._response_tools(value)
                if value.get('usage') is not None:
                    value['usage'] = self._usage(value['usage'], self.protocol)
            return value
        kind = value.get('type')
        index = value.get('index')
        if kind == 'content_block_start' and value.get('content_block', {}).get('type') == 'tool_use':
            self.has_tools = True
            block = value['content_block']
            self._call_identity(block)
            self.message_calls[index] = {'arguments': '', 'initial': block.get('input', {})}
        if kind == 'content_block_delta' and value.get('delta', {}).get('type') == 'input_json_delta':
            self.has_tools = True
            state = self.message_calls.setdefault(index, {'arguments': '', 'initial': {}})
            state['arguments'] += value['delta'].get('partial_json', '')
        if kind == 'content_block_stop' and index in self.message_calls:
            state = self.message_calls[index]
            self._checked_arguments(state['arguments'] or json.dumps(state['initial']))
        if kind == 'message_delta':
            for state in self.message_calls.values():
                self._checked_arguments(state['arguments'] or json.dumps(state['initial']))
            self._check_tool_end()
            if value.get('usage') is not None:
                value['usage'] = self._usage(value['usage'], 'messages')
        if kind == 'message_stop':
            self.terminal = True
        if kind == 'message':  # non-streaming response
            for block in value.get('content', []):
                if block.get('type') == 'tool_use':
                    self.has_tools = True
                    self._checked_arguments(json.dumps(block.get('input')))
            if value.get('usage') is not None:
                value['usage'] = self._usage(value['usage'], 'messages')
        return value

    def incomplete_response(self):
        response = copy.deepcopy(self.response) if self.response is not None else {
            'id': 'resp_r6_' + uuid.uuid4().hex, 'object': 'response',
            'created_at': int(time.time()), 'model': self.model,
        }
        # Never publish an unfinished function that an Agent could execute.
        # All tool frames remain withheld by the atomic output buffer.
        response.update(status='incomplete', error=None, output=[], usage=None,
                        incomplete_details={'reason': self.incomplete_reason()})
        return response

    def incomplete_frames(self):
        self.sequence += 1
        return [_frame('response.incomplete', {
            'type': 'response.incomplete', 'response': self.incomplete_response(),
            'sequence_number': self.sequence})]

    def errors(self, message, code):
        error = {'message': message, 'type': 'server_error', 'param': None, 'code': code}
        if self.protocol == 'chat_completions':
            return [_frame(None, {'error': error}), _frame(None, '[DONE]')]
        if self.protocol == 'messages':
            return [_frame('error', {'type': 'error', 'error': {'type': 'api_error', 'message': message}})]
        response = copy.deepcopy(self.response) if self.response is not None else {
            'id': 'resp_r6_' + uuid.uuid4().hex, 'object': 'response',
            'created_at': int(time.time()), 'model': self.model,
            'output': [], 'usage': None,
        }
        response.update(status='failed', error={'code': code, 'message': message}, output=[], usage=None)
        self.sequence += 1
        first = _frame('error', {'type': 'error', 'code': code, 'message': message, 'param': None, 'sequence_number': self.sequence})
        self.sequence += 1
        last = _frame('response.failed', {'type': 'response.failed', 'response': response, 'sequence_number': self.sequence})
        return [first, last]


async def _close(iterator):
    close = getattr(iterator, 'aclose', None)
    if close is None:
        return
    try:
        from anyio import CancelScope
    except ImportError:  # CPU-only native-library fixtures have no web stack.
        await asyncio.shield(close())
    else:
        with CancelScope(shield=True):
            await close()


async def guarded_response_body(request, infer, *, upstream_body, request_error_type=None):
    audit = NativeTokenUsage()
    token = CURRENT_AUDIT.set(audit)
    guard = _Guard(request, audit)
    processor_guard = _Guard(request, audit) if not request.stream else None
    processor_token = _PROCESSOR_GUARD.set(processor_guard)
    upstream = None
    pending = []
    holding = False
    last_tool_heartbeat = None
    closed = False
    try:
        upstream = upstream_body(request, infer)
        async for raw in upstream:
            if not request.stream:
                value = guard.process(json.loads(raw))
                pending.append(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
                continue
            # Upstream yields complete SSE frames, before transport segmentation.
            for frame in raw.split('\n\n'):
                if not frame.strip():
                    continue
                event = next((line.split(':', 1)[1].lstrip(' ') for line in frame.splitlines() if line.startswith('event:')), None)
                data = '\n'.join(line[5:].lstrip(' ') for line in frame.splitlines() if line.startswith('data:'))
                if data == '[DONE]':
                    if not guard.terminal:
                        raise OutputIntegrityError('Inference ended without a successful terminal event')
                    pending.append(_frame(event, data))
                elif data:
                    value = guard.process(json.loads(data))
                    holding = holding or _completes_call_or_response(request.protocol, value, guard)
                    encoded = _frame(event, value)
                    if holding:
                        pending.append(encoded)
                        if guard.has_tools:
                            now = _heartbeat_clock()
                            if last_tool_heartbeat is None or now - last_tool_heartbeat >= 5:
                                last_tool_heartbeat = now
                                # A standard SSE comment keeps the connection
                                # alive while atomic tool output is withheld.
                                # It carries no tool identity or arguments.
                                yield ': keepalive\n\n'
                    else:
                        yield encoded
        if request.stream and not guard.terminal:
            raise OutputIntegrityError('Inference ended without a terminal event')
        await _close(upstream)
        closed = True
        # Completed tool items and success terminals are released only after
        # the original iterator (including its finally/cleanup) completed.
        # This preserves order and avoids success followed by a late error.
        for encoded in pending:
            yield encoded
    except Exception as exc:
        # Cancellation/GeneratorExit are BaseException and must propagate.
        error_guard = processor_guard if processor_guard is not None else guard
        truncated = error_guard.truncated_tool(exc)
        diagnostic = error_guard.diagnostic(exc)
        diagnostic['classification'] = error_guard.incomplete_reason() if truncated else 'invalid_model_output' if isinstance(exc, (OutputIntegrityError, json.JSONDecodeError)) else 'inference_error'
        # Log correlation/length/hash metadata only. Exception text/tracebacks
        # from a native parser can contain private prompt or tool argument data.
        log.warning('R7 output diagnostic %s', json.dumps(diagnostic, separators=(',', ':')))
        code = diagnostic['classification']
        message = (f'Tool call was cut off by {error_guard.incomplete_reason()}; no completed tool call was returned'
                   if truncated else str(exc) if isinstance(exc, OutputIntegrityError)
                   else 'Inference could not complete; retry this request')
        if truncated and request.protocol == 'responses':
            # A normal backend length limit is incomplete, but a cleanup failure
            # must still become an explicit error rather than a dropped stream.
            try:
                await _close(upstream)
            except Exception as cleanup_error:
                truncated, code = False, 'inference_error'
                message = 'Inference cleanup could not complete; retry this request'
                cleanup_diagnostic = error_guard.diagnostic(cleanup_error)
                cleanup_diagnostic['classification'] = code
                log.warning('R7 output diagnostic %s', json.dumps(cleanup_diagnostic, separators=(',', ':')))
            else:
                closed = True
        if truncated and request.protocol == 'responses':
            if request.stream:
                for frame in error_guard.incomplete_frames():
                    yield frame
            else:
                yield json.dumps(error_guard.incomplete_response(), ensure_ascii=False, separators=(',', ':'))
        elif request.stream:
            for frame in guard.errors(message, code):
                yield frame
        elif request_error_type is not None:
            raise request_error_type(message, 502) from exc
        else:
            yield json.dumps({'error': {'message': message, 'type': 'server_error', 'code': code}}, separators=(',', ':'))
    finally:
        try:
            if not closed:
                try:
                    await _close(upstream)
                except Exception as cleanup_error:
                    log.error('R7 failed to close the original response iterator: %s', type(cleanup_error).__name__)
        finally:
            _PROCESSOR_GUARD.reset(processor_token)
            CURRENT_AUDIT.reset(token)
