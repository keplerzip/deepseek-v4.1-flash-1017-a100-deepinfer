"""R7 atomic tool-output contracts with actual recipe/ASGI, no GPUs/tools run."""
import asyncio
import json
import itertools
import os
import unittest
from unittest.mock import patch

import deepseek_recipe as recipe
from fastapi import FastAPI
import httpx

from adapter.official_server import install_routes
from adapter.stream_integrity import _arguments, OutputIntegrityError
from fixtures_official import FakeTokenEngine, PATHS, body, parse_sse, tokenizer_path, tool_schema


def dsml(name='shell', key='task', value='echo ok', closed=True):
    result = ('<｜DSML｜tool_calls><｜DSML｜invoke name="' + name + '">'
              '<｜DSML｜parameter name="' + key + '" string="true">' + value +
              '</｜DSML｜parameter></｜DSML｜invoke>')
    return result + ('</｜DSML｜tool_calls>' if closed else '')


def tool_frame(protocol, frame):
    if not isinstance(frame, dict):
        return False
    if protocol == 'responses':
        return (frame.get('type', '').startswith(('response.function_call_arguments.', 'response.custom_tool_call_input.'))
                or (frame.get('type') in ('response.output_item.added', 'response.output_item.done')
                    and frame.get('item', {}).get('type') in ('function_call', 'custom_tool_call', 'tool_search_call')))
    if protocol == 'chat_completions':
        return any(c.get('delta', {}).get('tool_calls') for c in frame.get('choices', []))
    return ((frame.get('type') == 'content_block_start' and frame.get('content_block', {}).get('type') == 'tool_use')
            or (frame.get('type') == 'content_block_delta' and frame.get('delta', {}).get('type') == 'input_json_delta'))


class R7AtomicToolTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = tokenizer_path()
        cls.tokenizer = recipe.Tokenizer.from_file(str(cls.path))

    async def asyncSetUp(self):
        self.engine = FakeTokenEngine(self.tokenizer, self.path)
        self.app = install_routes(FastAPI(), engine=self.engine, tokenizer=self.tokenizer)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')
        self.addAsyncCleanup(self.client.aclose)

    async def send(self, protocol, text, *, finish='stop', stream=True, custom=False):
        self.engine.plan(text, finish=finish)
        tools = [{'type': 'custom', 'name': 'apply_patch'}] if custom else tool_schema(protocol)
        return await self.client.post(PATHS[protocol], json=body(protocol, stream=stream, tools=tools))

    async def test_bad_json_never_exposes_tool_start_or_argument_fragments(self):
        broken = '<｜DSML｜tool_calls><｜DSML｜invoke name="shell"><｜DSML｜parameter name="task" string="true">PRIVATE_TOOL_FRAGMENT'
        for protocol in PATHS:
            with self.subTest(protocol=protocol):
                response = await self.send(protocol, broken)
                frames = [v for _, v in parse_sse(response)]
                self.assertFalse(any(tool_frame(protocol, f) for f in frames), response.text)
                self.assertNotIn('PRIVATE_TOOL_FRAGMENT', response.text)
                self.assertTrue(any(isinstance(f, dict) and (f.get('error') or f.get('type') == 'error') for f in frames))

    async def test_length_and_filter_do_not_publish_even_syntactically_valid_tools(self):
        for protocol in PATHS:
            for reason in ('length', 'content_filter'):
                for stream in (True, False):
                    with self.subTest(protocol=protocol, reason=reason, stream=stream):
                        # JSON is complete; the outer DSML block is unfinished.
                        response = await self.send(protocol, dsml(closed=False), finish=reason, stream=stream)
                        expected = 'max_output_tokens' if reason == 'length' else 'content_filter'
                        if stream:
                            frames = [v for _, v in parse_sse(response)]
                            self.assertFalse(any(tool_frame(protocol, f) for f in frames), response.text)
                            if protocol == 'responses':
                                terminal = frames[-1]
                                self.assertEqual(terminal['type'], 'response.incomplete')
                                self.assertEqual(terminal['response']['incomplete_details'], {'reason': expected})
                                self.assertEqual(terminal['response']['output'], [])
                            else:
                                self.assertIn(expected, response.text)
                        elif protocol == 'responses':
                            self.assertEqual(response.status_code, 200, response.text)
                            self.assertEqual(response.json()['status'], 'incomplete')
                            self.assertEqual(response.json()['incomplete_details'], {'reason': expected})
                            self.assertEqual(response.json()['output'], [])
                        else:
                            self.assertEqual(response.status_code, 502, response.text)
                            self.assertIn(expected, response.json()['error']['message'])

    async def test_custom_apply_patch_truncation_never_publishes_custom_input(self):
        for reason in ('length', 'content_filter'):
            for stream in (True, False):
                with self.subTest(reason=reason, stream=stream):
                    response = await self.send('responses', dsml('apply_patch', 'input', '*** Begin Patch\n+partial', False),
                        finish=reason, stream=stream, custom=True)
                    if stream:
                        frames = [v for _, v in parse_sse(response)]
                        self.assertFalse(any(tool_frame('responses', f) for f in frames), response.text)
                        self.assertEqual(frames[-1]['type'], 'response.incomplete')
                    else:
                        self.assertEqual(response.json()['status'], 'incomplete')
                        self.assertEqual(response.json()['output'], [])

    async def test_tool_output_is_atomic_until_cleanup_but_plain_text_keeps_streaming(self):
        for protocol, custom, ordinary in [('responses', False, False), ('responses', True, False),
                                          ('chat_completions', False, False), ('messages', False, False),
                                          ('responses', False, True)]:
            with self.subTest(protocol=protocol, custom=custom, ordinary=ordinary):
                engine = FakeTokenEngine(self.tokenizer, self.path)
                entered, release = asyncio.Event(), asyncio.Event()
                original_generate = engine.generate

                def generate(*args, **kwargs):
                    iterator = original_generate(*args, **kwargs)

                    async def produce():
                        try:
                            async for chunk in iterator:
                                yield chunk
                        finally:
                            entered.set()
                            await release.wait()
                            await iterator.aclose()
                    return produce()

                engine.generate = generate
                app = install_routes(FastAPI(), engine=engine, tokenizer=self.tokenizer)
                sent = []

                async def record_app(scope, receive, send):
                    async def capture(message):
                        if message['type'] == 'http.response.body':
                            sent.append(message.get('body', b''))
                        await send(message)
                    await app(scope, receive, capture)

                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=record_app), base_url='http://test') as client:
                    text = 'Visible ordinary text' if ordinary else dsml('apply_patch', 'input', '*** Begin Patch\n*** End Patch') if custom else dsml()
                    engine.plan(text)
                    tools = [] if ordinary else [{'type': 'custom', 'name': 'apply_patch'}] if custom else tool_schema(protocol)
                    clock = itertools.count(0, 3)
                    with patch('adapter.stream_integrity._heartbeat_clock', side_effect=lambda: next(clock)):
                        task = asyncio.create_task(client.post(PATHS[protocol], json=body(protocol, stream=True, tools=tools)))
                        try:
                            await asyncio.wait_for(entered.wait(), 5)
                            wire = b''.join(sent)
                            before = parse_sse(type('Response', (), {'text': wire.decode()})())
                            self.assertFalse(any(tool_frame(protocol, f) for _, f in before))
                            if ordinary:
                                self.assertTrue(any(isinstance(f, dict) and f.get('type') == 'response.output_text.delta' for _, f in before))
                                self.assertNotIn(b': keepalive\n\n', wire)
                            else:
                                self.assertGreaterEqual(wire.count(b': keepalive\n\n'), 2)
                        finally:
                            release.set()
                        response = await asyncio.wait_for(task, 5)
                frames = [f for _, f in parse_sse(response)]
                if not ordinary:
                    self.assertTrue(any(tool_frame(protocol, f) for f in frames), response.text)

    async def test_custom_cleanup_failure_does_not_leak_tool_frames(self):
        generate = self.engine.generate

        def failing_close(*args, **kwargs):
            iterator = generate(*args, **kwargs)

            async def produce():
                try:
                    async for chunk in iterator:
                        yield chunk
                finally:
                    await iterator.aclose()
                    raise RuntimeError('PRIVATE_CUSTOM_CLOSE')
            return produce()

        self.engine.generate = failing_close
        response = await self.send('responses', dsml('apply_patch', 'input', '*** Begin Patch\n*** End Patch'), custom=True)
        frames = [v for _, v in parse_sse(response)]
        self.assertFalse(any(tool_frame('responses', f) for f in frames), response.text)
        self.assertEqual(frames[-1]['type'], 'response.failed')
        self.assertNotIn('PRIVATE_CUSTOM_CLOSE', response.text)

    async def test_cancelling_held_tool_output_closes_and_aborts_without_hanging(self):
        sent = []

        async def record_app(scope, receive, send):
            async def capture(message):
                if message['type'] == 'http.response.body':
                    sent.append(message.get('body', b''))
                await send(message)
            await self.app(scope, receive, capture)

        self.engine.plan(dsml(), mode='block')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=record_app), base_url='http://test') as client:
            task = asyncio.create_task(client.post(PATHS['responses'], json=body('responses', stream=True,
                tools=tool_schema('responses'))))
            await asyncio.wait_for(self.engine.blocked.wait(), 5)
            self.assertIn(b': keepalive\n\n', b''.join(sent))
            frames = parse_sse(type('Response', (), {'text': b''.join(sent).decode()})())
            self.assertFalse(any(tool_frame('responses', f) for _, f in frames))
            request_id = self.engine.calls[-1]['request_id']
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        self.assertIn(request_id, self.engine.closed)
        self.assertIn(request_id, self.engine.aborted)

    def test_suffix_default_is_audit_only_for_valid_comments_and_heredocs(self):
        for command in (': # </parameter>', "cat <<'</parameter>'\nhello\n</parameter>\n"):
            with self.subTest(command=command), patch.dict(os.environ, {}, clear=True):
                with self.assertLogs('adapter.stream_integrity', level='INFO') as captured:
                    self.assertEqual(_arguments(json.dumps({'cmd': command})), {'cmd': command})
                row = json.loads(captured.output[0].split('R7 output diagnostic ', 1)[1])
                self.assertEqual(row['classification'], 'tool_suffix_observed')
                self.assertTrue(row['audit_only'])
                self.assertEqual(len(row['arguments_sha256']), 64)
                self.assertNotIn(command, '\n'.join(captured.output))

    def test_suffix_legacy_block_remains_explicit_opt_in(self):
        with patch.dict(os.environ, {'DEEPINFER_R6_TOOL_SUFFIX_GUARD': '1'}):
            with self.assertRaises(OutputIntegrityError):
                _arguments(json.dumps({'cmd': ': # </parameter>'}))

    def test_suffix_observation_is_deduplicated_by_request_argument_hash(self):
        from adapter.native_token_usage import NativeTokenUsage
        from adapter.stream_integrity import CURRENT_AUDIT
        token = CURRENT_AUDIT.set(NativeTokenUsage())
        try:
            with patch.dict(os.environ, {}, clear=True), self.assertLogs('adapter.stream_integrity', level='INFO') as captured:
                for _ in range(3):
                    _arguments(json.dumps({'cmd': ': # </parameter>'}))
            self.assertEqual(len(captured.output), 1)
        finally:
            CURRENT_AUDIT.reset(token)

    async def test_tool_search_incomplete_status_is_not_an_executable_empty_call(self):
        for valid, stream in ((False, True), (False, False), (True, True), (True, False)):
            with self.subTest(valid=valid, stream=stream):
                text = dsml('tool_search', 'query', 'shell')
                if not valid:
                    text = text.replace('string="true"', 'string="false"')
                self.engine.plan(text)
                response = await self.client.post(PATHS['responses'], json=body('responses', stream=stream,
                    tools=[{'type': 'tool_search', 'execution': 'client'}]))
                if valid:
                    self.assertEqual(response.status_code, 200, response.text)
                    if stream:
                        frames = [f for _, f in parse_sse(response)]
                        self.assertEqual(frames[-1]['type'], 'response.completed')
                        item = next(f['item'] for f in frames if f.get('type') == 'response.output_item.done')
                    else:
                        item = response.json()['output'][0]
                    self.assertEqual(item['arguments'], {'query': 'shell'})
                    self.assertEqual(item['status'], 'completed')
                elif stream:
                    frames = [f for _, f in parse_sse(response)]
                    self.assertFalse(any(tool_frame('responses', f) for f in frames), response.text)
                    self.assertEqual(frames[-1]['type'], 'response.failed')
                else:
                    self.assertEqual(response.status_code, 502, response.text)


if __name__ == '__main__':
    unittest.main()
