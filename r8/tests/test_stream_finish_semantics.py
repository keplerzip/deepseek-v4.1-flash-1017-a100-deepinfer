"""Real official routes/recipe, fake native IDs; no GPU or tool execution."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import deepseek_recipe as recipe
from fastapi import FastAPI
import httpx

from adapter.official_server import install_routes
from fixtures_official import FakeTokenEngine, PATHS, body, parse_sse, tool_schema, tokenizer_path


TRUNCATED = ('<｜DSML｜function_calls><｜DSML｜invoke name="shell">'
             '<｜DSML｜parameter name="task" string="true">PRIVATE_SECRET_MATERIAL')


class FinalBatchEngine(FakeTokenEngine):
    """Engine finish metadata and last tokens can share one DELTA yield."""
    def generate(self, prompt, sampling, request_id, **kwargs):
        output, mode, finish = self.plans.popleft()
        self.calls.append({'request_id': request_id, 'prompt': prompt,
                           'sampling': sampling, 'kwargs': kwargs})

        async def produce():
            try:
                yield SimpleNamespace(prompt_token_ids=prompt['prompt_token_ids'], num_cached_tokens=0,
                    outputs=[SimpleNamespace(index=0, token_ids=self.tokenizer.encode(output),
                        finish_reason=finish, stop_reason=None, logprobs=None)], finished=True)
            finally:
                self.closed.append(request_id)
        return produce()


class StreamFinishSemanticsTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = tokenizer_path()
        cls.tokenizer = recipe.Tokenizer.from_file(str(cls.path))

    async def asyncSetUp(self):
        self.engine = FakeTokenEngine(self.tokenizer, self.path)
        self.app = install_routes(FastAPI(), engine=self.engine, tokenizer=self.tokenizer)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')
        self.addAsyncCleanup(self.client.aclose)

    async def request(self, protocol='responses', *, stream=True, finish='length'):
        self.engine.plan(TRUNCATED, finish=finish)
        return await self.client.post(PATHS[protocol], json=body(protocol, stream=stream,
            tools=tool_schema(protocol)))

    async def test_responses_length_stream_keeps_incomplete_without_tool_done(self):
        response = await self.request()
        self.assertEqual(response.status_code, 200)
        frames = [v for _, v in parse_sse(response) if isinstance(v, dict)]
        terminal = frames[-1]
        self.assertEqual(terminal['type'], 'response.incomplete')
        self.assertEqual(terminal['response']['incomplete_details'], {'reason': 'max_output_tokens'})
        self.assertEqual(terminal['response']['status'], 'incomplete')
        self.assertEqual(terminal['response']['output'], [])
        self.assertEqual(terminal['response']['id'], frames[0]['response']['id'])
        self.assertFalse(any(v['type'] in ('error', 'response.failed', 'response.completed',
            'response.function_call_arguments.done', 'response.output_item.done') for v in frames))
        self.assertEqual(len(self.engine.closed), 1)

    async def test_responses_length_nonstream_is_incomplete_not_502(self):
        response = await self.request(stream=False)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data['status'], 'incomplete')
        self.assertEqual(data['incomplete_details'], {'reason': 'max_output_tokens'})
        self.assertEqual(data['output'], [])
        self.assertIsNone(data['error'])
        self.assertEqual(len(self.engine.closed), 1)

    async def test_stop_with_bad_json_stays_failed_in_all_protocols(self):
        for protocol in PATHS:
            for stream in (True, False):
                with self.subTest(protocol=protocol, stream=stream):
                    response = await self.request(protocol, stream=stream, finish='stop')
                    if not stream:
                        self.assertEqual(response.status_code, 502, response.text)
                        self.assertIn('invalid JSON', response.json()['error']['message'])
                    else:
                        frames = [v for _, v in parse_sse(response) if isinstance(v, dict)]
                        self.assertTrue(any(v.get('error') or v.get('type') == 'error' for v in frames), response.text)
                        self.assertFalse(any(v.get('type') == 'response.incomplete' for v in frames))

    async def test_chat_and_messages_length_are_explicit_unexecutable_errors(self):
        for protocol in ('chat_completions', 'messages'):
            with self.subTest(protocol=protocol):
                response = await self.request(protocol)
                frames = [v for _, v in parse_sse(response) if isinstance(v, dict)]
                errors = [v['error'] for v in frames if v.get('error')]
                self.assertTrue(errors, response.text)
                self.assertIn('max_output_tokens', errors[-1]['message'])
                if protocol == 'chat_completions':
                    self.assertEqual(errors[-1]['code'], 'max_output_tokens')
                    self.assertFalse(any(c.get('finish_reason') for v in frames for c in v.get('choices', [])))
                else:
                    self.assertFalse(any(v.get('type') in ('content_block_stop', 'message_stop') for v in frames))

    async def test_diagnostics_are_hashes_counts_and_positions_without_private_text(self):
        with self.assertLogs('adapter.stream_integrity', level='WARNING') as captured:
            await self.request(finish='stop')
        text = '\n'.join(captured.output)
        self.assertNotIn('PRIVATE_SECRET_MATERIAL', text)
        row = json.loads(text.split('R7 output diagnostic ', 1)[1])
        self.assertEqual(row['backend_finish_reason'], 'stop')
        self.assertTrue(row['backend_finished'])
        self.assertEqual(row['requested_max_tokens'], 512)
        self.assertEqual(row['effective_max_tokens'], 512)
        self.assertGreater(row['prompt_tokens'], 0)
        self.assertGreater(row['output_tokens'], 0)
        self.assertEqual(len(row['arguments_sha256']), 64)
        self.assertEqual(len(row['call_sha256']), 64)
        self.assertIsInstance(row['json_error_position'], int)

    async def test_length_on_final_token_batch_is_known_before_parser_flush(self):
        engine = FinalBatchEngine(self.tokenizer, self.path)
        app = install_routes(FastAPI(), engine=engine, tokenizer=self.tokenizer)
        engine.plan(TRUNCATED, finish='length')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post(PATHS['responses'], json=body('responses', stream=True,
                tools=tool_schema('responses'), max_output_tokens=4096))
        frames = [v for _, v in parse_sse(response) if isinstance(v, dict)]
        self.assertEqual(frames[-1]['type'], 'response.incomplete')
        self.assertEqual(engine.calls[0]['sampling'].max_tokens, 4096)
        self.assertEqual(len(engine.closed), 1)

    async def test_truncated_response_cleanup_failure_is_explicit_error(self):
        generate = self.engine.generate

        def failing_close(*args, **kwargs):
            iterator = generate(*args, **kwargs)

            async def produce():
                try:
                    async for chunk in iterator:
                        yield chunk
                finally:
                    await iterator.aclose()
                    raise RuntimeError('PRIVATE_CLOSE_DETAIL')
            return produce()

        self.engine.generate = failing_close
        with self.assertLogs('adapter.stream_integrity', level='WARNING') as captured:
            response = await self.request()
        frames = [v for _, v in parse_sse(response) if isinstance(v, dict)]
        self.assertEqual(frames[-1]['type'], 'response.failed')
        self.assertEqual(frames[-1]['response']['error']['code'], 'inference_error')
        self.assertFalse(any(v['type'] in ('response.incomplete', 'response.completed') for v in frames))
        self.assertNotIn('PRIVATE_CLOSE_DETAIL', response.text)
        self.assertNotIn('PRIVATE_CLOSE_DETAIL', '\n'.join(captured.output))

    def test_length_classification_requires_real_finished_backend(self):
        from adapter.native_token_usage import NativeTokenUsage
        from adapter.stream_integrity import _Guard, OutputIntegrityError
        audit = NativeTokenUsage()
        guard = _Guard(SimpleNamespace(protocol='responses', model='DeepSeek-Flash'), audit)
        error = OutputIntegrityError('bad JSON', kind='invalid_tool_json')
        for reason, finished, expected in [(None, False, False), ('stop', True, False),
                                          ('length', False, False), ('length', True, True)]:
            audit.backend_finish_reason, audit.backend_finished = reason, finished
            self.assertEqual(guard.truncated_tool(error), expected)


if __name__ == '__main__':
    unittest.main()
