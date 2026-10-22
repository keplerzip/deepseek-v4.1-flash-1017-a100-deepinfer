"""Actual upstream routes, official Rust recipe, and EngineClient-shaped inference."""
import asyncio
import hashlib
import io
import base64
import json
from pathlib import Path
import unittest

import deepseek_recipe as recipe
from fastapi import FastAPI
import httpx

from adapter.official_server import install_routes, ROOT
from test_http import FakeTokenEngine, tokenizer_path, body, dsml, tool_schema, response_text, PATHS, parse_sse


class OfficialServerTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = tokenizer_path()
        cls.tokenizer = recipe.Tokenizer.from_file(str(cls.path))

    async def asyncSetUp(self):
        self.engine = FakeTokenEngine(self.tokenizer, self.path)
        self.app = install_routes(FastAPI(), engine=self.engine, tokenizer=self.tokenizer)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)

    async def test_prebuilt_native_stack_preserves_cors_routes_and_acceptance_trace(self):
        from starlette.middleware.cors import CORSMiddleware
        from adapter.acceptance_trace import TraceMiddleware
        from unittest.mock import patch
        app=FastAPI()
        app.add_api_route('/health', lambda: {'status':'ok'}, methods=['GET'])
        app.add_middleware(CORSMiddleware, allow_origins=['http://client'])
        # Pinned SageMaker bootstrap does this before build_app returns.
        app.middleware_stack=app.build_middleware_stack()
        native_middleware=list(app.user_middleware)
        install_routes(app, engine=self.engine, tokenizer=self.tokenizer)
        install_routes(app, engine=self.engine, tokenizer=self.tokenizer)
        self.assertEqual(sum(row.cls is TraceMiddleware for row in app.user_middleware), 1)
        self.assertTrue(all(row in app.user_middleware for row in native_middleware))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            self.assertEqual((await client.get('/health')).json(), {'status':'ok'})
            with patch('adapter.acceptance_trace.active',return_value=True), patch('adapter.official_server.active',return_value=True):
                identity='d'*32
                response=await client.post('/v1/chat/completions',json=body('chat_completions',stream=True),
                    headers={'Origin':'http://client','X-Deepinfer-Acceptance-Id':identity})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.headers['access-control-allow-origin'],'http://client')
                trace=await client.get('/deepinfer/acceptance/trace/'+identity)
                self.assertEqual(trace.status_code,200,trace.text)
                self.assertTrue(trace.json()['engine_finished'])
            malformed=await client.post('/v1/chat/completions',content=b'{')
            self.assertEqual(malformed.status_code,400,malformed.text)

    async def test_acceptance_trace_records_exact_native_ids_without_changing_official_response(self):
        from unittest.mock import patch
        payload = body('chat_completions', stream=True)
        identity = 'a' * 32
        raw = json.dumps(payload).encode()
        with patch('adapter.acceptance_trace.active', return_value=True), patch('adapter.official_server.active', return_value=True):
            response = await self.client.post('/v1/chat/completions', content=raw,
                headers={'Content-Type': 'application/json', 'X-Deepinfer-Acceptance-Id': identity})
            self.assertEqual(response.status_code, 200, response.text)
            trace = (await self.client.get('/deepinfer/acceptance/trace/' + identity)).json()
            self.assertEqual(trace['request_body_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertTrue(trace['engine_finished'])
            self.assertEqual(trace['token_count'], sum(len(b['token_ids']) for b in trace['batches']))
            usage = next(frame[1]['usage'] for frame in parse_sse(response)
                         if isinstance(frame[1], dict) and frame[1].get('usage'))
            self.assertEqual(trace['token_count'], usage['completion_tokens'])
            self.assertEqual(trace['response_body_bytes'], len(response.content))
            self.assertEqual(trace['stream_boundaries'][-1]['token_count'], trace['token_count'])
            self.assertEqual((await self.client.get('/deepinfer/acceptance/trace/' + identity)).status_code, 409)
            self.assertNotIn('token_ids', response.text)
        self.assertEqual(self.engine.calls[-1]['sampling'].min_tokens, 0)

    async def test_acceptance_private_routes_and_fixed_budget_are_disabled_without_marker(self):
        from unittest.mock import patch
        headers={'X-Deepinfer-Acceptance-Id': 'b' * 32, 'X-Deepinfer-Acceptance-Fixed-Budget': '1'}
        with patch('adapter.acceptance_trace.active', return_value=False), patch('adapter.official_server.active', return_value=False):
            response = await self.client.post('/v1/chat/completions', json=body('chat_completions'), headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(self.engine.calls[-1]['sampling'].min_tokens, 0)
            self.assertEqual((await self.client.get('/deepinfer/acceptance/trace/'+'b'*32)).status_code, 404)
            self.assertEqual((await self.client.post('/deepinfer/acceptance/tokenize', json={})).status_code, 404)
        self.assertFalse(self.app.state.acceptance_traces.records)

    async def test_acceptance_tokenize_uses_same_official_prompt_and_fixed_budget_is_opt_in(self):
        from unittest.mock import patch
        payload = body('chat_completions', max_tokens=128, ignore_eos=True, return_token_ids=True, include_reasoning=True)
        with patch('adapter.acceptance_trace.active', return_value=True), patch('adapter.official_server.active', return_value=True):
            count = await self.client.post('/deepinfer/acceptance/tokenize', json=payload)
            self.assertEqual(count.status_code, 200, count.text)
            response = await self.client.post('/v1/chat/completions', json=payload,
                headers={'X-Deepinfer-Acceptance-Id': 'c'*32, 'X-Deepinfer-Acceptance-Fixed-Budget': '1'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(self.engine.calls[-1]['sampling'].min_tokens, 128)
            self.assertEqual(count.json()['count'], len(self.engine.calls[-1]['prompt']['prompt_token_ids']))

    async def test_startup_cache_reset_preserves_cpu_tier_and_closes_after_acceptance(self):
        from unittest.mock import AsyncMock, patch
        self.engine.reset_prefix_cache = AsyncMock(return_value=True)
        with patch('adapter.official_server.Path.is_file', return_value=False):
            denied = await self.client.post('/deepinfer/cache/reset-local')
        self.assertEqual(denied.status_code, 404)
        self.engine.reset_prefix_cache.assert_not_called()
        with patch('adapter.official_server.Path.is_file', return_value=True):
            accepted = await self.client.post('/deepinfer/cache/reset-local')
        self.assertEqual(accepted.status_code, 200)
        self.assertTrue(accepted.json()['success'])
        self.engine.reset_prefix_cache.assert_awaited_once_with(False, False)

    def test_upstream_source_is_unmodified_and_bound(self):
        manifest = json.loads((ROOT / "vendor/official-recipe/provenance.json").read_text())
        source = ROOT / "vendor/official-recipe" / manifest["server_path"]
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), manifest["server_sha256"])
        self.assertFalse(manifest["server_modified"])
        self.assertEqual(manifest["commit"], "935817c66f5d6e64838f419692545f06787e485f")

    async def test_all_protocols_text_usage_and_native_tokens(self):
        for protocol, path in PATHS.items():
            response = await self.client.post(path, json=body(protocol))
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertEqual(response_text(protocol, data), "你好 world")
            key = "completion_tokens" if protocol == "chat_completions" else "output_tokens"
            self.assertEqual(data["usage"][key], len(self.tokenizer.encode("你好 world")) + 1)
            self.assertIsNone(self.engine.calls[-1]["sampling"].structured_outputs)

    async def test_all_protocols_streaming(self):
        for protocol, path in PATHS.items():
            response = await self.client.post(path, json=body(protocol, stream=True))
            self.assertEqual(response.status_code, 200, response.text)
            frames = parse_sse(response)
            terminal = frames[-1][1]
            if protocol == "responses":
                self.assertEqual(terminal["type"], "response.completed")
            elif protocol == "chat_completions":
                self.assertEqual(terminal, "[DONE]")
            else:
                self.assertEqual(terminal["type"], "message_stop")

    async def test_all_protocols_tool_roundtrip_uses_official_parser(self):
        for protocol, path in PATHS.items():
            self.engine.plan(dsml("文件 run"))
            first = await self.client.post(path, json=body(protocol, tools=tool_schema(protocol)))
            self.assertEqual(first.status_code, 200, first.text)
            data = first.json()
            if protocol == "responses":
                call = next(i for i in data["output"] if i["type"] == "function_call")
                self.assertEqual(json.loads(call["arguments"]), {"task": "文件 run"})
                payload = body(protocol, tools=tool_schema(protocol), input=[{"role": "user", "content": "Hello"},
                    *data["output"], {"type": "function_call_output", "call_id": call["call_id"], "output": "ACK"}])
            elif protocol == "chat_completions":
                message = data["choices"][0]["message"]
                call = message["tool_calls"][0]
                self.assertEqual(json.loads(call["function"]["arguments"]), {"task": "文件 run"})
                payload = body(protocol, tools=tool_schema(protocol), messages=[{"role": "user", "content": "Hello"},
                    message, {"role": "tool", "tool_call_id": call["id"], "content": "ACK"}])
            else:
                call = next(i for i in data["content"] if i["type"] == "tool_use")
                self.assertEqual(call["input"], {"task": "文件 run"})
                payload = body(protocol, tools=tool_schema(protocol), messages=[{"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": data["content"]}, {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": call["id"], "content": "ACK"}]}])
            self.engine.plan("Acknowledged")
            second = await self.client.post(path, json=payload)
            self.assertEqual(second.status_code, 200, second.text)
            self.assertEqual(response_text(protocol, second.json()), "Acknowledged")

    async def test_strict_schema_composition_and_raw_string_are_not_locally_restricted(self):
        tools = [{"type": "function", "name": "shell", "strict": True,
                  "parameters": {"type": "object", "properties": {"task": {"anyOf": [
                      {"type": "string"}, {"type": "null"}]}}, "required": ["task"], "additionalProperties": False}}]
        self.engine.plan(dsml("unquoted string"))
        response = await self.client.post("/v1/responses", json=body("responses", tools=tools))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(json.loads(response.json()["output"][0]["arguments"]), {"task": "unquoted string"})
        self.assertIsNone(self.engine.calls[-1]["sampling"].structured_outputs)

    async def test_official_image_resolution_order_and_checkpoint_image_id(self):
        from PIL import Image
        data = io.BytesIO()
        Image.new("RGB", (32, 24), "red").save(data, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
        response = await self.client.post("/v1/responses", json=body("responses", input=[{
            "role": "user", "content": [{"type": "input_text", "text": "Read"},
                                         {"type": "input_image", "image_url": url}]}]))
        self.assertEqual(response.status_code, 200, response.text)
        prompt = self.engine.calls[-1]["prompt"]
        self.assertIn(129264, prompt["prompt_token_ids"])
        self.assertEqual(len(prompt["multi_modal_data"]["image"]), 1)

    async def test_cancellation_closes_and_aborts_backend(self):
        self.engine.plan("partial", mode="block")
        task = asyncio.create_task(self.client.post("/v1/responses", json=body("responses", stream=True)))
        await asyncio.wait_for(self.engine.blocked.wait(), 10)
        request_id = self.engine.calls[-1]["request_id"]
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIn(request_id, self.engine.aborted)
        self.assertIn(request_id, self.engine.closed)

    async def test_conversion_errors_are_upstream_http_errors(self):
        result = await self.client.post("/v1/responses", json=body("responses", model="unknown"))
        self.assertEqual(result.status_code, 400, result.text)
        self.assertIn("error", result.json())
        self.assertEqual(self.engine.calls, [])

    async def test_namespace_custom_patch_and_client_search_use_upstream_routes(self):
        def call(name, parameter, value):
            return (f'<｜DSML｜function_calls><｜DSML｜invoke name="{name}">'
                    f'<｜DSML｜parameter name="{parameter}" string="true">{value}'
                    '</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>')
        patch_text = '*** Begin Patch\n*** Add File: 文件.txt\n+hello\n*** End Patch'
        tools_and_outputs = [
            ([{'type':'namespace','name':'functions','tools':[{'type':'function','name':'shell',
                 'parameters':{'type':'object','properties':{'task':{'type':'string'}}}}]}],
             call('functions::shell','task','run'), 'function_call', 'function_call_output'),
            ([{'type':'custom','name':'apply_patch'}], call('apply_patch','input',patch_text),
             'custom_tool_call', 'custom_tool_call_output'),
            ([{'type':'tool_search','execution':'client','parameters':{'type':'object',
                 'properties':{'query':{'type':'string'}},'required':['query']}}],
             call('tool_search','query','shell'), 'tool_search_call', 'tool_search_output'),
        ]
        for tools, output, kind, result_kind in tools_and_outputs:
            self.engine.plan(output)
            response = await self.client.post('/v1/responses',json=body('responses',tools=tools,stream=True))
            self.assertEqual(response.status_code,200,response.text)
            final = parse_sse(response)[-1][1]['response']
            tool = next(i for i in final['output'] if i['type']==kind)
            if kind=='custom_tool_call':
                self.assertEqual(tool['input'],patch_text)
            if kind=='function_call':
                self.assertEqual(tool['namespace'],'functions')
            result = {'type':result_kind,'call_id':tool['call_id']}
            result['tools' if kind=='tool_search_call' else 'output'] = [] if kind=='tool_search_call' else 'OK'
            self.engine.plan('Accepted')
            second = await self.client.post('/v1/responses',json=body('responses',tools=tools,input=[
                {'role':'user','content':'Do it'},*final['output'],result]))
            self.assertEqual(second.status_code,200,second.text)
            self.assertEqual(response_text('responses',second.json()),'Accepted')
