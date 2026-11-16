"""Priority reaches EngineClient without rewriting official protocol bodies."""
import asyncio
import json
import unittest
import deepseek_recipe as recipe
import httpx
from fastapi import FastAPI
from adapter.official_server import install_routes,load_official,options,SPECIAL
from adapter.request_priority import CURRENT_PRIORITY
from fixtures_official import FakeTokenEngine,tokenizer_path,body,PATHS


class PriorityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        path=tokenizer_path();tokenizer=recipe.Tokenizer.from_file(str(path))
        self.engine=FakeTokenEngine(tokenizer,path)
        app=install_routes(FastAPI(),engine=self.engine,tokenizer=tokenizer)
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test')
        self.addAsyncCleanup(self.client.aclose)

    async def test_default_is_zero_for_all_protocols(self):
        for protocol,path in PATHS.items():
            response=await self.client.post(path,json=body(protocol))
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(self.engine.calls[-1]['kwargs']['priority'],0)

    async def test_explicit_priorities_are_request_local_under_concurrency(self):
        expected={}
        async def call(value):
            payload=body('chat_completions',stream=True,messages=[{'role':'user','content':'priority '+str(value)}])
            prepared=load_official().prepare_request('chat_completions',json.dumps(payload).encode(),options=options())
            ids=recipe.DeepseekV41Encoding().with_special_tokens(SPECIAL).with_tokenizer(self.engine.tokenizer).encode(prepared.conversation_request.conversation)
            expected[tuple(ids)]=value
            response=await self.client.post(PATHS['chat_completions'],json=payload,
                headers={'X-DeepInfer-Priority':str(value)})
            self.assertEqual(response.status_code,200,response.text)
        await asyncio.gather(*(call(x) for x in (-5,0,2,10,100,-100)))
        for row in self.engine.calls:
            priority=row['kwargs']['priority']
            self.assertEqual(priority,expected[tuple(row['prompt']['prompt_token_ids'])])
        self.assertCountEqual([x['kwargs']['priority'] for x in self.engine.calls],[-5,0,2,10,100,-100])
        self.assertEqual(CURRENT_PRIORITY.get(),0)
        response=await self.client.post(PATHS['chat_completions'],json=body('chat_completions'))
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.engine.calls[-1]['kwargs']['priority'],0)

    async def test_invalid_priority_rejected_before_model_and_context_reset(self):
        for value in ('nan','1.5','101','-101',''):
            response=await self.client.post(PATHS['responses'],json=body('responses'),
                headers={'X-DeepInfer-Priority':value})
            self.assertEqual(response.status_code,400,response.text)
        self.assertFalse(self.engine.calls)
        self.assertEqual(CURRENT_PRIORITY.get(),0)


if __name__=='__main__':unittest.main()
