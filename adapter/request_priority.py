"""Optional request-local scheduling priority; official bodies stay untouched."""
from contextvars import ContextVar

CURRENT_PRIORITY=ContextVar('deepinfer_request_priority',default=0)
PATHS={'/v1/responses','/v1/chat/completions','/v1/messages'}


class PriorityMiddleware:
    def __init__(self,app):
        self.app=app

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('path') not in PATHS:
            return await self.app(scope,receive,send)
        values=[value for key,value in scope.get('headers',[]) if key.lower()==b'x-deepinfer-priority']
        try:
            if len(values)>1:raise ValueError()
            value=0 if not values else int(values[0].decode('ascii'))
            if not -100<=value<=100:raise ValueError()
        except (ValueError,UnicodeError):
            from starlette.responses import JSONResponse
            return await JSONResponse({'error':{'message':'X-DeepInfer-Priority must be one integer in [-100,100]',
                                              'type':'invalid_request_error'}},status_code=400)(scope,receive,send)
        token=CURRENT_PRIORITY.set(value)
        try:
            await self.app(scope,receive,send)
        finally:
            CURRENT_PRIORITY.reset(token)
