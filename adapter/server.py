"""Install official-recipe API routes in the pinned vLLM server process."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

import deepseek_recipe as recipe
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .backend import VllmBackend
from .protocol import ProtocolError, ProtocolStream


logger = logging.getLogger(__name__)
MAX_BODY_BYTES = 128 * 1024**2
REPLACED = {"/v1/responses", "/v1/chat/completions", "/v1/messages", "/messages",
            "/v1/models", "/tokenize", "/v1/messages/count_tokens", "/messages/count_tokens"}


class RequestTooLarge(ProtocolError):
    pass


async def read_body(request):
    body, length = bytearray(), 0
    async for part in request.stream():
        length += len(part)
        if length > MAX_BODY_BYTES:
            raise RequestTooLarge("Request body exceeds 128 MiB")
        body.extend(part)
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ProtocolError("Request body must be an object")
    if data.get("n", 1) != 1:
        raise ProtocolError("Only n=1 is supported")
    return data


def error_response(exc, status=400):
    if isinstance(exc, RequestTooLarge):
        status = 413
    if isinstance(exc, recipe.ConversionError):
        try:
            body = json.loads(exc.body)
        except (TypeError, ValueError):
            body = {"error": {"message": str(exc), "type": "invalid_request_error"}}
        return JSONResponse(body, status_code=exc.status_code)
    return JSONResponse({"error": {"message": str(exc), "type": "invalid_request_error" if status < 500 else "server_error"}}, status_code=status)


def install_routes(app: FastAPI, backend=None):
    """Native health/metrics stay in vLLM; protocol routes share its EngineClient."""
    app.router.routes[:] = [route for route in app.router.routes
                           if getattr(route, "path", None) not in REPLACED
                           and not (getattr(route, "path", "") or "").startswith("/v1/responses/")]
    if backend is not None:
        app.state.deepinfer_backend = backend

    async def get_backend(request):
        if not hasattr(request.app.state, "deepinfer_backend"):
            request.app.state.deepinfer_backend = VllmBackend(request.app.state.engine_client)
        return request.app.state.deepinfer_backend

    async def events(request, backend, generation, stream, request_id):
        iterator = backend.chunks(generation, request_id)
        pending = None
        try:
            while not stream.finished:
                pending = asyncio.create_task(anext(iterator))
                while not pending.done():
                    done, _ = await asyncio.wait({pending}, timeout=15)
                    if not done:
                        yield ": keep-alive\n\n"
                try:
                    chunk = pending.result()
                except StopAsyncIteration:
                    for frame in stream.sse(stream.finish()):
                        yield frame
                    break
                finally:
                    pending = None
                for frame in stream.sse(stream.push(chunk)):
                    yield frame
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Generation failed: %s", request_id)
            for frame in stream.sse(stream.fail("Backend generation failed; inspect engine logs", code="backend_error")):
                yield frame
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await iterator.aclose()
            stream.close()

    async def generate(request, protocol):
        try:
            body = await read_body(request)
            backend = await get_backend(request)
            generation = await backend.prepare(protocol, body)
            stream = ProtocolStream(generation.prepared)
        except (ProtocolError, recipe.ConversionError, ValueError) as exc:
            return error_response(exc)
        except Exception:
            logger.exception("Request preparation failed")
            return error_response(RuntimeError("Backend is unavailable; inspect engine logs"), 503)
        request_id = "recipe-" + uuid.uuid4().hex
        if generation.prepared.stream:
            return StreamingResponse(events(request, backend, generation, stream, request_id),
                                     media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        iterator = backend.chunks(generation, request_id)
        collector = watcher = None
        try:
            async def collect():
                async for chunk in iterator:
                    stream.push(chunk)
                    if stream.finished:
                        break
                if not stream.finished:
                    stream.finish()

            async def disconnected():
                while not collector.done():
                    if await request.is_disconnected():
                        return
                    await asyncio.sleep(0.25)

            collector = asyncio.create_task(collect())
            watcher = asyncio.create_task(disconnected())
            done, _ = await asyncio.wait({collector, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if collector not in done:
                raise asyncio.CancelledError()
            await collector
            return JSONResponse(stream.response, status_code=200 if stream.successful else 502)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Generation failed: %s", request_id)
            stream.fail("Backend generation failed; inspect engine logs")
            return JSONResponse(stream.response, status_code=502)
        finally:
            for task in (collector, watcher):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*(t for t in (collector, watcher) if t is not None), return_exceptions=True)
            await iterator.aclose()
            stream.close()

    async def responses(request: Request):
        return await generate(request, "responses")

    async def chat(request: Request):
        return await generate(request, "chat_completions")

    async def messages(request: Request):
        return await generate(request, "messages")

    async def models():
        return {"object": "list", "data": [{"id": "DeepSeek-Flash", "object": "model", "owned_by": "local"}]}

    async def tokenize(request: Request):
        try:
            body = await read_body(request)
            backend = await get_backend(request)
            protocol = "chat_completions" if "messages" in body else "responses"
            body["stream"] = False
            if protocol == "chat_completions":
                body["max_tokens"] = 1
            else:
                body["max_output_tokens"] = 1
                if "prompt" in body:
                    body["input"] = body.pop("prompt")
            generation = await backend.prepare(protocol, body)
            return {"count": generation.prepared.prompt_tokens, "max_model_len": generation.prepared.context_limit,
                    "tokens": generation.engine_input["prompt_token_ids"]}
        except (ProtocolError, recipe.ConversionError, ValueError) as exc:
            return error_response(exc)
        except Exception:
            logger.exception("Tokenization preparation failed")
            return error_response(RuntimeError("Backend is unavailable; inspect engine logs"), 503)

    async def count_tokens(request: Request):
        try:
            body = await read_body(request)
            body["max_tokens"] = 1
            backend = await get_backend(request)
            generation = await backend.prepare("messages", body)
            return {"input_tokens": generation.prepared.prompt_tokens}
        except (ProtocolError, recipe.ConversionError, ValueError) as exc:
            return error_response(exc)
        except Exception:
            logger.exception("Token-count preparation failed")
            return error_response(RuntimeError("Backend is unavailable; inspect engine logs"), 503)

    async def compatibility():
        return {"adapter": "deepinfer-sm80-recipe", "recipe_version": recipe.__version__,
                "gpu_engine": "pinned-vllm-sm80-r2", "native_deepinfer_runtime": False,
                "responses_stateful": False, "tool_execution": "client", "remote_image_fetch": False}

    for path, endpoint in [("/v1/responses", responses), ("/v1/chat/completions", chat),
                           ("/v1/messages", messages), ("/messages", messages),
                           ("/tokenize", tokenize), ("/v1/messages/count_tokens", count_tokens),
                           ("/messages/count_tokens", count_tokens)]:
        app.add_api_route(path, endpoint, methods=["POST"])
    app.add_api_route("/v1/models", models, methods=["GET"])
    app.add_api_route("/deepinfer/compatibility", compatibility, methods=["GET"])
    app.openapi_schema = None
    return app


def main():
    from vllm.entrypoints.launchers.api_server import entry

    native_build_app = entry.build_app

    def build_app(*args, **kwargs):
        return install_routes(native_build_app(*args, **kwargs))

    entry.build_app = build_app
    entry.main()


if __name__ == "__main__":
    main()
