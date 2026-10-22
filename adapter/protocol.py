"""Official recipe conversion with an explicit, failure-safe backend contract.

No HTTP, model execution, image fetch, or tool execution happens here. Backends
must preserve generated control tokens and report Ready and Finish explicitly.
The recipe parser remains the single source for streaming and full responses.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import deepseek_recipe as recipe


PROTOCOL_TYPES = {
    "responses": (recipe.ResponsesRequest, recipe.ResponsesResponse),
    "chat_completions": (recipe.ChatCompletionRequest, recipe.ChatCompletionResponse),
    "messages": (recipe.MessagesRequest, recipe.MessagesResponse),
}
FLASH_ALIASES = (
    "DeepSeek-Flash", "deepseek-flash", "DeepSeek-V4.1-Flash",
    "DeepSeek-V4.1-Flash-1017", "deepseek-v4.1-flash-1017",
)
# The 1017 checkpoint's tokenizer uses this spelling for token ID 129264.
# Recipe's generic V4.1 default (<｜image｜>) is absent from this vocabulary.
SPECIAL_TOKENS = recipe.SpecialTokens(image="<｜deepseek_image｜>")


class ProtocolError(ValueError):
    """A request or backend update violates this adapter's contract."""


@dataclass(frozen=True)
class PreparedRequest:
    protocol: str
    model: str
    converted: Any
    prompt_token_ids: list[int]
    rendered: Any
    image_sources: list[Any]
    inference_options: Any
    tokenizer: Any
    max_tokens: int
    prompt_tokens: int
    context_limit: int

    @property
    def stream(self) -> bool:
        return self.converted.stream


def conversion_options() -> Any:
    """Use the official thinking/tool round-trip policy for this Flash backend."""
    return recipe.ConversionOptions(
        reasoning_content_behavior=recipe.ReasoningContentBehavior.Reject,
    ).with_special_tokens(SPECIAL_TOKENS).with_model_versions_by_name({
        alias: recipe.ModelVersion.DeepseekV41Flash for alias in FLASH_ALIASES
    })


def prepare(
    protocol: str,
    body: Mapping[str, Any] | str | bytes,
    tokenizer: Any,
    *,
    context_limit: int = 524288,
    image_token_adjustment: int = 0,
) -> PreparedRequest:
    """Validate and encode using recipe's special-token-aware prompt encoder.

    ``inference_options`` is the native object; ``max_tokens`` is separately
    resolved after encoding and MUST be passed to the backend. Image sources
    are returned without I/O; multimodal backends must preprocess them and
    supply the resulting token adjustment before accepting inference.
    """
    if protocol not in PROTOCOL_TYPES:
        raise ProtocolError(f"Unsupported protocol: {protocol}")
    if tokenizer is None:
        raise ProtocolError("A matching recipe Tokenizer is required")
    if context_limit <= 0 or image_token_adjustment < 0:
        raise ProtocolError("Invalid context limit or image token adjustment")
    data = json.loads(body) if isinstance(body, (str, bytes)) else copy.deepcopy(dict(body))
    if not isinstance(data, dict):
        raise ProtocolError("Request body must be an object")
    # DeepSeek Responses is stateless. Reject stateful requests explicitly
    # instead of silently ignoring a continuation identifier and losing history.
    if protocol == "responses":
        for field in ("previous_response_id", "conversation"):
            if data.get(field) is not None:
                raise ProtocolError(f"{field} is unsupported; send complete input history")
        if data.get("store") is True or data.get("background") is True:
            raise ProtocolError("Stored/background responses are unsupported")
    request_type, _ = PROTOCOL_TYPES[protocol]
    request = request_type(data)
    model = request.model
    converted = request.convert(conversion_options())
    encoding = recipe.DeepseekV41Encoding().with_special_tokens(SPECIAL_TOKENS).with_tokenizer(tokenizer)
    rendered = encoding.render_conversation(converted.conversation)
    prompt_token_ids = encoding.encode(converted.conversation)
    prompt_tokens = len(prompt_token_ids) + image_token_adjustment
    if prompt_tokens >= context_limit:
        raise ProtocolError("Prompt leaves no room for output in the context window")
    max_tokens = converted.resolve_max_tokens(prompt_tokens, context_limit)
    return PreparedRequest(
        protocol, model, converted, prompt_token_ids, rendered,
        list(rendered.image_sources), converted.inference_options, tokenizer,
        max_tokens, prompt_tokens, context_limit,
    )


class ProtocolStream:
    """One backend generation, producing JSON events and their full response.

    A backend EOF without Finish is failure, never recipe's implicit success.
    Token ID chunks are preferred. Text chunks require truthful token counts.
    After ``finished``, cancel/close the backend iterator without reading more.
    ``close`` abandons the stream without inventing any terminal event.
    """

    def __init__(self, prepared: PreparedRequest, *, response_id: str | None = None):
        self.prepared = prepared
        self.response_id = response_id or "resp_" + uuid.uuid4().hex
        request_type, self._response_type = PROTOCOL_TYPES[prepared.protocol]
        generator = request_type.chunk_generator(
            prepared.converted, self.response_id, prepared.model,
        )
        self._processor = recipe.StreamProcessor(
            generator, prepared.converted.parsing_options, prepared.tokenizer,
        )
        self._accumulator = self._response_type(
            self.response_id, prepared.model, int(time.time()), 0, 0,
        )
        self._ready = False
        self._explicit_finish = False
        self._finished = False
        self._closed = False
        self._successful = False
        self._sequence = 0
        self._tokens_seen = 0
        self._final: dict[str, Any] | None = None
        self._sse_done_sent = False

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def successful(self) -> bool:
        """True for completed/incomplete normal endings, false for failures."""
        return self._successful

    @property
    def response(self) -> dict[str, Any]:
        return copy.deepcopy(self._final) if self._final is not None else json.loads(
            self._accumulator.to_json()
        )

    def _consume(self, chunks: Iterable[Any]) -> list[dict[str, Any]]:
        events = []
        for chunk in chunks:
            event = json.loads(chunk.to_json())
            self._accumulator.accumulate(chunk)
            if "sequence_number" in event:
                self._sequence = event["sequence_number"] + 1
            events.append(event)
        return events

    @staticmethod
    def _validate_completed_tools(response: dict[str, Any]) -> None:
        """Do not certify unfinished JSON as an executable completed call.

        This is validation, not repair: schema/wrapper contents remain exactly
        as the official decoder produced them. Length-truncated calls are
        intentionally allowed in an incomplete response.
        """
        if response.get("status") == "incomplete":
            return
        calls = [item for item in response.get("output", []) if item.get("type") == "function_call"]
        for choice in response.get("choices", []):
            calls.extend(call.get("function", {}) for call in
                         choice.get("message", {}).get("tool_calls", []))
        for call in calls:
            try:
                arguments = json.loads(call.get("arguments", ""))
            except (ValueError, TypeError) as exc:
                raise ProtocolError("Backend stopped with unfinished or invalid tool arguments") from exc
            if not isinstance(arguments, dict):
                raise ProtocolError("Completed function arguments must be a JSON object")
        for item in response.get("output", []):
            if item.get("type") == "tool_search_call" and item.get("status") != "completed":
                raise ProtocolError("Backend stopped with invalid tool_search arguments")

    def push(self, chunk: Any) -> list[dict[str, Any]]:
        if self._finished or self._closed:
            raise ProtocolError("Generation is already finished or closed")
        try:
            kind = chunk.kind
            if kind == "ready":
                if self._ready:
                    raise ProtocolError("Backend sent Ready more than once")
                usage = chunk.prompt_usage
                if usage is None or usage.prompt_tokens != self.prepared.prompt_tokens:
                    raise ProtocolError("Backend prompt usage does not match encoded prompt")
                if not 0 <= usage.prompt_cache_hit_tokens <= usage.prompt_tokens:
                    raise ProtocolError("Backend cache token count is invalid")
                self._ready = True
            elif not self._ready:
                raise ProtocolError("Backend must report Ready with actual prompt usage first")
            if kind in ("text", "tokens"):
                count = len(chunk.token_ids) if kind == "tokens" else chunk.token_count
                if count is None or count < 0 or (kind == "text" and chunk.content and count == 0):
                    raise ProtocolError("Text chunks require an actual generated token count")
                if self._tokens_seen + count > self.prepared.max_tokens:
                    raise ProtocolError("Backend exceeded the requested output token limit")
                self._tokens_seen += count
            if kind == "finish":
                self._explicit_finish = True
            chunks = self._processor.push(chunk)
            if self.prepared.protocol == "responses":
                for native_event in chunks:
                    event = json.loads(native_event.to_json())
                    if event.get("type") == "response.completed":
                        self._validate_completed_tools(event["response"])
            events = self._consume(chunks)
            if self._processor.finished:
                # The official parser may itself honor a requested stop
                # sequence. This is a positive termination signal, unlike an
                # unmarked backend EOF. The caller then cancels the engine.
                parser_stop = kind in ("tokens", "text") and bool(
                    self.prepared.converted.parsing_options.stop_sequences
                )
                if not self._explicit_finish and not parser_stop:
                    raise ProtocolError("Parser ended before an explicit backend Finish")
                if chunk.finish_reason != recipe.InferenceFinishReason.Length:
                    self._validate_completed_tools(json.loads(self._accumulator.to_json()))
                self._finished = True
                self._successful = True
                self._final = json.loads(self._accumulator.to_json())
                self._processor.close()
            return events
        except Exception as exc:
            return self.fail(str(exc), code="backend_protocol_error")

    def finish(self) -> list[dict[str, Any]]:
        """Notify transport EOF; never let recipe turn unexpected EOF into success."""
        if self._finished:
            return []
        if self._closed:
            raise ProtocolError("Generation is closed")
        return self.fail("Backend stream ended without an explicit Finish", code="backend_eof")

    def fail(self, message: str, *, code: str = "backend_error") -> list[dict[str, Any]]:
        if self._finished:
            return []
        if self._closed:
            raise ProtocolError("Generation is closed")
        self._processor.close()
        self._finished = True
        error = {"code": code, "message": message, "type": "server_error"}
        if self.prepared.protocol == "responses":
            response = json.loads(self._accumulator.to_json())
            response.update(status="failed", error={"code": code, "message": message},
                            incomplete_details=None, completed_at=None, usage=None)
            for item in response.get("output", []):
                if item.get("status") == "in_progress":
                    item["status"] = "incomplete"
            self._final = response
            return [{"type": "response.failed", "response": copy.deepcopy(response),
                     "sequence_number": self._sequence}]
        if self.prepared.protocol == "messages":
            self._final = {"type": "error", "error": error}
        else:
            self._final = {"error": error}
        return [copy.deepcopy(self._final)]

    def sse(self, events: Iterable[dict[str, Any]]) -> list[str]:
        """Frame a push/finish/fail batch; emit Chat [DONE] once on normal end."""
        frames = []
        for event in events:
            name = None if self.prepared.protocol == "chat_completions" else event.get("type")
            prefix = f"event: {name}\n" if name else ""
            frames.append(prefix + "data: " + json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n\n")
        if self._finished and self._successful and not self._sse_done_sent:
            done = self._response_type.sse_done_data()
            if done is not None:
                frames.append(f"data: {done}\n\n")
            self._sse_done_sent = True
        return frames

    def close(self) -> None:
        if not self._closed:
            self._processor.close()
            self._closed = True

    def __enter__(self) -> ProtocolStream:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
