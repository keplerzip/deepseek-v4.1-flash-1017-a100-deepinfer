#!/usr/bin/env python3
"""Port of the previous stable smoke/benchmark/long-context suite.

Uses unchanged official protocol endpoints, plus marker-gated engine evidence.
Raw engine token counts are bound to unchanged SSE byte offsets and client
receipt times. Client-visible TTFT is recorded separately. Python standard library only.

Run inside the delivered image. No model response is fabricated. Long-context
and full suites are explicit, potentially expensive target-GPU operations.
"""
import argparse
from bisect import bisect_left
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib

MODEL = "DeepSeek-Flash"
CONTEXT = 524288
CONCURRENCY = 32
FONT = {
    "0": (14, 17, 19, 21, 25, 17, 14), "1": (4, 12, 4, 4, 4, 4, 14),
    "2": (14, 17, 1, 2, 4, 8, 31), "3": (30, 1, 1, 14, 1, 1, 30),
    "4": (2, 6, 10, 18, 31, 2, 2), "5": (31, 16, 16, 30, 1, 1, 30),
    "6": (14, 16, 16, 30, 17, 17, 14), "7": (31, 1, 2, 4, 8, 8, 8),
    "8": (14, 17, 17, 14, 17, 17, 14), "9": (14, 17, 17, 15, 1, 1, 14),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1))] if ordered else None


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def png_code(code):
    """Real, deterministic black digit glyphs, rendered without Pillow/fonts."""
    scale, margin = 14, 35
    width, height = margin * 2 + (6 * len(code) - 1) * scale, margin * 2 + 7 * scale
    pixels = [bytearray([255] * width) for _ in range(height)]
    for position, char in enumerate(code):
        for row, bits in enumerate(FONT[char]):
            for column in range(5):
                if bits & (1 << (4 - column)):
                    left = margin + (position * 6 + column) * scale
                    for y in range(margin + row * scale, margin + (row + 1) * scale):
                        pixels[y][left:left + scale] = bytes(scale)
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    raw = b"".join(b"\x00" + row for row in pixels)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def image_block(code):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png_code(code)).decode()}}


def normalized_answer(text):
    return re.sub(r"\s+", "", text).strip("`\"'")


def validate_usage(usage):
    require(isinstance(usage, dict), "Missing server token usage")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(type(usage.get(key)) is int and usage[key] >= 0, "Invalid usage field: " + key)
    require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Usage totals do not add up")
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens")
    if reasoning is not None:
        require(type(reasoning) is int and 0 <= reasoning <= usage["completion_tokens"], "Invalid reasoning token usage")
    return usage


def client_token_batches(trace, arrivals):
    """Bind raw native token totals at unchanged SSE byte boundaries to receipt.

    Counts come from EngineClient IDs, never text/event counts. Byte offsets are
    in the HTTP body after transfer decoding. Partial lines and coalesced writes
    map to the first client read that contains the complete server boundary.
    """
    require(arrivals, 'No client byte-arrival evidence')
    offsets = [row['end_offset'] for row in arrivals]
    require(all(a < b for a, b in zip([0] + offsets, offsets)), 'Client byte offsets are not increasing')
    require(all(a['seconds'] <= b['seconds'] for a, b in zip(arrivals, arrivals[1:])), 'Client clock decreased')
    require(offsets[-1] == trace['response_body_bytes'], 'Server/client response byte totals differ')
    batches, count, previous_offset = [], 0, 0
    for row in trace['stream_boundaries']:
        offset, total = row['end_offset'], row['token_count']
        require(previous_offset < offset <= offsets[-1] and count <= total <= trace['token_count'],
                'Invalid server token/byte boundary')
        index = bisect_left(offsets, offset)
        if total > count:
            if batches and batches[-1]['client_read_index'] == index:
                batches[-1]['count'] += total - count
            else:
                batches.append({'seconds': arrivals[index]['seconds'], 'count': total-count,
                                'client_read_index': index, 'end_offset': offset})
        count, previous_offset = total, offset
    require(count == trace['token_count'], 'Not all native tokens have client byte-arrival evidence')
    return batches


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host, unix_socket, **kwargs):
        super().__init__(host, **kwargs)
        self.unix_socket = unix_socket

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.unix_socket)


class UnixHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, unix_socket):
        super().__init__()
        self.unix_socket = unix_socket

    def http_open(self, request):
        return self.do_open(lambda host, **kwargs: UnixHTTPConnection(host, self.unix_socket, **kwargs), request)


class Client:
    def __init__(self, base, model, timeout, key="", unix_socket=None):
        self.base, self.model, self.timeout, self.key = base.rstrip("/"), model, timeout, key
        self.unix_socket = str(unix_socket) if unix_socket else None
        self.ledger, self.lock = [], threading.Lock()

    def request(self, path, body=None, raw=False):
        if path == "/tokenize":
            path = "/deepinfer/acceptance/tokenize"
        trace_id = uuid.uuid4().hex if path == "/v1/chat/completions" else None
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if trace_id:
            headers["X-Deepinfer-Acceptance-Id"] = trace_id
            if body.get("ignore_eos"):
                headers["X-Deepinfer-Acceptance-Fixed-Budget"] = "1"
        if self.key:
            headers.update(Authorization="Bearer " + self.key, **{"x-api-key": self.key})
        request = urllib.request.Request(self.base + path,
            data=None if body is None else json.dumps(body, ensure_ascii=False).encode(), headers=headers)
        start = time.monotonic()
        handlers = [urllib.request.ProxyHandler({})]
        if self.unix_socket:
            handlers.append(UnixHTTPHandler(self.unix_socket))
        opener = urllib.request.build_opener(*handlers)
        try:
            with opener.open(request, timeout=self.timeout) as response:
                if raw:
                    return response.read().decode()
                result = self._stream(response, start) if body and body.get("stream") else json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read(16384).decode(errors="replace")
            raise RuntimeError("HTTP %s %s: %s" % (error.code, path, detail)) from error
        elapsed = time.monotonic() - start
        require(not result.get("error"), "Endpoint error: " + str(result.get("error")))
        if path == "/v1/chat/completions":
            usage = validate_usage(result.get("usage"))
            trace = self.request('/deepinfer/acceptance/trace/' + trace_id)
            require(trace.get('engine_finished') and trace.get('http_complete') and trace.get('http_status') == 200,
                    'Engine trace did not finish')
            require(not trace.get('overflow'), 'Engine trace overflow; cannot sign off token timing')
            encoded = json.dumps(body, ensure_ascii=False).encode()
            require(trace.get('request_body_sha256') == hashlib.sha256(encoded).hexdigest(), 'Engine trace body identity differs')
            batches = trace.get('batches') or []
            require(batches and all(b['count'] == len(b['token_ids']) and all(type(t) is int for t in b['token_ids']) for b in batches),
                    'Invalid native token batches')
            require(sum(b['count'] for b in batches) == trace['token_count'] == usage['completion_tokens'],
                    'Native engine token IDs and official completion usage differ')
            require(trace['prompt_tokens'] == usage['prompt_tokens'], 'Native prompt count and official usage differ')
            trace['token_ids_sha256'] = digest([batch['token_ids'] for batch in batches])
            for batch in batches:
                del batch['token_ids']
            result['engine_trace'] = trace
            if body.get('stream'):
                result['token_batches'] = client_token_batches(trace, result['byte_arrivals'])
                result['token_timing_source'] = 'Client SSE body arrival, bound to native IDs by unchanged ASGI byte offsets'
            require(result.get("model") == self.model, "Response model identity differs")
            result["elapsed_seconds"] = elapsed
            row = {"request_id": result.get("id"), "request_sha256": digest(body), "usage": usage,
                   "elapsed_seconds": elapsed, "stream": bool(body.get("stream")), "status": "RECEIVED",
                   "engine_trace_id": trace_id, "native_token_count": trace["token_count"],
                   "request_body_sha256": trace["request_body_sha256"],
                   "native_token_ids_sha256": trace["token_ids_sha256"],
                   "native_batches": batches}
            with self.lock:
                self.ledger.append(row)
        return result

    def _stream(self, response, start):
        result = {"text": "", "reasoning": "", "token_batches": [], "usage": None,
                  "finished": False, "done": False, "sse_events": 0, "first_visible_seconds": None,
                  "byte_arrivals": []}
        data_lines = []
        def event():
            if not data_lines:
                return
            data = b"\n".join(data_lines).strip()
            data_lines.clear()
            if data == b"[DONE]":
                result["done"] = True
                return
            value = json.loads(data)
            require(not value.get("error"), "SSE error: " + str(value.get("error")))
            result["sse_events"] += 1
            if value.get("model") is not None:
                result["model"] = value["model"]
            if value.get("id"):
                result["id"] = value["id"]
            if value.get("usage"):
                result["usage"] = value["usage"]
            for choice in value.get("choices", []):
                require(choice.get("index", 0) == 0, "Unexpected multi-choice stream")
                delta = choice.get("delta") or {}
                result["text"] += delta.get("content") or ""
                result["reasoning"] += delta.get("reasoning") or delta.get("reasoning_content") or ""
                now = time.monotonic() - start
                if result["first_visible_seconds"] is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                    result["first_visible_seconds"] = now
                token_ids = choice.get("token_ids")
                if token_ids:
                    require(isinstance(token_ids, list) and all(type(x) is int for x in token_ids), "Invalid streamed token IDs")
                    result["token_batches"].append({"seconds": now, "count": len(token_ids)})
                if choice.get("finish_reason") is not None:
                    result["finished"] = True
                    result["finish_reason"] = choice["finish_reason"]
        received_bytes = 0
        for line in response:
            received_bytes += len(line)
            line = line.rstrip(b"\r\n")
            if not line:
                event()
            elif line.startswith(b"data:"):
                data_lines.append(line[5:].lstrip())
            result['byte_arrivals'].append({'end_offset': received_bytes, 'seconds': time.monotonic()-start})
        event()
        require(result["done"] and result["finished"], "Incomplete SSE: terminal event or [DONE] missing")
        return result

    def body(self, text=None, mode="off", max_tokens=128, **extra):
        result = {"model": self.model, "messages": [{"role": "user", "content": text}],
                  "temperature": 0, "seed": 42, "max_tokens": max_tokens,
                  "thinking": {"type": "enabled" if mode == "high" else "disabled"},
                  "reasoning_effort": "high" if mode == "high" else "none"}
        result.update(extra)
        return result

    def chat(self, text=None, mode="off", max_tokens=128, stream=True, **extra):
        if stream:
            extra.update(stream=True, stream_options={"include_usage": True})
        result = self.request("/v1/chat/completions", self.body(text, mode, max_tokens, **extra))
        if stream:
            batches = result["token_batches"]
            require(batches, "Acceptance trace produced no native output tokens")
            require(sum(b["count"] for b in batches) == result["usage"]["completion_tokens"], "Engine token IDs and completion usage differ; timings cannot be signed off")
            require("\ufffd" not in result["text"] and not re.search(r"<[｜|]?DSML[｜|]?", result["text"]), "Visible output contains invalid/unparsed protocol data")
        return result


def stream_stats(result):
    batches, usage = result["token_batches"], result["usage"]
    first, last = batches[0], batches[-1]
    interval = last["seconds"] - first["seconds"]
    measured_tokens = usage["completion_tokens"] - first["count"]
    return {"request_id": result.get("id"), "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "ttft_seconds": first["seconds"], "first_visible_seconds": result["first_visible_seconds"],
            "token_timing_source": result.get('token_timing_source', result["engine_trace"]["timing_source"]),
            "engine_trace_id": result["engine_trace"]["trace_id"],
            "e2e_seconds": result["elapsed_seconds"], "decode_seconds": interval,
            "decode_measured_tokens": measured_tokens,
            "decode_tokens_per_second": measured_tokens / interval if interval > 0 and measured_tokens > 0 else None,
            "e2e_tokens_per_second": usage["completion_tokens"] / result["elapsed_seconds"],
            "first_token_batch_size": first["count"], "token_bearing_events": len(batches),
            "finish_reason": result["finish_reason"], "usage": usage,
            "output_sha256": digest({"text": result["text"], "reasoning": result["reasoning"]})}


def parse_metrics(text):
    totals = {}
    for line in text.splitlines():
        match = re.fullmatch(r"(vllm:[a-zA-Z0-9_:]+)(?:\{.*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?", line)
        if match:
            totals[match[1]] = totals.get(match[1], 0) + float(match[2])
    return totals


def settle(client, seconds=12):
    previous = None
    deadline = time.monotonic() + seconds
    while True:
        metrics = parse_metrics(client.request("/metrics", raw=True))
        require("vllm:generation_tokens_total" in metrics, "Missing generation-token counter in /metrics")
        require("vllm:num_requests_running" in metrics and "vllm:num_requests_waiting" in metrics, "Missing running/waiting gauges in /metrics")
        counters = {k: v for k, v in metrics.items() if k.endswith("_total")}
        if metrics["vllm:num_requests_running"] == metrics["vllm:num_requests_waiting"] == 0 and counters == previous:
            return {"settled": True, "metrics": metrics}
        if time.monotonic() >= deadline:
            return {"settled": False, "metrics": metrics}
        previous = counters
        time.sleep(0.5)


def accounting(before, after, rows, require_dspark=False):
    old, new = before["metrics"], after["metrics"]
    delta = {k: v - old.get(k, 0) for k, v in new.items() if k.endswith("_total")}
    expected = sum(row["usage"]["completion_tokens"] for row in rows)
    observed = delta.get("vllm:generation_tokens_total")
    issues = []
    if not before["settled"] or not after["settled"]:
        issues.append("Service was busy or counters did not settle; possible external traffic")
    if observed != expected:
        issues.append("Global generation tokens %s != this test's server usage %s; possible traffic, dropped request, counter reset or accounting error" % (observed, expected))
    if any(v < 0 for v in delta.values()):
        issues.append("Metric counter decreased during the test")
    dspark = {name: delta.get("vllm:spec_decode_num_" + name + "_total") for name in ("drafts", "draft_tokens", "accepted_tokens")}
    if require_dspark:
        if not all(value is not None and value > 0 for value in dspark.values()):
            issues.append("Missing positive DSpark draft/proposed/accepted counters")
        elif dspark["accepted_tokens"] > dspark["draft_tokens"] or dspark["draft_tokens"] > dspark["drafts"] * 5 + 1e-6:
            issues.append("DSpark counters conflict with native k=5")
    return {"expected_generation_tokens": expected, "observed_generation_tokens": observed,
            "counter_deltas": delta, "dspark": dspark, "issues": issues}


def recorded_capacity_evidence(path):
    """Validate initialized pools without treating capacity as live residency.

    TP ranks shard the same logical pool. Compute the minimum observed window
    count within each DP pool, then add the independent pool minima. A DP2
    C32 launch needs at least sixteen full windows in *each* pool.
    """
    raw = path.read_bytes()
    capacity = json.loads(raw)
    records = capacity.get("capacity_records") or []
    issues, pools, topology = [], {}, None
    for key, wanted in (("configured_context", CONTEXT), ("configured_scheduler_concurrency", CONCURRENCY)):
        if type(capacity.get(key)) is not int or capacity[key] != wanted:
            issues.append("Unexpected " + key)
    if capacity.get("status") != "PASS" or capacity.get("gpu_executed") is not True:
        issues.append("Source is not a passed GPU capacity report")
    if capacity.get("errors"):
        issues.append("Source reports capacity errors")
    if not isinstance(records, list) or not records:
        issues.append("Missing capacity records")
        records = []
    for row in records:
        if not isinstance(row, dict):
            issues.append("Invalid capacity record")
            continue
        tp, dp, local, rank = (row.get(key) for key in ("tensor_parallel_size", "data_parallel_size", "configured_max_num_seqs", "data_parallel_rank"))
        integers = all(type(value) is int for value in (tp, dp, local, rank))
        if not integers or (tp, dp) not in ((8, 1), (4, 2)) or local * dp != CONCURRENCY or not 0 <= rank < dp:
            issues.append("Unreviewed or incomplete DP topology")
            continue
        current = (tp, dp, local)
        if topology is None:
            topology = current
        elif topology != current:
            issues.append("Capacity records mix different launch topologies")
        if type(row.get("max_model_len")) is not int or row["max_model_len"] != CONTEXT:
            issues.append("Capacity record context differs")
        groups = row.get("groups")
        if not isinstance(groups, list) or not groups or any(
                not isinstance(group, dict) or type(group.get("full_window_blocks")) is not int
                or group["full_window_blocks"] <= 0 for group in groups):
            issues.append("Invalid grouped full-window block accounting")
            continue
        blocks = sum(group["full_window_blocks"] for group in groups)
        windows, available, declared = (row.get(key) for key in ("full_windows", "pool_blocks", "blocks_per_full_window"))
        if any(type(value) is not int for value in (windows, available, declared)) or windows < 1 or available < 0 or declared != blocks or available // blocks != windows:
            issues.append("Capacity window count disagrees with grouped blocks")
            continue
        pools.setdefault(rank, []).append(windows)
    if topology is None or set(pools) != set(range(topology[1])):
        issues.append("Missing or unexpected DP pool records")
    per_pool = {str(rank): min(windows) for rank, windows in pools.items()}
    recomputed = sum(per_pool.values())
    measured = capacity.get("measured_full_window_capacity")
    if type(measured) is not int or measured != recomputed:
        issues.append("Reported capacity differs from the sum of per-pool minima")
    provided_pools = capacity.get("per_dp_full_window_capacity")
    if provided_pools is not None and (not isinstance(provided_pools, dict)
            or set(provided_pools) != set(per_pool)
            or any(type(value) is not int or value != per_pool[key] for key, value in provided_pools.items())):
        issues.append("Reported per-DP capacities disagree")
    sufficient = bool(topology and set(pools) == set(range(topology[1]))
                      and all(windows >= topology[2] for windows in per_pool.values()))
    valid = not issues
    return {"status": "VALID_RECORDED_CAPACITY" if valid else "UNVERIFIED",
            "source_path": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "recorded_status": capacity.get("status"), "measured_full_window_capacity": measured,
            "recomputed_full_window_capacity": recomputed,
            "per_dp_full_window_capacity": per_pool,
            "topology": {"tensor_parallel_size": topology[0], "data_parallel_size": topology[1],
                         "per_pool_concurrency": topology[2]} if topology else None,
            "capacity_sufficient_for_32_full_windows": sufficient if valid else None,
            "validation_issues": issues,
            "simultaneous_full_window_residency_status": "NOT_PROVEN",
            "freshness": "Caller-supplied launch report; HTTP cannot prove this file belongs to the current engine launch",
            "scope": "Independent initialized KV capacity; does not prove concurrent full-window residency"}


class Suite:
    def __init__(self, args):
        self.args = args
        key = args.api_key_file.read_text().strip() if args.api_key_file else os.environ.get("VLLM_API_KEY", "")
        self.client = Client(args.base_url, args.model, args.timeout, key, args.unix_socket)
        self.report = {"schema": 1, "status": "RUNNING", "suite": args.suite, "started_at": utc_now(),
                       "model": args.model, "base_url": args.base_url, "unix_socket": str(args.unix_socket) if args.unix_socket else None,
                       "context": CONTEXT, "concurrency": CONCURRENCY,
                       "scope": "Real HTTP inference only; startup GPU topology, CUDA graph and model bytes require separate launch gates",
                       "long_context_explicitly_requested": args.suite in ("long-context", "full", "full-window-c32"),
                       "timing_method": "Official usage includes reasoning. Marker-gated EngineClient raw IDs must sum to usage and global generation metrics. Cumulative raw counts at unchanged SSE body byte offsets are bound to client receipt/parser times. Decode=(completion_tokens-first_client_batch)/(last-first client batch arrival), including HTTP/parser overhead as before. first_visible_seconds is first text/reasoning delta. Batching follows official SSE. Events/text characters never supply token counts; no strict cross-version speedup is claimed.",
                       "fixed_budget_method": "Acceptance-only min_tokens=max_tokens masks EOS until budget; ordinary API sampling is unchanged.",
                       "test_provenance": "Ported from previous stable tests/acceptance.py; official protocol conversion replaces legacy chat_template_kwargs",
                       "tests": [], "contamination": [], "requests": self.client.ledger}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        self.trace = args.output.with_suffix(".requests.jsonl")
        self.trace.write_text("")
        self.save()

    def save(self):
        temp = self.args.output.with_name(self.args.output.name + ".tmp")
        temp.write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + "\n")
        temp.replace(self.args.output)

    def test(self, name, function):
        print("START " + name, flush=True)
        start = time.monotonic()
        try:
            details = function()
            row = {"name": name, "status": "PASS", "details": details}
        except Exception as error:
            row = {"name": name, "status": "FAIL", "error_type": type(error).__name__, "error": str(error)[:6000]}
        row["seconds"] = time.monotonic() - start
        self.report["tests"].append(row)
        self.save()
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    def identity(self):
        response = self.client.request("/v1/models")
        ids = [model["id"] for model in response["data"]]
        require(ids == [self.args.model], "Expected exactly requested model, got " + repr(ids))
        return {"ids": ids}

    def text(self, mode):
        result = self.client.chat("Compute 19 multiplied by 23. Give the result, then briefly explain how you checked it.", mode=mode, max_tokens=512)
        require(result["text"] or result["reasoning"], "No text or reasoning in stream")
        if mode == "off":
            require("437" in result["text"], "Off-mode arithmetic answer is wrong")
            require(not result["reasoning"], "Off mode unexpectedly produced reasoning")
        else:
            require(result["reasoning"] or (result["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) > 0,
                    "High mode did not expose any reasoning")
            require(result["text"] and "437" in result["text"], "High-mode final answer missing or wrong")
        return stream_stats(result)

    def vision(self, count):
        codes = ["739261", "482905", "615837"][:count]
        prompt = "Read the six digit number in each attached image in image order. Return only the numbers, separated by commas."
        content = [{"type": "text", "text": prompt}] + [image_block(code) for code in codes]
        result = self.client.chat(messages=[{"role": "user", "content": content}], max_tokens=128)
        expected = ",".join(codes)
        require(normalized_answer(result["text"]) == expected, "OCR differs: expected %r, got %r" % (expected, result["text"]))
        return {"images": count, "fixture_sha256": [hashlib.sha256(png_code(code)).hexdigest() for code in codes],
                "expected": expected, "actual": result["text"], **stream_stats(result)}

    def multimodal_tool_order(self):
        codes = {"A": "394726", "B": "850163"}
        calls = [{"id": "panel_" + name, "type": "function", "function": {"name": "get_panel", "arguments": json.dumps({"panel": name})}} for name in ("A", "B")]
        messages = [{"role": "user", "content": "Fetch panel A then panel B with get_panel. Read their images and return only A=<six digits>,B=<six digits>. Results can arrive in reverse order; use tool call identity."},
                    {"role": "assistant", "content": None, "tool_calls": calls}]
        for name in ("B", "A"):
            messages.append({"role": "tool", "tool_call_id": "panel_" + name,
                             "content": [{"type": "text", "text": "Image returned for panel " + name}, image_block(codes[name])]})
        tools = [{"type": "function", "function": {"name": "get_panel", "description": "Read a panel image", "parameters": {"type": "object", "properties": {"panel": {"type": "string"}}, "required": ["panel"]}}}]
        result = self.client.chat(messages=messages, tools=tools, max_tokens=128)
        expected = "A=" + codes["A"] + ",B=" + codes["B"]
        require(normalized_answer(result["text"]) == expected, "Reverse-arrival image/tool association differs: expected %r, got %r" % (expected, result["text"]))
        return {"tool_call_order": ["A", "B"], "tool_result_arrival_order": ["B", "A"], "expected": expected,
                "actual": result["text"], **stream_stats(result)}

    def cache(self):
        prompt = "Unique cache probe " + uuid.uuid4().hex + "\n" + "\n".join(
            "Record %d: unchanged reference material for cache validation." % index for index in range(512))
        prompt += "\nReply only CACHE_READY."
        cold = self.client.chat(prompt, max_tokens=32)
        hot = self.client.chat(prompt, max_tokens=32)
        require(cold["text"].strip() == hot["text"].strip() == "CACHE_READY", "Cache response is wrong")
        before = (cold["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
        after = (hot["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
        require(type(before) is int and type(after) is int, "Missing prompt_tokens_details.cached_tokens")
        require(0 <= before < after <= hot["usage"]["prompt_tokens"], "Repeated exact prefix did not increase cached tokens")
        return {"cold_usage": cold["usage"], "hot_usage": hot["usage"], "cold_seconds": cold["elapsed_seconds"],
                "hot_seconds": hot["elapsed_seconds"], "hot_cache_hit_ratio": after / hot["usage"]["prompt_tokens"]}

    def concurrent_tools(self):
        barrier = threading.Barrier(CONCURRENCY)
        tool = {"type": "function", "function": {"name": "record_code", "description": "Record the exact supplied code.",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False}}}
        def one(index):
            code = "C32_" + uuid.uuid4().hex[:12]
            try:
                barrier.wait(timeout=30)
                response = self.client.chat("Call record_code exactly once with code " + code + ". Do not answer in prose.",
                                            stream=False, tools=[tool], tool_choice="auto", max_tokens=128)
                choice = response["choices"][0]
                calls = choice["message"].get("tool_calls") or []
                require(len(calls) == 1 and calls[0]["function"]["name"] == "record_code", "Expected one parsed record_code tool call")
                require(json.loads(calls[0]["function"]["arguments"]) == {"code": code}, "Tool arguments differ")
                require(choice.get("finish_reason") in ("tool_calls", "stop"), "Unexpected tool finish reason")
                return {"index": index, "status": "PASS", "elapsed_seconds": response["elapsed_seconds"], "usage": response["usage"]}
            except Exception as error:
                return {"index": index, "status": "FAIL", "error": str(error)[:2000]}
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            rows = list(pool.map(one, range(CONCURRENCY)))
        self.report["tool_c32_requests"] = rows
        self.save()
        failures = [row for row in rows if row["status"] != "PASS"]
        require(not failures, "C32 tool requests failed: " + json.dumps(failures)[:3000])
        return {"concurrency": CONCURRENCY, "successful_requests": len(rows), "wall_seconds": time.monotonic() - start,
                "latency_p50_seconds": percentile([row["elapsed_seconds"] for row in rows], .5),
                "latency_p95_seconds": percentile([row["elapsed_seconds"] for row in rows], .95),
                "scope": "32 simultaneous short tool requests; full-window residency is not asserted"}

    def dspark(self):
        before = settle(self.client)
        index = len(self.client.ledger)
        response = self.client.chat("Explain binary search tree invariants and give detailed code examples. Continue until the output limit.",
                                    max_tokens=256, ignore_eos=True)
        after = settle(self.client)
        result = accounting(before, after, self.client.ledger[index:], require_dspark=True)
        require(not result["issues"], json.dumps(result))
        return {"accounting": result, "sample": stream_stats(response)}

    def benchmark(self, mode):
        args, client = self.args, self.client
        before = settle(client)
        ledger_start = len(client.ledger)
        rows, phases = [], {}
        for phase, count, concurrency in (("warmup", args.warmups, 1), ("single", args.single_samples, 1),
                                           ("c32", args.load_requests, CONCURRENCY)):
            phase_start = time.monotonic()
            barrier = threading.Barrier(min(CONCURRENCY, count)) if phase == "c32" else None
            def one(index):
                tag = uuid.uuid4().hex
                try:
                    if barrier is not None and index < CONCURRENCY:
                        barrier.wait(timeout=30)
                    result = client.chat("Benchmark " + tag + ". Explain a robust LRU cache implementation, concurrency invariants, edge cases, testing and code. Continue with detailed examples until the output limit.",
                                         mode=mode, max_tokens=args.output_tokens, ignore_eos=True)
                    row = {"phase": phase, "mode": mode, "index": index, "status": "PASS", **stream_stats(result)}
                    require(row["completion_tokens"] == args.output_tokens, "Benchmark output ended before its fixed token budget")
                    require(row["decode_tokens_per_second"] is not None, "Insufficient streamed token batches for decode timing")
                    return row
                except Exception as error:
                    return {"phase": phase, "mode": mode, "index": index, "status": "FAIL", "error": str(error)[:3000]}
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for future in as_completed([pool.submit(one, index) for index in range(count)]):
                    row = future.result()
                    rows.append(row)
                    with self.trace.open("a") as output:
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print("benchmark %s %s %d/%d %s" % (mode, phase, sum(r["phase"] == phase for r in rows), count, row["status"]), flush=True)
            phases[phase] = {"requests": count, "concurrency": concurrency, "wall_seconds": time.monotonic() - phase_start}
            self.report.setdefault("benchmark_progress", {})[mode] = {"phases": phases, "samples": rows}
            self.save()
        after = settle(client)
        audit = accounting(before, after, client.ledger[ledger_start:], require_dspark=True)
        failures = [row for row in rows if row["status"] != "PASS"]
        singles = [row for row in rows if row["phase"] == "single" and row["status"] == "PASS"]
        load = [row for row in rows if row["phase"] == "c32" and row["status"] == "PASS"]
        def median(key):
            return statistics.median(row[key] for row in singles) if singles else None
        details = {"mode": mode, "phases": phases, "accounting": audit, "samples": rows,
                   "single_decode_tokens_per_second_median": median("decode_tokens_per_second"),
                   "single_ttft_seconds_median": median("ttft_seconds"), "single_e2e_seconds_median": median("e2e_seconds"),
                   "single_e2e_tokens_per_second_median": median("e2e_tokens_per_second"),
                   "c32_total_output_tokens": sum(row["completion_tokens"] for row in load),
                   "c32_aggregate_tokens_per_second": sum(row["completion_tokens"] for row in load) / phases["c32"]["wall_seconds"],
                   "c32_latency_p50_seconds": percentile([row["e2e_seconds"] for row in load], .5),
                   "c32_latency_p95_seconds": percentile([row["e2e_seconds"] for row in load], .95),
                   "c32_ttft_p95_seconds": percentile([row["ttft_seconds"] for row in load], .95),
                   "c32_scope": "Entire load-phase wall time including queueing, prefill and internal trace-evidence retrieval; native usage includes reasoning; not a strict previous-version timing A/B",
                   "cache_policy": "Different nonce appears at the beginning of every request; fixed synthetic workload, no private history"}
        self.report.setdefault("benchmarks", {})[mode] = details
        self.report["contamination"].extend("benchmark %s: %s" % (mode, issue) for issue in audit["issues"])
        self.save()
        require(not failures, "Benchmark requests failed; partial samples retained")
        require(not audit["issues"], "Benchmark accounting contaminated; samples retained")
        return {key: value for key, value in details.items() if key != "samples"}

    def near_full_prompt(self, nonce, nonce_first=False, report_key="long_context_tokenizer_probes"):
        target = CONTEXT - 512
        values = ["VALUE_%s_%d" % (nonce, index) for index in range(5)]
        records = ["\nPOSITION_%d=%s\n" % (index, value) for index, value in enumerate(values)]
        filler = "Archive record: unrelated background material; retrieve only the named POSITION values.\n"
        template = {"model": self.args.model, "thinking": {"type": "disabled"}, "max_tokens": 512}
        def build(repeats):
            chunk = filler * repeats
            return ((nonce + "\n" if nonce_first else "")
                    + "Find POSITION_0 through POSITION_4 in this archive. Preserve their numeric order.\n"
                    + chunk.join(records) + "\nReturn only the five exact POSITION values separated by commas.")
        probes = []
        def count(repeats):
            prompt = build(repeats)
            response = self.client.request("/tokenize", dict(template, messages=[{"role": "user", "content": prompt}]))
            actual = response.get("count")
            require(type(actual) is int and actual > 0, "Invalid /tokenize count")
            probes.append({"repeats_per_gap": repeats, "actual_prompt_tokens": actual})
            self.report[report_key] = probes
            self.save()
            return prompt, actual
        small_prompt, small = count(64)
        _, zero = count(0)
        per_repeat = (small - zero) / 64
        require(per_repeat > 0, "Filler did not increase tokenized length")
        repeats = max(1, int((target - zero - 128) / per_repeat))
        prompt, actual = count(repeats)
        low, high = (0, repeats) if actual > target else (repeats, max(repeats + 1, repeats * 2))
        for _ in range(24):
            if target - 1024 <= actual <= target:
                break
            if actual > target:
                high = repeats
            else:
                low = repeats
            repeats = (low + high) // 2
            prompt, actual = count(repeats)
        require(target - 1024 <= actual <= target, "Cannot construct a near-512Ki-token input: %s" % actual)
        return prompt, actual, values, probes

    def long_context(self):
        reserve, target = 512, CONTEXT - 512
        prompt, actual, values, probes = self.near_full_prompt(uuid.uuid4().hex[:16])
        result = self.client.chat(prompt, max_tokens=reserve)
        expected = ",".join(values)
        require(normalized_answer(result["text"]) == expected, "Long-context retrieval failed at prompt token count %d; output=%r" % (actual, result["text"]))
        require(result["usage"]["prompt_tokens"] == actual, "Serving input differs from /tokenize; truncation or template mismatch")
        return {"requested_context": CONTEXT, "prompt_token_target": target, "actual_prompt_tokens": actual,
                "output_budget": reserve, "input_plus_output_budget": actual + reserve,
                "retrieval_positions": [0, .25, .5, .75, 1], "expected": expected, "actual": result["text"],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "tokenizer_probes": probes,
                "scope": "One nearly full window, tested separately; does not assert 32 simultaneous full windows",
                **stream_stats(result)}

    def full_window_c32(self):
        """Explicit stress: concurrent near-full requests, never a residency shortcut."""
        reserve, target = 512, CONTEXT - 512
        detail = {"request_concurrency_status": "RUNNING",
                  "simultaneous_full_window_residency_status": "NOT_PROVEN",
                  "context": CONTEXT, "concurrency": CONCURRENCY,
                  "prompt_token_target_per_request": target, "output_budget_per_request": reserve,
                  "residency_note": "Successful queued requests, max_num_seqs=32, or running gauge=32 do not prove 32 near-full KV windows were simultaneously resident after prefill. Capacity is a separate necessary condition; no per-request KV residency telemetry is available to this HTTP test.",
                  "prefix_policy": "Each user content starts immediately with a different nonce, before all shared instructions and filler. Unavoidable chat-template headers may share a few tokens; long input prefixes cannot be reused across these 32 requests.",
                  "prepared_requests": [], "samples": [], "scheduler_trajectory": [],
                  "scheduler_metric_errors": [], "capacity_evidence": {"status": "NOT_PROVIDED"}}
        self.report["full_window_c32"] = detail
        capacity_path = getattr(self.args, "capacity_report", None)
        if capacity_path:
            detail["capacity_evidence"] = recorded_capacity_evidence(capacity_path)
        self.save()
        prepared = []
        nonces = set()
        for index in range(CONCURRENCY):
            nonce = uuid.uuid4().hex
            require(nonce not in nonces, "Duplicate request nonce")
            nonces.add(nonce)
            key = "full_window_c32_tokenizer_probes_%02d" % index
            prompt, actual, values, probes = self.near_full_prompt(nonce, nonce_first=True, report_key=key)
            require(prompt.startswith(nonce + "\n"), "Unique nonce must begin the user content")
            prepared.append((prompt, actual, values))
            detail["prepared_requests"].append({"index": index, "nonce": nonce,
                "actual_prompt_tokens": actual, "input_plus_output_budget": actual + reserve,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "tokenizer_probes": probes,
                "retrieval_positions": [0, .25, .5, .75, 1]})
            self.report.pop(key, None)
            self.save()
            print("full-window-c32 prepared %d/%d: %d prompt tokens" % (index + 1, CONCURRENCY, actual), flush=True)
        require(len({row["prompt_sha256"] for row in detail["prepared_requests"]}) == CONCURRENCY,
                "Full-window requests are not distinct")
        before = settle(self.client)
        ledger_start = len(self.client.ledger)
        start = time.monotonic()
        trajectory_path = self.args.output.with_suffix(".scheduler.jsonl")
        detail["scheduler_trajectory_file"] = str(trajectory_path)
        trajectory_path.write_text("")
        stop = threading.Event()
        monitor_client = Client(self.client.base, self.client.model, min(self.client.timeout, 10),
                                self.client.key, self.client.unix_socket)

        def monitor():
            while not stop.is_set():
                row = {"seconds": time.monotonic() - start}
                try:
                    metrics = parse_metrics(monitor_client.request("/metrics", raw=True))
                    require("vllm:num_requests_running" in metrics and "vllm:num_requests_waiting" in metrics,
                            "Scheduler running/waiting gauges missing")
                    row["metrics"] = {key: value for key, value in metrics.items()
                                      if key in ("vllm:num_requests_running", "vllm:num_requests_waiting",
                                                 "vllm:num_requests_swapped", "vllm:kv_cache_usage_perc",
                                                 "vllm:gpu_cache_usage_perc", "vllm:generation_tokens_total")
                                      or "preempt" in key}
                    detail["scheduler_trajectory"].append(row)
                except Exception as error:
                    row["error"] = str(error)[:2000]
                    detail["scheduler_metric_errors"].append(row)
                with trajectory_path.open("a") as output:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                if stop.wait(1):
                    return

        thread = threading.Thread(target=monitor, name="full-window-c32-metrics", daemon=True)
        thread.start()
        barrier = threading.Barrier(CONCURRENCY)

        def one(index):
            prompt, actual, values = prepared[index]
            row = {"index": index, "actual_prompt_tokens": actual, "output_budget": reserve,
                   "expected": ",".join(values), "status": "FAIL"}
            try:
                barrier.wait(timeout=60)
                row["submitted_seconds"] = time.monotonic() - start
                response = self.client.chat(prompt, max_tokens=reserve)
                row.update(stream_stats(response), actual=response["text"],
                           finished_seconds=time.monotonic() - start)
                require(response["usage"]["prompt_tokens"] == actual,
                        "Actual serving prompt differs from /tokenize; truncation or template mismatch")
                require(normalized_answer(response["text"]) == row["expected"], "Five-position long retrieval differs")
                require(target - 1024 <= response["usage"]["prompt_tokens"] <= target,
                        "Request is not near the specified full input window")
                cached = (response["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
                require(type(cached) is int and 0 <= cached <= 1024,
                        "Missing cache detail or unexpectedly long cached prefix despite unique leading nonce")
                row["status"] = "PASS"
            except Exception as error:
                row.update(error=str(error)[:3000], finished_seconds=time.monotonic() - start)
            return row

        self.save()
        try:
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futures = [pool.submit(one, index) for index in range(CONCURRENCY)]
                for future in as_completed(futures):
                    row = future.result()
                    detail["samples"].append(row)
                    with self.trace.open("a") as output:
                        output.write(json.dumps(dict(row, phase="full-window-c32"), ensure_ascii=False) + "\n")
                    self.save()
                    print("full-window-c32 completed %d/%d %s" % (len(detail["samples"]), CONCURRENCY, row["status"]), flush=True)
            wall = time.monotonic() - start
            after = settle(self.client)
        finally:
            stop.set()
            thread.join(timeout=12)
        audit = accounting(before, after, self.client.ledger[ledger_start:], require_dspark=True)
        samples = detail["samples"]
        successful = [row for row in samples if row["status"] == "PASS"]
        trajectories = detail["scheduler_trajectory"]
        peaks = {name: max((row["metrics"].get("vllm:num_requests_" + name, 0) for row in trajectories), default=None)
                 for name in ("running", "waiting")}
        submitted = sorted(row["submitted_seconds"] for row in samples if "submitted_seconds" in row)
        simultaneous_in_flight_peak = 0
        events = [(row["submitted_seconds"], 1) for row in samples if "submitted_seconds" in row]
        events += [(row["finished_seconds"], -1) for row in samples if "submitted_seconds" in row]
        in_flight = 0
        for _, step in sorted(events, key=lambda event: (event[0], -event[1])):
            in_flight += step
            simultaneous_in_flight_peak = max(simultaneous_in_flight_peak, in_flight)
        issues = list(audit["issues"])
        if detail["scheduler_metric_errors"]:
            issues.append("Scheduler metric sampling failed; trajectory is incomplete")
        if not trajectories:
            issues.append("No scheduler trajectory was captured")
        self.report["contamination"].extend("full-window-c32: " + issue for issue in issues)
        detail.update(accounting=audit, wall_seconds=wall, successful_requests=len(successful),
            client_in_flight_peak=simultaneous_in_flight_peak,
            submission_spread_seconds=submitted[-1] - submitted[0] if submitted else None,
            observed_running_peak=peaks["running"], observed_waiting_peak=peaks["waiting"],
            preemption_counter_deltas={key: value for key, value in audit["counter_deltas"].items() if "preempt" in key},
            observed_preemption_metric_names=sorted({key for row in trajectories for key in row["metrics"] if "preempt" in key}),
            ttft_p50_seconds=percentile([row["ttft_seconds"] for row in successful], .5),
            ttft_p95_seconds=percentile([row["ttft_seconds"] for row in successful], .95),
            e2e_latency_p50_seconds=percentile([row["e2e_seconds"] for row in successful], .5),
            e2e_latency_p95_seconds=percentile([row["e2e_seconds"] for row in successful], .95),
            aggregate_output_tokens_per_second=sum(row["completion_tokens"] for row in successful) / wall,
            request_concurrency_status="PASS" if len(successful) == CONCURRENCY and simultaneous_in_flight_peak == CONCURRENCY and not issues else "FAIL")
        detail["preemption_telemetry_status"] = "OBSERVED" if detail["observed_preemption_metric_names"] else "NOT_EXPOSED"
        self.save()
        require(len(successful) == CONCURRENCY, "One or more near-full requests failed; all completed samples retained")
        require(simultaneous_in_flight_peak == CONCURRENCY, "Did not observe 32 overlapping client requests")
        require(not issues, "Full-window concurrency accounting/trajectory contaminated; report retained")
        return {key: value for key, value in detail.items() if key not in ("samples", "scheduler_trajectory", "prepared_requests")}

    def run(self):
        before = None
        ledger_start = len(self.client.ledger)
        self.test("model-identity", self.identity)
        try:
            before = settle(self.client)
            self.report["initial_metrics"] = before
        except Exception as error:
            self.report["contamination"].append("Cannot obtain initial accounting: " + str(error))
        if self.args.suite in ("smoke", "full"):
            self.test("text-SSE-off", lambda: self.text("off"))
            self.test("text-SSE-high", lambda: self.text("high"))
            self.test("single-image-OCR", lambda: self.vision(1))
            self.test("multi-image-OCR", lambda: self.vision(3))
            self.test("reverse-tool-result-image-association", self.multimodal_tool_order)
            self.test("prefix-cache-cold-hot-details", self.cache)
            self.test("C32-auto-tool-calls", self.concurrent_tools)
            self.test("DSpark-native-k5-counters", self.dspark)
        if self.args.suite in ("benchmark", "full"):
            for mode in ("off", "high"):
                self.test("benchmark-" + mode, lambda value=mode: self.benchmark(value))
        if self.args.suite in ("long-context", "full"):
            self.test("independent-near-524288-token-retrieval", self.long_context)
        if self.args.suite == "full-window-c32":
            self.test("explicit-C32-near-full-window-retrieval", self.full_window_c32)
        if before is not None:
            try:
                after = settle(self.client)
                audit = accounting(before, after, self.client.ledger[ledger_start:])
                self.report["final_metrics"] = after
                self.report["global_accounting"] = audit
                self.report["contamination"].extend(audit["issues"])
            except Exception as error:
                self.report["contamination"].append("Cannot finalize accounting: " + str(error))
        failed = any(row["status"] != "PASS" for row in self.report["tests"])
        self.report["status"] = "FAIL" if failed else "CONTAMINATED" if self.report["contamination"] else "PASS"
        if (self.report["status"] == "PASS" and self.args.suite == "full-window-c32"
                and self.report["full_window_c32"]["simultaneous_full_window_residency_status"] != "PASS"):
            self.report["status"] = "NOT_PROVEN"
        self.report["finished_at"] = utc_now()
        self.save()
        print("%s: %s" % (self.report["status"], self.args.output), flush=True)
        return 0 if self.report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--unix-socket", type=Path, help="Connect all HTTP requests through the engine Unix socket, e.g. /state/api.sock")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--suite", choices=("smoke", "benchmark", "long-context", "full", "full-window-c32"), default="smoke")
    parser.add_argument("--capacity-report", type=Path, help="Optional independent runtime.py capacity JSON for the explicit full-window-c32 report")
    parser.add_argument("--output", type=Path, default=Path("acceptance-report.json"))
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--timeout", type=float, default=7200, help="Per-request read timeout; 512K prefill can be slow")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--single-samples", type=int, default=5)
    parser.add_argument("--load-requests", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=512)
    args = parser.parse_args()
    require(not args.unix_socket or args.base_url.startswith("http://"), "Unix socket mode requires an http:// placeholder base URL")
    require(args.timeout > 0 and args.warmups >= 1 and args.single_samples >= 1, "Positive timeout, warmups and samples required")
    require(args.load_requests >= CONCURRENCY and args.output_tokens >= 128, "Benchmark needs at least 32 load requests and 128 output tokens")
    suite = Suite(args)
    try:
        return suite.run()
    except (KeyboardInterrupt, Exception) as error:
        suite.report.update(status="FAIL", fatal_error=type(error).__name__ + ": " + str(error)[:6000], finished_at=utc_now())
        suite.save()
        print("FAIL: " + str(error), flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
