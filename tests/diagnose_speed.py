#!/usr/bin/env python3
"""Read-only configuration diagnostics against a live local inference endpoint.

This tool sends real requests but never changes model, context, image, scheduler,
or speculative-decoding settings. It does not fetch models or media itself.
Token timing binds native IDs to client SSE byte-boundary arrival; end-to-end is client-side.
Counters cannot identify GPU causes. Reports save counts/times and token hashes,
not raw output token IDs.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

from acceptance import Client, MODEL, digest, require, stream_stats


PROMPTS = {
    "text": "Explain the practical tradeoffs between a queue and a stack, with concrete examples and edge cases. Continue until the output limit.",
    "code": "Implement a thread-safe bounded LRU cache in Python. Explain the invariants, cancellation, error handling, race conditions, and tests. Continue with concrete code until the output limit.",
}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_metrics(raw):
    totals, positions = {}, {}
    for line in raw.splitlines():
        match = re.fullmatch(r'(vllm:[a-zA-Z0-9_:]+)(?:\{(.*)\})?\s+([-+0-9.eE]+)(?:\s+\d+)?', line)
        if not match:
            continue
        name, labels, number = match.groups()
        value = float(number)
        if not math.isfinite(value):
            continue
        totals[name] = totals.get(name, 0) + value
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            position = re.search(r'(?:^|,)\s*position="(\d+)"', labels or "")
            if position:
                key = position[1]
                positions[key] = positions.get(key, 0) + value
    return {"totals": totals, "accepted_positions": positions}


def metrics_snapshot(client, wait_seconds=12):
    """Keep the final raw snapshot even if settling fails or metrics disappear."""
    previous = None
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            raw = client.request("/metrics", raw=True)
            parsed = parse_metrics(raw)
        except Exception as error:
            return {"settled": False, "raw": "", "totals": {}, "accepted_positions": {},
                    "captured_at": utc_now(), "captured_time_ns": time.time_ns(),
                    "error": type(error).__name__ + ": " + str(error)[:2000]}
        totals = parsed["totals"]
        required = ("vllm:generation_tokens_total", "vllm:num_requests_running", "vllm:num_requests_waiting")
        missing = [key for key in required if key not in totals]
        counters = {key: value for key, value in totals.items() if key.endswith("_total")}
        idle = not missing and totals[required[1]] == totals[required[2]] == 0
        settled = idle and counters == previous
        result = {"settled": settled, "raw": raw, "captured_at": utc_now(),
                  "captured_time_ns": time.time_ns(), **parsed}
        if missing:
            result["error"] = "Missing counters/gauges: " + ", ".join(missing)
        if settled or time.monotonic() >= deadline:
            return result
        previous = counters
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))


def account(before, after, completion_tokens, expect_dspark=True, expected_k=5):
    old, new = before["totals"], after["totals"]
    delta = {key: value - old.get(key, 0) for key, value in new.items() if key.endswith("_total")}
    positions = {key: value - before["accepted_positions"].get(key, 0)
                 for key, value in after["accepted_positions"].items()}
    issues = []
    if not before["settled"] or not after["settled"]:
        issues.append("Counters did not settle at idle; concurrent traffic or incomplete metrics are possible")
    observed = delta.get("vllm:generation_tokens_total")
    if completion_tokens is None or observed != completion_tokens:
        issues.append("Generation accounting differs: metrics=%r, received usage=%r" % (observed, completion_tokens))
    if any(value < 0 for value in delta.values()) or any(value < 0 for value in positions.values()):
        issues.append("Counter reset/decrease detected")
    for key in old:
        if key.endswith("_total") and key not in new:
            issues.append("Counter disappeared: " + key)
    counters = {name: delta.get("vllm:spec_decode_num_" + name + "_total")
                for name in ("drafts", "draft_tokens", "accepted_tokens")}
    drafts, proposed, accepted = (counters[name] for name in ("drafts", "draft_tokens", "accepted_tokens"))
    active = drafts is not None and proposed is not None and accepted is not None and drafts > 0 and proposed > 0
    if expect_dspark and not active:
        issues.append("Expected speculative decoding has no positive draft/proposed counters")
    if active:
        if not 0 <= accepted <= proposed <= drafts * expected_k + 1e-6:
            issues.append("Speculative counters violate accepted <= proposed <= drafts * configured k")
        if positions and abs(sum(positions.values()) - accepted) > 1e-6:
            issues.append("Per-position accepted totals differ from accepted-token counter")
    return {"status": "CLEAN" if not issues else "CONTAMINATED", "issues": issues,
            "generation_tokens_expected": completion_tokens, "generation_tokens_observed": observed,
            "counter_deltas": delta, "drafts": drafts, "draft_tokens": proposed,
            "accepted_tokens": accepted, "accepted_tokens_by_position": positions,
            "acceptance_rate": accepted / proposed if active else None,
            "accepted_per_draft": accepted / drafts if active else None,
            "mean_draft_length": proposed / drafts if active else None,
            "activity_observed": active,
            "attribution_limit": "Exact global token accounting detects common contamination, but cannot prove absence of unrelated traffic with coincident token totals. Prefill-only/cancelled traffic can disturb timing without a generation-token mismatch."}


def inspect_graphs(directory):
    if directory is None:
        return {"status": "NOT_PROVIDED", "records": {}, "errors": []}
    records, errors = {}, []
    if not directory.is_dir():
        return {"status": "UNAVAILABLE", "records": {}, "errors": ["Observation directory does not exist: " + str(directory)]}
    for path in sorted(directory.glob("*.json")):
        try:
            raw = path.read_bytes()
            row = json.loads(raw)
            if row.get("kind") in ("graph-capture", "graph-replay", "dspark", "capacity"):
                records[path.name] = {"source_sha256": hashlib.sha256(raw).hexdigest(), "record": row}
        except Exception as error:
            errors.append({"file": path.name, "error": str(error)[:1000]})
    return {"status": "OBSERVED" if records else "EMPTY", "records": records, "errors": errors}


def graph_delta(before, after):
    fresh = {key: value for key, value in after["records"].items() if key not in before["records"]}
    return {"status": after["status"], "new_records": fresh, "errors": after["errors"],
            "new_first_shape_replay_records": sum(row["record"].get("kind") == "graph-replay" for row in fresh.values()),
            "limit": "Current observer records the first replay per manager/shape. Zero new records can mean a previously seen graph. These records cannot establish replay frequency, eager fallback fraction, or kernel time."}


def make_body(client, workload, mode, phase, index, seed, output_tokens, template=None):
    if workload == "request":
        require(template is not None, "request workload requires --request-file")
        body = copy.deepcopy(template)
        require(isinstance(body.get("messages"), list) and body["messages"], "Request file requires nonempty messages")
        require(body.get("model", client.model) == client.model, "Request file model differs from --model")
    else:
        tag = hashlib.sha256((seed + "/" + workload + "/" + phase + "/" + str(index)).encode()).hexdigest()[:24]
        body = client.body("Measurement " + tag + ". " + PROMPTS[workload])
    for key in ("max_completion_tokens", "stop", "thinking", "reasoning_effort"):
        body.pop(key, None)
    body.update(model=client.model, n=1, max_tokens=output_tokens, temperature=0, seed=42,
                ignore_eos=True, stream=True, return_token_ids=True, include_reasoning=True)
    body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
    body.pop("chat_template_kwargs", None)
    body.update(thinking={"type": "enabled" if mode == "high" else "disabled"},
                reasoning_effort="high" if mode == "high" else "none")
    return body


def timing(result, expected_output_tokens):
    batches = result["token_batches"]
    require(batches, "No native engine token batches; SSE events and characters cannot replace tokens")
    require(sum(batch["count"] for batch in batches) == result["usage"]["completion_tokens"],
            "Native token batches do not sum to completion usage")
    require(result["usage"]["completion_tokens"] == expected_output_tokens, "Output ended before fixed measurement token budget")
    stats = stream_stats(result)
    require(stats["decode_tokens_per_second"] is not None, "Insufficient distinct token-bearing arrivals")
    gaps = [right["seconds"] - left["seconds"] for left, right in zip(batches, batches[1:])]
    # These are batch gaps, not per-token latency. DSpark emits variable batches.
    ordered = sorted(gaps)
    stats.update(token_batches=batches, token_batch_gap_p50_seconds=statistics.median(gaps) if gaps else None,
                 token_batch_gap_p95_seconds=ordered[max(0, math.ceil(len(ordered) * .95) - 1)] if ordered else None,
                 first_visible_note="Inherited Client's first_visible_seconds means first text OR reasoning delta; it is not the first final-answer token. Final-answer-only speed is not measured.",
                 gap_note="Gaps are between client token-bearing byte boundaries, not individual generated tokens")
    return stats


def round_analysis(stats, audit, baseline_tps=60, target_tps=None):
    measured = stats.get("decode_tokens_per_second")
    answer = {"baseline_decode_tps": baseline_tps,
              "observed_decode_tps_over_baseline": measured / baseline_tps if measured else None,
              "status": "NOT_ATTRIBUTABLE", "speculation_speedup_proven": False,
              "next_measurement": "Paired same-request speculation on/off (separate operator-selected launches) and GPU profiling are required to establish speedup and identify kernel/communication/host-memory causes"}
    if audit["status"] != "CLEAN" or not audit["activity_observed"] or not measured:
        return answer
    drafts = audit["drafts"]
    effective = 1 + audit["accepted_per_draft"]
    # Prefill/first-token and final-round boundaries differ from the decode
    # interval. This is an accounting estimate, not GPU-event timing.
    round_seconds = stats["decode_seconds"] / drafts
    answer.update(status="ACCOUNTED_CLIENT_ESTIMATE", effective_tokens_per_draft_round_estimate=effective,
                  observed_completion_tokens_per_draft=stats["completion_tokens"] / drafts,
                  client_decode_interval_per_draft_ms_estimate=1000 * round_seconds,
                  estimate_limit="One bonus token per draft round is assumed. Prefill/first-batch/final-round boundaries and scheduling affect these estimates; neither value is a GPU timing measurement.")
    if target_tps is not None:
        answer["hypothetical_target"] = {"decode_tps": target_tps,
            "required_effective_tokens_per_round_at_same_cost": effective * target_tps / measured,
            "required_round_ms_at_same_effective_tokens": 1000 * round_seconds * measured / target_tps,
            "required_round_time_reduction_fraction": 1 - measured / target_tps,
            "note": "Arithmetic sensitivity only; no achievable speed or independence between acceptance and round cost is claimed"}
    return answer


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def persist_snapshot(snapshot, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    raw = snapshot["raw"].encode()
    path = directory / (name + ".prom")
    path.write_bytes(raw)
    return {**{key: value for key, value in snapshot.items() if key != "raw"},
            "raw_metrics_path": str(path.resolve()), "raw_metrics_sha256": hashlib.sha256(raw).hexdigest()}


def summarize(samples):
    groups = {}
    for row in samples:
        if row["phase"] != "sample":
            continue
        key = row["workload"] + "/" + row["mode"]
        group = groups.setdefault(key, {"samples": 0, "failed": 0, "contaminated": 0, "rates": [], "e2e_rates": [], "ttfts": []})
        group["samples"] += 1
        if row["status"] != "PASS":
            group["failed"] += 1
        if row.get("accounting", {}).get("status") != "CLEAN":
            group["contaminated"] += 1
        if row.get("timing_valid"):
            group["rates"].append(row["timing"]["decode_tokens_per_second"])
            group["e2e_rates"].append(row["timing"]["e2e_tokens_per_second"])
            group["ttfts"].append(row["timing"]["ttft_seconds"])
    for group in groups.values():
        for source, dest in (("rates", "decode_tps_median"), ("e2e_rates", "e2e_tps_median"), ("ttfts", "ttft_seconds_median")):
            values = group.pop(source)
            group[dest] = statistics.median(values) if values else None
        group["measurement_status"] = "CLEAN" if not group["failed"] and not group["contaminated"] else "NOT_SIGNOFF"
    return groups


def run(args):
    args.output = args.output.resolve()
    for source in (args.request_file, args.runtime_report):
        require(source is None or source.resolve() != args.output, "Output must not overwrite a supplied request/runtime report")
    key = args.api_key_file.read_text().strip() if args.api_key_file else os.environ.get("VLLM_API_KEY", "")
    client = Client(args.base_url, args.model, args.timeout, key, args.unix_socket)
    template = json.loads(args.request_file.read_text()) if args.request_file else None
    workloads = args.workloads or ["text", "code"] + (["request"] if template is not None else [])
    require("request" not in workloads or template is not None, "request workload requires --request-file")
    require(len(workloads) == len(set(workloads)) and len(args.modes) == len(set(args.modes)), "Duplicate workloads/modes would overwrite raw metric evidence")
    report = {"schema": 1, "status": "RUNNING", "started_at": utc_now(), "model": args.model,
        "base_url": args.base_url, "unix_socket": str(args.unix_socket) if args.unix_socket else None,
        "configuration_changed": False, "weights_downloaded": False,
        "scope": "Sequential C1 requests against the operator's existing launch. C32 scheduler, 524288 context and native multimodal configuration remain untouched; this tool does not prove their capacity or quality.",
        "timing_method": "Raw EngineClient IDs, official usage and global counters must match including reasoning. Native counts at unchanged SSE body byte offsets are bound to client receipt/parser times. Decode and TTFT use client token-batch arrivals including HTTP/parser overhead as before. first_visible_seconds is first visible text/reasoning delta. Official SSE batching can differ; no strict cross-version speedup is claimed.",
        "workloads": workloads, "modes": args.modes, "warmups_per_case": args.warmups,
        "samples_per_case": args.samples, "output_tokens": args.output_tokens, "seed": args.seed,
        "request_file_sha256": hashlib.sha256(args.request_file.read_bytes()).hexdigest() if args.request_file else None,
        "request_file_policy": "Original messages, images and tools retained; repeated private request shares prefix cache, reported cached_tokens must be inspected. Only measurement generation/stream/thinking controls are replaced. No prompt or response text is saved.",
        "expected_draft_k": args.expected_draft_k, "expect_dspark": args.expect_dspark,
        "samples": [], "initial_graph_observations": inspect_graphs(args.observations)}
    if args.runtime_report:
        raw = args.runtime_report.read_bytes()
        report["supplied_runtime_report"] = {"path": str(args.runtime_report.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(), "report": json.loads(raw),
            "limit": "Caller-supplied launch report; current endpoint identity/freshness is not proven by this file"}
    save_report(args.output, report)
    raw_directory = args.output.with_name(args.output.stem + ".metrics")
    try:
        for workload in workloads:
            for mode in args.modes:
                for phase, count in (("warmup", args.warmups), ("sample", args.samples)):
                    for index in range(count):
                        name = "%s-%s-%s-%03d" % (workload, mode, phase, index)
                        print("START " + name, flush=True)
                        body = make_body(client, workload, mode, phase, index, args.seed, args.output_tokens, template)
                        row = {"workload": workload, "mode": mode, "phase": phase, "index": index,
                               "request_sha256": digest(body), "status": "FAIL", "timing_valid": False,
                               "started_at": utc_now()}
                        before_graph = inspect_graphs(args.observations)
                        before = metrics_snapshot(client, args.settle_seconds)
                        row["metrics_before"] = persist_snapshot(before, raw_directory, name + "-before")
                        result = None
                        try:
                            row["request_started_at"] = utc_now()
                            result = client.request("/v1/chat/completions", body)
                            row["received_usage"] = result.get("usage")
                            row["received_token_batches"] = result.get("token_batches", [])
                            row["timing"] = timing(result, args.output_tokens)
                            row["timing_valid"] = True
                            row["status"] = "PASS"
                        except Exception as error:
                            row["error"] = type(error).__name__ + ": " + str(error)[:3000]
                        finally:
                            after = metrics_snapshot(client, args.settle_seconds)
                            row["metrics_after"] = persist_snapshot(after, raw_directory, name + "-after")
                            completion = (result or {}).get("usage", {}).get("completion_tokens")
                            row["accounting"] = account(before, after, completion, args.expect_dspark, args.expected_draft_k)
                            row["graphs"] = graph_delta(before_graph, inspect_graphs(args.observations))
                            if row["timing_valid"]:
                                row["analysis"] = round_analysis(row["timing"], row["accounting"], args.baseline_tps, args.target_tps)
                            row["finished_at"] = utc_now()
                            report["samples"].append(row)
                            report["summary"] = summarize(report["samples"])
                            save_report(args.output, report)
                        print(json.dumps({"case": name, "status": row["status"], "accounting": row["accounting"]["status"],
                            "decode_tps": row.get("timing", {}).get("decode_tokens_per_second"),
                            "ttft_seconds": row.get("timing", {}).get("ttft_seconds"),
                            "accepted_per_draft": row["accounting"]["accepted_per_draft"]}), flush=True)
        failed = any(row["status"] != "PASS" for row in report["samples"])
        contaminated = any(row["accounting"]["status"] != "CLEAN" for row in report["samples"])
        report["status"] = "FAIL" if failed else "CONTAMINATED" if contaminated else "PASS"
    except (KeyboardInterrupt, Exception) as error:
        report.update(status="FAIL", fatal_error=type(error).__name__ + ": " + str(error)[:3000])
    report["finished_at"] = utc_now()
    save_report(args.output, report)
    print("%s: %s" % (report["status"], args.output), flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--unix-socket", type=Path)
    parser.add_argument("--model", default=MODEL, help="Public API model identity; defaults to acceptance.MODEL")
    parser.add_argument("--output", type=Path, default=Path("speed-diagnostics.json"))
    parser.add_argument("--request-file", type=Path, help="Existing local Chat Completions JSON; preserves messages/images/tools")
    parser.add_argument("--workloads", nargs="+", choices=("text", "code", "request"))
    parser.add_argument("--modes", nargs="+", choices=("off", "high"), default=["off", "high"])
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--observations", type=Path, help="Optional current launch observer directory, read only")
    parser.add_argument("--runtime-report", type=Path, help="Optional capacity/launch JSON, read only and explicitly unverified for freshness")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--settle-seconds", type=float, default=12)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--seed", default="a100-speed-diagnosis-v1")
    parser.add_argument("--expected-draft-k", type=int, default=5)
    parser.add_argument("--expect-dspark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-tps", type=float, default=60, help="Comparison reference only; does not invent a measurement")
    parser.add_argument("--target-tps", type=float, help="Optional hypothetical sensitivity arithmetic, never a speed forecast")
    args = parser.parse_args()
    require(args.timeout > 0 and args.settle_seconds >= 0 and args.samples >= 1 and args.warmups >= 0, "Invalid timeout/sample count")
    require(args.output_tokens >= 128 and args.expected_draft_k >= 1 and args.baseline_tps > 0, "Positive token/k/reference values required")
    require(args.target_tps is None or args.target_tps > 0, "Target TPS must be positive")
    require(not args.unix_socket or args.base_url.startswith("http://"), "Unix socket requires an http:// placeholder URL")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
