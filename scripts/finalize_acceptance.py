"""Seal startup acceptance only after every real-machine report passes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

FULL_CASES = {'model-identity', 'text-SSE-off', 'text-SSE-high', 'single-image-OCR',
              'multi-image-OCR', 'reverse-tool-result-image-association',
              'prefix-cache-cold-hot-details', 'C32-auto-tool-calls',
              'DSpark-native-k5-counters', 'benchmark-off', 'benchmark-high',
              'independent-near-524288-token-retrieval'}


def validate_extended(full, diagnosis):
    assert full['status'] == 'PASS' and full['suite'] == 'full'
    assert full['context'] == 524288 and full['concurrency'] == 32
    assert not full['contamination']
    assert {row['name'] for row in full['tests']} == FULL_CASES
    assert all(row['status'] == 'PASS' for row in full['tests'])
    assert set(full['benchmarks']) == {'off', 'high'}
    for mode, report in full['benchmarks'].items():
        assert not report['accounting']['issues'], mode
        for phase, count in [('warmup', 2), ('single', 10), ('c32', 128)]:
            assert report['phases'][phase]['requests'] == count
            rows = [row for row in report['samples'] if row['phase'] == phase]
            assert len(rows) == count
            assert all(row['status'] == 'PASS' and row['completion_tokens'] == 1024 for row in rows)
    assert diagnosis['status'] == 'PASS'
    assert diagnosis['warmups_per_case'] == 2 and diagnosis['samples_per_case'] == 10
    assert diagnosis['output_tokens'] == 1024
    assert set(diagnosis['summary']) == {'text/off', 'text/high', 'code/off', 'code/high'}
    assert all(row['samples'] == 10 and row['measurement_status'] == 'CLEAN'
               for row in diagnosis['summary'].values())


def finalize(root):
    required = ['preflight.json', 'beta-mla-gpu.json', 'selected-sm80-gpu.json',
                'live-acceptance.json', 'cpu-kv-acceptance.json', 'full-acceptance.json',
                'speed-diagnosis.json', 'c32-acceptance.json', 'long-context-acceptance.json',
                'memory-reserve.json']
    plan = json.loads((root/'cpu-kv-plan.json').read_text())
    reports = {name: json.loads((root/name).read_text()) for name in required}
    for name, report in reports.items():
        if name in ('c32-acceptance.json', 'long-context-acceptance.json'):
            assert report['all_requests_finished'] and report['all_needles_found'], name
        elif name == 'cpu-kv-acceptance.json' and plan['effective_gib'] == 0:
            assert report['status'] == 'SKIP', name
        else:
            assert report['status'] == 'PASS', name
    validate_extended(reports['full-acceptance.json'], reports['speed-diagnosis.json'])
    profile = json.loads((root/'release-profile.json').read_text())
    profile.update(gpu_validation='PASS', startup_acceptance='PASS',
                   cpu_kv_gib=plan['effective_gib'], cpu_kv_policy=plan['eviction_policy'],
                   available_ram_reserve_gib=500, completed_at=datetime.now(timezone.utc).isoformat(),
                   checks=required, actual_app_gateway='NOT_RUN',
                   image_id=(root/'image-id').read_text().strip(),
                   effective_config_sha256=hashlib.sha256((root/'effective-config.json').read_bytes()).hexdigest(),
                   performance_claim='MEASURED_ONLY_NO_PROMISED_SPEEDUP',
                   benchmark_summary={mode: {key: value for key, value in report.items() if key != 'samples'}
                                      for mode, report in reports['full-acceptance.json']['benchmarks'].items()},
                   diagnosis_summary=reports['speed-diagnosis.json']['summary'])
    (root/'release-profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    return profile


if __name__ == '__main__':
    print(json.dumps(finalize(Path('/state')), indent=2))
