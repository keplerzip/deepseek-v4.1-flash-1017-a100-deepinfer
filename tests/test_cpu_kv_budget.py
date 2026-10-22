import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cpu_kv_budget',ROOT/'scripts/cpu_kv_budget.py')
budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)
settings=json.loads((ROOT/'configs/cpu-kv.json').read_text())
GIB=1<<30

class CPUKVBudgetTests(unittest.TestCase):
    def test_total_capacity_cap_and_startup_reserve(self):
        # An already running model has ~1.5 TiB available; the stopped model
        # releases Engram, so model startup headroom must be budgeted separately.
        for available in (1536,1728,2048):
            p=budget.plan(settings,available*GIB,1024*GIB)
            self.assertLessEqual(p['effective_gib'],768)
            self.assertGreaterEqual(available-p['effective_gib']-384-16,500)
            self.assertEqual(p['effective_gib']%16,0)
        self.assertEqual(budget.plan(settings,1728*GIB,1024*GIB)['effective_gib'],768)

    def test_shm_caps_pool_and_no_room_fails(self):
        self.assertEqual(budget.plan(settings,2048*GIB,258*GIB)['effective_gib'],256)
        with self.assertRaises(ValueError):budget.plan(settings,900*GIB,1024*GIB)
        with self.assertRaises(ValueError):budget.plan(settings,2048*GIB,8*GIB)

    def test_explicit_off_and_invalid_sizes(self):
        self.assertEqual(budget.plan(settings,1000*GIB,1024*GIB,0)['effective_gib'],0)
        for value in (-1,769,True,1.5):
            with self.subTest(value=value),self.assertRaises(ValueError):
                budget.plan(settings,2048*GIB,1024*GIB,value)

    def test_cgroup_headroom_is_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mem=root/'meminfo';cg=root/'cg';cg.mkdir()
            mem.write_text(f'MemAvailable: {2048*1024*1024} kB\n')
            (cg/'memory.max').write_text(str(1400*GIB));(cg/'memory.current').write_text(str(200*GIB))
            self.assertEqual(budget.available_bytes(mem,cg),1200*GIB)


    def test_policy_default_control_and_invalid_choice(self):
        self.assertEqual(budget.plan(settings,2048*GIB,1024*GIB)['eviction_policy'],'arc')
        self.assertEqual(budget.plan(settings,2048*GIB,1024*GIB,policy='lru')['eviction_policy'],'lru')
        with self.assertRaises(ValueError):budget.plan(settings,2048*GIB,1024*GIB,policy='random')
