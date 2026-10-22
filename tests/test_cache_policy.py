import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import check_cache_policy as check

class CachePolicyTests(unittest.TestCase):
    def test_hot_dialogue_resistance_without_regressing_stable_working_set(self):
        r=check.compare()['policies']
        self.assertGreater(r['arc']['scan_with_hot_dialogues']['hits'],r['lru']['scan_with_hot_dialogues']['hits'])
        self.assertEqual(r['arc']['stable_working_set']['hits'],r['lru']['stable_working_set']['hits'])

    def test_both_policies_preserve_active_transfers_and_atomic_failed_eviction(self):
        for policy in ('arc','lru'):
            with self.subTest(policy=policy):check.validate_safety(policy)
