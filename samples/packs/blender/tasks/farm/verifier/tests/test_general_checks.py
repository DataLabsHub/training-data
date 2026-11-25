import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'verifier'))
sys.path.insert(0, str(ROOT / 'verifier'))
from general_contract import CHECKS, check, result, publish
from reward import main


class GeneralTests(unittest.TestCase):
    def checks(self):
        return [check(k, 'PASS', 'verified') for k in CHECKS]

    def test_pass_is_eligibility_not_quality_reward(self):
        r = result(self.checks())
        self.assertEqual(r['general_status'], 'PASS')
        self.assertIsNone(r['reward'])
        self.assertEqual(r['training_mask'], 0)
        self.assertIsNone(r['accepted'])

    def test_diagnostics_do_not_fail_gate(self):
        checks = self.checks()
        checks[3]['status'] = 'FAIL'
        self.assertEqual(result(checks)['general_status'], 'PASS')

    def test_resource_failure_is_negative_training_example(self):
        checks = self.checks()
        checks[2]['status'] = 'FAIL'
        checks[4]['status'] = 'UNKNOWN'
        r = result(checks)
        self.assertEqual((r['reward'], r['training_mask']), (0, 1))

    def test_unknown_does_not_become_zero(self):
        checks = self.checks()
        checks[1]['status'] = 'UNKNOWN'
        r = result(checks)
        self.assertEqual(r['status'], 'ERROR')
        self.assertIsNone(r['reward'])

    def test_no_dependencies_is_not_failure(self):
        checks = self.checks()
        checks[2]['status'] = 'N_A'
        self.assertEqual(result(checks)['general_status'], 'PASS')

    def test_task_reward_requires_general_pass(self):
        checks = self.checks()
        task = {'status': 'PASS', 'reward': 1, 'training_mask': 1}
        self.assertEqual(result(checks, task='fence-grounding-v0', task_result=task)['reward'], 1)
        checks[0]['status'] = 'UNKNOWN'
        self.assertIsNone(result(checks, task='fence-grounding-v0', task_result=task)['reward'])

    def test_blocked_task_has_no_reward(self):
        self.assertEqual(result(self.checks(), task='farm-reconstruction-v0')['status'], 'BLOCKED')

    def test_stale_reward_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'reward.txt').write_text('1', encoding="utf-8")
            publish(p, result(self.checks()))
            self.assertFalse((p/'reward.txt').exists())

    def test_missing_submission_has_complete_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            self.assertEqual(main(['--submission', str(p/'input/missing.blend'), '--reports', str(p/'report')]), 0)
            r=json.loads((p/'report/result.json').read_text(encoding="utf-8"))
            self.assertEqual(len(r['checks']), 5)
            self.assertEqual(r['reward'], 0)

    def test_missing_runtime_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'input').mkdir(); (p/'input/model.blend').write_bytes(b'BLENDER-fixture')
            with patch('reward.shutil.which', return_value=None):
                self.assertEqual(main(['--submission', str(p/'input/model.blend'), '--reports', str(p/'report')]), 2)
            self.assertFalse((p/'report/reward.txt').exists())
