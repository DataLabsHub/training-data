"""Contract/regression tests using simulated extracted facts, not Blender execution."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'verifier'))
from constraint_check import TARGETS, score
from task_check import publish, verify


def facts(solved=False):
    result = {'schema_version': 1, 'blender_version': 'test_fixture_not_real_blender',
              'scene': {'camera': 'Camera'}, 'prohibited': [], 'objects': {}}
    for name in ('Ground', 'Barn', *TARGETS):
        m = [[float(i == j) for j in range(4)] for i in range(4)]
        m[2][3] = 0.5 if solved else 0.56
        result['objects'][name] = {'type': 'MESH', 'geometry': name + '_hash',
            'bounds': {'min': [0, 0, 0 if solved else 0.06], 'max': [1, 1, 1]},
            'matrix': m, 'data_properties': None, 'visibility': [False, False, False, True]}
        if name not in TARGETS:
            result['objects'][name]['matrix'][2][3] = 0
    return result


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.before = facts()
        self.after = facts(True)

    def fail_score(self):
        result = score(self.before, self.after)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['reward'], 0)
        self.assertEqual(result['training_mask'], 1)

    def test_valid_repair_passes(self):
        self.assertEqual(score(self.before, self.after)['reward'], 1)

    def test_unchanged_scene_fails(self):
        self.after = self.before
        self.fail_score()

    def test_missing_post_fails(self):
        del self.after['objects'][TARGETS[0]]
        self.fail_score()

    def test_moved_ground_fails(self):
        self.after['objects']['Ground']['matrix'][2][3] = 0.06
        self.fail_score()

    def test_changed_mesh_fails(self):
        self.after['objects'][TARGETS[0]]['geometry'] = 'squashed'
        self.fail_score()

    def test_hidden_post_fails(self):
        self.after['objects'][TARGETS[0]]['visibility'][0] = True
        self.fail_score()

    def test_changed_xy_fails(self):
        self.after['objects'][TARGETS[0]]['matrix'][0][3] = 0.2
        self.fail_score()

    def test_added_object_fails(self):
        self.after['objects']['extra'] = copy.deepcopy(self.after['objects']['Ground'])
        self.fail_score()

    def test_penetration_fails(self):
        self.after['objects'][TARGETS[0]]['bounds']['min'][2] = -0.03
        self.fail_score()

    def test_nan_fails(self):
        self.after['objects'][TARGETS[0]]['bounds']['min'][2] = float('nan')
        self.fail_score()

    def test_script_features_fail(self):
        self.after['prohibited'] = ['texts']
        self.fail_score()

    def test_scene_change_fails(self):
        self.after['scene']['camera'] = 'different'
        self.fail_score()

    def test_version_mismatch_is_grader_error(self):
        self.after['blender_version'] = 'different'
        with self.assertRaises(ValueError):
            score(self.before, self.after)


class RunnerTests(unittest.TestCase):
    def test_error_removes_stale_reward(self):
        with tempfile.TemporaryDirectory() as d:
            publish(d, {'status': 'PASS', 'reward': 1, 'training_mask': 1})
            publish(d, {'status': 'ERROR', 'training_mask': 0})
            self.assertFalse((Path(d) / 'reward.txt').exists())

    def test_missing_runtime_is_error_not_fail(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input.blend'
            source.write_bytes(b'BLENDER-fixture')
            rc = verify(source, d, 'definitely_not_installed_blender_7341')
            result = json.loads((Path(d) / 'result.json').read_text(encoding="utf-8"))
            self.assertEqual(rc, 2)
            self.assertEqual(result['status'], 'ERROR')
            self.assertEqual(result['training_mask'], 0)
            self.assertFalse((Path(d) / 'reward.txt').exists())

    def test_verifier_fixture_matches_public_fixture(self):
        self.assertEqual((ROOT / 'verifier/fixture.py').read_bytes(),
                         (ROOT / 'assets/create_base.py').read_bytes())


if __name__ == '__main__':
    unittest.main()
