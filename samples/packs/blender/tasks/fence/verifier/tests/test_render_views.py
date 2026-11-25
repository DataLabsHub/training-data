"""Camera contract tests. Real Blender smoke evidence is recorded separately."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'verifier'))
sys.path.insert(0, str(ROOT / 'verifier'))
from view_plan import validate, fit_camera
from render_eval import main


class ViewTests(unittest.TestCase):
    def test_original_request_is_supported(self):
        request = json.loads((ROOT / 'runner/requests/review_01_render_views.request.json').read_text(encoding="utf-8"))
        args = validate(request)
        self.assertEqual(args['mode'], 'explicit')
        self.assertEqual(len(args['views']), 5)

    def test_default_is_auto(self):
        self.assertEqual(validate({'action': 'render.views'})['mode'], 'auto')

    def test_bad_camera_input(self):
        for location in ([0, 0, 0], [float('nan'), 0, 2]):
            with self.assertRaises(ValueError):
                validate({'action': 'render.views', 'args': {'views': [
                    {'name': 'test', 'location': location, 'target': [0, 0, 0]}]}})

    def test_paths_and_duplicate_names_rejected(self):
        view = {'name': '../escape', 'location': [1, 1, 1], 'target': [0, 0, 0]}
        with self.assertRaises(ValueError):
            validate({'action': 'render.views', 'args': {'views': [view]}})
        view['name'] = 'same'
        with self.assertRaises(ValueError):
            validate({'action': 'render.views', 'args': {'views': [view, copy.deepcopy(view)]}})

    def test_bounding_sphere_fits_wide_and_tall_frames(self):
        points = [[x, y, z] for x in (-10, 10) for y in (-1, 1) for z in (0, 3)]
        for resolution in ([1600, 200], [200, 1600], [640, 480]):
            camera = fit_camera(points, -60, 25, 50, resolution)
            distance = math.dist(camera['location'], camera['target'])
            half_fov = math.atan(min(36, 36 / (resolution[0] / resolution[1])) / 100)
            self.assertLess(math.asin(camera['fit_radius'] / distance), half_fov)

    def test_runner_failure_has_no_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'input').mkdir()
            source = base / 'input/input.blend'
            source.write_bytes(b'BLENDER-fixture')
            with patch('render_eval.shutil.which', return_value='/bin/false'), patch('render_eval.WorkerRuntime.run') as run:
                run.return_value = 3
                code = main(['--submission', str(source), '--output', str(base / 'out'), '--backend', 'local'])
            self.assertEqual(code, 2)
            result = json.loads((base / 'out/render_result.json').read_text(encoding="utf-8"))
            self.assertEqual(result['status'], 'ERROR')
            self.assertIsNone(result['reward'])
            self.assertFalse((base / 'out/reward.txt').exists())

    def test_runner_timeout_is_error(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'input').mkdir()
            source = base / 'input/input.blend'
            source.write_bytes(b'BLENDER-fixture')
            with patch('render_eval.shutil.which', return_value='blender'), patch('render_eval.WorkerRuntime.run', side_effect=subprocess.TimeoutExpired('blender', 1)):
                self.assertEqual(main(['--submission', str(source), '--output', str(base / 'out'), '--backend', 'local']), 2)
            result = json.loads((base / 'out/render_result.json').read_text(encoding="utf-8"))
            self.assertEqual(result['reason'], 'render_timeout')


if __name__ == '__main__':
    unittest.main()
