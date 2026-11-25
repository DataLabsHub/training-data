import json, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

class CompatibilityTests(unittest.TestCase):
    def test_full_farm_blocked_without_reward(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'reward.txt').write_text('1', encoding="utf-8")
            run = subprocess.run([sys.executable, str(ROOT / 'verifier/compat/farm_blocked.py'), '--report-dir', d])
            self.assertEqual(run.returncode, 2)
            result = json.loads((Path(d) / 'result.json').read_text(encoding="utf-8"))
            self.assertEqual(result['status'], 'BLOCKED')
            self.assertEqual(result['training_mask'], 0)
            self.assertFalse((Path(d) / 'reward.txt').exists())

