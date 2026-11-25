"""Explicitly blocked: do not invent visual scores for the missing farm."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--report-dir', type=Path, default=Path('/logs/verifier'))
args = parser.parse_args()
args.report_dir.mkdir(parents=True, exist_ok=True)
(args.report_dir / 'reward.txt').unlink(missing_ok=True)
(args.report_dir / 'result.json').write_text(json.dumps({
    'status': 'BLOCKED', 'training_mask': 0,
    'reason': 'full_farm_verifier_not_implemented_or_calibrated',
    'missing': ['authorized_reference_and_assets', 'validated_Blender_runtime',
                'inspected_reference_blend', 'task_specific_visual_and_structural_rubric',
                'positive_and_adversarial_negative_examples']}, indent=2), encoding="utf-8")
raise SystemExit(2)
