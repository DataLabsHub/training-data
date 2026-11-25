"""Pure scorer; its input must come from trusted extraction, never the agent."""
import math

TARGETS = tuple(f'FencePost_{i:02d}' for i in range(1, 5))
CONTACT_TOLERANCE_M = 0.002
PRESERVATION_TOLERANCE = 0.000001


def finite_tree(x):
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, list):
        return all(finite_tree(v) for v in x)
    return True


def score(before, after):
    if after.get("invalid_submission"):
        return {"status": "FAIL", "reward": 0.0, "training_mask": 1, "failures": [after["invalid_submission"]]}
    if not finite_tree(after):
        return {'status': 'FAIL', 'reward': 0.0, 'training_mask': 1,
                'failures': ['nonfinite_measurement']}
    failures = []
    if before.get("protected_state") != after.get("protected_state"):
        failures.append("protected_state_changed")
    original, candidate = before['objects'], after['objects']
    if before['blender_version'] != after['blender_version']:
        raise ValueError('baseline/submission extractor version mismatch')
    if after['prohibited']:
        failures.append('prohibited_features')
    if set(original) != set(candidate):
        failures.append('object_set_changed')
    if before['scene'] != after['scene']:
        failures.append('scene_changed')
    for name in sorted(set(original) & set(candidate)):
        a, b = original[name], candidate[name]
        for field in ('type', 'geometry', 'data_properties', 'visibility'):
            if a[field] != b[field]:
                failures.append(f'{name}:{field}_changed')
        for row in range(4):
            for col in range(4):
                if name in TARGETS and (row, col) == (2, 3):
                    continue
                if abs(a['matrix'][row][col] - b['matrix'][row][col]) > PRESERVATION_TOLERANCE:
                    failures.append(f'{name}:forbidden_transform')
                    break
    gaps = {}
    for name in TARGETS:
        obj = candidate.get(name)
        if not obj or not obj.get('bounds'):
            failures.append(f'{name}:missing_geometry')
            continue
        gap = obj['bounds']['min'][2]  # Fixture ground top is fixed at z=0.
        gaps[name] = gap
        if abs(gap) > CONTACT_TOLERANCE_M:
            failures.append(f'{name}:not_grounded')
    failures = sorted(set(failures))
    return {'status': 'FAIL' if failures else 'PASS', 'reward': float(not failures),
            'training_mask': 1, 'failures': failures, 'gaps_m': gaps,
            'scope': 'procedural_grounding_only_no_visual_fidelity_score'}
