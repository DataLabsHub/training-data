"""Blender-side general eligibility checks. Does not modify the source file."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from general_contract import CHECKS, check
from render_views import execute, sha256
from support import native_file_preflight, InvalidSubmission, native_preflight, BudgetExceeded, POLICY


def reachable_ids(scene):
    # IDs referenced transitively by the active scene, not all unused datablocks.
    users = bpy.data.user_map()
    reached = {scene}
    while True:
        added = {item for item, owners in users.items() if owners & reached} - reached
        if not added:
            return reached
        reached |= added


def resources(scene):
    snapshot_root = Path(bpy.data.filepath).resolve().parent
    deps, missing, unknown = [], [], []
    reached = reachable_ids(scene)
    for item in reached:
        if isinstance(item, bpy.types.Image):
            if item.source in ('GENERATED', 'VIEWER'):
                continue
            packed = bool(item.packed_file) or bool(getattr(item, 'packed_files', []))
            if item.source in ('TILED', 'SEQUENCE', 'MOVIE'):
                unknown.append(item.name + ':unsupported_' + item.source)
                continue
            if packed:
                deps.append({'name': item.name, 'kind': 'image', 'packed': True})
                continue
            path = Path(bpy.path.abspath(item.filepath, library=item.library))
            if not item.filepath or not path.is_file() or snapshot_root not in path.resolve().parents:
                missing.append({'name': item.name, 'path': str(path), 'kind': 'image'})
            else:
                deps.append({'name': item.name, 'path': str(path), 'kind': 'image', 'sha256': sha256(path)})
        if getattr(item, 'library', None):
            lib = item.library
            path = Path(bpy.path.abspath(lib.filepath, library=lib.parent))
            if not path.is_file() or snapshot_root not in path.resolve().parents:
                missing.append({'name': lib.name, 'path': str(path), 'kind': 'library'})
            else:
                entry = {'name': lib.name, 'path': str(path), 'kind': 'library', 'sha256': sha256(path)}
                if entry not in deps:
                    deps.append(entry)
        if getattr(item, 'is_missing', False):
            missing.append({'name': item.name, 'kind': 'missing_linked_id'})
        if isinstance(item, (bpy.types.Volume, bpy.types.CacheFile, bpy.types.MovieClip, bpy.types.Sound)):
            unknown.append(item.name + ':unsupported_external_dependency_type')
    status = 'FAIL' if missing else ('UNKNOWN' if unknown else ('PASS' if deps else 'N_A'))
    return check('resources_complete', status, 'dependency_audit',
                 {'dependencies': deps, 'missing': missing, 'unsupported': unknown})


from geometry_check import geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resolution', type=int, nargs=2, default=[320, 240])
    parser.add_argument('--samples', type=int, default=4)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checks = {k: check(k) for k in CHECKS}
    payload = {'checks': list(checks.values()), 'artifact_hash': sha256(args.submission), 'blender_version': bpy.app.version_string}
    def save():
        payload['checks'] = list(checks.values())
        (out / 'general_checks.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    save()
    if '.'.join(map(str, bpy.app.version)) != POLICY['blender_version']:
        checks['project_readable'] = check('project_readable', 'UNKNOWN', 'blender_version_mismatch')
        save()
        return
    try:
        # A fresh factory scene verifies API startup before opening the submission.
        bpy.ops.wm.read_factory_settings(use_empty=True)
        try:
            native_file_preflight(args.submission)
            bpy.ops.wm.open_mainfile(filepath=str(Path(args.submission).resolve()), use_scripts=False)
            native_preflight(bpy)
            bpy.context.view_layer.update()
        except InvalidSubmission as exc:
            checks['project_readable'] = check('project_readable', 'FAIL', str(exc))
            return
        except BudgetExceeded as exc:
            checks['project_readable'] = check('project_readable', 'PASS', 'opened')
            checks['geometry_valid'] = check('geometry_valid', 'FAIL', str(exc))
            return
        except Exception as exc:
            checks['project_readable'] = check('project_readable', 'UNKNOWN', 'open_failed_needs_triage:' + str(exc))
            return
        checks['project_readable'] = check('project_readable', 'PASS', 'opened_and_read_scene', {'scene': bpy.context.scene.name})
        save()
        try:
            checks['geometry_valid'], checks['mesh_diagnostics'] = geometry(bpy.context.scene)
        except Exception as exc:
            checks['geometry_valid'] = check('geometry_valid', 'UNKNOWN', 'geometry_audit_failed:' + str(exc))
        save()
        try:
            checks['resources_complete'] = resources(bpy.context.scene)
        except Exception as exc:
            checks['resources_complete'] = check('resources_complete', 'UNKNOWN', 'resource_audit_failed:' + str(exc))
        save()
        if any(checks[k]['status'] == 'FAIL' for k in ('geometry_valid', 'resources_complete')):
            checks['inspection_render'] = check('inspection_render', 'UNKNOWN', 'skipped_after_confirmed_failure')
            return
        request = {'action': 'render.views', 'args': {'mode': 'auto', 'max_closeups': 0,
                   'resolution': args.resolution, 'samples': args.samples, 'engine': 'CYCLES'}}
        req = out / 'render_request.json'
        req.write_text(json.dumps(request), encoding="utf-8")
        try:
            manifest = execute(args.submission, req, out / 'views')
            loaded = []
            for view in manifest['views']:
                image = bpy.data.images.load(str(out / 'views' / view['image']), check_existing=False)
                size = list(image.size)
                bpy.data.images.remove(image)
                if size != args.resolution:
                    raise ValueError('render dimensions mismatch')
                loaded.append({'image': 'views/' + view['image'], 'size': size, 'framing_verified': view.get('framing_verified')})
            if not loaded:
                raise ValueError('no rendered views')
            checks['inspection_render'] = check('inspection_render', 'PASS', 'images_decoded_and_dimensions_verified',
                 {'images': loaded, 'visual_quality_assessed': False, 'warnings': manifest['warnings']}, ['views/manifest.json'])
        except Exception as exc:
            checks['inspection_render'] = check('inspection_render', 'UNKNOWN', 'render_failed_needs_triage:' + str(exc))
        # External resources are part of the evidence identity as well.
        for dep in checks['resources_complete']['measurements'].get('dependencies', []):
            if dep.get('sha256') and (not Path(dep['path']).is_file() or sha256(dep['path']) != dep['sha256']):
                checks['resources_complete'] = check('resources_complete', 'UNKNOWN', 'dependency_changed_during_evaluation')
                break
        if sha256(args.submission) != payload['artifact_hash']:
            checks['project_readable'] = check('project_readable', 'UNKNOWN', 'artifact_changed_during_evaluation')
    finally:
        save()


if __name__ == '__main__':
    main()
