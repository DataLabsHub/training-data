"""Blender worker: plan inspection views, render, and bind evidence to input hash.

Runs on an in-memory copy. Never saves or overwrites the submitted .blend.
Auto visibility is a geometric ray heuristic, not a visual quality judgement.
"""
import argparse
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from view_plan import bounds, fit_camera, validate, visibility_targets
from support import native_file_preflight, native_preflight, POLICY


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def matches(name, patterns):
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def geometry(scene, graph):
    result = {}
    for obj in scene.objects:
        if obj.type not in ('MESH', 'CURVE', 'SURFACE', 'FONT', 'META', 'VOLUME') or obj.hide_render or not obj.visible_get():
            continue
        evaluated = obj.evaluated_get(graph)
        if obj.type == 'MESH' and not len(evaluated.data.vertices):
            continue
        result[obj.name] = [list(evaluated.matrix_world @ Vector(p)) for p in evaluated.bound_box]
    return result


def visibility(scene, graph, location, names, boxes):
    """Sample bounding-box center/face centers; require first hit on a target."""
    hits, count = 0, 0
    target_set = set(names)
    for name in visibility_targets(names, boxes):
        center, _ = bounds(boxes[name])
        points = [center] + [[(p[i] + center[i]) / 2 for i in range(3)] for p in boxes[name]]
        for point in points:
            delta = Vector(point) - Vector(location)
            if delta.length < 1e-9:
                continue
            found, _, _, _, obj, _ = scene.ray_cast(graph, Vector(location), delta.normalized(), distance=delta.length * 1.01)
            count += 1
            if found and obj and obj.original.name in target_set:
                hits += 1
    return hits / count if count else 0.0


def auto_plan(args, scene, graph, boxes):
    eligible = [n for n in boxes if not matches(n, args['exclude_objects'])]
    if not eligible:
        eligible = list(boxes)
    if not eligible:
        raise ValueError('no visible supported geometry')
    groups = args.get('groups', [])
    if not groups:
        # Largest individual objects are a fallback, not semantic recognition.
        ranked = sorted(eligible, key=lambda n: (-bounds(boxes[n])[1], n))
        groups = [{'name': 'detail_%02d' % (i + 1), 'resolved': [n], 'lens': 50}
                  for i, n in enumerate(ranked[:args['max_closeups']])]
    plans = []
    for group in [{'name': 'overview', 'resolved': eligible, 'lens': 50}, *groups]:
        names = group.get('resolved')
        if names is None:
            names = [n for n in boxes if matches(n, group['objects'])]
        if not names:
            raise ValueError('no visible geometry matches group ' + group['name'])
        points = [p for n in names for p in boxes[n]]
        candidates = []
        for elevation in (20, 35):
            for azimuth in (-60, -120, 0, 60, 120, 180):
                candidate = fit_camera(points, azimuth, elevation, group.get('lens', 50), args['resolution'])
                candidate.update(azimuth=azimuth, elevation=elevation)
                candidate['visibility_proxy'] = visibility(scene, graph, candidate['location'], names, boxes)
                candidates.append(candidate)
        best = max(candidates, key=lambda v: v['visibility_proxy'])
        best.update(name=group['name'], targets=names,
                    evidence_type='beauty' if group['name'] == 'overview' else 'closeup',
                    selection_method='bbox_fit_and_bounded_ray_visibility', candidate_count=len(candidates),
                    visibility_sample_targets=len(visibility_targets(names, boxes)), visibility_max_rays_per_candidate=576)
        plans.append(best)
        if group['name'] == 'overview':
            reverse = fit_camera(points, best['azimuth'] + 180, best['elevation'], best['lens'], args['resolution'])
            reverse.update(name='reverse', targets=names, evidence_type='beauty',
                           selection_method='opposite_overview',
                           visibility_sample_targets=len(visibility_targets(names, boxes)), visibility_max_rays_per_candidate=576,
                           visibility_proxy=visibility(scene, graph, reverse['location'], names, boxes))
            plans.append(reverse)
    return plans


def execute(source, request, output):
    args = validate(json.loads(Path(request).read_text(encoding="utf-8")))
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('output directory must be new')
    artifact_hash = sha256(source)
    output.mkdir(parents=True)
    manifest = {'schema': 'render.evidence.v1', 'artifact_hash': artifact_hash,
                'blender_version': bpy.app.version_string, 'status': 'ERROR', 'views': [],
                'quality_stage': args.get('quality_stage', 'evidence'),
                'mode': args['mode'], 'resolution': args['resolution'],
                'engine': args['engine'], 'samples': args['samples'],
                'request_hash': sha256(request), 'warnings': [],
                'scope': 'inspection_evidence_not_reference_match_or_quality_score'}
    try:
        native_file_preflight(source)
        bpy.ops.wm.open_mainfile(filepath=str(source), use_scripts=False)
        native_preflight(bpy)
        scene = bpy.context.scene
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        boxes = geometry(scene, graph)
        manifest['submission_camera'] = scene.camera.name if scene.camera else None
        if any(o.instance_type != 'NONE' or o.type == 'CURVES' for o in scene.objects):
            manifest['warnings'].append('Auto framing does not fully cover instances or native hair; inspect images and use explicit views as needed.')
        views = auto_plan(args, scene, graph, boxes) if args['mode'] == 'auto' else [dict(v, selection_method='explicit_request') for v in args['views']]
        if any(v.get('visibility_proxy', 1) < 0.5 for v in views):
            manifest['warnings'].append('Some selected targets have low ray visibility; additional views or review are required.')
        if args.get('gray_override', False):
            gray = bpy.data.materials.new('EvaluatorGray')
            gray.use_nodes = True
            shader = gray.node_tree.nodes.get('Principled BSDF')
            shader.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)
            shader.inputs['Roughness'].default_value = 0.8
            bpy.context.view_layer.material_override = gray
            manifest['material_override'] = 'neutral_gray'
        # Ignore request.output_dir: host/container output is owned by the runner.
        scene.render.engine = args['engine']
        if args['engine'] == 'CYCLES':
            scene.cycles.samples = args['samples']
            scene.cycles.seed = 0
            scene.cycles.use_animated_seed = False
            scene.cycles.device = 'CPU'
            scene.cycles.use_denoising = True
        scene.render.resolution_x, scene.render.resolution_y = args['resolution']
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
        scene.render.image_settings.file_format = 'PNG'
        scene.render.use_border = False
        scene.render.use_crop_to_border = False
        # Keep submission lighting/materials; bypass compositor/VSE for camera evidence.
        scene.render.use_compositing = False
        scene.render.use_sequencer = False
        data = bpy.data.cameras.new('EvaluatorInspectionCamera')
        camera = bpy.data.objects.new('EvaluatorInspectionCamera', data)
        scene.collection.objects.link(camera)
        scene.camera = camera
        data.type = 'PERSP'
        data.sensor_fit = 'HORIZONTAL'
        data.sensor_width = 36
        data.dof.use_dof = False
        for view in views:
            started = time.monotonic()
            camera.location = view['location']
            direction = Vector(view['target']) - camera.location
            camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
            data.lens = view['lens']
            data.clip_start = view.get('clip_start', max(0.00001, min(0.01, direction.length / 1000)))
            data.clip_end = view.get('clip_end', max(1000, direction.length * 10))
            bpy.context.view_layer.update()
            # Verify the auto fit against Blender's actual camera projection.
            if view.get('targets'):
                projected = [world_to_camera_view(scene, camera, Vector(p)) for n in view['targets'] for p in boxes[n]]
                view['framing_verified'] = all(p.z > 0 and 0 <= p.x <= 1 and 0 <= p.y <= 1 for p in projected)
                if not view['framing_verified']:
                    raise ValueError('auto camera projection failed for ' + view['name'])
            image_path = output / (view['name'] + '.png')
            scene.render.filepath = str(image_path)
            bpy.ops.render.render(write_still=True)
            if not image_path.is_file():
                raise RuntimeError('render did not produce ' + image_path.name)
            manifest['views'].append(dict(view, image=image_path.name, image_sha256=sha256(image_path),
                                          seconds=round(time.monotonic() - started, 3)))
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if sha256(source) != artifact_hash:
            raise RuntimeError('source changed during evidence collection')
        manifest['status'] = 'COMPLETE'
        return manifest
    except Exception as exc:
        manifest['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    execute(args.submission, args.request, args.output)
