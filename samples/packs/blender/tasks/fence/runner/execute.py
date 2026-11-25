"""Blender-side script adapter. Preserve the submitted scene for native grading."""
import argparse
from pathlib import Path
import runpy
import sys
import tempfile
import bpy

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--export-glb', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.initial:
        bpy.ops.wm.open_mainfile(filepath=str(args.initial), use_scripts=False)
    elif (ROOT / 'assets/optional_base.blend').exists():
        bpy.ops.wm.open_mainfile(filepath=str(ROOT / 'assets/optional_base.blend'), use_scripts=False)
    elif (ROOT / 'assets/create_base.py').exists():
        fixture = runpy.run_path(str(ROOT / 'assets/create_base.py'))
        with tempfile.TemporaryDirectory(prefix='blender-initial-') as temporary:
            fixture['create'](Path(temporary) / 'initial.blend')
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    # A build() callback is optional; plain top-level bpy scripts also work.
    old_argv = sys.argv[:]
    sys.argv = [str(args.script)]
    try:
        scope = runpy.run_path(str(args.script), run_name='__main__')
        if callable(scope.get('build')):
            scope['build'](ROOT / 'assets', ROOT / 'references')
    finally:
        sys.argv = old_argv
    bpy.context.view_layer.update()
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / 'result.blend'))
    # Optional exports are made AFTER saving the native artifact and never saved back.
    if args.export_glb:
        bpy.ops.export_scene.gltf(filepath=str(args.output / 'result.glb'), export_format='GLB')
    if args.render:
        scene = bpy.context.scene
        if scene.camera is None:
            raise RuntimeError('Preview requested but the submission has no active camera')
        renders = args.output / 'renders'
        renders.mkdir(exist_ok=True)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = str(renders / 'view_01.png')
        bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
