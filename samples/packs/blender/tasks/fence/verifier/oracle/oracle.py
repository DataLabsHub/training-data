"""Reference solution for the procedural fixture only; evaluator-only file."""
import sys
from pathlib import Path
import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:]
src, dst = args if len(args) == 2 else ('/workspace/initial.blend', '/workspace/submission.blend')
bpy.ops.wm.open_mainfile(filepath=str(Path(src).resolve()), use_scripts=False)
for i in range(1, 5):
    obj = bpy.data.objects[f'FencePost_{i:02d}']
    bottom = min((obj.matrix_world @ Vector(v)).z for v in obj.bound_box)
    obj.location.z -= bottom
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(Path(dst).resolve()))
