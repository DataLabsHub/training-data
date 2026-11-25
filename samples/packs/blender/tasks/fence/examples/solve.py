"""Evaluator-owned reference solution. Keep out of the agent image."""
import bpy
from mathutils import Vector


def build(assets_dir, references_dir):
    for index in range(1, 5):
        obj = bpy.data.objects[f'FencePost_{index:02d}']
        obj.location.z -= min((obj.matrix_world @ Vector(v)).z for v in obj.bound_box)
    bpy.context.view_layer.update()
