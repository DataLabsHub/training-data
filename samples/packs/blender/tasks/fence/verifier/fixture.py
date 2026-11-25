"""Procedural integration fixture, NOT a recovered farm checkpoint.

Run with Blender --background --factory-startup --python fixture.py -- OUTPUT.
One Blender unit is one metre. Uses only built-in primitives and materials.
"""
import sys
from pathlib import Path
import bpy
from mathutils import Vector


def create(output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new('World')
    scene.world.color = (0.25, 0.25, 0.25)
    for name, loc, dims, color in [
        ('Ground', (0, 0, -0.1), (8, 6, 0.2), (0.18, 0.35, 0.12, 1)),
        ('Barn', (0, 1, 0.75), (2, 1.6, 1.5), (0.50, 0.08, 0.05, 1)),
        ('FencePost_01', (-2.4, -1.7, 0.56), (0.12, 0.12, 1), (0.4, 0.23, 0.1, 1)),
        ('FencePost_02', (-0.8, -1.7, 0.54), (0.12, 0.12, 1), (0.4, 0.23, 0.1, 1)),
        ('FencePost_03', (0.8, -1.7, 0.58), (0.12, 0.12, 1), (0.4, 0.23, 0.1, 1)),
        ('FencePost_04', (2.4, -1.7, 0.55), (0.12, 0.12, 1), (0.4, 0.23, 0.1, 1)),
    ]:
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        obj = bpy.context.object
        obj.name = name
        obj.dimensions = dims
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        mat = bpy.data.materials.new(name + '_Material')
        mat.diffuse_color = color
        obj.data.materials.append(mat)
    bpy.ops.object.camera_add(location=(7, -10, 7))
    camera = bpy.context.object
    camera.name = 'Camera'
    camera.rotation_euler = (Vector((0, 0, 0.5)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 10
    scene.camera = camera
    bpy.ops.object.light_add(type='AREA', location=(0, -2, 7))
    bpy.context.object.name = 'KeyLight'
    bpy.context.object.data.energy = 1200
    bpy.context.object.data.shape = 'DISK'
    bpy.context.object.data.size = 5
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == '__main__':
    create(sys.argv[sys.argv.index('--') + 1])
