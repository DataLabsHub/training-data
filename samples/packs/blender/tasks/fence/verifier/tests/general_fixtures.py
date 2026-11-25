"""Generate trusted positive/negative native Blender fixtures for general checks."""
from pathlib import Path
import sys
import bpy
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'assets'))
from create_base import create
out = Path(sys.argv[sys.argv.index('--') + 1])
out.mkdir(parents=True, exist_ok=False)
create(out / 'baseline.blend')
for obj in bpy.context.scene.objects:
    if obj.name.startswith('FencePost_'):
        obj.location.z = 0.5
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'solved.blend'))
# Node material and a referenced packed image must be allowed by general checks.
mat = bpy.data.objects['Barn'].data.materials[0]
mat.use_nodes = True
image = bpy.data.images.new('used_image', width=2, height=2)
image.filepath_raw = str(out / 'texture.png'); image.file_format = 'PNG'; image.save()
image = bpy.data.images.load(str(out / 'texture.png'))
tex = mat.node_tree.nodes.new('ShaderNodeTexImage'); tex.image = image
shader = mat.node_tree.nodes.get('Principled BSDF')
mat.node_tree.links.new(tex.outputs['Color'], shader.inputs['Base Color'])
image.pack()
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'packed.blend'))
image.unpack(method='REMOVE')
image.filepath = str(out / 'confirmed_missing.png')
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'missing_resource.blend'))
mat.node_tree.nodes.remove(tex)
image.use_fake_user = True
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'unused_missing_resource.blend'))
# Indexed duplicate faces and a collinear face are diagnostic only.
mesh = bpy.data.meshes.new('DiagnosticMesh')
mesh.from_pydata([(0,0,0),(1,0,0),(2,0,0),(0,1,0)], [], [(0,1,3),(0,1,3),(0,1,2)])
obj = bpy.data.objects.new('DiagnosticMesh', mesh); bpy.context.scene.collection.objects.link(obj)
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'diagnostic.blend'))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'empty.blend'))
