# -*- coding: utf-8 -*-
"""Export the two subject meshes with their shared standard PBR materials.
Run in Blender Text Editor, or blender -b asset_jadevase.blend -P jadevase_export.py.
Exports FBX/GLB and synchronizes the embedded web copy. Locale independent.
"""
import bpy
from pathlib import Path
import base64

PARTS = ("ewer_body", "ewer_lid")
OUT_BASE = "asset_jadevase"

def _call(op, **kwargs):
    valid = set(op.get_rna_type().properties.keys())
    return op(**{k: v for k, v in kwargs.items() if k in valid})

def run():
    if not bpy.data.filepath:
        raise RuntimeError("Save the supplied Blender project before export")
    root = Path(bpy.data.filepath).resolve().parents[2]
    out = root / "mesh"
    out.mkdir(exist_ok=True)
    selected = list(bpy.context.selected_objects)
    active = bpy.context.view_layer.objects.active
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for name in PARTS:
            ob = bpy.data.objects[name]
            ob.select_set(True)
            for slot in ob.material_slots:
                if not slot.material: continue
                for n in slot.material.node_tree.nodes:
                    if n.type == "TEX_IMAGE" and n.image and min(n.image.size) < 2048:
                        raise RuntimeError("Missing or sub-2K texture: " + n.image.name)
        bpy.context.view_layer.objects.active = bpy.data.objects[PARTS[0]]
        _call(bpy.ops.export_scene.fbx, filepath=str(out / (OUT_BASE + ".fbx")),
              use_selection=True, object_types={"MESH"}, use_mesh_modifiers=True,
              mesh_smooth_type="FACE", use_tspace=True, bake_space_transform=True,
              path_mode="COPY", embed_textures=True, bake_anim=False)
        _call(bpy.ops.export_scene.gltf, filepath=str(out / (OUT_BASE + ".glb")),
              export_format="GLB", use_selection=True, export_apply=True,
              export_yup=True, export_animations=False, export_image_format="AUTO")
        payload = base64.b64encode((out / (OUT_BASE + ".glb")).read_bytes()).decode("ascii")
        (root / "source/threejs/js/model_data.js").write_text(
            '/* Same bytes as mesh/' + OUT_BASE + '.glb */\nwindow.JADEVASE_GLB_BASE64="' + payload + '";\n')
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for ob in selected: ob.select_set(True)
        bpy.context.view_layer.objects.active = active
    print("Exported FBX, GLB and synchronized web model")

if __name__ == "__main__":
    run()
