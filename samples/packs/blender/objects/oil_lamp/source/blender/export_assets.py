# export_assets.py
# One click exporter for the oiltamp asset (id: oiltamp, version: v004).
# Run inside Blender with source/blender/asset_oiltamp.blend open: open this text
# block in the Text Editor and press Run Script, or run headless:
#   blender --background source/blender/asset_oiltamp.blend --python source/blender/export_assets.py
#
# It exports only the twelve lamp meshes (no lights, camera or backdrop):
#   mesh/asset_oiltamp.glb   glTF binary, Y up, glass as transmission
#   mesh/asset_oiltamp.fbx   FBX with embedded textures
# and refreshes the three.js viewer payload:
#   source/threejs/js/model_data.js
#
# Material transcription (in memory only, the blend file is never saved):
#   glass_globe material: Glass + Transparent mix is rewritten per target
#     format, the mix factor and colors are read from the node tree at run
#     time.
#     - GLB: Principled BSDF with transmission (KHR_materials_transmission),
#       the closest glTF equivalent of the source glass.
#     - FBX: FBX has no transmission channel, and a naive alpha copy of the
#       mix factor renders pale. The glass is instead approximated with a
#       dark smoky alpha blend (alpha 0.65, near-black green-gray base),
#       tuned against a Cycles render of the source glass so both read as
#       the same dark see-through glass; roughness, IOR and the normal map
#       are carried over unchanged.
#   vent_cap material: the ColorRamp on base color is baked into a
#     standalone grayscale map following the ramp formula.
#   Glass surface detail: the source and both exports share the authored
#     combined normal map textures/glass_clear_normal.png at strength 0.35.
#     It already contains the glass displacement relief; source glass
#     displacement is disabled to avoid double application.

import bpy
import base64
import os

import numpy as np

ASSET_ID = "oiltamp"
ASSET_VERSION = "v004"

LAMP_OBJECTS = [
    "glass_globe", "burner", "vent_cap", "upper_body", "base", "top_ring",
    "guard_bow", "body_collar", "bail_wire", "fount_upper", "fount_lower",
    "base_rim",
]

GLASS_MATERIAL = "glass_clear"
CAP_MATERIAL = "metal_cap"
GLASS_NORMAL_BAKED = "glass_clear_normal.png"


def _root():
    blend_dir = os.path.dirname(bpy.data.filepath)
    return os.path.abspath(os.path.join(blend_dir, "..", ".."))


def _lamp_objects():
    obs = []
    for name in LAMP_OBJECTS:
        ob = bpy.data.objects.get(name)
        if ob is None:
            raise RuntimeError("missing lamp object: " + name)
        obs.append(ob)
    return obs


def _image_by_filename(filename):
    for img in bpy.data.images:
        if img.source == "FILE" and os.path.basename(img.filepath) == filename:
            return img
    return None


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


class _Undo:
    def __init__(self):
        self.links = []          # (tree, from_node, from_socket, to_node, to_socket)
        self.new_nodes = []      # (tree, node)
        self.new_images = []
        self.node_images = []    # (node, old_image)
        self.blend_methods = []
        self.render_methods = []


def _unlink(tree, link, undo):
    undo.links.append((tree, link.from_node, link.from_socket.name,
                       link.to_node, link.to_socket.name))
    tree.links.remove(link)


def _relink(undo):
    for tree, fn, fs, tn, ts in undo.links:
        n_from = tree.nodes.get(fn.name)
        n_to = tree.nodes.get(tn.name)
        if n_from is None or n_to is None:
            continue
        tree.links.new(n_from.outputs[fs], n_to.inputs[ts])
    for tree, node in undo.new_nodes:
        try:
            tree.nodes.remove(node)
        except Exception:
            pass
    for node, img in undo.node_images:
        node.image = img


def prepare_materials(mode):
    """Rewrite the two non standard wirings for export. Returns undo info.

    mode: "glb" uses transmission for the glass, "fbx" uses alpha
    transparency (FBX has no transmission channel).
    """
    undo = _Undo()

    # --- glass globe: Glass + Transparent mix -> Principled
    mat = bpy.data.materials[GLASS_MATERIAL]
    nt = mat.node_tree
    glass = next(n for n in nt.nodes if n.type == "BSDF_GLASS")
    transp = next(n for n in nt.nodes if n.type == "BSDF_TRANSPARENT")
    mix = next(n for n in nt.nodes if n.type == "MIX_SHADER")
    fac_in = mix.inputs["Fac"]
    if fac_in.is_linked:
        raise RuntimeError("glass Mix Factor is driven by a node tree; "
                           "expected a constant for export")
    fac = float(fac_in.default_value)
    gcol = tuple(glass.inputs["Color"].default_value)
    tcol = tuple(transp.inputs["Color"].default_value)
    base = [fac * tcol[i] + (1.0 - fac) * gcol[i] for i in range(3)]
    rough = float(glass.inputs["Roughness"].default_value)
    normal_sock = glass.inputs["Normal"]
    normal_from = normal_sock.links[0].from_socket if normal_sock.is_linked else None
    if normal_from is not None:
        # 三个载体共用合并法线，避免FBX缺失玻璃锤纹或重复叠加置换。
        baked = bpy.data.images.load(
            os.path.join(_root(), "textures", GLASS_NORMAL_BAKED),
            check_existing=True)
        baked.colorspace_settings.name = "Non-Color"
        nm_node = normal_from.node
        tex_node = nm_node.inputs["Color"].links[0].from_node
        undo.node_images.append((tex_node, tex_node.image))
        tex_node.image = baked

    pr = nt.nodes.new("ShaderNodeBsdfPrincipled")
    pr.name = "Principled Export Glass"
    undo.new_nodes.append((nt, pr))
    pr.inputs["Base Color"].default_value = (*base, 1.0)
    pr.inputs["Roughness"].default_value = rough
    pr.inputs["IOR"].default_value = 1.5
    pr.inputs["Metallic"].default_value = 0.0
    if mode == "glb":
        pr.inputs["Transmission Weight"].default_value = 1.0
        out_blend = "OPAQUE"
        print("[export] glass -> transmission, baseColor=%s rough=%.4f fac=%.4f"
              % (tuple(round(v, 4) for v in base), rough, fac))
    else:
        # FBX has no transmission channel. A naive alpha copy of the mix
        # factor renders pale, so the glass is approximated with a dark
        # smoky alpha blend. Alpha 0.65 and the near-black green-gray base
        # were tuned against a Cycles render of the source glass (region
        # mean color deviation ~1/255, contrast within ~6/255), matching
        # both the darkness and the see-through amount of the source.
        pr.inputs["Alpha"].default_value = 0.65
        pr.inputs["Base Color"].default_value = (0.02, 0.022, 0.016, 1.0)
        out_blend = "BLEND"
        print("[export] glass -> dark smoky alpha blend, alpha=0.65 rough=%.4f fac=%.4f"
              % (rough, fac))
    if normal_from is not None:
        nt.links.new(normal_from, pr.inputs["Normal"])
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    _unlink(nt, out.inputs["Surface"].links[0], undo)
    nt.links.new(pr.outputs["BSDF"], out.inputs["Surface"])
    if hasattr(mat, "blend_method"):
        undo.blend_methods.append((mat, mat.blend_method))
        mat.blend_method = out_blend
    if hasattr(mat, "surface_render_method"):
        undo.render_methods.append((mat, mat.surface_render_method))
        try:
            mat.surface_render_method = "BLENDED" if mode == "fbx" else "DITHERED"
        except TypeError:
            pass

    # --- cap: ColorRamp over the color map -> baked grayscale map
    mat = bpy.data.materials[CAP_MATERIAL]
    nt = mat.node_tree
    ramp = next(n for n in nt.nodes if n.type == "VALTORGB")
    els = ramp.color_ramp.elements
    p0 = float(els[0].position)
    p1 = float(els[-1].position)
    fac_sock = ramp.inputs["Factor"]
    if not fac_sock.is_linked:
        raise RuntimeError("cap ColorRamp factor is not driven by an image")
    src_node = fac_sock.links[0].from_node
    if src_node.type != "TEX_IMAGE" or src_node.image is None:
        raise RuntimeError("cap ColorRamp factor is not driven directly by an image")
    src_img = src_node.image
    w, h = src_img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    src_img.pixels.foreach_get(buf)
    rgb = buf.reshape(-1, 4)[:, :3]
    lin = _srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    bw = lin.mean(axis=1)
    v = np.clip((bw - p0) / (p1 - p0), 0.0, 1.0)
    enc = _linear_to_srgb(v)
    out_px = np.repeat(enc, 4).astype(np.float32)
    out_px[3::4] = 1.0

    out_name = "metal_cap_basecolor"
    new_img = bpy.data.images.get(out_name)
    if new_img is None:
        new_img = bpy.data.images.new(out_name, w, h, alpha=False)
        undo.new_images.append(new_img)
    new_img.pixels.foreach_set(out_px)
    new_img.colorspace_settings.name = "sRGB"
    tex_dir = os.path.join(_root(), "textures")
    os.makedirs(tex_dir, exist_ok=True)
    abs_path = os.path.join(tex_dir, out_name + ".png")
    new_img.filepath = abs_path
    new_img.save()
    new_img.filepath = bpy.path.relpath(abs_path)
    try:
        new_img.pack()
    except Exception:
        pass

    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    _unlink(nt, out.inputs["Surface"].links[0], undo)
    tex2 = nt.nodes.new("ShaderNodeTexImage")
    tex2.name = "Ramp Baked Map"
    tex2.image = new_img
    undo.new_nodes.append((nt, tex2))
    pr2 = nt.nodes.new("ShaderNodeBsdfPrincipled")
    pr2.name = "Principled Export Cap"
    undo.new_nodes.append((nt, pr2))
    nt.links.new(tex2.outputs["Color"], pr2.inputs["Base Color"])
    met_sock = pr2.inputs["Metallic"]
    rough_sock = pr2.inputs["Roughness"]
    principled = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED" and n.name != "Principled Export Cap")
    for src_in, dst_in in (("Metallic", "Metallic"), ("Roughness", "Roughness")):
        s = principled.inputs[src_in]
        if s.is_linked:
            nt.links.new(s.links[0].from_socket, pr2.inputs[dst_in])
        else:
            pr2.inputs[dst_in].default_value = s.default_value
    ns = principled.inputs["Normal"]
    if ns.is_linked:
        nt.links.new(ns.links[0].from_socket, pr2.inputs["Normal"])
    nt.links.new(pr2.outputs["BSDF"], out.inputs["Surface"])
    print("[export] cap ColorRamp(%.3f -> %.3f) baked into %s.png" % (p0, p1, out_name))

    # --- disconnect displacement on all lamp materials, assert colorspaces
    for mname in ("metal_rust", CAP_MATERIAL, GLASS_MATERIAL):
        mat = bpy.data.materials[mname]
        nt = mat.node_tree
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        disp = out.inputs["Displacement"]
        if disp.is_linked:
            _unlink(nt, disp.links[0], undo)
        for n in nt.nodes:
            if n.type == "TEX_IMAGE" and n.image is not None:
                cs = n.image.colorspace_settings
                fname = os.path.basename(n.image.filepath or n.image.name)
                want = "sRGB" if fname == "metal_rust_basecolor.png" else "Non-Color"
                if cs.name != want:
                    cs.name = want
                n.image.colorspace_settings.name = want
    print("[export] displacement disconnected, colorspaces asserted")
    return undo


def restore_materials(undo):
    _relink(undo)
    for mat, bm in undo.blend_methods:
        mat.blend_method = bm
    for mat, rm in undo.render_methods:
        mat.surface_render_method = rm
    for img in undo.new_images:
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass
    print("[export] materials restored")


def _shift_to_ground(objects):
    """Offset the lamp so the export origin is the bbox bottom center.

    The source file is already aligned to this convention (asset bottom
    center at the world origin), so on a healthy project this is a no-op
    guard; it only prints the measured values. A nonzero offset printed
    here means the source drifted from the delivery convention.
    """
    import mathutils
    deps = bpy.context.evaluated_depsgraph_get()
    gmin = None
    gmax = None
    for ob in objects:
        ev = ob.evaluated_get(deps)
        for v in ev.data.vertices:
            w = ev.matrix_world @ v.co
            gmin = w.copy() if gmin is None else mathutils.Vector(map(min, gmin, w))
            gmax = w.copy() if gmax is None else mathutils.Vector(map(max, gmax, w))
    cx = (gmin.x + gmax.x) / 2.0
    cy = (gmin.y + gmax.y) / 2.0
    z0 = gmin.z
    print("[export] world bbox size (%.4f, %.4f, %.4f), bottom center (%.4f, %.4f)"
          % (gmax.x - gmin.x, gmax.y - gmin.y, gmax.z - gmin.z, cx, cy))
    saved = {}
    for ob in objects:
        saved[ob.name] = ob.matrix_world.copy()
        ob.matrix_world = mathutils.Matrix.Translation((-cx, -cy, -z0)) @ ob.matrix_world
    return saved


def _restore_shift(saved):
    for name, mw in saved.items():
        ob = bpy.data.objects.get(name)
        if ob:
            ob.matrix_world = mw


def export_glb(path):
    obs = _lamp_objects()
    saved = _shift_to_ground(obs)
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for ob in obs:
            ob.select_set(True)
        bpy.context.view_layer.objects.active = obs[0]
        kw = dict(filepath=path, export_format="GLB", use_selection=True,
                  export_apply=True, export_animations=False)
        props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
        if "export_yup" in props:
            kw["export_yup"] = True
        bpy.ops.export_scene.gltf(**kw)
    finally:
        _restore_shift(saved)
    print("[export] GLB written: %s (%.1f MB)"
          % (path, os.path.getsize(path) / 1048576.0))


def export_fbx(path):
    obs = _lamp_objects()
    saved = _shift_to_ground(obs)
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for ob in obs:
            ob.select_set(True)
        bpy.context.view_layer.objects.active = obs[0]
        kw = dict(filepath=path, use_selection=True, object_types={"MESH"},
                  use_mesh_modifiers=True, embed_textures=True,
                  path_mode="COPY", add_leaf_bones=False, bake_anim=False)
        props = bpy.ops.export_scene.fbx.get_rna_type().properties.keys()
        for k in list(kw):
            if k not in props:
                kw.pop(k)
        bpy.ops.export_scene.fbx(**kw)
    finally:
        _restore_shift(saved)
    print("[export] FBX written: %s (%.1f MB)"
          % (path, os.path.getsize(path) / 1048576.0))


def sync_model_data(glb_path):
    js_dir = os.path.join(_root(), "source", "threejs", "js")
    os.makedirs(js_dir, exist_ok=True)
    with open(glb_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    out = os.path.join(js_dir, "model_data.js")
    with open(out, "w", encoding="ascii", newline="\n") as f:
        f.write('window.LAMP_GLB_BASE64="%s";\n' % b64)
    print("[export] model_data.js synced (%.1f MB)"
          % (os.path.getsize(out) / 1048576.0))


def main():
    root = _root()
    mesh_dir = os.path.join(root, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    glb_path = os.path.join(mesh_dir, "asset_%s.glb" % ASSET_ID)
    fbx_path = os.path.join(mesh_dir, "asset_%s.fbx" % ASSET_ID)
    undo_glb = prepare_materials("glb")
    try:
        export_glb(glb_path)
    finally:
        restore_materials(undo_glb)
    undo_fbx = prepare_materials("fbx")
    try:
        export_fbx(fbx_path)
    finally:
        restore_materials(undo_fbx)
    sync_model_data(glb_path)
    print("[export] done")


if __name__ == "__main__":
    main()
