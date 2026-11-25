"""Trusted extractor. Executed by Blender with auto-execution disabled."""
import hashlib
import json
import math
import sys
from pathlib import Path
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import native_file_preflight, InvalidSubmission, native_preflight, BudgetExceeded, POLICY
from snapshot_state import snapshot, NonfiniteState, value as finite_value
from constraint_check import TARGETS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def node_snapshot(tree):
    """Protect simple default node graphs too (Blender 5.2 always uses nodes)."""
    if tree is None:
        return None
    def value(v):
        return finite_value(v)
    nodes=[]
    for node in tree.nodes:
        properties={}
        for prop in node.bl_rna.properties:
            if prop.type in ('BOOLEAN', 'INT', 'FLOAT', 'STRING', 'ENUM') and prop.identifier != 'rna_type':
                properties[prop.identifier]=value(getattr(node, prop.identifier))
        nodes.append({'name':node.name,'type':node.bl_idname,'properties':properties,
                      'inputs':[[s.identifier,value(s.default_value)] for s in node.inputs if hasattr(s,'default_value')],
                      'outputs':[[s.identifier,value(s.default_value)] for s in node.outputs if hasattr(s,'default_value')]})
    return {'nodes':nodes,'links':[[l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier] for l in tree.links]}


def material(m):
    if m is None:
        return None
    return dict(name=m.name, color=list(m.diffuse_color), nodes=m.use_nodes,
                metallic=m.metallic, roughness=m.roughness,
                specular=m.specular_intensity, node_tree=node_snapshot(m.node_tree))


def layer_tree(layer):
    return [layer.name, layer.exclude, layer.hide_viewport,
            layer.collection.hide_render, layer.collection.hide_viewport,
            [layer_tree(child) for child in layer.children]]


def extract():
    native_preflight(bpy)
    protected_state, state_prohibited = snapshot(TARGETS)
    scene = bpy.context.scene
    objects = {}
    prohibited = list(state_prohibited)
    if bpy.data.libraries or bpy.data.images or bpy.data.texts or bpy.data.actions:
        prohibited.append('external_resources_images_texts_or_actions')
    if len(bpy.data.scenes) != 1 or len(scene.view_layers) != 1:
        prohibited.append('extra_scenes_or_view_layers')
    for obj in scene.objects:
        if obj.modifiers or obj.constraints or obj.parent or obj.animation_data:
            prohibited.append(obj.name + ':modifiers_constraints_parent_or_animation')
        geometry = None
        bounds = None
        data_properties = None
        if obj.type == 'MESH':
            mesh = obj.data
            if mesh.shape_keys:
                prohibited.append(obj.name + ':shape_keys')
            verts = [list(v.co) for v in mesh.vertices]
            world = [list(obj.matrix_world @ v.co) for v in mesh.vertices]
            if world:
                bounds = {'min': [min(v[i] for v in world) for i in range(3)],
                          'max': [max(v[i] for v in world) for i in range(3)]}
            geometry = digest({'v': verts, 'e': [list(e.vertices) for e in mesh.edges],
                'f': [[list(p.vertices), p.material_index, p.use_smooth] for p in mesh.polygons],
                'materials': [material(m) for m in mesh.materials]})
        elif obj.type == 'CAMERA':
            c = obj.data
            data_properties = [c.type, c.lens, c.ortho_scale, c.clip_start, c.clip_end,
                               c.shift_x, c.shift_y, c.dof.use_dof]
        elif obj.type == 'LIGHT':
            light = obj.data
            data_properties = [light.type, list(light.color), light.energy,
                               getattr(light, 'shape', ''), getattr(light, 'size', 0)]
        else:
            prohibited.append(obj.name + ':unsupported_object_type')
        objects[obj.name] = dict(type=obj.type, geometry=geometry, bounds=bounds,
            matrix=[list(row) for row in obj.matrix_world], data_properties=data_properties,
            visibility=[obj.hide_render, obj.hide_viewport, obj.hide_get(), obj.visible_get()])
    world = scene.world
    if world and world.animation_data:
        prohibited.append('world_animation')
    if any(m.animation_data for m in bpy.data.materials):
        prohibited.append('material_animation')
    if scene.animation_data:
        prohibited.append('scene_animation')
    # Nested groups are outside this deliberately small fixture contract.
    if bpy.data.node_groups:
        prohibited.append('node_groups')
    scene_properties = dict(camera=scene.camera.name if scene.camera else None,
        world_color=list(world.color) if world else None,
        world_nodes=node_snapshot(world.node_tree) if world else None,
        compositor_nodes=node_snapshot(getattr(scene, 'node_tree', None)),
        render=[scene.render.engine, scene.render.resolution_x, scene.render.resolution_y,
                scene.render.resolution_percentage, scene.cycles.samples],
        units=[scene.unit_settings.system, scene.unit_settings.scale_length],
        frame=scene.frame_current, layers=layer_tree(bpy.context.view_layer.layer_collection),
        color=[scene.view_settings.view_transform, scene.view_settings.look,
               scene.view_settings.exposure, scene.view_settings.gamma])
    return dict(schema_version=2, protected_state=protected_state, blender_version=bpy.app.version_string,
                objects=objects, scene=scene_properties, prohibited=prohibited)


if __name__ == '__main__':
    source, target = sys.argv[sys.argv.index('--') + 1:]
    try:
        native_file_preflight(source)
        bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()), use_scripts=False)
        native_preflight(bpy)
        bpy.context.view_layer.update()
        facts = extract()
        encoded = json.dumps(facts, ensure_ascii=False, allow_nan=False)
    except (NonfiniteState, BudgetExceeded, InvalidSubmission) as exc:
        encoded = json.dumps({'schema_version': 2, 'invalid_submission': str(exc)}, allow_nan=False)
    Path(target).write_text(encoded, encoding='utf-8')
