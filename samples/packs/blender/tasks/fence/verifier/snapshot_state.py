"""Strict preservation snapshot for the small, fixed fence fixture.

This contract covers supported Blender RNA and all persistent datablock inventories;
unsupported dynamic features are rejected, not silently ignored.
"""
import hashlib
import json
import math
import bpy


class NonfiniteState(ValueError):
    pass


def value(v):
    if isinstance(v, float):
        if not math.isfinite(v):
            raise NonfiniteState('nonfinite_blender_state')
        return v
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    if isinstance(v, bpy.types.ID):
        return [v.bl_rna.identifier, v.name]
    if hasattr(v, 'to_dict'):
        return {k: value(x) for k, x in sorted(v.to_dict().items())}
    if isinstance(v, set):
        return sorted(value(x) for x in v)
    if isinstance(v, dict):
        return {k: value(x) for k, x in sorted(v.items())}
    try:
        return [value(x) for x in v]
    except TypeError:
        raise ValueError('unsupported_snapshot_value:' + type(v).__name__)


def custom(v):
    return {k: value(v[k]) for k in sorted(v.keys())} if hasattr(v, 'keys') else {}


def rna(v, skip=()):
    result = {}
    if v is None:
        return None
    for prop in v.bl_rna.properties:
        key = prop.identifier
        if key in skip or key == 'rna_type' or prop.is_readonly or prop.type == 'COLLECTION':
            continue
        obj = getattr(v, key)
        if prop.type == 'POINTER' and obj is not None and not isinstance(obj, bpy.types.ID):
            continue
        result[key] = value(obj)
    return result


def hash_value(v):
    return hashlib.sha256(json.dumps(value(v), sort_keys=True, allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def mesh_state(mesh):
    attrs = {}
    for attr in mesh.attributes:
        attrs[attr.name] = {'domain': attr.domain, 'type': attr.data_type,
                            'values': hash_value([rna(element) for element in attr.data])}
    return {'rna': rna(mesh), 'custom': custom(mesh), 'attributes': attrs,
            'uv': {layer.name: hash_value([list(item.uv) for item in layer.data]) for layer in mesh.uv_layers},
            'vertex_weights': hash_value([[(g.group, g.weight) for g in vertex.groups] for vertex in mesh.vertices]),
            'custom_normals': bool(mesh.has_custom_normals),
            'corner_normals': hash_value([list(n.vector) for n in mesh.corner_normals]),
            'vertices': hash_value([rna(x) for x in mesh.vertices]),
            'edges': hash_value([rna(x) for x in mesh.edges]),
            'polygons': hash_value([rna(x) for x in mesh.polygons])}


def snapshot(targets):
    scene = bpy.context.scene
    state = {'objects': {}, 'meshes': {}, 'ids': {}, 'settings': {}, 'custom': {}, 'collections': {}, 'node_trees': {}}
    prohibited = []
    ids = []
    for prop in bpy.data.bl_rna.properties:
        if prop.type != 'COLLECTION':
            continue
        collection = getattr(bpy.data, prop.identifier)
        values = [x for x in collection if isinstance(x, bpy.types.ID)]
        state['ids'][prop.identifier] = sorted((x.name, x.bl_rna.identifier, bool(x.use_fake_user)) for x in values)
        ids.extend(values)
    visited = set()
    for item in ids:
        key = item.bl_rna.identifier + ':' + item.name
        state['custom'][key] = custom(item)
        candidates = [item, getattr(item, 'node_tree', None), getattr(item, 'compositing_node_group', None)]
        for candidate in candidates:
            if candidate is None or candidate.as_pointer() in visited:
                continue
            visited.add(candidate.as_pointer())
            if getattr(candidate, 'animation_data', None) is not None:
                prohibited.append('animation_or_driver:' + key)
            if hasattr(candidate, 'nodes'):
                state['custom'][key + ':node_tree'] = custom(candidate)
                state['node_trees'][key] = {
                    'nodes': {node.name: {'rna': rna(node),
                        'inputs': [[s.identifier, value(s.default_value)] for s in node.inputs if hasattr(s, 'default_value')],
                        'outputs': [[s.identifier, value(s.default_value)] for s in node.outputs if hasattr(s, 'default_value')]}
                        for node in candidate.nodes},
                    'links': sorted([link.from_node.name, link.from_socket.identifier, link.to_node.name, link.to_socket.identifier]
                                    for link in candidate.links)}
                for node in candidate.nodes:
                    state['custom'][key + ':node:' + node.name] = custom(node)
    for obj in bpy.data.objects:
        loc = value(obj.location)
        if obj.name in targets:
            loc[2] = 'ALLOWED_WORLD_Z_TRANSLATION'
        state['objects'][obj.name] = {'rna': rna(obj, ('matrix_world', 'matrix_local', 'matrix_basis', 'location')),
            'location': loc, 'collections': sorted(c.name for c in obj.users_collection),
            'vertex_groups': [(g.name, g.index, g.lock_weight) for g in obj.vertex_groups],
            'custom': custom(obj), 'data': [obj.data.bl_rna.identifier, obj.data.name] if obj.data else None}
        if obj.type == 'CAMERA':
            state['objects'][obj.name]['camera'] = {'rna': rna(obj.data), 'dof': rna(obj.data.dof), 'stereo': rna(obj.data.stereo)}
        if obj.type == 'LIGHT':
            state['objects'][obj.name]['light'] = rna(obj.data)
    for mesh in bpy.data.meshes:
        state['meshes'][mesh.name] = mesh_state(mesh)
    for collection in bpy.data.collections:
        state['collections'][collection.name] = {'rna': rna(collection), 'custom': custom(collection),
            'objects': sorted(o.name for o in collection.objects), 'children': sorted(c.name for c in collection.children)}
    state['settings'] = {'scene': rna(scene), 'render': rna(scene.render), 'cycles': rna(scene.cycles),
        'image': rna(scene.render.image_settings), 'view': rna(scene.view_settings), 'units': rna(scene.unit_settings),
        'display': rna(scene.display_settings), 'world': rna(scene.world),
        'view_layers': {layer.name: rna(layer) for layer in scene.view_layers},
        'materials': {material.name: rna(material) for material in bpy.data.materials}}
    return state, prohibited
