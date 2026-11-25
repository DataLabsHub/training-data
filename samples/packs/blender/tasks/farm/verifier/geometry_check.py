"""Measured geometry validity and advisory mesh diagnostics; run in Blender."""
import math
import bpy
from mathutils import Vector
from general_contract import check
from support import POLICY

def geometry(scene):
    graph = bpy.context.evaluated_depsgraph_get()
    valid, invalid, unsupported, diagnostics = [], [], [], []
    evaluated_vertices = 0
    for instance in graph.object_instances:
        obj = instance.object
        if obj.hide_render or not obj.original.visible_get():
            continue
        if obj.type in ('CAMERA', 'LIGHT', 'EMPTY', 'ARMATURE', 'LATTICE', 'SPEAKER', 'LIGHT_PROBE'):
            continue
        if obj.type not in ('MESH', 'CURVE', 'SURFACE', 'FONT', 'META'):
            unsupported.append(obj.name + ':' + obj.type)
            continue
        matrix = instance.matrix_world
        if not all(math.isfinite(x) for row in matrix for x in row):
            invalid.append(obj.name + ':nonfinite_transform')
            continue
        mesh = None
        try:
            mesh = obj.to_mesh()
            if mesh is None:
                unsupported.append(obj.name + ':mesh_conversion_failed')
                continue
            evaluated_vertices += len(mesh.vertices)
            if evaluated_vertices > POLICY['max_evaluated_vertices']:
                invalid.append('evaluated_vertex_budget_exceeded')
                break
            points = [matrix @ v.co for v in mesh.vertices]
            if not all(math.isfinite(x) for p in points for x in p):
                invalid.append(obj.name + ':nonfinite_vertices')
                continue
            if not points or not mesh.polygons:
                continue
            valid.append(obj.name)
            lo = Vector([min(p[i] for p in points) for i in range(3)])
            hi = Vector([max(p[i] for p in points) for i in range(3)])
            epsilon = max((hi - lo).length * 1e-8, 1e-12)
            short = sum((points[e.vertices[0]] - points[e.vertices[1]]).length <= epsilon for e in mesh.edges)
            mesh.calc_loop_triangles()
            areas = [0.0] * len(mesh.polygons)
            for tri in mesh.loop_triangles:
                a, b, c = [points[i] for i in tri.vertices]
                areas[tri.polygon_index] += (b - a).cross(c - a).length / 2
            # Indexed duplicate faces: same vertex-index set, independent of winding.
            seen, duplicate = set(), 0
            for face in mesh.polygons:
                key = tuple(sorted(face.vertices))
                duplicate += key in seen
                seen.add(key)
            diagnostics.append({'object': obj.name, 'vertices': len(points), 'faces': len(mesh.polygons),
                                'length_epsilon_scene_units': epsilon,
                                'zero_length_edges': short, 'zero_area_faces': sum(a <= epsilon ** 2 for a in areas),
                                'duplicate_indexed_faces': duplicate})
        except Exception as exc:
            unsupported.append(obj.name + ':' + type(exc).__name__)
        finally:
            if mesh is not None:
                obj.to_mesh_clear()
    status = 'FAIL' if invalid else ('UNKNOWN' if unsupported else ('PASS' if valid else 'FAIL'))
    reason = 'nonfinite_geometry' if invalid else ('unsupported_geometry' if unsupported else ('nonempty_surface_geometry' if valid else 'no_nonempty_surface_geometry'))
    g = check('geometry_valid', status, reason, {'valid_objects': valid, 'invalid': invalid, 'unsupported': unsupported})
    has_defects = any(d[k] for d in diagnostics for k in ('zero_length_edges', 'zero_area_faces', 'duplicate_indexed_faces'))
    d = check('mesh_diagnostics', 'FAIL' if has_defects else ('UNKNOWN' if unsupported else ('PASS' if diagnostics else 'N_A')),
              'advisory_only_not_a_hard_gate', {'objects': diagnostics, 'unsupported': unsupported,
              'duplicate_scope': 'same_vertex_indices_only; coincident disconnected vertices are not merged',
              'open_boundaries_normals_intersections': 'not_scored'})
    return g, d

