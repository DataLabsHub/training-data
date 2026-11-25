"""Farm evidence tools. No quality inferred from object names or render success."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_views import execute, sha256, geometry, matches
from geometry_check import geometry as diagnose
from support import native_file_preflight, native_preflight


def simple(v):
    if isinstance(v,(str,int,float,bool)) or v is None:return v
    if isinstance(v,bpy.types.ID):return [v.bl_rna.identifier,v.name]
    try:return list(v)
    except TypeError:return str(v)


def nodes(tree):
    if tree is None:return None
    return {'nodes':[{'name':n.name,'type':n.bl_idname,
             'inputs':[[s.identifier,simple(s.default_value)] for s in n.inputs if hasattr(s,'default_value')],
             'group':getattr(getattr(n,'node_tree',None),'name',None)} for n in tree.nodes],
            'links':[[l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier] for l in tree.links]}


def snapshot():
    scene=bpy.context.scene
    graph=bpy.context.evaluated_depsgraph_get()
    boxes=geometry(scene,graph)
    objects=[]
    for o in sorted(scene.objects,key=lambda o:o.name):
        row={'name':o.name,'type':o.type,'matrix':[list(r) for r in o.matrix_world],
             'bounds':boxes.get(o.name),'materials':[s.material.name if s.material else None for s in o.material_slots],
             'hide_render':o.hide_render,'editable':bool(o.data and not o.data.library),
             'parent':o.parent.name if o.parent else None}
        if o.type=='MESH':
            payload={'vertices':[list(v.co) for v in o.data.vertices],
                     'faces':[list(p.vertices) for p in o.data.polygons]}
            row.update(vertices=len(o.data.vertices),faces=len(o.data.polygons),
                mesh_sha256=hashlib.sha256(json.dumps(payload,allow_nan=False).encode()).hexdigest())
        if o.type=='CAMERA':row['camera']={k:simple(getattr(o.data,k)) for k in ('type','lens','ortho_scale','shift_x','shift_y','clip_start','clip_end')}
        if o.type=='LIGHT':row['light']={'type':o.data.type,'energy':o.data.energy,'color':list(o.data.color),'nodes':nodes(o.data.node_tree)}
        objects.append(row)
    return {'objects':objects,'camera':scene.camera.name if scene.camera else None,
            'materials':[{'name':m.name,'color':list(m.diffuse_color),'roughness':m.roughness,'metallic':m.metallic,'nodes':nodes(m.node_tree)} for m in sorted(bpy.data.materials,key=lambda m:m.name)],
            'node_groups':[[g.name,nodes(g)] for g in sorted(bpy.data.node_groups,key=lambda g:g.name)],
            'world':{'color':list(scene.world.color),'nodes':nodes(scene.world.node_tree)} if scene.world else None}


def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False), encoding="utf-8")


def initial(source,out,resolution,samples):
    bpy.ops.wm.open_mainfile(filepath=str(source),use_scripts=False);native_preflight(bpy);bpy.context.view_layer.update()
    snap=snapshot();dump(out/'scene_inventory.json',snap)
    g,d=diagnose(bpy.context.scene);dump(out/'mesh_diagnostics.json',{'geometry':g,'diagnostics':d})
    # Temporary round trip; compare measured fields, not unmeasured Blender properties.
    temp=out/'roundtrip.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(temp),copy=True)
    bpy.ops.wm.open_mainfile(filepath=str(temp),use_scripts=False);native_preflight(bpy);bpy.context.view_layer.update()
    after=snapshot()
    dump(out/'roundtrip.json',{'status':'PASS' if snap==after else 'FAIL','scope':'mesh_vertices_faces_material_nodes_transforms_camera_lights_sampled_fields',
                              'before_sha256':hashlib.sha256(json.dumps(snap,sort_keys=True).encode()).hexdigest(),
                              'after_sha256':hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest()})
    temp.unlink()
    bpy.ops.wm.open_mainfile(filepath=str(source),use_scripts=False);native_preflight(bpy);bpy.context.view_layer.update()
    scene=bpy.context.scene
    if scene.camera:
        # Preserve submission camera and image aspect. Do not optimize the main shot.
        aspect=(scene.render.resolution_x*scene.render.pixel_aspect_x)/(scene.render.resolution_y*scene.render.pixel_aspect_y)
        width=max(16,min(resolution[0],round(resolution[1]*aspect)))
        height=max(16,min(resolution[1],round(width/aspect)))
        scene.render.resolution_x=width;scene.render.resolution_y=height;scene.render.resolution_percentage=100
        scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1
        scene.render.use_border=False;scene.render.use_crop_to_border=False
        scene.cycles.seed=0;scene.cycles.use_animated_seed=False
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples
        scene.render.image_settings.file_format='PNG';scene.render.filepath=str(out/'main.png')
        scene.render.use_sequencer=False
        # Direct camera evidence uses the submitted lighting/materials, without compositing.
        scene.render.use_compositing=False
        bpy.ops.render.render(write_still=True)
        dump(out/'main_camera.json',{'status':'PASS','camera':snap['camera'],'image':'main.png','aspect':aspect,
             'camera_parameters':next(o for o in snap['objects'] if o['name']==snap['camera']),
             'compositor_disabled':True,'image_sha256':sha256(out/'main.png')})
    else:dump(out/'main_camera.json',{'status':'UNKNOWN','reason':'submission_camera_missing'})
    names=[o['name'] for o in snap['objects'] if o['bounds']]
    aliases={'buildings':['*Barn*','*barn*','*Cottage*','*House*','*住宅*','*谷仓*'],
             'garden':['*Greenhouse*','*Plot*','*Garden*','*温室*','*菜*'],
             'livestock':['*Cow*','*Cattle*','*Chicken*','*Coop*','*牛*','*鸡*'],
             'vehicles':['*Tractor*','*Trailer*','*拖拉机*','*拖车*'],
             'landscape':['*Tree*','*Grass*','*Lake*','*树*','*湖*']}
    groups=[{'name':name,'objects':patterns} for name,patterns in aliases.items() if any(matches(n,patterns) for n in names)]
    req={'action':'render.views','args':{'mode':'auto','groups':groups,'resolution':resolution,'samples':samples,'max_closeups':4}}
    path=out/'auto_request.json';dump(path,req);execute(source,path,out/'color')
    req['args']['groups']=[];req['args']['max_closeups']=0;req['args']['gray_override']=True
    path=out/'gray_request.json';dump(path,req);execute(source,path,out/'gray')


def main():
    p=argparse.ArgumentParser();p.add_argument('--submission',required=True);p.add_argument('--output',required=True)
    p.add_argument('--request');p.add_argument('--resolution',nargs=2,type=int,default=[960,720]);p.add_argument('--samples',type=int,default=16)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);source=Path(a.submission).resolve();out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    native_file_preflight(source)
    before=sha256(source)
    if not a.request:initial(source,out,a.resolution,a.samples)
    else:
        request=json.loads(Path(a.request).read_text(encoding="utf-8"))
        if request['action']=='render.views':execute(source,a.request,out/'render')
        elif request['action']=='inspect.geometry':
            bpy.ops.wm.open_mainfile(filepath=str(source),use_scripts=False);native_preflight(bpy);bpy.context.view_layer.update()
            targets=request.get('targets',[]);snap=snapshot()
            rows=[o for o in snap['objects'] if not targets or o['name'] in targets]
            dump(out/'geometry.json',{'objects':rows,'missing_targets':sorted(set(targets)-{o['name'] for o in rows}),
                'scope':'bounds_counts_and_geometry_fingerprints_not_contact_certification'})
        else:raise ValueError('unsupported inspection action')
    if sha256(source)!=before:raise RuntimeError('artifact_changed')
    dump(out/'complete.json',{'status':'COMPLETE','artifact_hash':before,'blender_version':bpy.app.version_string})


if __name__=='__main__':main()
