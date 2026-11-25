"""Editable procedural farm for rendering/integration acceptance, not a gold label.

All geometry and materials are generated locally. No photograph is placed in the
scene. This example is evaluator-owned and is never copied into the agent image.
"""
import math
import random
import bpy
from mathutils import Vector

R = random.Random(20261008)
M = {}


def material(name, color, roughness=.7, metallic=0, texture=0, scale=5):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bs = nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*color, 1)
    bs.inputs['Roughness'].default_value = roughness
    bs.inputs['Metallic'].default_value = metallic
    if texture:
        tex = nodes.new('ShaderNodeTexNoise'); tex.inputs['Scale'].default_value = scale
        tex.inputs['Detail'].default_value = 3
        coordinates = nodes.new('ShaderNodeNewGeometry')
        links.new(coordinates.outputs['Position'], tex.inputs['Vector'])
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position = .22
        ramp.color_ramp.elements[0].color = (*(v*.55 for v in color), 1)
        ramp.color_ramp.elements[1].position = .8
        ramp.color_ramp.elements[1].color = (*(min(1, v*1.3) for v in color), 1)
        links.new(tex.outputs['Fac'], ramp.inputs[0]); links.new(ramp.outputs[0], bs.inputs['Base Color'])
        bump = nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = texture
        bump.inputs['Distance'].default_value = .075
        links.new(tex.outputs['Fac'], bump.inputs['Height']); links.new(bump.outputs[0], bs.inputs['Normal'])
    M[name] = mat
    return mat


def finish(obj, name, mat):
    obj.name = name
    obj.data.materials.append(M[mat])
    return obj


def box(name, loc, dims, mat, bevel=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = finish(bpy.context.object, name, mat); obj.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = obj.modifiers.new('Soft edges', 'BEVEL'); mod.width = bevel; mod.segments = 2
    return obj


def ellipsoid(name, loc, scale, mat, segments=12):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=8, radius=1, location=loc)
    obj = finish(bpy.context.object, name, mat); obj.scale = scale
    for face in obj.data.polygons: face.use_smooth = True
    return obj


def cylinder(name, loc, radius, depth, mat, vertices=16):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc)
    obj = finish(bpy.context.object, name, mat)
    return obj


def beam(name, start, end, width, mat, rectangular=True):
    a, b = Vector(start), Vector(end)
    obj = box(name, (a+b)/2, (width, width, (b-a).length), mat) if rectangular else cylinder(name, (a+b)/2, width/2, (b-a).length, mat, 8)
    obj.rotation_euler = (b-a).to_track_quat('Z','Y').to_euler()
    return obj


def roof(name, x, y, width, length, eave, ridge, mat='roof', gambrel=False):
    profile = [(-width/2,eave), (-width*.32,ridge-.6), (0,ridge), (width*.32,ridge-.6), (width/2,eave)] if gambrel else [(-width/2,eave),(0,ridge),(width/2,eave)]
    for i, (a,b) in enumerate(zip(profile,profile[1:])):
        mid = ((a[0]+b[0])/2+x,y,(a[1]+b[1])/2)
        angle = -math.atan2(b[1]-a[1],b[0]-a[0])
        panel = box(name+'_panel',mid,(math.dist(a,b)+.09,length+.35,.13),mat)
        panel.rotation_euler.y = angle
        for j in range(int(length/.25)+1):
            yy=y-length/2+j*.25
            beam(name+'_seam',(x+a[0],yy,a[1]+.085),(x+b[0],yy,b[1]+.085),.024,'metal')
    for yy in (y-length/2-.06,y+length/2+.06):
        verts=[(x+a,yy,z) for a,z in profile]+[(x+width/2,yy,eave-.1),(x-width/2,yy,eave-.1)]
        mesh=bpy.data.meshes.new(name+'_gable_mesh');mesh.from_pydata(verts,[],[tuple(range(len(verts)))]);mesh.update()
        obj=bpy.data.objects.new(name+'_gable',mesh);bpy.context.collection.objects.link(obj);obj.data.materials.append(M['red' if gambrel else 'white'])
        solid=obj.modifiers.new('Solid gable','SOLIDIFY');solid.thickness=.1
        for a,b in zip(profile,profile[1:]):beam(name+'_fascia',(x+a[0],yy-.025,a[1]),(x+b[0],yy-.025,b[1]),.12,'white')


def window(name, x, y, z, w=.7, h=.9):
    box(name+'_glass',(x,y,z),(w,.065,h),'window')
    for xx in (x-w/2,x,x+w/2):box(name+'_frame',(xx,y-.05,z),(.06,.07,h+.12),'white')
    for zz in (z-h/2,z,z+h/2):box(name+'_frame',(x,y-.05,zz),(w+.12,.07,.06),'white')


def fence(name, a, b, white=False, rails=3):
    a,b=Vector(a),Vector(b);n=max(1,math.ceil((b-a).length/1.45));mat='white' if white else 'wood'
    for i in range(n+1):
        p=a.lerp(b,i/n);box(name+'_post',(p.x,p.y,.62),(.12,.12,1.24),mat,.02)
    for height in ([.35,.7,1.05] if rails==3 else [.4,.9]):
        beam(name+'_rail',(a.x,a.y,height),(b.x,b.y,height),.085,mat)


def enclosure(name, x, y, w, d, white=False):
    for a,b in [((x-w/2,y-d/2,0),(x+w/2,y-d/2,0)),((x+w/2,y-d/2,0),(x+w/2,y+d/2,0)),((x+w/2,y+d/2,0),(x-w/2,y+d/2,0)),((x-w/2,y+d/2,0),(x-w/2,y-d/2,0))]:fence(name,a,b,white)


def tree(name,x,y,h=6):
    beam(name+'_trunk',(x,y,0),(x+.1,y,h*.72),h*.09,'bark',False)
    for i in range(7):
        angle=i*2.4;dx=math.cos(angle)*h*.16;dy=math.sin(angle)*h*.16
        beam(name+'_branch',(x,y,h*.42),(x+dx,y+dy,h*.72),h*.035,'bark',False)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=1,location=(x+dx,y+dy,h*(.7+R.uniform(-.07,.15))))
        o=finish(bpy.context.object,name+'_crown','leaf'+str(i%3));o.scale=(h*.23,h*.22,h*.25)
        for v in o.data.vertices:v.co*=R.uniform(.88,1.12)
        for f in o.data.polygons:f.use_smooth=True


def cow(index,x,y,brown=False):
    name=f'Cow_{index:02d}';color='brown' if brown else 'white'
    ellipsoid(name+'_body',(x,y,1.15),(1.05,.42,.57),color)
    for xx in (-.66,.66):
        for yy in (-.27,.27):
            beam(name+'_leg',(x+xx,y+yy,.12),(x+xx*.9,y+yy,1.12),.14,color,False)
            box(name+'_hoof',(x+xx,y+yy,.1),(.19,.2,.2),'black',.025)
    beam(name+'_neck',(x+.7,y,1.24),(x+1.05,y,1.42),.36,color,False)
    ellipsoid(name+'_head',(x+1.1,y,1.5),(.37,.24,.35),color)
    ellipsoid(name+'_muzzle',(x+1.37,y,1.35),(.25,.23,.14),'pink')
    for yy in (-.3,.3):
        ellipsoid(name+'_ear',(x+1.05,y+yy,1.72),(.21,.16,.06),color)
        ellipsoid(name+'_eye',(x+1.25,y+yy*.72,1.6),(.045,.035,.045),'black')
    beam(name+'_tail',(x-1,y,1.38),(x-1.14,y,.64),.055,color,False)
    if not brown:
        for j in range(3):
            for side in (-1,1):ellipsoid(name+'_spot',(x-.55+j*.5,y+side*.397,1.2+R.uniform(-.08,.12)),(.26,.035,.3),'black')


def chicken(index,x,y):
    name=f'Chicken_{index:02d}';color='white' if index%3==0 else 'brown'
    ellipsoid(name+'_body',(x,y,.34),(.22,.16,.23),color)
    ellipsoid(name+'_head',(x+.15,y,.61),(.105,.095,.12),color)
    ellipsoid(name+'_comb',(x+.15,y,.74),(.07,.035,.045),'red')
    beam(name+'_beak',(x+.22,y,.61),(x+.34,y,.59),.06,'gold',False)
    for dy in (-.07,.07):beam(name+'_leg',(x,y+dy,.02),(x,y+dy,.22),.028,'gold',False)
    tail=ellipsoid(name+'_tail',(x-.23,y,.49),(.08,.08,.23),color);tail.rotation_euler.y=-.6


def hay(name,x,y,z=.43,round_bale=True):
    if round_bale:
        o=cylinder(name,(x,y,z),.43,.72,'hay',24);o.rotation_euler.x=math.pi/2
        for yy in (y-.29,y+.29):
            bpy.ops.mesh.primitive_torus_add(major_segments=20,minor_segments=5,location=(x,yy,z),major_radius=.433,minor_radius=.013,rotation=(math.pi/2,0,0))
            finish(bpy.context.object,name+'_twine','rope')
    else:box(name,(x,y,z),(1,.58,.55),'hay',.03)


def tractor(x,y):
    box('Tractor_chassis',(x,y,.6),(1.1,2.2,.3),'black',.05)
    box('Tractor_hood',(x,y-.5,1),(1,1.25,.6),'tractor_red',.08)
    box('Tractor_grille',(x,y-1.15,.92),(.85,.035,.4),'metal')
    for xx in (-.25,.25):ellipsoid('Tractor_headlight',(x+xx,y-1.19,1.22),(.1,.05,.1),'white')
    box('Tractor_seat',(x,y+.55,1.08),(.6,.55,.15),'black',.04)
    for yy,r in ((y+.65,.64),(y-.85,.4)):
        for side in (-1,1):
            xx=x+side*.72
            wheel=cylinder('Tractor_tire',(xx,yy,r),r,.3,'black',24);wheel.rotation_euler.y=math.pi/2
            hub=cylinder('Tractor_wheel_hub',(xx+side*.16,yy,r),r*.48,.04,'tractor_red',16);hub.rotation_euler.y=math.pi/2
            for j in range(12):
                a=j*math.tau/12
                tread=box('Tractor_tread',(xx,yy+math.sin(a)*(r-.01),r+math.cos(a)*(r-.01)),(.34,.12,.08),'black')
                tread.rotation_euler.x=-a
    for dx in (-.45,.45):
        for dy in (.2,.95):beam('Tractor_canopy_support',(x+dx,y+dy,1.1),(x+dx,y+dy,2.15),.055,'metal')
    box('Tractor_canopy',(x,y+.58,2.18),(1.2,1.2,.1),'tractor_red',.04)
    cylinder('Tractor_exhaust',(x+.3,y-.4,1.9),.055,1,'metal')
    ty=y+3.1
    beam('Trailer_hitch',(x,y+1,.4),(x,ty-1.2,.4),.12,'metal')
    box('Trailer_floor',(x,ty,.72),(1.7,2.4,.12),'wood')
    for side in (-1,1):
        for z in (1,1.25):box('Trailer_plank',(x+side*.82,ty,z),(.07,2.4,.2),'wood')
        wheel=cylinder('Trailer_tire',(x+side*.94,ty,.42),.42,.22,'black');wheel.rotation_euler.y=math.pi/2
    for j in range(3):hay('Trailer_hay',x,ty-.7+j*.65,.98,False)


def build(assets_dir, references_dir):
    R.seed(20261008);M.clear();bpy.ops.wm.read_factory_settings(use_empty=True)
    material('grass',(.24,.34,.075),texture=.35,scale=8)
    material('soil',(.24,.125,.055),texture=.5,scale=14)
    material('road',(.49,.37,.22),texture=.5,scale=20)
    material('red',(.42,.075,.045),texture=.3,scale=18)
    material('tractor_red',(.58,.035,.016),.32)
    material('white',(.83,.8,.66),texture=.12,scale=25)
    material('roof',(.19,.22,.22),.65,texture=.3,scale=25)
    material('wood',(.39,.27,.13),texture=.35,scale=15)
    material('bark',(.18,.105,.046),texture=.6,scale=9)
    for i,c in enumerate([(.17,.27,.035),(.29,.37,.055),(.1,.21,.034)]):material('leaf'+str(i),c,texture=.2,scale=6)
    material('hay',(.62,.43,.11),texture=.7,scale=35)
    material('rope',(.26,.2,.07))
    material('metal',(.36,.4,.39),.4,.6)
    material('black',(.025,.03,.026),.85)
    material('brown',(.3,.095,.025),texture=.2,scale=12)
    material('pink',(.47,.3,.22));material('gold',(.78,.4,.04))
    material('window',(.1,.23,.28),.15,.25)
    water=material('water',(.07,.23,.25),.12,.15,texture=.06,scale=5)
    glass=material('glass',(.48,.66,.56),.12)
    glass.node_tree.nodes.get('Principled BSDF').inputs['Transmission Weight'].default_value=.72
    box('Terrain',(0,8,-.24),(200,200,.45),'grass')
    box('Farm_access_road',(0,-3.5,.005),(3.5,25,.06),'road')
    box('Farm_courtyard',(3,0,.01),(15,8,.06),'road')
    road=box('Front_country_lane',(0,-16,.03),(160,2.7,.065),'road');road.rotation_euler.z=-.05
    # White cottage, clapboards, porch and two dormers.
    x,y=-7,2
    box('House_foundation',(x,y,.18),(4.6,4.6,.36),'metal')
    box('House_walls',(x,y,1.9),(4.5,4.5,3.5),'white')
    for z in [i*.17+.3 for i in range(20)]:
        box('House_clapboard',(x,y-2.26,z),(4.5,.025,.025),'wood')
    roof('House_roof',x,y,4.9,4.8,3.6,5.3)
    for xx in (-8.3,-5.7):
        window('House_front_window',xx,y-2.29,2.35,.75,1.1)
        box('House_dormer',(xx,y-.7,4.43),(1.1,1.3,1.2),'white')
        roof('House_dormer_roof',xx,y-.7,1.3,1.5,4.95,5.4)
        window('House_dormer_window',xx,y-1.36,4.48,.6,.8)
    box('House_door',(x,y-2.3,1.3),(.8,.07,2.2),'wood')
    box('House_porch',(x,y-3.1,.28),(5,1.8,.3),'wood')
    for xx in (x-2.3,x-1,x+1,x+2.3):beam('House_porch_post',(xx,y-3.75,.4),(xx,y-3.75,2.8),.12,'white')
    panel=box('House_porch_roof',(x,y-3.15,2.83),(5.2,2.2,.12),'roof');panel.rotation_euler.x=.14
    for xx in [x-2.35+i*.19 for i in range(26)]:
        if abs(xx-x)>.6:box('House_porch_baluster',(xx,y-3.85,.92),(.045,.045,.9),'white')
    for z in (.5,1.35):
        for dx in (-1.5,1.5):box('House_porch_rail',(x+dx,y-3.85,z),(1.7,.08,.075),'white')
    for i in range(3):box('House_step',(x,y-4.1-i*.25,.22-i*.065),(1.2,.3,.16),'wood')
    box('House_chimney',(x-.8,y+1,5),( .45,.6,1.65),'white')
    enclosure('House_picket',x,y-1,7.7,8.7,True)
    # Barn with a gambrel roof and separately editable doors and siding.
    x,y=3.2,5
    box('Barn_foundation',(x,y,.12),(6.4,6,.24),'metal')
    box('Barn_main',(x,y,2.25),(6.2,5.8,4.3),'red')
    roof('Barn_roof',x,y,6.7,6.25,4.35,6.6,gambrel=True)
    for xx in [x-3+i*.2 for i in range(31)]:box('Barn_vertical_siding',(xx,y-2.92,2.2),(.035,.05,4.2),'wood')
    for dx in (-.94,.94):
        xx=x+dx;yy=y-3
        box('Barn_door',(xx,yy,1.62),(1.85,.16,3.05),'red')
        for z in (.12,3.12):box('Barn_door_trim',(xx,yy-.1,z),(1.9,.065,.1),'white')
        for off in (-.88,.88):box('Barn_door_trim',(xx+off,yy-.1,1.62),(.1,.06,3.1),'white')
        beam('Barn_door_cross',(xx-.82,yy-.14,.2),(xx+.82,yy-.14,3),.075,'white')
        beam('Barn_door_cross',(xx+.82,yy-.14,.2),(xx-.82,yy-.14,3),.075,'white')
    window('Barn_loft',x,y-2.99,5.06,1,1.1)
    box('Barn_cupola',(x,y+.6,6.66),(.75,.8,.85),'white');roof('Barn_cupola_roof',x,y+.6,1,1,7,7.4)
    box('Barn_lean_to',(x+4.1,y,1.23),(2.2,5.5,2.35),'red')
    o=box('Barn_lean_to_roof',(x+4.1,y,2.61),(2.7,5.8,.13),'roof');o.rotation_euler.y=.2
    for yy in (y-1.4,y+.2):window('Barn_annex_window',x+4.1,yy-1.3,1.4,.5,.65)
    # Silo, roof ribs, ladder and open windmill tower.
    x,y=8.3,8
    cylinder('Silo_body',(x,y,3.3),1.15,6.6,'metal',40)
    ellipsoid('Silo_dome',(x,y,6.6),(1.16,1.16,.8),'metal',24)
    for z in [i*.58+.2 for i in range(12)]:
        bpy.ops.mesh.primitive_torus_add(major_segments=32,minor_segments=5,location=(x,y,z),major_radius=1.16,minor_radius=.025);finish(bpy.context.object,'Silo_band','roof')
    for dx in (-.2,.2):beam('Silo_ladder',(x+dx,y-1.19,.1),(x+dx,y-1.19,6.8),.045,'roof')
    for i in range(23):beam('Silo_rung',(x-.2,y-1.21,.25+i*.28),(x+.2,y-1.21,.25+i*.28),.035,'roof')
    wx,wy=12,9
    for sx in (-1,1):
        for sy in (-1,1):beam('Windmill_tower',(wx+sx*.8,wy+sy*.8,0),(wx+sx*.22,wy+sy*.22,7.6),.08,'metal')
    for z in range(1,7):
        r=.8-z*.076
        for side in (-1,1):beam('Windmill_brace',(wx-r,wy+side*r,z),(wx+r-.08,wy+side*(r-.08),z+1),.04,'metal')
    for i in range(12):
        a=i*math.tau/12
        beam('Windmill_spoke',(wx,wy-.32,7.7),(wx+math.sin(a)*1.15,wy-.32,7.7+math.cos(a)*1.15),.055,'metal')
        blade=box('Windmill_blade',(wx+math.sin(a)*.91,wy-.34,7.7+math.cos(a)*.91),(.33,.045,.65),'metal');blade.rotation_euler.y=a
    # Cow enclosure, shelter, exactly three cows, and water trough.
    enclosure('Cattle_fence',11,-2.5,10,7)
    box('Cattle_yard',(11,-2.5,.015),(9.8,6.8,.045),'soil')
    for xx in (7.4,11,14.6):
        for yy in (-.15,1):beam('Cattle_shed_post',(xx,yy,0),(xx,yy,2.6),.16,'wood')
    box('Cattle_shed_roof',(11,.4,2.65),(8,2.3,.16),'roof')
    for i in range(5):hay('Cattle_feed',8+i*1.25,.3,.43)
    for i,(xx,yy) in enumerate(((8.1,-3),(11.2,-2.7),(13.1,-4.4)),1):cow(i,xx,yy,i==3)
    cylinder('Cattle_water_trough',(10.5,-5.25,.35),.65,.7,'metal',28);cylinder('Cattle_trough_water',(10.5,-5.25,.705),.57,.015,'water',28)
    # Garden, vegetables and greenhouse framing.
    enclosure('Garden_fence',-5.4,-8.2,9,6)
    for row in range(5):
        yy=-10.4+row*1.03
        box('Garden_plot',(-6.4,yy,.1),(6.2,.72,.2),'soil')
        for j in range(14):
            xx=-9.1+j*.42
            for k in range(3):
                leaf=ellipsoid('Garden_leaf',(xx+R.uniform(-.1,.1),yy+R.uniform(-.1,.1),.22+k*.07),(.13,.1,.14),'leaf'+str(k))
            if row==3:ellipsoid('Garden_pumpkin',(xx,yy,.31),(.15,.15,.13),'gold')
    gx,gy=-2.3,-6.8
    box('Greenhouse_base',(gx,gy,.12),(2.7,3.1,.24),'wood')
    for xx in (gx-1.25,gx+1.25):
        box('Greenhouse_glass_side',(xx,gy,1.1),(.035,3,1.9),'glass')
        for j in range(5):beam('Greenhouse_frame',(xx,gy-1.5+j*.75,.2),(xx,gy-1.5+j*.75,2.05),.045,'metal')
    for j in range(5):
        yy=gy-1.5+j*.75
        beam('Greenhouse_roof_frame',(gx-1.25,yy,2.05),(gx,yy,2.9),.045,'metal')
        beam('Greenhouse_roof_frame',(gx,yy,2.9),(gx+1.25,yy,2.05),.045,'metal')
    for side in (-1,1):
        pane=box('Greenhouse_roof_glass',(gx+side*.625,gy,2.475),(1.52,3,.025),'glass');pane.rotation_euler.y=side*math.atan2(.85,1.25)
    beam('Greenhouse_ridge',(gx,gy-1.5,2.9),(gx,gy+1.5,2.9),.055,'metal')
    # Raised chicken coop with ramp and flock.
    enclosure('Chicken_fence',5,-10.5,6.5,5)
    box('Chicken_run',(5,-10.5,.015),(6.3,4.8,.05),'soil')
    cx,cy=5,-9.3
    for dx in (-.8,.8):
        for dy in (-.6,.6):beam('Coop_leg',(cx+dx,cy+dy,0),(cx+dx,cy+dy,.7),.12,'wood')
    box('Coop_house',(cx,cy,1.27),(2,1.6,1.5),'red');roof('Coop_roof',cx,cy,2.3,1.9,2,2.65)
    box('Coop_door',(cx-.35,cy-.82,1),(.55,.07,1),'wood');window('Coop_window',cx+.55,cy-.85,1.5,.45,.55)
    ramp=box('Coop_ramp',(cx-.35,cy-1.5,.3),(.65,1.65,.06),'wood');ramp.rotation_euler.x=.4
    for j in range(7):box('Coop_ramp_rung',(cx-.35,cy-2.2+j*.2,.025+j*.085),(.66,.055,.04),'white')
    for i in range(8):chicken(i,3.1+R.random()*3.6,-12.3+R.random()*1.6)
    tractor(.2,-1.4)
    for i in range(7):hay('Barn_hay',6.4+(i%3)*.9,1.7+(i//3)*.8,.43)
    for i in range(4):hay('Barn_stacked_bale',6.4+(i%2)*1.05,4+(i//2)*.65,.33,False)
    # Lake, irregular shore stones, dock, background trees and rolling terrain.
    ellipsoid('Lake_water',(-8,19,-.13),(18,6,.2),'water',48)
    for i in range(40):
        a=i*math.tau/40
        ellipsoid('Lake_shore_rock',(-8+math.cos(a)*18,19+math.sin(a)*6,.02),(R.uniform(.2,.55),R.uniform(.2,.4),R.uniform(.1,.3)),'metal',8)
    for j in range(14):box('Lake_dock_board',(-8,12+j*.27,.25),(1.6,.25,.12),'wood')
    for yy in (12.3,14,15.4):
        for xx in (-8.75,-7.25):beam('Lake_dock_post',(xx,yy,-.3),(xx,yy,.8),.09,'wood')
    for x,y,h in [(-13,4,6.8),(-14,-5,6),(-11,10,6),(0,12,7),(18,8,6),(-13,-12,5.5),(22,-1,6),(5,15,7)]:tree('Tree_foreground',x,y,h)
    for i in range(36):
        xx=R.uniform(-33,33);yy=R.uniform(28,47)
        tree('Tree_background',xx,yy,R.uniform(4,8))
    for i in range(8):
        ellipsoid('Hill',(-36+i*11,48+R.uniform(0,9),-.7),(R.uniform(8,15),R.uniform(8,13),R.uniform(2,5)),'grass',16)
    for yy in (11,-14):fence('Boundary',(-18,yy,0),(18,yy,0))
    # Grass clumps use a single mesh rather than thousands of objects.
    verts=[];faces=[]
    for _ in range(12000):
        x,y=R.uniform(-22,22),R.uniform(-19,26)
        if abs(x)<2 or (-11<x<-1 and -12<y<-4) or (0<x<17 and -7<y<12) or (-12<x<-2 and -4<y<8) or ((x+8)/18)**2+((y-19)/6)**2<1:continue
        h=R.uniform(.06,.2);w=.025;i=len(verts)
        verts.extend([(x-w,y,0),(x+w,y,0),(x+R.uniform(-.08,.08),y+.025,h)])
        faces.append((i,i+1,i+2))
    mesh=bpy.data.meshes.new('Grass_blades');mesh.from_pydata(verts,[],faces);mesh.update()
    obj=bpy.data.objects.new('Grass_clumps',mesh);bpy.context.collection.objects.link(obj);obj.data.materials.append(M['leaf1'])
    scene=bpy.context.scene
    scene.world=bpy.data.worlds.new('Daylight');scene.world.use_nodes=True
    bg=scene.world.node_tree.nodes.get('Background');bg.inputs[0].default_value=(.54,.7,1,1);bg.inputs[1].default_value=.38
    bpy.ops.object.light_add(type='SUN',location=(-20,-15,30));sun=bpy.context.object;sun.name='Sun_warm'
    sun.rotation_euler=(math.radians(28),math.radians(-24),math.radians(-35));sun.data.energy=2.7;sun.data.angle=.09;sun.data.color=(1,.83,.63)
    bpy.ops.object.camera_add(location=(28,-43,29));camera=bpy.context.object;camera.name='Camera_main'
    camera.rotation_euler=(Vector((0,3,1.4))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=43;scene.camera=camera
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=32;scene.cycles.use_denoising=True;scene.cycles.seed=20261008
    scene.render.threads_mode='FIXED';scene.render.threads=4
    scene.render.resolution_x=1440;scene.render.resolution_y=1080;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
    scene.view_settings.view_transform='AgX'
    scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    bpy.context.view_layer.update()
