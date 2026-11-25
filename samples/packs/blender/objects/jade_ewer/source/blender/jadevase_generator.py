# -*- coding: utf-8 -*-
# ============================================================
# 玉壶参数化生成器 / Jade Ewer Parametric Generator
# ------------------------------------------------------------
# 用法 / Usage:
#   1. 打开「asset_jadevase.blend」，在 Text Editor 中选中本脚本，点 Run Script
#   2. 按 N 键，侧边栏「玉壶」页签里有「玉壶生成器」面板
#   3. 拖动 6 个参数（单位：米），点「生成」；点「恢复默认」回到基准
#   4. 需要每次开文件自动出现面板：勾选 Text Editor 里的 Register
#
# 参数全部为真实世界尺寸（米），由工程内实测几何换算：
#   1 建模单位 = 1 m（交付版已应用 0.1 倍缩放，通高 0.3106 米）
#   ┌────────────┬──────────────┬──────────┬────────────┬──────────┐
#   │ 参数        │ 实测(单位)    │ 基准(米)  │ 可用范围(米) │ 实物参照  │
#   ├────────────┼──────────────┼──────────┼────────────┼──────────┤
#   │ 器身高度     │ 2.6361       │ 0.2636   │ 0.19–0.34  │ 底至口沿  │
#   │ 腹径        │ 1.7778       │ 0.1778   │ 0.13–0.23  │ 腹部最宽  │
#   │ 足径        │ 1.0360       │ 0.1036   │ 0.075–0.135│ 圈足外径  │
#   │ 盖高        │ 0.2170       │ 0.0217   │ 0.015–0.030│ 瓷盖穹高  │
#   │ 钮径        │ 0.2208       │ 0.0221   │ 0.014–0.032│ 金钮直径  │
#   │ 壁厚        │ 0.0300       │ 0.0030   │ 0.0015–0.0045│ 瓷胎厚 │
#   └────────────┴──────────────┴──────────┴────────────┴──────────┘
#   通高（派生）= 器身高度 + 盖高 + 钮露出高度 ≈ 0.311 m
#
# 原理 / How it works:
#   - 首次运行把壶身/壶盖的基准矩阵与**全部基准顶点世界坐标**存入文本
#     数据块 jadevase_generator_base.json
#   - 每次生成：先把顶点整体还原到基准，再按参数在**世界空间**做变形
#     （壶身：XY 等比 / Z 向拉伸 / 足部径向；壶盖：随口沿 XY、Z 向穹高；
#      金钮：以钮底为轴均匀缩放并随盖顶抬升；瓷盖与金钮共用的接缝环
#      半径随金钮、高度随穹顶，保证钮径参数在全范围准确且网格不开缝；
#      盖顶护圈逐环兜底，极端组合下钮座不挤出折叠）
#   - 腹径与足径互不干扰：底部 1.5 cm 内按足径比例，向上平滑过渡到腹径
#     比例，因此改腹径不会把圈足一起拉粗，改足径也只动底部
#   - 顶点级变形而非物体缩放：Solidify 壁厚保持各向同性，UV 与材质槽
#     完全不动；每次从基准重算，反复点击不累积误差
#   - 壶盖随口沿等比缩放，保证任何腹径下盖子都盖得住
# ============================================================

import bpy
import json
import math
from mathutils import Vector, Matrix

# ---------- 对象名称映射（改名后只改这里） ----------
BODY = "ewer_body"         # 壶身（含流与把）
LID = "ewer_lid"           # 壶盖（含金钮），父级为壶身
GOLD_MAT = "gold"          # 金钮材质，用于识别钮顶点

UNIT_TO_M = 1.0            # 建模单位 -> 米（交付版已应用 0.1 倍：1 单位 = 1 米）
FOOT_BAND = 0.015          # 足部形变带高度（米）= 1.5 cm
FOOT_EPS = 0.002           # 足径测量带（米）= 2 mm

SNAP_NAME = "jadevase_generator_base.json"
SNAP_VERSION = 2           # v2：交付基准（米制、正面 +Y、原点在包围盒底部中心）

PARAM_KEYS = ("body_height", "belly_diameter", "foot_diameter",
              "lid_height", "finial_diameter", "wall_thickness")

PARAM_LABELS = {
    "body_height": "器身高度",
    "belly_diameter": "腹径",
    "foot_diameter": "足径",
    "lid_height": "盖高",
    "finial_diameter": "钮径",
    "wall_thickness": "壁厚",
}

PARAM_DESC = {
    "body_height": "壶底至口沿的高度（拉伸壶身，壶盖随之抬升）",
    "belly_diameter": "腹部最宽处直径（流与把同步外扩，壶盖随口沿等比跟随）",
    "foot_diameter": "圈足外径（仅影响底部 1.5 cm 形变带）",
    "lid_height": "瓷盖穹顶高度，不含金钮",
    "finial_diameter": "盖顶金钮直径（以钮底为轴等比缩放）",
    "wall_thickness": "瓷胎壁厚（Solidify 修改器，壶身与壶盖同步）",
}

PARAM_RANGE = {          # 米，按真实执壶工艺给范围
    "body_height": (0.190, 0.340),
    "belly_diameter": (0.130, 0.230),
    "foot_diameter": (0.075, 0.135),
    "lid_height": (0.015, 0.030),
    "finial_diameter": (0.014, 0.032),
    "wall_thickness": (0.0015, 0.0045),
}

_last_clamped = []


# ---------- 基础工具 ----------

def _obj(name):
    return bpy.data.objects.get(name)


def _check_parts():
    miss = [n for n in (BODY, LID) if _obj(n) is None]
    return (False, "错误: 找不到对象 %s（可能被改名，请更新脚本顶部名称映射）"
            % ", ".join(miss)) if miss else (True, "")


def _mat_to_list(m):
    return [[float(c) for c in row] for row in m]


def _text_write(t, s):
    """兼容写入文本: Blender <=4.x 用 body, 5.x 用 clear()+write()"""
    if hasattr(t, "body"):
        t.body = s
    else:
        t.clear()
        t.write(s)


def _world_pts(o):
    mw = o.matrix_world
    return [mw @ v.co for v in o.data.vertices]


def _mat_vert_groups(o, gold_name):
    """按材质槽把壶盖顶点分成三组（返回字典，索引列表）：
       porcelain  瓷盖全部顶点（含接缝环）
       finial     金钮全部顶点（含接缝环）
       joint      接缝环：同时被瓷盖与金钮多边形引用的顶点
    """
    slots = [s.material.name if s.material else "" for s in o.material_slots]
    gi = slots.index(gold_name) if gold_name in slots else (len(slots) - 1)
    por, fin = set(), set()
    for p in o.data.polygons:
        (fin if p.material_index == gi else por).update(p.vertices)
    joint = por & fin
    return {"porcelain": sorted(por), "finial": sorted(fin), "joint": sorted(joint)}


def _lid_groups(snap):
    """从快照取壶盖三组顶点；旧快照无 joint 字段时按交集补算（向后兼容）"""
    g = snap["parts"]["lid"]["groups"]
    por, fin = set(g["porcelain"]), set(g["finial"])
    joint = set(g.get("joint")) if g.get("joint") is not None else (por & fin)
    return por, fin, joint


def _lid_collar(snap, por_only, joint, sxy, sf):
    """盖顶护圈：接缝环半径随金钮缩放后，若宽于下方瓷环（如钮径最大+
    腹径最小时），钮座会挤出折叠。逐环把 XY 比例抬到「不小于上一环」，
    形成一小段直壁钮座；自然宽于护圈后即恢复 sxy。返回 {顶点索引: 比例}。
    基准态与常见组合下护圈不触发（返回空表）。"""
    flat = snap["parts"]["lid"]["verts"]
    ax, ay = snap["geo"]["axis"]

    def rad(i):
        return math.hypot(flat[3 * i] - ax, flat[3 * i + 1] - ay)

    prev_r = max(rad(i) for i in joint) * sf     # 锚点：接缝环目标半径
    lv = {}
    for i in por_only:
        lv.setdefault(round(flat[3 * i + 2], 6), []).append(i)

    collar = {}
    for z in sorted(lv, reverse=True):
        r = max(rad(i) for i in lv[z])
        if r * sxy >= prev_r - 1e-9:             # 已自然宽于护圈，更深的环只会更宽
            break
        s = prev_r / r
        for i in lv[z]:
            collar[i] = s
        prev_r = r * s
    return collar


# ---------- 基准快照 ----------

def _build_snapshot():
    ok, msg = _check_parts()
    if not ok:
        return None
    body, lid = _obj(BODY), _obj(LID)
    groups = _mat_vert_groups(lid, GOLD_MAT)

    snap = {"version": SNAP_VERSION, "unit_to_m": UNIT_TO_M, "parts": {}}
    for key, o in (("body", body), ("lid", lid)):
        pts = _world_pts(o)
        snap["parts"][key] = {
            "name": o.name,
            "matrix": _mat_to_list(o.matrix_world),
            "verts": [round(c, 6) for p in pts for c in p],
        }
    snap["parts"]["lid"]["groups"] = groups

    bp = [Vector(v) for v in _chunks(snap["parts"]["body"]["verts"])]
    lp = [Vector(v) for v in _chunks(snap["parts"]["lid"]["verts"])]
    lpor = [lp[i] for i in groups["porcelain"]]
    lfin = [lp[i] for i in groups["finial"]]

    bz0 = min(p.z for p in bp)
    bz1 = max(p.z for p in bp)
    lz0 = min(p.z for p in lp)
    pz1 = max(p.z for p in lpor)
    fz0 = min(p.z for p in lfin)

    # 竖直轴取口沿（瓷盖）中心：壶盖与壶身共用同一轴，缩放后仍同心
    ax = (min(p.x for p in lpor) + max(p.x for p in lpor)) / 2.0
    ay = (min(p.y for p in lpor) + max(p.y for p in lpor)) / 2.0

    def dia(pts, axis):
        return max(getattr(p, axis) for p in pts) - min(getattr(p, axis) for p in pts)

    geo = {
        "axis": [ax, ay],
        "body_z0": bz0, "body_z1": bz1,
        "belly_dia_u": dia(bp, "x"),           # 正面(+Y)为流向，流/把沿 Y，X 向即纯腹径
        "foot_dia_u": dia([p for p in bp if p.z < bz0 + FOOT_EPS], "x"),
        "lid_z0": lz0, "lid_pz1": pz1,
        "lid_dia_u": dia(lpor, "x"),
        "finial_z0": fz0,
        "finial_dia_u": (dia(lfin, "x") + dia(lfin, "y")) / 2.0,
        "wall_u": _solidify_thickness(lid) or 0.03,
    }
    snap["geo"] = geo
    snap["base_m"] = {
        "body_height": (bz1 - bz0) * UNIT_TO_M,
        "belly_diameter": geo["belly_dia_u"] * UNIT_TO_M,
        "foot_diameter": geo["foot_dia_u"] * UNIT_TO_M,
        "lid_height": (pz1 - lz0) * UNIT_TO_M,
        "finial_diameter": geo["finial_dia_u"] * UNIT_TO_M,
        "wall_thickness": geo["wall_u"] * UNIT_TO_M,
    }
    return snap


def _chunks(flat):
    for i in range(0, len(flat), 3):
        yield flat[i], flat[i + 1], flat[i + 2]


def _solidify_thickness(o):
    for m in o.modifiers:
        if m.type == 'SOLIDIFY':
            return m.thickness
    return None


def ensure_snapshot():
    """读取基准快照；不存在/版本不符/顶点数变化则重建"""
    t = bpy.data.texts.get(SNAP_NAME)
    if t and t.as_string().strip():
        try:
            snap = json.loads(t.as_string())
            if snap.get("version") == SNAP_VERSION:
                ok = all(len(snap["parts"][k]["verts"]) // 3 ==
                         len(_obj(snap["parts"][k]["name"]).data.vertices)
                         for k in ("body", "lid"))
                if ok:
                    return snap
        except Exception:
            pass

    snap = _build_snapshot()
    if snap is None:
        return None
    t = bpy.data.texts.get(SNAP_NAME) or bpy.data.texts.new(SNAP_NAME)
    _text_write(t, json.dumps(snap, indent=1))
    return snap


# ---------- 变形核心 ----------

def _restore(key, snap):
    rec = snap["parts"][key]
    o = _obj(rec["name"])
    o.matrix_world = Matrix(rec["matrix"])
    bpy.context.view_layer.update()
    mw = o.matrix_world
    inv = mw.inverted_safe()
    flat = rec["verts"]
    for i, v in enumerate(o.data.vertices):
        v.co = inv @ Vector((flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]))
    o.data.update()


def _deform(key, snap, fn):
    """按基准世界坐标重写顶点：fn(world_vec, index) -> world_vec"""
    rec = snap["parts"][key]
    o = _obj(rec["name"])
    mw = Matrix(rec["matrix"])
    inv = mw.inverted_safe()
    flat = rec["verts"]
    for i, v in enumerate(o.data.vertices):
        w = Vector((flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]))
        v.co = inv @ fn(w, i)
    o.data.update()


def _smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def regenerate(params):
    """按参数重建玉壶。params: 6 个参数名 -> 米。返回 (成功, 消息)"""
    global _last_clamped
    _last_clamped = []

    ok, msg = _check_parts()
    if not ok:
        return False, msg
    snap = ensure_snapshot()
    if snap is None:
        return False, "错误: 无法建立基准快照"

    base, geo = snap["base_m"], snap["geo"]
    ax, ay = geo["axis"]

    # 1. 裁剪到工艺可用范围
    val = {}
    for k in PARAM_KEYS:
        v = float(params[k])
        lo, hi = PARAM_RANGE[k]
        vc = max(lo, min(hi, v))
        if abs(vc - v) > 1e-9:
            _last_clamped.append("%s %.4f→%.4f" % (PARAM_LABELS[k], v, vc))
        val[k] = vc

    sxy = val["belly_diameter"] / base["belly_diameter"]
    sz = val["body_height"] / base["body_height"]
    sfoot = val["foot_diameter"] / base["foot_diameter"]
    lz = val["lid_height"] / base["lid_height"]
    sf = val["finial_diameter"] / base["finial_diameter"]

    # 2. 全部还原到基准（幂等：不累积）
    for key in ("body", "lid"):
        _restore(key, snap)

    # 3. 壶身：腹径 XY 等比 + 器身高度 Z 拉伸 + 圈足径向（底部形变带）
    bz0, bz1 = geo["body_z0"], geo["body_z1"]
    zb, zt = bz0, bz0 + FOOT_BAND

    def body_fn(w, i):
        # 底部 1.5 cm 由足径取值，向上平滑过渡到腹径比例（两者互不干扰）
        t = _smoothstep((w.z - zb) / (zt - zb))
        s = sfoot + (sxy - sfoot) * t
        return Vector((ax + (w.x - ax) * s,
                       ay + (w.y - ay) * s,
                       bz0 + (w.z - bz0) * sz))

    _deform("body", snap, body_fn)
    body_top_new = bz0 + (bz1 - bz0) * sz
    dz = body_top_new - bz1           # 壶盖随口沿抬升量

    # 4. 壶盖：XY 随口沿等比（保证盖得住），瓷盖 Z 向外穹高，金钮等比放大
    lz0, pz1, fz0 = geo["lid_z0"], geo["lid_pz1"], geo["finial_z0"]
    por, fin, joint = _lid_groups(snap)
    por_top_new = (lz0 + dz) + (pz1 - lz0) * lz
    collar = _lid_collar(snap, por - joint, joint, sxy, sf)

    def lid_fn(w, i):
        if i in joint:
            # 接缝环（瓷盖与金钮共用）：半径随金钮 sf 缩放，否则钮径参数会被
            # 这圈钉死；高度随瓷盖穹顶走（环位于盖顶平面 fz0=pz1，
            # 两种映射在该平面重合，网格保持焊死不开缝）
            x = ax + (w.x - ax) * sf
            y = ay + (w.y - ay) * sf
            z = (lz0 + dz) + (w.z - lz0) * lz
        elif i in fin:
            # 金钮：随盖顶抬升，以钮底为轴等比
            x = ax + (w.x - ax) * sf
            y = ay + (w.y - ay) * sf
            z = por_top_new + (w.z - fz0) * sf
        else:
            # 瓷盖：随口沿 XY（护圈环按需抬升防折叠），Z 向穹高
            s = collar.get(i, sxy)
            x = ax + (w.x - ax) * s
            y = ay + (w.y - ay) * s
            z = (lz0 + dz) + (w.z - lz0) * lz
        return Vector((x, y, z))

    _deform("lid", snap, lid_fn)

    # 5. 壁厚：Solidify 同步（本地单位，物体缩放为 1 时即世界厚度）
    th = val["wall_thickness"] / UNIT_TO_M
    for key in ("body", "lid"):
        o = _obj(snap["parts"][key]["name"])
        for m in o.modifiers:
            if m.type == 'SOLIDIFY':
                m.thickness = th

    bpy.context.view_layer.update()

    m = measure()
    msg = "生成完成: 通高 %.1f mm · 腹径 %.1f mm · 盖径 %.1f mm · 壁厚 %.2f mm" % (
        m["total_height"] * 1000, m["belly_diameter"] * 1000,
        m["lid_diameter"] * 1000, m["wall_thickness"] * 1000)
    if _last_clamped:
        msg += " | 超范围已裁剪: " + "; ".join(_last_clamped)
    return True, msg


def measure():
    """当前尺寸（米），直接读网格顶点（不含修改器），用于面板显示与验证"""
    body, lid = _obj(BODY), _obj(LID)
    if not body or not lid:
        return {}
    bp = _world_pts(body)
    lp = _world_pts(lid)
    groups = _mat_vert_groups(lid, GOLD_MAT)
    lpor = [lp[i] for i in groups["porcelain"]]
    lfin = [lp[i] for i in groups["finial"]]
    bz0 = min(p.z for p in bp)

    def dia(pts, axis):
        return max(getattr(p, axis) for p in pts) - min(getattr(p, axis) for p in pts)

    return {
        "body_height": dia(bp, "z") * UNIT_TO_M,
        "belly_diameter": dia(bp, "x") * UNIT_TO_M,
        "foot_diameter": dia([p for p in bp if p.z < bz0 + FOOT_EPS], "x") * UNIT_TO_M,
        "lid_height": (max(p.z for p in lpor) - min(p.z for p in lpor)) * UNIT_TO_M,
        "lid_diameter": dia(lpor, "x") * UNIT_TO_M,
        "finial_diameter": (dia(lfin, "x") + dia(lfin, "y")) / 2.0 * UNIT_TO_M,
        "total_height": (max(p.z for p in lp) - bz0) * UNIT_TO_M,
        "wall_thickness": (_solidify_thickness(lid) or 0.0) * UNIT_TO_M,
    }


# ---------- UI ----------

class JadeVaseProps(bpy.types.PropertyGroup):
    body_height: bpy.props.FloatProperty(
        name="器身高度 (m)", description=PARAM_DESC["body_height"],
        subtype='DISTANCE', min=0.19, max=0.34, precision=4, default=0.2636)
    belly_diameter: bpy.props.FloatProperty(
        name="腹径 (m)", description=PARAM_DESC["belly_diameter"],
        subtype='DISTANCE', min=0.13, max=0.23, precision=4, default=0.1778)
    foot_diameter: bpy.props.FloatProperty(
        name="足径 (m)", description=PARAM_DESC["foot_diameter"],
        subtype='DISTANCE', min=0.075, max=0.135, precision=4, default=0.1036)
    lid_height: bpy.props.FloatProperty(
        name="盖高 (m)", description=PARAM_DESC["lid_height"],
        subtype='DISTANCE', min=0.015, max=0.030, precision=4, default=0.0217)
    finial_diameter: bpy.props.FloatProperty(
        name="钮径 (m)", description=PARAM_DESC["finial_diameter"],
        subtype='DISTANCE', min=0.014, max=0.032, precision=4, default=0.0221)
    wall_thickness: bpy.props.FloatProperty(
        name="壁厚 (m)", description=PARAM_DESC["wall_thickness"],
        subtype='DISTANCE', min=0.0015, max=0.0045, precision=4, default=0.0030)


def _props():
    return bpy.context.scene.jadevase_gen


def _params_from_props():
    p = _props()
    return {k: getattr(p, k) for k in PARAM_KEYS}


class JADEVASE_OT_generate(bpy.types.Operator):
    bl_idname = "jadevase.generate"
    bl_label = "生成"
    bl_description = "按当前参数重建玉壶（每次从基准重算，结果可复现）"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ok, msg = regenerate(_params_from_props())
        self.report({'INFO' if ok else 'ERROR'}, msg)
        return {'FINISHED'}


class JADEVASE_OT_reset(bpy.types.Operator):
    bl_idname = "jadevase.reset"
    bl_label = "恢复默认"
    bl_description = "回到实测基准尺寸并重新生成"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        snap = ensure_snapshot()
        if snap is None:
            self.report({'ERROR'}, "无法建立基准快照")
            return {'CANCELLED'}
        p = _props()
        for k in PARAM_KEYS:
            setattr(p, k, snap["base_m"][k])
        regenerate({k: snap["base_m"][k] for k in PARAM_KEYS})
        self.report({'INFO'}, "已恢复实测基准尺寸")
        return {'FINISHED'}


class VIEW3D_PT_jadevase_gen(bpy.types.Panel):
    bl_label = "玉壶生成器"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "玉壶"

    def draw(self, context):
        lay = self.layout
        p = _props()
        for k in PARAM_KEYS:
            lay.prop(p, k)
        row = lay.row(align=True)
        row.operator("jadevase.generate", icon='FILE_REFRESH')
        row.operator("jadevase.reset", icon='LOOP_BACK')

        snap = ensure_snapshot()
        if not snap:
            lay.label(text="找不到壶身/壶盖对象", icon='ERROR')
            return
        box = lay.box()
        box.label(text="实测基准（米）", icon='INFO')
        for k in PARAM_KEYS:
            box.label(text="%s  %.4f" % (PARAM_LABELS[k], snap["base_m"][k]))
        lay.label(text="1 建模单位 = %.1f m" % UNIT_TO_M)

        m = measure()
        if m:
            box = lay.box()
            box.label(text="当前尺寸（米）", icon='DRIVER')
            box.label(text="通高      %.4f" % m["total_height"])
            box.label(text="腹径      %.4f" % m["belly_diameter"])
            box.label(text="足径      %.4f" % m["foot_diameter"])
            box.label(text="盖径      %.4f" % m["lid_diameter"])
            box.label(text="高径比    %.2f" % (m["total_height"] / max(m["belly_diameter"], 1e-6)))


# ---------- 注册 ----------

_CLASSES = (JadeVaseProps, JADEVASE_OT_generate, JADEVASE_OT_reset, VIEW3D_PT_jadevase_gen)
_DNS_KEY = "jadevase_gen_registered"
_DEFAULTS_KEY = "_jadevase_defaults_v"


def _init_defaults(snap):
    sc = bpy.context.scene
    if sc.get(_DEFAULTS_KEY) == SNAP_VERSION:
        return
    if snap:
        for k in PARAM_KEYS:
            setattr(sc.jadevase_gen, k, snap["base_m"][k])
    sc[_DEFAULTS_KEY] = SNAP_VERSION


def register():
    # Hot reload replaces old registered operators, including scripts from older revisions.
    old_values = None
    if hasattr(bpy.types.Scene, "jadevase_gen"):
        old_values = {k: getattr(bpy.context.scene.jadevase_gen, k) for k in PARAM_KEYS}
        del bpy.types.Scene.jadevase_gen
    old_classes = bpy.app.driver_namespace.pop(_DNS_KEY + "_classes", ())
    if not old_classes:
        found = []
        for cls, parent in zip(_CLASSES, (bpy.types.PropertyGroup, bpy.types.Operator,
                                         bpy.types.Operator, bpy.types.Panel)):
            old = parent.bl_rna_get_subclass_py(cls.__name__)
            if old is not None:
                found.append(old)
        old_classes = tuple(found)
    for old in reversed(old_classes):
        try:
            bpy.utils.unregister_class(old)
        except (RuntimeError, ValueError):
            pass
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.jadevase_gen = bpy.props.PointerProperty(type=JadeVaseProps)
    bpy.app.driver_namespace[_DNS_KEY] = True
    bpy.app.driver_namespace[_DNS_KEY + "_classes"] = _CLASSES
    snap = ensure_snapshot()
    for key, value in (old_values or snap["base_m"]).items():
        setattr(bpy.context.scene.jadevase_gen, key, value)


def unregister():
    if hasattr(bpy.types.Scene, "jadevase_gen"):
        del bpy.types.Scene.jadevase_gen
    for cls in reversed(bpy.app.driver_namespace.pop(_DNS_KEY + "_classes", _CLASSES)):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass
    bpy.app.driver_namespace.pop(_DNS_KEY, None)


register()
print("[jadevase_generator] 已注册: N 面板 -> 玉壶 -> 玉壶生成器")
