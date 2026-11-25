# -*- coding: utf-8 -*-
# ============================================================
# 油灯参数化生成器 / Oil Lamp Parametric Generator
# 资产编号 oiltamp / asset id: oiltamp   版本 v004
# ------------------------------------------------------------
# 用法 / Usage:
#   1. 打开本 .blend 后, 在 Text Editor 中选择本脚本并点击 Run Script
#   2. 按 N 键, 在侧边栏找到「油灯生成器」面板
#   3. 修改 6 个参数 (单位: 米), 点击「生成」
#   4. 「恢复默认」一键回到基准尺寸 (与原模型逐位一致)
#
# 原理:
#   - 首次运行时把 12 个部件的顶点基准坐标存为快照
#     (快照保存在文本 datablock oiltamp_generator_base.json 中)
#   - 每次生成: 先把全部顶点还原到基准快照, 再按参数施加绕支点的
#     轴向缩放 (只写顶点坐标, 对象变换保持单位矩阵, 便于导出)
#     因此同一组参数永远得到同一结果, 不累积误差
#   - 参数范围: 基准尺寸的 90% ~ 110% (满足交付要求 ±10%)
#
# 坐标约定: Blender Z 轴向上, 灯轴近似在 x=0, y=0
#   导出 glTF 后: +Y 向上, 正面朝 -Z
# ============================================================

import base64
import json
import zlib

import bpy
import numpy as np
from mathutils import Matrix

# ---- 部件名单 (对象改名后只需改这里) ----
GLASS = "glass_globe"
PARTS = ["glass_globe", "burner", "vent_cap", "upper_body", "base", "top_ring",
         "guard_bow", "body_collar", "bail_wire", "fount_upper", "fount_lower",
         "base_rim"]

SNAP_NAME = "oiltamp_generator_base.json"

PARAM_KEYS = ["total_height", "globe_diameter", "globe_height",
              "upper_body_height", "base_diameter", "fount_height"]
PARAM_LABELS = {"total_height": "总高 (含提梁)",
                "globe_diameter": "灯罩直径",
                "globe_height": "灯罩高度",
                "upper_body_height": "灯头柱高度",
                "base_diameter": "底座直径",
                "fount_height": "油壶高度"}
RATIO_MIN, RATIO_MAX = 0.9, 1.1      # 相对基准的可用范围

# 基准值 (米), 快照缺失时使用的兜底值; 正常运行时以快照为准
DEFAULTS = {"total_height": 0.320000,
            "globe_diameter": 0.097269,
            "globe_height": 0.106278,
            "upper_body_height": 0.109930,
            "base_diameter": 0.148470,
            "fount_height": 0.057528}

_last_clamped = []      # 记录上次生成时被范围裁剪的参数
_coords_cache = {}      # 解码后的基准坐标 (numpy float32), 按部件名缓存
_snap_cache = None      # 快照内存缓存


# ---------- 工具函数 ----------

def _find(name):
    o = bpy.data.objects.get(name)
    return o if (o and o.type == 'MESH') else None


def _managed():
    return [n for n in PARTS if _find(n)]


def _text_write(t, s):
    """兼容写入文本: Blender <=4.x 用 body, 5.x 用 clear()+write()"""
    if hasattr(t, "body"):
        t.body = s
    else:
        t.clear()
        t.write(s)


def _decode(rec):
    """快照记录 -> (n, 3) float32 数组"""
    raw = zlib.decompress(base64.b64decode(rec["coords"]))
    return np.frombuffer(raw, dtype=np.float32).reshape(rec["n"], 3)


def _encode(arr):
    return base64.b64encode(zlib.compress(arr.astype(np.float32).tobytes(), 9)).decode("ascii")


def _read_co(me):
    arr = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", arr)
    return arr.reshape(len(me.vertices), 3)


def _write_co(me, arr):
    me.vertices.foreach_set("co", np.ascontiguousarray(arr, dtype=np.float32).ravel())
    me.update()


def _eval_bbox(name):
    """求值后 (含修改器) 的世界包围盒"""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    o = bpy.data.objects[name]
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    mw = o.matrix_world
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    m = np.array([[mw[i][j] for j in range(4)] for i in range(4)])
    w = co @ m[:3, :3].T + m[:3, 3]
    mn, mx = w.min(axis=0), w.max(axis=0)
    ev.to_mesh_clear()
    return mn, mx


def _bbox(rec):
    """快照记录 -> 世界包围盒 (对象变换近似单位矩阵)"""
    co = _decode(rec).astype(np.float64)
    return co.min(axis=0), co.max(axis=0)


# ---------- 基准快照 ----------

def PART_NAMES_PRESENT(snap):
    return [n for n in PARTS if n in snap.get("objects", {})]


def _build_snapshot():
    """从当前场景创建基准快照 (只在快照缺失时执行一次)"""
    snap = {"asset": "oiltamp", "version": "v004", "objects": {}, "base": {}, "pivots": {}}
    for n in PARTS:
        o = _find(n)
        if not o:
            continue
        co = _read_co(o.data)
        snap["objects"][n] = {
            "n": int(len(co)),
            "coords": _encode(co),
            "matrix": [float(v) for row in o.matrix_world for v in row],
        }
    if not snap["objects"]:
        return snap

    # 基准尺寸取求值后的真实尺寸 (含修改器), 与文档/统计一致
    def size(n, axis):
        mn, mx = _eval_bbox(n)
        return float(mx[axis] - mn[axis])

    gmin = min(_eval_bbox(n)[0][2] for n in PART_NAMES_PRESENT(snap))
    gmax = max(_eval_bbox(n)[1][2] for n in PART_NAMES_PRESENT(snap))
    fmt = _eval_bbox(GLASS)
    bmn, bmx = _eval_bbox("base")
    f_lo = min(_eval_bbox("fount_lower")[0][2], _eval_bbox("fount_upper")[0][2])
    f_hi = max(_eval_bbox("fount_lower")[1][2], _eval_bbox("fount_upper")[1][2])
    ub = _eval_bbox("upper_body")

    snap["base"] = {
        "total_height": gmax - gmin,
        "globe_diameter": max(fmt[1][0] - fmt[0][0], fmt[1][1] - fmt[0][1]),
        "globe_height": fmt[1][2] - fmt[0][2],
        "upper_body_height": ub[1][2] - ub[0][2],
        "base_diameter": max(bmx[0] - bmn[0], bmx[1] - bmn[1]),
        "fount_height": f_hi - f_lo,
    }
    # 支点: 底缘定底、顶缘定顶, 接缝不脱开
    snap["pivots"] = {
        "total_height": [0.0, 0.0, gmin],
        "globe_diameter": [float((fmt[0][0] + fmt[1][0]) / 2.0),
                           float((fmt[0][1] + fmt[1][1]) / 2.0), 0.0],
        "globe_height": [0.0, 0.0, float(fmt[1][2])],
        "upper_body_height": [0.0, 0.0, float(ub[1][2])],
        "base_diameter": [float((bmn[0] + bmx[0]) / 2.0),
                          float((bmn[1] + bmx[1]) / 2.0), 0.0],
        "fount_height": [0.0, 0.0, float(f_lo)],
    }
    return snap


def ensure_snapshot():
    """读取基准快照; 不存在则从当前场景创建。返回 dict。"""
    global _snap_cache
    if _snap_cache is not None:
        return _snap_cache
    t = bpy.data.texts.get(SNAP_NAME)
    if t and t.as_string().strip():
        _snap_cache = json.loads(t.as_string())
        return _snap_cache
    snap = _build_snapshot()
    if snap.get("objects"):
        t = bpy.data.texts.get(SNAP_NAME) or bpy.data.texts.new(SNAP_NAME)
        _text_write(t, json.dumps(snap, indent=1))
    _snap_cache = snap
    return snap


# ---------- 核心生成逻辑 ----------

def _restore(snap):
    """把全部部件还原到基准快照 (顶点坐标 + 对象变换)"""
    for n in PARTS:
        rec = snap["objects"].get(n)
        o = _find(n)
        if not rec or not o:
            continue
        co = _coords_cache.get(n)
        if co is None:
            co = _decode(rec)
            _coords_cache[n] = co
        _write_co(o.data, co)
    bpy.context.view_layer.update()


def regenerate(params):
    """按参数重建油灯。params: 6 个参数名 -> 米 (float)。
    返回 (成功, 提示消息)。"""
    global _last_clamped
    snap = ensure_snapshot()
    base = snap.get("base") or DEFAULTS
    piv = snap.get("pivots") or {}
    _last_clamped = []

    # 0. 部件存在性检查: 缺失时明确报错而不是静默无效
    missing = [n for n in PARTS if not _find(n)]
    if missing:
        return False, "错误: 找不到部件 %s — 对象可能被改名, 请更新脚本顶部名单" % ", ".join(missing)
    missing_rec = [n for n in PARTS if n not in snap.get("objects", {})]
    if missing_rec:
        return False, "错误: 基准快照缺少部件 %s — 请删除文本块 %s 后重新运行脚本" % (
            ", ".join(missing_rec), SNAP_NAME)

    # 1. 参数裁剪到可用范围, 计算缩放比
    ratios = {}
    for k in PARAM_KEYS:
        b = float(base[k])
        r = float(params[k]) / b if b > 1e-9 else 1.0
        rc = max(RATIO_MIN, min(RATIO_MAX, r))
        if abs(rc - r) > 1e-9:
            _last_clamped.append("%s(%.4f)" % (PARAM_LABELS[k], params[k]))
        ratios[k] = rc

    # 2. 整体还原到基准快照 (保证幂等: 不累积缩放/不叠加)
    _restore(snap)

    # 2b. 全部为默认值时, 直接以快照收尾 (避免单精度浮点连乘漂移)
    if all(abs(r - 1.0) < 1e-12 for r in ratios.values()):
        return True, "已恢复默认尺寸"

    # 3. 逐部件合成轴向缩放: x' = a*x + b (绕支点缩放的等价仿射)
    aff = {n: [np.ones(3), np.zeros(3)] for n in PARTS}

    def add(names, axes, f, pivot):
        if abs(f - 1.0) < 1e-12:
            return
        for n in names:
            a, b = aff[n]
            for i in range(3):
                if axes[i]:
                    a[i] *= f
                    b[i] = b[i] * f + float(pivot[i]) * (1.0 - f)

    AX_X = (True, False, False)
    AX_Y = (False, True, False)
    AX_Z = (False, False, True)
    AX_XY = (True, True, False)

    # 3a. 灯罩: 直径沿灯轴径向缩放, 高度绕灯罩顶缘向下伸缩,
    #     顶缘不动, 灯罩始终坐在灯头柱上
    add([GLASS], AX_XY, ratios["globe_diameter"], piv["globe_diameter"])
    add([GLASS], AX_Z, ratios["globe_height"], piv["globe_height"])

    # 3b. 灯头柱: 高度绕顶缘向下伸缩, 顶盖跟随, 不脱开
    add(["upper_body"], AX_Z, ratios["upper_body_height"], piv["upper_body_height"])

    # 3c. 底盘的最大直径独立变化；承托口与底缘跟随玻璃接合面。
    # 上半部的联动在写入顶点时做平滑过渡，不把整座上移或改变地面原点。
    add(["base"], AX_XY, ratios["base_diameter"], piv["base_diameter"])

    # 3d. 油壶: 高度绕底缘向上伸缩, 底缘始终插在底座碗内
    add(["fount_upper", "fount_lower"], AX_Z, ratios["fount_height"], piv["fount_height"])

    # 3e. 写入局部变形后再做整体高度缩放，使接缝同步。
    for n in PARTS:
        a, b = aff[n]
        co = _coords_cache.get(n)
        if co is None:
            co = _decode(snap["objects"][n])
            _coords_cache[n] = co
        original = co.astype(np.float64)
        changed = original * a + b
        if n in ("base", "base_rim", "guard_bow"):
            # 玻璃与承托口使用同一局部映射，灯罩变短时承托口随之升高。
            # 底盘下部仍只服从 base_diameter，保持最大底径独立可控。
            coupled = original.copy()
            center = np.array(piv["globe_diameter"], dtype=np.float64)
            coupled[:, :2] = (original[:, :2] - center[:2]) * ratios["globe_diameter"] + center[:2]
            ztop = float(piv["globe_height"][2])
            coupled[:, 2] = (original[:, 2] - ztop) * ratios["globe_height"] + ztop
            if n == "base_rim":
                weight = np.ones(len(original))
            elif n == "guard_bow":
                # 护弓下端随承托口移动；燃烧器及上部固定处保持原位。
                weight = np.clip((0.095 - original[:, 2]) / 0.010, 0.0, 1.0)
                weight = weight * weight * (3.0 - 2.0 * weight)
            else:
                # 35–50 mm 是底盘肩部过渡，50 mm以上完整跟随承托口。
                weight = np.clip((original[:, 2] - 0.035) / 0.015, 0.0, 1.0)
                weight = weight * weight * (3.0 - 2.0 * weight)
            changed = changed * (1.0 - weight[:, None]) + coupled * weight[:, None]
        # 保留金属围箍与玻璃的径向配合；其高度仍独立按 fount_height 测量。
        if n in ("fount_upper", "fount_lower"):
            center = np.array(piv["globe_diameter"], dtype=np.float64)
            changed[:, :2] = (changed[:, :2] - center[:2]) * ratios["globe_diameter"] + center[:2]
        zground = float(piv["total_height"][2])
        changed[:, 2] = (changed[:, 2] - zground) * ratios["total_height"] + zground
        _write_co(bpy.data.objects[n].data, changed)
    bpy.context.view_layer.update()

    msg = "生成完成"
    if _last_clamped:
        msg += " (以下参数超出 ±10%% 范围已裁剪: %s)" % ", ".join(_last_clamped)
    return True, msg


def measured_extents():
    """当前各参数对应的实测世界尺寸 (含修改器), 用于验证"""
    out = {}
    out["total_height"] = 0.0
    lo = None
    hi = None
    for n in PARTS:
        mn, mx = _eval_bbox(n)
        lo = mn[2] if lo is None else min(lo, mn[2])
        hi = mx[2] if hi is None else max(hi, mx[2])
    out["total_height"] = float(hi - lo)
    gmn, gmx = _eval_bbox(GLASS)
    out["globe_diameter"] = float(gmx[0] - gmn[0])
    out["globe_height"] = float(gmx[2] - gmn[2])
    umn, umx = _eval_bbox("upper_body")
    out["upper_body_height"] = float(umx[2] - umn[2])
    bmn, bmx = _eval_bbox("base")
    out["base_diameter"] = float(bmx[0] - bmn[0])
    fl = min(_eval_bbox("fount_lower")[0][2], _eval_bbox("fount_upper")[0][2])
    fh = max(_eval_bbox("fount_lower")[1][2], _eval_bbox("fount_upper")[1][2])
    out["fount_height"] = float(fh - fl)
    return out


def default_params():
    snap = ensure_snapshot()
    base = snap.get("base") or DEFAULTS
    return {k: float(base[k]) for k in PARAM_KEYS}


# ---------- UI ----------

class OiltampGenProps(bpy.types.PropertyGroup):
    total_height: bpy.props.FloatProperty(
        name="总高 (含提梁)", description="地面到提梁最高点的垂直距离",
        subtype='DISTANCE', unit='LENGTH', min=0.01, max=100.0,
        precision=4, default=DEFAULTS["total_height"])
    globe_diameter: bpy.props.FloatProperty(
        name="灯罩直径", description="玻璃灯罩最宽处直径",
        subtype='DISTANCE', unit='LENGTH', min=0.01, max=100.0,
        precision=4, default=DEFAULTS["globe_diameter"])
    globe_height: bpy.props.FloatProperty(
        name="灯罩高度", description="玻璃灯罩的垂直高度",
        subtype='DISTANCE', unit='LENGTH', min=0.01, max=100.0,
        precision=4, default=DEFAULTS["globe_height"])
    upper_body_height: bpy.props.FloatProperty(
        name="灯头柱高度", description="灯罩上方灯头柱的垂直高度",
        subtype='DISTANCE', unit='LENGTH', min=0.01, max=100.0,
        precision=4, default=DEFAULTS["upper_body_height"])
    base_diameter: bpy.props.FloatProperty(
        name="底座直径", description="底座最宽处的直径",
        subtype='DISTANCE', unit='LENGTH', min=0.01, max=100.0,
        precision=4, default=DEFAULTS["base_diameter"])
    fount_height: bpy.props.FloatProperty(
        name="油壶高度", description="灯罩下方金属油壶的垂直高度",
        subtype='DISTANCE', unit='LENGTH', min=0.01, max=100.0,
        precision=4, default=DEFAULTS["fount_height"])


def _props():
    return bpy.context.scene.oiltamp_gen


def _params_from_props():
    p = _props()
    return {k: float(getattr(p, k)) for k in PARAM_KEYS}


def _set_defaults():
    p = _props()
    for k, v in default_params().items():
        setattr(p, k, v)


class OILTAMP_OT_generate(bpy.types.Operator):
    bl_idname = "oiltamp.generate"
    bl_label = "生成"
    bl_description = "按当前参数重建油灯 (每次从同一基准生成, 结果可复现)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ok, msg = regenerate(_params_from_props())
        self.report({'INFO' if ok else 'ERROR'}, msg)
        return {'FINISHED'}


class OILTAMP_OT_reset(bpy.types.Operator):
    bl_idname = "oiltamp.reset"
    bl_label = "恢复默认"
    bl_description = "恢复到基准默认尺寸并重新生成"

    def execute(self, context):
        _set_defaults()
        ok, msg = regenerate(default_params())
        self.report({'INFO' if ok else 'ERROR'}, "已恢复默认尺寸")
        return {'FINISHED'}


class VIEW3D_PT_oiltamp_gen(bpy.types.Panel):
    bl_label = "油灯生成器"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "油灯"

    def draw(self, context):
        lay = self.layout
        p = _props()
        for k in PARAM_KEYS:
            lay.prop(p, k)
        row = lay.row(align=True)
        row.operator("oiltamp.generate", icon='FILE_REFRESH')
        row.operator("oiltamp.reset", icon='LOOP_BACK')
        snap = ensure_snapshot()
        base = snap.get("base") or DEFAULTS
        box = lay.box()
        box.label(text="基准尺寸 (米):", icon='INFO')
        for k in PARAM_KEYS:
            box.label(text="%s: %.4f" % (PARAM_LABELS[k], base[k]))
        box.label(text="可用范围: 基准 ±10%")


# ---------- 注册 ----------

_CLASSES = (OiltampGenProps, OILTAMP_OT_generate, OILTAMP_OT_reset, VIEW3D_PT_oiltamp_gen)
_DNS_KEY = "oiltamp_gen_registered"


def register():
    # 会话级守卫: 同一 Blender 会话内只注册一次 (重复运行脚本只刷新默认值)
    # 注: Blender 5.x 中 hasattr(bpy.types, 类名) 对已注册类也可能返回 False,
    #     所以不能用 bpy.types 查询做守卫, 改用 driver_namespace
    if bpy.app.driver_namespace.get(_DNS_KEY):
        if not hasattr(bpy.types.Scene, "oiltamp_gen"):
            setattr(bpy.types.Scene, "oiltamp_gen",
                    bpy.props.PointerProperty(type=OiltampGenProps))
        return "已注册 (会话内)"

    for c in _CLASSES:
        bpy.utils.register_class(c)
    if not hasattr(bpy.types.Scene, "oiltamp_gen"):
        setattr(bpy.types.Scene, "oiltamp_gen",
                bpy.props.PointerProperty(type=OiltampGenProps))
    bpy.app.driver_namespace[_DNS_KEY] = True
    _set_defaults()
    return "已注册"


def unregister():
    for c in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
    if hasattr(bpy.types.Scene, "oiltamp_gen"):
        delattr(bpy.types.Scene, "oiltamp_gen")
    bpy.app.driver_namespace.pop(_DNS_KEY, None)


register()
print("[ID_generator] 已注册: N 面板 -> 油灯生成器")
