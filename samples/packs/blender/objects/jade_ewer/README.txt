玉壶 / The Jade Ewer
资产 ID / Asset ID: jadevase    版本 / Version: v004

目录与打开方法
本文件所在的 asset_jadevase 即资产根目录。description.txt 为中英描述，stats.json 为统计与校验清单。
source/blender/asset_jadevase.blend：Blender 5.2.2 LTS 可编辑工程，保留基础网格、Solidify / Subdivision 修改器、参数基准、展示相机与灯光。
mesh/asset_jadevase.glb、mesh/asset_jadevase.fbx：仅含壶身和壶盖，贴图内嵌。
source/threejs/index.html：three.js r147 查看器，依赖均在 js/ 下。直接打开可使用同字节内嵌 GLB；或在资产根目录执行 python -m http.server 8000，访问 http://localhost:8000/source/threejs/index.html。支持旋转、缩放、复位与背景切换。

生成与导出
在 Blender 文本编辑器选择 jadevase_generator.py 并运行，按 N 打开“玉壶”面板。调整参数后点“生成”，点“恢复默认”还原随包基准；无需关闭脚本安全保护。基准保存在工程文本块 jadevase_generator_base.json，请勿删除。
六参数单位米（默认值；允许范围）：
器身高度 0.2636；0.190–0.340
腹径     0.1778；0.130–0.230
足径     0.1036；0.075–0.135
盖高     0.0217；0.015–0.030
钮径     0.0221；0.014–0.032
壁厚     0.0030；0.0015–0.0045
尺寸按基础控制网格测量；细分后表面略有收缩，例如钮径设 14 mm，最终表面约 13.5 mm。壁厚由 Solidify 控制；壶盖随口沿移动，共享钮座边界随钮径变化。每次从基准计算，不累计变形。
运行内嵌 jadevase_export.py，或在资产根目录执行：
blender --background source/blender/asset_jadevase.blend --python source/blender/jadevase_export.py
导出器更新 mesh/ 下两种模型与 source/threejs/js/model_data.js，仅导出两个主体。正式导出前先恢复默认。外置与内嵌脚本一致；make_embed.py 也可单独刷新网页备用模型。

默认模型与材质
2 个部件：ewer_body（壶身含流、把）、ewer_lid（可拆壶盖含金钮）。默认求值网格 117568 三角面、58782 个源 / FBX 顶点、63376 个 GLB 顶点（含 UV / 法线接缝拆分），不含展示辅助物。
约 0.217×0.176×0.311 m，1 单位=1 m，Scale=1，原点位于包围盒底部中心。Blender Z 向上、正面 +Y；glTF Y 向上、正面 -Z。
textures/ 有八个 2K 文件：porcelain_basecolor.jpg、porcelain_normal.png、porcelain_roughness.png；gold_basecolor.png、gold_normal.png、gold_roughness.png、gold_metalness.png；porcelain_crackle_source.png 为合成基色的原始裂纹来源，不直接接入材质。活动贴图已打包且与外部文件一致，采用相对路径。瓷身金属度为 0，源工程、GLB、FBX 共用标准 PBR 采样。
UV 有意共享：壶身两侧基础 UV 岛部分重叠，超出 0–1 区域按 Repeat 采样；Solidify 生成的内壁复用对应外壁 UV。瓷盖共用瓷纹，金钮使用独立金属材质。详细区域说明见 description.txt。该布局保留重复表面纹理，不是全模型独占图集；单侧文字或逐面唯一烘焙需另做独占 UV。

展示媒体
renders/ 含八张 2048×2048 PNG：front、back、left、right、top、hero_45，以及 detail_lid_finial（金钮）与 detail_belly_crackle（壶身）。
turntable.mp4：H.264、1920×1080、24 fps、240 帧、10 秒，默认模型匀速 360° 环绕。
delivery_presentation 集合保存相机、灯光与背景；delivery_camera 为相机，delivery_orbit 为环绕控制。第 1 帧 0°、第 241 帧 360°，线性插值；只输出 1–240 帧以避免重复终点。渲染动画会在资产根目录 turntable_frames/ 输出 PNG 序列。展示辅助物不导出到 GLB / FBX。

来源与权利
Metal048C 与 Onyx015 据原交付声明来自 ambientCG，CC0，允许修改、商用和再分发。three.js 及官方组件使用 MIT 许可，全文见 source/threejs/LICENSE_THREEJS.txt。
委托方于 2026-10-19 确认裂纹源图为设计师原创，允许修改、商用及再交付；合成基色沿用该许可。模型与其他原图权利依据仍以提供方原交付声明或合同为准。
v004 仅整理最终交付命名与必要依赖，保留可编辑性、几何、UV、材质采样和展示媒体内容。

English
The asset root is asset_jadevase. description.txt describes the work; stats.json contains counts and checksums. Open source/blender/asset_jadevase.blend in Blender 5.2.2 LTS. It retains the editable base meshes, Solidify/Subdivision modifiers, parameter baseline and presentation setup.
Run the embedded jadevase_generator.py, then use the Jade Ewer sidebar. Generate recalculates from the baseline; Reset Defaults restores it. The baseline is the embedded jadevase_generator_base.json text block. Automatic script execution is not required.
Six parameters use meters: body height 0.2636 [0.190–0.340], belly diameter 0.1778 [0.130–0.230], foot diameter 0.1036 [0.075–0.135], lid height 0.0217 [0.015–0.030], finial diameter 0.0221 [0.014–0.032], wall thickness 0.0030 [0.0015–0.0045]. Dimensions refer to the base control cage; subdivision slightly contracts the surface (14 mm finial setting gives approximately 13.5 mm on the final surface). Solidify sets wall thickness; the lid and shared finial seat follow their related parameters. Regeneration does not accumulate deformation.
Run jadevase_export.py in Blender, or use the command above from the asset root. It exports mesh/asset_jadevase.glb and mesh/asset_jadevase.fbx, with embedded textures, and synchronizes source/threejs/js/model_data.js. Only ewer_body and ewer_lid are exported. Restore defaults before formal export. External and embedded scripts match; source/threejs/make_embed.py can refresh the web fallback separately.
The two evaluated subjects total 117568 triangles, 58782 source/FBX vertices and 63376 GLB vertices after UV/normal seam splits. Default dimensions are approximately 0.217×0.176×0.311 m, one unit per meter, Scale=1, with the origin at the bottom bounding-box center. Blender is Z-up/front +Y; glTF is Y-up/front -Z. Helpers are excluded from these counts and dimensions.
Eight 2K texture files are included: porcelain base color, normal and roughness; gold base color, normal, roughness and metallic; and porcelain_crackle_source.png, retained as the source of the composed base color, not an active shader input. Active maps are packed, byte-identical to their external files and use relative paths. Porcelain metalness is zero; Blender and both exported formats share the standard PBR sampling.
UV reuse is intentional: the two base body islands partially overlap, out-of-0–1 body UVs repeat, and Solidify inner surfaces inherit outer UVs. The porcelain lid shares porcelain maps while the finial uses its own metal material. See description.txt for the precise regions. This is a reusable surface-texture layout, not a unique asset-wide atlas; one-sided decals or unique per-face baking need a separate UV layout.
Open source/threejs/index.html directly for the embedded-model fallback, or serve the asset root with python -m http.server 8000 and visit /source/threejs/index.html. All three.js r147 dependencies are local. Orbit, zoom, reset and background switching are supported.
The renders directory contains eight 2048×2048 PNGs: six views and two named details. turntable.mp4 is H.264, 1920×1080, 24 fps, 240 frames, 10 seconds, with a constant-speed 360-degree camera orbit. delivery_presentation stores the setup; delivery_camera is the camera and delivery_orbit controls rotation. Linear keys span frames 1–241 (0–360 degrees); output is frames 1–240 to avoid duplicating the endpoint. Rendering animation writes PNGs to turntable_frames/ inside the asset root. Presentation helpers are excluded from mesh exports.
According to the supplied credits, Metal048C and Onyx015 are ambientCG CC0 resources, and three.js is MIT licensed (source/threejs/LICENSE_THREEJS.txt). On 2026-10-19 the commissioning party confirmed the designer’s original crackle texture may be modified, commercially used and re-delivered; this permission covers the derived base color. Rights to original geometry and other original images remain governed by the provider’s declaration or contract.
v004 normalizes final-delivery names and dependencies while preserving editability, geometry, UVs, material sampling and media contents.
