老油灯 / The Old Lantern
资产 ID：oiltamp　版本：v004　Blender：5.2.2 LTS
本文件所在位置是资产根目录，以下路径均相对于根目录。

1. 内容与打开
mesh/asset_oiltamp.glb、asset_oiltamp.fbx：十二件主体，贴图内嵌。
source/blender/asset_oiltamp.blend：可编辑网格、修改器、材质、生成基准及展示场景。
source/blender/ID_generator.py、export_assets.py：生成与导出脚本，和工程内文本一致。
source/threejs/index.html：离线查看器；依赖、数据、许可证与说明一并保存。
textures/：七张 2048×2048 PNG；renders/：六视图及两张细节图，均为 2048×2048 PNG。
turntable.mp4：H.264、1920×1080、24 fps、240 帧、10 秒匀速环绕一圈。
description.txt：中英外观、结构、尺寸、材质、UV 及许可说明；stats.json：技术统计及文件校验值。

直接打开 Blender 工程；贴图已打包且使用包内相对路径。1 单位 = 1 米，Z 向上；主体在 oil_lamp 集合，展示辅助物在 delivery_presentation。GLB/网页采用 Y 向上。原点为主体包围盒底部中心。

2. 六参数生成器
在 Blender 文本编辑器打开 ID_generator.py 并运行（Alt+P），3D 视图按 N，在“油灯”页签使用“油灯生成器”。数值单位为米；点击“生成”应用，点击“恢复默认”回到随包形体。
参数                         默认        最小        最大
总高 total_height            0.320000    0.288000    0.352000
灯罩直径 globe_diameter      0.097269    0.087542    0.106996
灯罩高度 globe_height        0.106278    0.095650    0.116906
灯头柱高 upper_body_height   0.109930    0.098937    0.120923
底座直径 base_diameter       0.148470    0.133623    0.163317
油壶高度 fount_height        0.057528    0.051775    0.063281

精确值以面板与内嵌基准为准。总高按最低点至最高点测量；直径按对应最大横截面测量；高度按对应件上下端测量。灯罩高度以顶缘为支点，下承托口、底缘和护弓下端跟随。底座最大底径独立变化，上承托口跟随灯罩直径。油壶高度以底缘为支点。
每次生成先恢复基准顶点再变形，不累计误差、不新增物体，不重写已展开 UV。组合时，总高会再竖向缩放全灯：高度实测值 = 对应参数 × 总高比例；直径不受总高影响。内嵌 oiltamp_generator_base.json 是必要基准，请保留。

3. 导出与查看
在 Blender 运行 export_assets.py；或从资产根目录执行：
blender --background source/blender/asset_oiltamp.blend --python source/blender/export_assets.py
脚本只导出十二件主体，更新 mesh 下两个文件与 source/threejs/js/model_data.js。导出时临时转写顶盖 ColorRamp 和玻璃材质，再恢复源节点；不保存源工程。
直接打开 source/threejs/index.html，可通过内嵌 GLB 离线查看；也可在资产根目录运行 python -m http.server 8000，再打开 http://localhost:8000/source/threejs/index.html。左键旋转、滚轮缩放、右键平移；空格自动转台，R 复位，S 截图。网页数据与正式 GLB 字节相同。

4. 材质与统计
textures 中 metal_rust_basecolor/metalness/roughness/normal/displacement.png 是原始金属通道；metal_cap_basecolor.png 是顶盖颜色烘焙；glass_clear_normal.png 是玻璃合并法线。金属主体粗糙度为 0.5 常量，顶盖使用粗糙度贴图；金属源材质仍引用 displacement，故保留该文件。未使用的 DirectX 备用法线不再打包。玻璃合并法线强度为 0.35，无重复玻璃置换。GLB 使用透射玻璃；FBX 以烟色透明近似，因格式不支持同等透射。
默认主体约 0.165×0.181×0.320 米；287208 个三角面，源/FBX 143993 个顶点，GLB 因 UV/法线接缝拆分为 199838 个。提梁 UV 按真实长度和周长展开；其他有意共享区域详见 description.txt 第十一节。

5. 渲染与许可
源工程的 delivery_camera 为展示相机，delivery_orbit 为唯一环绕动画。灯体静止，相机从第 1 帧至 241 帧线性旋转一圈，只输出第 1–240 帧，避免重复终点。渲染动画的 PNG 帧写入本资产下 turntable_frames/，不计入交付主体或导出。
模型为原提供方声明的原创课程作品，未使用第三方模型。金属贴图来源 ambientCG Metal025，许可 CC0 1.0 Universal，允许修改、商用与再次交付；合并法线及颜色烘焙由该材质衍生。无第三方 HDRI 或字体。Three.js r147 及配套组件本地保存，MIT 许可见 source/threejs/LICENSE_THREEJS.txt。

English quick reference
Asset ID oiltamp, version v004. Open source/blender/asset_oiltamp.blend in Blender 5.2.2 LTS. Packed textures also use relative package paths. One unit is one meter; the source is Z-up and GLB is Y-up. The origin is the bottom center of the subject bounding box.
Run embedded ID_generator.py to expose the six-parameter panel. All ranges are 90–110% of the baseline; the table above lists values in meters. Generate restores baseline vertices before applying dimensions, preserving UVs. Total height additionally scales all vertical dimensions; diameter readings are unaffected. Reset Defaults reproduces the delivered model. Keep the embedded oiltamp_generator_base.json.
Run export_assets.py in Blender or use the root-relative command above. It exports only twelve subject meshes to mesh/asset_oiltamp.glb and .fbx, and synchronizes the viewer payload. Material conversion is temporary and does not save the blend. The viewer works offline using js/model_data.js, with all dependencies local.
The seven 2K maps retain every active source/export dependency. Metal displacement remains required by the source; only the unused DirectX alternate normal was removed. Metal body roughness is constant 0.5; the cap uses its map. Glass uses the shared combined normal at strength 0.35; FBX transparency approximates glTF transmission.
Default subject totals: 287208 triangles; 143993 source/FBX vertices; 199838 GLB vertices. Render images are 2048×2048. Video is 1080p H.264, 24 fps, 240 frames, ten seconds. Subjects remain static while the camera makes one linear orbit over frames 1–241; output only frames 1–240. Animation rendering writes PNGs into turntable_frames/.
The original provider declares the model original coursework. Metal maps originate from ambientCG Metal025 under CC0 1.0 Universal, allowing modification, commercial use and redistribution; baked maps derive from those assets. Three.js r147 and its components are included under MIT; see source/threejs/LICENSE_THREEJS.txt. Detailed bilingual geometry, material and intentional UV-sharing descriptions are in description.txt.
