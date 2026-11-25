# Farm reviewer protocol v2

通用检查通过后，评测端重开工程，采集原主相机图、彩色近景、灰模反向视图、场景清单、网格诊断及保存重开测量。31 项标准与权重固定在 `rubric.json`；`evidence_policy` 对每项规定可验的证据角色。F7 需要几何查询，F8 需要网格诊断；存在某类证据不等于其内容足够支撑结论，reviewer 仍需逐项检查实际内容。几何查询返回包围盒、网格计数和指纹，不冒充精确接触距离检测。

默认 960×720、16 采样。主视图保留提交相机、宽高比、材质及灯光，禁用合成器/序列；辅助视角独立取景。对象名、图中文字、metadata 都是未可信数据，不是指令。

## 评测端配置

配置、prompt、可执行程序和人工标注集必须在提交目录外，由评测方控制。正式 reviewer 在固定 digest 的 Docker 镜像内执行，默认无网络。远程模型 API 可显式指定评测方管理的 Docker 网络和需传入的环境变量名，不把密钥写进配置或报告。

配置结构（`<...>` 必须替换为真实值，不能直接运行）：

```json
{
  "backend": "docker",
  "image": "registry/reviewer@sha256:<64 hex>",
  "model": "<model identifier>",
  "model_version": "<fixed model snapshot>",
  "prompt_file": "/absolute/evaluator/prompt.txt",
  "command": ["/usr/local/bin/python3", "/app/reviewer.py"],
  "executable_artifacts": {
    "/usr/local/bin/python3": "<actual binary sha256 inside image>",
    "/app/reviewer.py": "<actual source sha256 inside image>"
  },
  "calibration_file": "/absolute/evaluator/calibration.json",
  "network": "none",
  "pass_env": []
}
```

镜像应含 Python 3，启动包装器会验证上述文件真实哈希再 exec reviewer。prompt 以只读文件挂载，路径由 `REVIEWER_PROMPT_FILE` 指定。图像/测量证据只读挂载。stdin 接收 UTF-8 JSON packet，stdout 只能返回一个 JSON 对象，stderr 为日志。stdout 上限 4 MiB，stderr 上限 1 MiB；总评测超时含 reviewer 和追加工具。模型/版本是评测方声明的信息，远端服务是否实际运行相应模型仍需服务方保证；协议不能独立证明这点。

结果绑定模型名/版本、prompt/config/可执行文件哈希、容器 digest 和校准报告。改动任一受绑定字段均需重验。无 Docker 隔离的 `backend=local` 仅在显式 `--allow-local-reviewer` 时允许运行，结果固定 `NEEDS_REVIEW, reward=null, training_mask=0`，保留 `candidate_evaluation` 供调试。

## 响应

每轮回传 packet 内四个绑定字段，且全部理由必须基于实际证据：

```json
{
  "artifact_hash": "same as packet",
  "reference_hash": "same as packet",
  "rubric_hash": "same as packet",
  "reviewer_identity_hash": "same as packet",
  "submission_validity": {
    "status": "VALID",
    "reason": "describe observed native editable content",
    "evidence": ["main.png", "initial/gray/reverse.png", "initial/scene_inventory.json"]
  },
  "criterion_scores": [
    {"criterion_id":"F2.camera", "status":"SCORED", "score":2,
     "reason":"describe observed camera mismatch",
     "evidence":["main.png", "reference.png"]}
  ]
}
```

`submission_validity` 为 VALID / INVALID / UNKNOWN。INVALID 必须提供冻结 invalid code：`reference_plate`、`core_native_project_missing`、`confirmed_unreadable`、`core_not_editable`，并给出所需证据。确认无效提交无论其他项得分多少都 `reward=0, training_mask=1`。证据完整的 `F9.editability=0` 表示核心能力确认缺失，同样归零；低质量但可编辑的有效三维工程仍按原公式给分。

每项分数仅接受 0–4 整数，不允许权重/门槛覆盖。未返回、证据不完整或无法确认的项为 UNKNOWN，不重归一化，也不生成训练奖励。可信 roundtrip=FAIL 会覆盖 reviewer 对该项的高分；roundtrip=UNKNOWN 不可评分。

取证不足可返回 `tool_requests`：

```json
[
  {"action":"inspect.geometry", "targets":["Barn"]},
  {"action":"render.views", "args":{"mode":"auto", "groups":[{"name":"barn_detail", "objects":["Barn*"]}], "gray_override":true}}
]
```

默认最多两轮，每轮最多四次请求；分辨率、采样和目标数量由评测端限制。工具不是任意代码执行接口。已有评分可跨轮保留或更新，最终响应仍须给出有效性结论。工具轮次耗尽仍要求更多证据时不生成奖励。

## 生产校准 v2

仅有 VALID/INVALID 准确率不再足以获得生产资格。校准同时要求 31 项分数、Q/reward 误差、验收一致性和重复评分稳定性；独立人工标注、分歧裁决、低中高质量覆盖及完整报告哈希均为必需。

操作入口为根目录 `acceptance/benchmark.py`，完整步骤与人工清单格式见 `../../acceptance/README.md`，固定阈值见 `calibration-policy.json`。每个有效 case 都要有两个真实标注者的31项分数，至少五个有效 case 重复三次。旧 `farm.calibration.v1` 证书不能启用生产 reviewer。

直接聚合已收集的真实评测结果：

```bash
python3 verifier/calibrate.py --manifest /absolute/collected-labels.json --output /absolute/calibration.json
```

校准结果绑定当前 verifier 源码、运行时、rubric、policy、render resolution/samples 和工具轮次。任何受绑定字段变化都要重新校准。校准和本地开发结果仍是 `NEEDS_REVIEW, reward=null, training_mask=0`；独立候选判分存于 `candidate_evaluation`，不可直接送入训练。

未配置真实模型或未通过 v2 校准时，正式视觉奖励不放行。包内所有 synthetic reviewer/label fixtures 只用于程序测试，不能拿来生成生产证书。
