# Harbor review bundle

This snapshot contains 5 independently validated Harbor 1.2 tasks. Every included case has valid Kimi K3, GPT-5.6-Sol, and Claude Opus 5's trajectory.

Each case includes the Agent-visible workspace, verbatim user task, frozen binary rubrics, reference deliverables, all three candidate outputs, score reports, sanitized Executor trajectories, and successful Judge trajectories. 

## 生产力与 Case 概览

### 产能

多模态任务生产已经完成 Scale。按当前生产流程，可根据单任务体量与复杂度在一周内供给约 100 个任务。

### 当前 Case 与行业分布

- 当前 Case 数量：5 个，均为经过独立验证的 Harbor 1.2 多模态任务。
- 当前 Case 覆盖：金融、法律、医疗、艺术、学术，各 1 个。
- 可生产行业范围：金融、法律、医疗、教育、艺术、学术；教育行业当前 Case 为 0。

| 行业 | 当前 Case 数 | 占比 |
| --- | ---: | ---: |
| 金融 | 1 | 20% |
| 法律 | 1 | 20% |
| 医疗 | 1 | 20% |
| 教育 | 0 | 0% |
| 艺术 | 1 | 20% |
| 学术 | 1 | 20% |

### 输入/输出文件数量分布

统计口径：输入文件按每个任务的 `data/input_files` 计算，不重复计算 `environment/input_files` 中的镜像副本；输出文件按 reference deliverables 计算。

- 输入文件：共 100 个，单任务 17–23 个，平均 20.0 个。
- 要求输出文件：共 7 个，单任务 1–2 个，平均 1.4 个。
- 输出 1 个文件的任务：3 个，占 60%；输出 2 个文件的任务：2 个，占 40%。
- 三个模型分别生成同等数量的候选输出，合计 21 个候选交付文件。

| 任务 | 行业 | 输入文件数 | 要求输出文件数 |
| --- | --- | ---: | ---: |
| `photographic_archive_rights` | 艺术 | 21 | 1 |
| `procurement_protest_record_review` | 法律 | 23 | 2 |
| `pension_funding_contribution_control` | 金融 | 20 | 1 |
| `home_health_authorization_control` | 医疗 | 19 | 1 |
| `publication_figure_data_consistency` | 学术 | 17 | 2 |
| **合计** | — | **100** | **7** |

### Rubrics 数量

- 独立 rubric：5 套，每个任务 1 套。
- 评估标准：共 95 条，其中程序校验 7 条、LLM Judge 88 条。
- 下表不重复计算 `tests/rubric.json` 中的镜像副本。

| 任务 | Rubric 条目数 | 程序校验 | LLM Judge |
| --- | ---: | ---: | ---: |
| `photographic_archive_rights` | 18 | 1 | 17 |
| `procurement_protest_record_review` | 23 | 2 | 21 |
| `pension_funding_contribution_control` | 21 | 1 | 20 |
| `home_health_authorization_control` | 19 | 1 | 18 |
| `publication_figure_data_consistency` | 14 | 2 | 12 |
| **合计** | **95** | **7** | **88** |

### 模型跑分

分数取自本包 `scores.csv`，指标为 unweighted binary claim pass rate。平均分为 5 个任务的简单算术平均。

| 任务 | Kimi K3 | GPT-5.6-Sol | Claude Opus 5 | 单任务最高分 |
| --- | ---: | ---: | ---: | ---: |
| `photographic_archive_rights` | 50.00% | 55.56% | 44.44% | 55.56% |
| `procurement_protest_record_review` | 43.48% | 39.13% | 39.13% | 43.48% |
| `pension_funding_contribution_control` | 28.57% | 38.10% | 52.38% | 52.38% |
| `home_health_authorization_control` | 57.89% | 42.11% | 63.16% | 63.16% |
| `publication_figure_data_consistency` | 21.43% | 42.86% | 50.00% | 50.00% |
| **平均分** | **40.27%** | **43.55%** | **49.82%** | **52.91%** |
