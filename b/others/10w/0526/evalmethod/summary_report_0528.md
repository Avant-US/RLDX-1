# 全模型对比评估报告与真机部署推荐

本报告对比当前所有可用的离线策略评估结果，给出真机部署推荐。

数据来源（均为 `offline_policy_diagnostics_v2.py` 产出）：

| 来源目录 | 包含模型 |
|---|---|
| `offline_policy_result/` | `rldx_1`、`pi0_5`、`molmoact2`、`fastwam (旧)` |
| `0526FASTWAM/` | `fastwam (新, 0526)` ← 本目录 |
| `0526FASTWAM_20000/` | `fastwam (0526-20k)`，与新版几乎一致 |

---

## 1. 摘要结论（先看这里）

**真机部署首选：`fastwam (新, 0526)`**

| 维度 | 推荐 | 一句话理由 |
|---|---|---|
| 16-step 真机部署综合最优 | **`fastwam (新, 0526)`** | `PQS Closed = 79.38` 为同口径最高，chunk decay 显著改善 |
| 安全优先备选 | `rldx_1` | `Safety = 92.04`、`Tracking = 96.30` 最稳健 |
| 短 horizon (≤10 step) 部署 | `molmoact2` | `chunk_decay = 51.94` 但 chunk size 仅 10，先天占优 |
| 不推荐部署 | `pi0_5` / `fastwam (旧)` | closed-loop 综合最低，无明显领先项 |

**`fastwam (新, 0526)` 唯一短板**：`left_gripper` 起始段 spike 风险高，**部署必须先加 gripper 安全约束**。

---

## 2. 评估模型清单

| 模型 | chunk size | 数据来源 | 说明 |
|---|---:|---|---|
| `rldx_1` | 16 | offline_policy_result | 16-step 基线，最保守 |
| `pi0_5` | 16 | offline_policy_result | 16-step 基线，与 rldx_1 类似 |
| `molmoact2` | 10 | offline_policy_result | 10-step chunk，decay 先天占优 |
| `fastwam (旧)` | 16 | offline_policy_result | 旧版 fastwam |
| **`fastwam (新)`** | 16 | **0526FASTWAM** | **0526 重训练，本报告主体** |
| `fastwam (0526-20k)` | 16 | 0526FASTWAM_20000 | 同模型 20k step checkpoint，与新版差距 < 0.2 分 |

下文 "fastwam (新)" 默认指 `0526FASTWAM`，并已与 `0526FASTWAM_20000` 比较确认基本等价。

---

## 3. PQS 总览（all-group）

| 模型 | chunk size | PQS Open | PQS Closed |
|---|---:|---:|---:|
| `fastwam (旧)` | 16 | **91.38** | 75.68 |
| `fastwam (新)` | 16 | 91.18 | **79.38** |
| `fastwam (0526-20k)` | 16 | 91.15 | 79.25 |
| `rldx_1` | 16 | 91.14 | 76.86 |
| `pi0_5` | 16 | 91.10 | 75.82 |
| `molmoact2` | 10 | 90.76 | 78.91 |

观察：

- 所有模型 `PQS Open` 都在 90-91 之间，open-loop 模仿精度无法拉开差距。
- `PQS Closed` 差距更显著，从 75.68 到 79.38，差 3.7 分。
- 16-step 同口径下，新版 `fastwam` 第一（79.38），优于 `molmoact2`（78.91，但只有 10-step），优于 `rldx_1` / `pi0_5` 约 2-3 分。

---

## 4. 模块分数对比（all-group）

| 模型 | Tracking | Smoothness | Stability | Safety | Chunk Decay | Task |
|---|---:|---:|---:|---:|---:|---:|
| `fastwam (新)` | 95.08 | 76.02 | 96.31 | 88.55 | **49.11** | 99.62 |
| `fastwam (0526-20k)` | 95.11 | 75.94 | 96.20 | 88.57 | 48.66 | 99.57 |
| `fastwam (旧)` | 95.90 | 74.50 | 97.52 | 89.84 | 33.89 | **99.79** |
| `rldx_1` | **96.30** | 74.46 | 94.88 | **92.04** | 36.72 | 99.69 |
| `pi0_5` | 96.13 | 73.87 | 96.41 | 91.76 | 33.55 | 99.42 |
| `molmoact2` | 93.60 | **77.97** | **98.05** | 85.13 | 51.94* | 96.21 |

`*` `molmoact2` 的 chunk decay 在 chunk size = 10 的口径下计算，先天占优，不可直接对比 16-step 模型。

各模块谁最优：

| 模块 | 最优模型 | 含义 |
|---|---|---|
| Tracking | `rldx_1` (96.30) | 模仿精度最高 |
| Smoothness | `molmoact2` (77.97) | 动作平滑度最好 |
| Stability | `molmoact2` (98.05) | 方向翻转最少 |
| Safety | `rldx_1` (92.04) | 综合安全风险最低 |
| Chunk Decay (16-step) | **`fastwam (新)`** (49.11) | 16-step horizon 下衰减最缓 |
| Task | `fastwam (旧)` (99.79) | 任务完成度最高 |

整体看，**没有任何单一模型在所有模块全面领先**：

- `rldx_1`：tracking / safety 双冠，但 chunk decay 偏弱。
- `fastwam (新)`：chunk decay 大幅领先，但 tracking / safety 略弱。
- `molmoact2`：smoothness / stability 双冠，但 tracking 最低、chunk 短。

---

## 5. 分组对比

### 5.1 左臂 PQS Closed

| 模型 | PQS Closed | Tracking | Safety | Chunk Decay |
|---|---:|---:|---:|---:|
| `fastwam (新)` | **77.86** | 95.72 | 93.23 | **37.64** |
| `rldx_1` | 76.58 | 95.86 | 92.40 | 33.78 |
| `molmoact2` | 77.61 | **96.76** | 90.78 | 33.63 |
| `pi0_5` | 74.38 | 95.98 | 92.60 | 26.32 |
| `fastwam (旧)` | 73.25 | 96.12 | **93.30** | 19.73 |

`fastwam (新)` 左臂综合最好；`molmoact2` 左臂 tracking 最高。

### 5.2 右臂 PQS Closed

| 模型 | PQS Closed | Tracking | Safety | Chunk Decay |
|---|---:|---:|---:|---:|
| `fastwam (新)` | **83.89** | 93.89 | 88.98 | **65.44** |
| `molmoact2` | 82.63 | **95.81** | 90.03 | 61.08 |
| `rldx_1` | 80.10 | 95.82 | 89.63 | 50.67 |
| `pi0_5` | 80.02 | 95.22 | 89.43 | 50.85 |
| `fastwam (旧)` | 79.54 | 95.55 | **91.41** | 46.71 |

`fastwam (新)` 右臂 chunk decay 大幅领先，但 tracking 较低、stability 退步。

### 5.3 Gripper PQS Closed

| 模型 | PQS Closed | Safety | Chunk Decay | mean_spike_risk |
|---|---:|---:|---:|---:|
| `molmoact2` | **72.95** | 50.00 | **82.53** | 1.00 |
| `fastwam (旧)` | 70.96 | 73.11 | 38.36 | 0.51 |
| `fastwam (新)` | 69.39 | 71.50 | 32.93 | 0.52 |
| `rldx_1` | 67.03 | **98.88** | 0.00 | 0.02 |
| `pi0_5` | 66.61 | 96.71 | 0.00 | 0.02 |

Gripper 没有干净的赢家：

- `rldx_1` / `pi0_5`：spike 风险极低（safety 98.88、96.71），但 chunk decay = 0，gripper 越往 horizon 末端误差越大。
- `fastwam (新)` / `fastwam (旧)`：decay 较好，但 spike_risk ≈ 0.5。
- `molmoact2`：spike_risk = 1.0（最差），但 chunk 短，PQS 名义最高。

**结论：所有模型都需要对 gripper 单独加安全门限**，不存在裸跑安全的模型。

---

## 6. Chunk Offset 衰减（all-group 归一化 RMSE）

| 模型 | offset 0 | offset 5 | offset 9 | offset 15 | 末端 / 起点 |
|---|---:|---:|---:|---:|---:|
| `rldx_1` | 0.030 | 0.075 | 0.094 | 0.103 | 3.39x |
| `pi0_5` | 0.032 | 0.077 | 0.101 | 0.112 | 3.47x |
| `fastwam (旧)` | 0.040 | 0.079 | 0.102 | 0.113 | 2.81x |
| `fastwam (新)` | 0.045 | 0.083 | 0.102 | **0.107** | **2.40x** |
| `molmoact2` | 0.055 | 0.082 | 0.110 | — | 2.00x (在 offset 9) |

观察：

- `rldx_1` / `pi0_5` 起点最低，但 horizon 拉长后增长很快，末端反而被 `fastwam (新)` 反超。
- `fastwam (新)` 起点偏高（0.045），但增长曲线最平，是 16-step 模型里末端 RMSE 最低的。
- `molmoact2` 只有 10 步 horizon，不能直接和 16-step 模型比较。

> **closed-loop 实际部署看末端误差，而非起点误差。** `fastwam (新)` 的衰减曲线最适合长 horizon 闭环执行。

---

## 7. Phase 分布对比

### 7.1 all-group RMSE

| 模型 | Q1 | Q2 | Q3 | Q4 | 主要问题 phase |
|---|---:|---:|---:|---:|---|
| `rldx_1` | 0.073 | 0.066 | **0.108** | 0.041 | Q3 |
| `pi0_5` | 0.039 | 0.043 | **0.081** | 0.029 | Q3 |
| `fastwam (新)` | **1.464** | 0.067 | 0.160 | 0.022 | Q1 |
| `fastwam (旧)` | **1.450** | **1.329** | 0.161 | 0.032 | Q1 / Q2 |
| `molmoact2` | **1.454** | **2.789** | **1.643** | 0.071 | Q1 / Q2 / Q3 |

- `rldx_1` / `pi0_5` 所有 phase 都在 0.1 量级以内，但 Q3 是它们最弱段。
- `fastwam (新)` 把 `fastwam (旧)` 的 Q2 从 1.33 修到 0.07，**唯一遗留问题是 Q1**。
- `molmoact2` 前 75% 段误差都极高。

### 7.2 Gripper Phase RMSE（关键）

| 模型 | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| `rldx_1` | 0.40 | 0.35 | 0.73 | 0.29 |
| `pi0_5` | 0.17 | 0.21 | 0.47 | 0.17 |
| `fastwam (新)` | **11.54** | 0.36 | 1.05 | 0.13 |
| `fastwam (旧)` | **11.46** | **10.47** | 1.10 | 0.22 |
| `molmoact2` | **11.49** | **22.15** | **12.99** | 0.51 |

- `rldx_1` / `pi0_5` gripper 在所有 phase 量级稳定。
- `fastwam (新)` gripper Q1 仍约 11.54，是新版唯一没解决的 phase；Q2 / Q3 / Q4 已经修到接近 `rldx_1`。
- `molmoact2` gripper 全程不可信。

---

## 8. 主要风险点对比

| 风险 | 影响模型 | 严重程度 |
|---|---|---|
| Gripper 起始段 spike (Q1 ~11) | `fastwam (新)` / `fastwam (旧)` / `molmoact2` | 高 |
| Gripper 中段大幅偏差 (Q2 > 10) | `fastwam (旧)` / `molmoact2` | 高，新 fastwam 已修 |
| Gripper chunk decay = 0 | `rldx_1` / `pi0_5` | 中，长 horizon 末端越差 |
| Right arm Q3 误差 | 所有模型，新 fastwam 最高 (0.057) | 中 |
| 整体 chunk decay 偏低 | `rldx_1` / `pi0_5` / `fastwam (旧)` | 中 |
| `left_gripper` spike_risk = 0.99 | `fastwam (新)` | 高 |

---

## 9. 真机部署推荐

### 9.1 选模型

**推荐：`fastwam (新, 0526)`，前提是先加 gripper 安全约束。**

理由：

1. **16-step closed-loop 综合最强**：`PQS Closed = 79.38`，最高。
2. **chunk decay 显著改善**：49.11，比同口径其它 16-step 模型高 12-15 分；末端 offset 15 的 RMSE 是 16-step 模型里最低。
3. **轨迹中段已修好**：Q2 from 1.33 to 0.07，gripper Q2 from 10.47 to 0.36。
4. **唯一明确短板可控**：`left_gripper` Q1 起始 spike，可以用外部约束兜底。

不推荐的原因：

- `rldx_1`：tracking / safety 是稳的，但 chunk decay 几乎不如新 fastwam 的一半，长 horizon 末端误差更大。仅作为安全优先的备选。
- `pi0_5`：所有模块都不领先，PQS Closed 最低，没有部署优先级。
- `fastwam (旧)`：被新版全面替代。
- `molmoact2`：chunk size 10 不能等同 16-step，gripper Q1/Q2/Q3 全程不可信，不适合裸跑。

### 9.2 必加的安全约束

针对 `fastwam (新)` 部署，必须配合以下措施：

1. **左 gripper 起始保护**  
   - episode 前 25% 时间（Q1 段）禁用或限速 `left_gripper` 输出。
   - 单步 `left_gripper` 命令变化幅度做 clip，例如 ≤ 5°/step。
   - 加 hysteresis：连续 3 个 step 同方向才允许翻转。

2. **左 gripper spike clip**  
   - `spike_risk = 0.986`，建议加 ±2σ 边界的 hard clip。
   - 配合 `right_gripper` 的 spike_risk = 0.058（很干净）做差异化阈值。

3. **右臂 Q3 段限速**  
   - 右臂 Q3 RMSE 偏高（0.057），可以在 episode 中后段对右臂加低通滤波或动作幅度限速。

4. **chunk horizon 截断（可选）**  
   - 即使 chunk decay 大幅改善，offset 13-15 RMSE 仍最高。
   - 仍建议用 receding horizon，每次只执行 chunk 前 8-10 步，避免末端误差累积。

5. **Q1 段保守速度**  
   - 整段 episode 的 Q1 段（前 25% 时间）使用更低的执行速度，给 gripper 起始误差留缓冲。

### 9.3 二选模型（如果不允许 gripper 外部约束）

如果系统不允许加任何 gripper 安全约束，必须裸跑全 chunk：

**改用 `rldx_1`**。理由：

- `gripper Safety = 98.88`，spike_risk = 0.02，是所有模型里 gripper 最干净的。
- `gripper chunk_decay = 0` 是问题，但因为 spike 几乎不发生，真机风险低于 fastwam 裸跑。
- 代价：long-horizon 末端 RMSE 更高，但不至于直接 estop。

---

## 10. 最终结论

| 部署条件 | 推荐模型 | 备注 |
|---|---|---|
| **能加 gripper 安全约束（推荐路径）** | **`fastwam (新, 0526)`** | closed-loop 综合最强 |
| 不能加任何外部约束，必须裸跑 | `rldx_1` | gripper spike 最少，整体最保守 |
| chunk size = 10 的短 horizon 系统 | `molmoact2` | 但仍需 gripper 状态机 |
| 任何场景 | 都不建议 `pi0_5` / `fastwam (旧)` | 无领先项 |

**最终建议：以 `fastwam (新, 0526)` 作为 16-step 真机部署首选模型，配合 §9.2 的左 gripper 安全约束清单上线；并行保留 `rldx_1` 作为安全 fallback。**
