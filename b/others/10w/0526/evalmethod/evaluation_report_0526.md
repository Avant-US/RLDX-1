# Offline Policy 评估报告

## 1. 评估结论摘要

本次评估比较了 `rldx_1`、`pi0_5`、`fastwam`、`molmoact2` 四个策略模型，评估结果来自 `offline_policy_result` 下的 v2 指标文件，重点区分了 open-loop 模仿质量与 closed-loop 部署风险。

总体结论：

| 模型 | PQS Open | PQS Closed | 结论 |
|---|---:|---:|---|
| `molmoact2` | 90.76 | **78.91** | closed-loop 综合最高，chunk decay 最好，但 gripper safety 仍是最大风险 |
| `rldx_1` | 91.14 | 76.86 | 跟踪、安全性均衡，是更保守的部署候选 |
| `pi0_5` | 91.10 | 75.82 | open-loop 接近 `rldx_1`，但闭环 decay 与 gripper 风险偏弱 |
| `fastwam` | **91.38** | 75.68 | open-loop 最高，但闭环受 gripper spike 与左臂 decay 拖累 |

如果目标是**离线模仿验收 / checkpoint 粗筛**，`fastwam` 的 open-loop 分数略高，但优势很小；`molmoact2` 的 all open-loop 略低，主要被 gripper 拉低，双臂单独看并不差。如果目标是**chunk 闭环部署优先**，`molmoact2` 的 `PQS Closed` 最高，主要胜在 chunk decay；但需要注意它的 chunk size 只有 10，而其它模型是 16，因此它的 closed-loop 分数不能简单理解为“完整 16-step horizon 更好”。若采用更保守的安全优先策略，`rldx_1` 仍是稳定候选。

## 2. PQS 计算方式

PQS 是 Policy Quality Score 的缩写，用来把多个离线评估指标合成为 0-100 分的综合分数。分数越高表示策略质量越好。v2 评估里不再只给一个总分，而是拆成两套口径：

- `PQS Open`：偏向 open-loop 模仿质量，用于训练验收、checkpoint 对比。
- `PQS Closed`：偏向 closed-loop 部署风险，用于真机部署候选筛选。

### 2.1 原子指标如何转成分数

大部分误差类指标会先归一化到 0-1 区间，再转成 0-100 分：

```text
score = 100 * clamp(1 - normalized_error, 0, 1)
```

因此误差越小，分数越高；当归一化误差接近 0 时，分数接近 100。

各模块分数的计算方式如下：

| 模块 | 计算方式 | 含义 |
|---|---|---|
| `tracking_score` | `score((nrmse_open + ndtw + p95_norm) / 3)` | 预测动作与 GT 的接近程度 |
| `smoothness_score` | `score((mean_jerk + peak_jerk + tv_norm) / 3)` | 动作平滑性，包含平均 jerk、峰值 jerk 和总变化量 |
| `stability_score` | `score(excess_flip_rate)` | 预测相对 GT 多出来的方向翻转程度 |
| `safety_score` | `0.5 * score(excess_flip_rate) + 0.5 * score(spike_risk)` | 同时考虑多余翻转和尖峰风险 |
| `chunk_decay_score` | `100 * clamp(1 - (decay_ratio - 1) / 3, 0, 1)` | chunk 末端相对起点的误差衰减程度 |
| `task_score` | `100 * (0.5 * success + 0.5 * clamp(1 - final_err_norm, 0, 1))` | 是否完成任务以及最终误差 |

其中 `decay_ratio = rmse_at_max_offset / rmse_at_0`。当末端误差是起点误差的 4 倍时，`chunk_decay_score` 会降到 0。

### 2.2 PQS Open

`PQS Open` 更强调模仿精度，公式为：

```text
PQS Open =
  0.45 * tracking_score
+ 0.25 * smoothness_score
+ 0.15 * stability_score
+ 0.15 * task_score
```

这套权重里 tracking 占比最高，所以它适合回答“模型是否像 GT”这个问题，但不能充分反映闭环部署时的误差累积。

### 2.3 PQS Closed

`PQS Closed` 更强调部署安全性和 chunk 末端退化，公式为：

```text
PQS Closed =
  0.15 * tracking_score
+ 0.25 * chunk_decay_score
+ 0.20 * safety_score
+ 0.20 * smoothness_score
+ 0.20 * task_score
```

这套权重把 `chunk_decay_score` 放到最高权重，因此即使一个模型 open-loop 跟踪很好，只要 chunk 后半段误差增长快，closed-loop 分数也会被明显拉低。

注意：如果某个模型文件只有单步 open-loop 输出、没有 chunk offset，本脚本可以把它适配为 `offset=0`。这种情况下无法真实计算 chunk 末端衰减，`chunk_decay_score` 会因为没有衰减轴而等于 100，因此 `PQS Closed` 需要单独解读。本次更新后的 `molmoact2` 已包含 `chunk_offset`，但 chunk size 是 10，其它模型是 16；因此它可以用于 chunk 口径评估，但 `PQS Closed` 与 16-step 模型比较时要考虑 horizon 长度差异。

### 2.4 All 分数如何聚合

单个 group 的 PQS 先由该 group 下所有 signal 的模块分数平均得到；`all` 不是简单把所有 signal 直接平均，而是按 group family 加权聚合，避免 gripper 的离散动作量级压过 arm 指标。

默认 family 权重为：

| Family | 权重 |
|---|---:|
| arm | 0.50 |
| gripper | 0.15 |
| torso | 0.15 |
| chassis | 0.15 |
| action | 0.05 |

本次结果主要包含 `left_arm`、`right_arm`、`gripper` 三组，因此 `all` 分数会按 `left_arm=0.50`、`right_arm=0.50`、`gripper=0.15` 再归一化后聚合。这样可以让双臂指标在总体评分中占主导，同时保留 gripper 风险对总分的影响。

## 3. 总体指标对比

| 模型 | Tracking | Smoothness | Stability | Safety | Chunk Decay | Task |
|---|---:|---:|---:|---:|---:|---:|
| `rldx_1` | **96.30** | 74.46 | 94.88 | **92.04** | **36.72** | 99.69 |
| `pi0_5` | 96.13 | 73.87 | 96.41 | 91.76 | 33.55 | 99.42 |
| `fastwam` | 95.90 | 74.50 | 97.52 | 89.84 | 33.89 | **99.79** |
| `molmoact2` | 93.60 | **77.97** | **98.05** | 85.13 | **51.94** | 96.31 |

关键观察：

- `fastwam`、`rldx_1`、`pi0_5` 的 `PQS Open` 都在 91 分左右，差距小于 0.3 分；`molmoact2` 稍低，但仍在 90 分以上。
- `molmoact2` 的 `PQS Closed` 最高，主要来自最好的 chunk decay 和较高的 smoothness / stability；但它只有 10-step chunk，较短 horizon 会让 decay 指标相对更容易。
- `fastwam` 虽然 stability、smoothness 和 task 分数最高，但 safety 分数最低，说明它存在部署风险项。
- `molmoact2` 的 tracking 低于 `rldx_1` / `pi0_5`，safety 也低于二者，主要问题集中在 gripper。
- 除 `molmoact2` 外，其它模型的 `Chunk Decay` 分数都偏低，说明 chunk offset 增大后误差会明显累积。

## 4. 分组表现

### 4.1 Gripper

| 模型 | PQS Open | PQS Closed | Safety | Chunk Decay | 风险说明 |
|---|---:|---:|---:|---:|---|
| `rldx_1` | **90.15** | 67.03 | **98.88** | 0.00 | 安全性好，但 chunk 末端衰减严重 |
| `pi0_5` | 89.53 | 66.61 | 96.71 | 0.00 | 与 `rldx_1` 类似，decay 问题更明显 |
| `fastwam` | 88.24 | **70.96** | 73.11 | **38.36** | decay 相对好，但有明显 spike 风险 |
| `molmoact2` | 75.11 | 70.81 | 50.00 | **82.53** | gripper decay 最好，但 spike 风险最高 |

Gripper 是所有模型共同的主要风险来源。`rldx_1` 和 `pi0_5` 在 gripper 的 chunk decay score 为 0，说明 offset 增大后误差增长非常严重。`molmoact2` 的 gripper decay 最好，但 safety 只有 50.00，`left_gripper/right_gripper` 的 spike_risk 都达到 1.0。`fastwam` 也有明显 gripper spike 风险。因此 gripper 不能只看 closed 分数，必须单独加安全门限。

### 4.2 Left Arm

| 模型 | PQS Closed | Tracking | Safety | Chunk Decay |
|---|---:|---:|---:|---:|
| `molmoact2` | **77.61** | **96.76** | 90.78 | 33.63 |
| `rldx_1` | 76.58 | 95.86 | 92.40 | **33.78** |
| `pi0_5` | 74.38 | 95.98 | 92.60 | 26.32 |
| `fastwam` | 73.25 | 96.12 | **93.30** | 19.73 |

左臂上 `molmoact2` 和 `rldx_1` 非常接近，`molmoact2` closed-loop 略高，主要来自更好的 tracking；`rldx_1` 的 safety 和 chunk decay 略好。`fastwam` 左臂 tracking 不差，但 chunk decay 最弱。

### 4.3 Right Arm

| 模型 | PQS Closed | Tracking | Safety | Chunk Decay |
|---|---:|---:|---:|---:|
| `molmoact2` | **82.63** | 95.81 | 90.03 | **61.08** |
| `rldx_1` | 80.10 | **95.82** | 89.63 | 50.67 |
| `pi0_5` | 80.02 | 95.22 | 89.43 | 50.85 |
| `fastwam` | 79.54 | 95.55 | **91.41** | 46.71 |

右臂是三组中最稳定的部分。`molmoact2` 的右臂 closed-loop 最高，主要来自明显更好的 chunk decay；`rldx_1` tracking 略高但差距极小，`fastwam` safety 最好但 decay 略弱。

## 5. Chunk Offset 衰减分析

从 offset 0 到各模型最大 offset 的归一化 RMSE 变化如下。注意 `molmoact2` 的最大 offset 是 9，其它模型最大 offset 是 15：

| 模型 | All offset 0 | 最大 offset | 末端 RMSE | 增长倍数 |
|---|---:|---:|---:|---:|
| `rldx_1` | 0.0304 | 15 | 0.1030 | 3.39x |
| `pi0_5` | 0.0323 | 15 | 0.1120 | 3.47x |
| `fastwam` | 0.0403 | 15 | 0.1132 | 2.81x |
| `molmoact2` | 0.0549 | 9 | 0.1100 | 2.00x |

整体上，四个模型的误差都会随 chunk offset 增大而上升。`molmoact2` 的 horizon 目前到 offset 9，末端增长倍数约 2.00x，是四者中最缓的；但它的 chunk 更短，不能直接等价外推到 offset 15。按共同区间 offset 0-9 粗略比较，`rldx_1`、`pi0_5`、`fastwam` 的 all-group 增长倍数分别约为 3.08x、3.12x、2.54x，`molmoact2` 仍然较缓，但它 offset 0 和 offset 9 的绝对误差都更高。

按执行组看：

- `rldx_1` 左臂从 0.0317 增至 0.0849，右臂从 0.0363 增至 0.0847，双臂 decay 相对最好。
- `pi0_5` 左臂从 0.0316 增至 0.0973，右臂从 0.0413 增至 0.0969，末端 offset 误差偏高。
- `fastwam` 左臂从 0.0293 增至 0.0978，右臂从 0.0376 增至 0.0950，左臂衰减尤其明显。
- Gripper 的 offset 衰减最严重，`rldx_1` 和 `pi0_5` 的 offset 0 很低，但 offset 15 均上升到约 0.22 以上；`fastwam` 的 offset 0 已经达到 0.0881，初始动作误差偏大。
- `molmoact2` 左臂从 0.0286 增至 0.0853，右臂从 0.0378 增至 0.0752，右臂 decay 最好；但 gripper 从 0.2071 增至 0.3180，初始误差和末端误差都很高。

## 6. Phase 分析

| 模型 | 主要高误差阶段 | 说明 |
|---|---|---|
| `rldx_1` | Q3 | all-group Q3 RMSE 最高，主要来自 gripper Q3 |
| `pi0_5` | Q3 | Q3 同样是主要误差阶段，gripper 与右臂均有贡献 |
| `fastwam` | Q1 / Q2 | all-group Q1、Q2 RMSE 极高，主要由 gripper 早期误差造成 |
| `molmoact2` | Q1 / Q2 | all-group Q1、Q2 RMSE 极高，主要来自 gripper；Q2 最严重 |

`fastwam` 和 `molmoact2` 的 phase 分布最异常。`fastwam` 的 all-group Q1 和 Q2 RMSE 分别达到 1.45 和 1.33，而 gripper Q1 / Q2 分别达到 11.46 和 10.47。`molmoact2` 的 all-group Q1 / Q2 为 1.45 和 2.79，gripper Q1 / Q2 达到 11.49 和 22.15。这说明两者的 gripper 在轨迹早期存在显著偏差，不适合直接用作真机早期动作输出。

## 7. 主要风险点

1. **Chunk decay 仍是主要闭环风险。** `molmoact2` 的 0-9 decay 明显最好，但它 chunk size 只有 10；其它三个模型需要承担到 offset 15 的更长 horizon，末端误差增长更明显。
2. **Gripper 是最大风险组。** gripper 的离散动作导致误差、jerk、spike 更敏感，所有模型都需要单独处理 gripper 的决策与平滑逻辑。
3. **`fastwam` 存在 safety spike 风险。** 特别是 `left_gripper` 出现 spike_count=2，spike_risk 接近 0.97，虽然它 open-loop 总分最高，但真机部署风险不能忽略。
4. **`molmoact2` 的 gripper spike 风险最高。** `left_gripper/right_gripper` 的 spike_risk 都为 1.0，且 `left_gripper` spike_count=8、`right_gripper` spike_count=2，说明 gripper 输出仍有明显尖峰。
5. **`pi0_5` 的 closed-loop 综合表现略弱。** 它 open-loop 与 `rldx_1` 接近，但 closed-loop decay 和 gripper 表现不足。
6. **单一 PQS Open 会掩盖部署风险。** 仅看 open-loop 会倾向选择 `fastwam`，但 closed-loop 指标显示 `molmoact2` 和 `rldx_1` 更值得作为部署候选；最终选择仍取决于能否约束 gripper spike。

## 8. 部署建议

推荐策略：

| 场景 | 推荐模型 | 理由 |
|---|---|---|
| 离线模仿质量对比 | `fastwam` / `rldx_1` | open-loop 分数接近，`fastwam` 略高 |
| 真机部署候选 | `molmoact2` / `rldx_1` | `molmoact2` 10-step closed-loop 最高，`rldx_1` 16-step safety 更保守 |
| 右臂动作优先任务 | `molmoact2` | 右臂 closed-loop 和 chunk decay 最优 |
| 左臂动作优先任务 | `molmoact2` / `rldx_1` | 两者左臂 closed-loop 接近 |
| gripper 敏感任务 | 暂不建议直接部署 | 四个模型均存在 gripper decay、早期误差或 spike 风险 |

具体上线建议：

- 如果部署策略使用 10-step chunk，且能先加 gripper 安全门限，`molmoact2` 是当前最强的 closed-loop 候选；如果需要 16-step chunk 或安全优先且不希望承担 gripper spike 风险，优先使用 `rldx_1`。
- 对 gripper 单独加安全门限，例如限制单步变化、加入 spike 检测、对开合状态做 hysteresis 或 action hold。
- 部署时建议采用短 horizon 或 receding horizon，只执行 chunk 前半段，避免 offset 末端误差累积。
- 对 `fastwam`，必须先排查 gripper 早期动作误差和 `left_gripper` spike，再考虑真机测试。
- 对 `molmoact2`，必须先处理 gripper spike，尤其是 `left_gripper/right_gripper` 的 spike_risk=1.0。
- 下一轮训练或调参应重点优化 chunk decay，而不是继续只优化 open-loop RMSE。

## 9. 最终结论

本次更新后，`molmoact2` 已具备 chunk 维度，并在 10-step 口径下的 `PQS Closed` 排名第一，主要优势来自更好的 chunk decay，尤其是右臂表现。但它不能直接证明在 16-step horizon 下也优于其它模型。`rldx_1` 仍然是更保守的安全候选，tracking 和 safety 更均衡。`fastwam` 可以作为 open-loop 表现较好的参考模型，但需要先解决 gripper spike 和早期 gripper 偏差；`pi0_5` 表现稳定但没有明显领先项。

最终建议：**若部署采用 10-step chunk 且能先补 gripper 安全约束，优先测试 `molmoact2`；若需要 16-step chunk 或希望风险更保守，优先测试 `rldx_1`。**
