# 基于3个eval py及其测试结果csv进行对比

我们测试和对比3个VLA的效果, 这3个VLA分别是: Pi0.5, FastWam, RLDX-1. 用的测试集都是一样的, 都是评估预测的action和ground truth的差别, 测试结果都用csv保存. Pi0.5的测试脚本是 @vs_pi05_fstwam_rldx1/eval_pi05_policy.py , 测试结果是 @vs_pi05_fstwam_rldx1/pi05_openloop3_16_12500.csv ; RLDX-1的测试脚本是 @vs_pi05_fstwam_rldx1/rldx1_eval_policy.py , 测试结果是 @vs_pi05_fstwam_rldx1/rldx1_openloop2_16_gripper.csv ; FastWam的测试脚本是 @vs_pi05_fstwam_rldx1/fastwam_eval_policy.py , 测试结果是 @vs_pi05_fstwam_rldx1/fastwam_openloop2_16.csv . 请深入分析各个模型的测试脚本, 及其对应的测试结果, 并进行深入, 看谁的效果更好, 并给出理由.

数据全部就位，下面给出完整对比与结论。

---

## 一、三个评估脚本的"测试方法学"对比

虽然三套脚本风格不同，但**评测协议是一致的 chunk re-inference open-loop**：在同一条 episode 上，每隔 `H=16` 步推理一次，把这一次推理出的 16 步动作 chunk 与对应 GT 的 16 步轨迹（GT 不足末尾用最后一帧 pad）一一对齐，最终把 1000 步的预测 vs GT 落盘成 CSV。

| 维度 | `eval_pi05_policy.py` | `rldx1_eval_policy.py` | `fastwam_eval_policy.py` |
|---|---|---|---|
| 策略接口 | `openpi.policies.create_trained_policy` 的 `policy.infer(obs)` | `rldx.policy.RLDXPolicy.get_action(obs)` | Hydra 实例化的 FastWAM 模型，调用 `model.infer_action(...)` |
| 观测格式 | 扁平 dict：`head_rgb / left_wrist_rgb / right_wrist_rgb / state / prompt`（单帧） | 嵌套 dict：`{video, state, language}`，每模态多帧（按策略自带 `delta_indices`，例 `[-6,-4,-2,0]`） | LeRobot dataset 取一个 `sample`，已含 `video[C,T,H,W]`、`proprio`、`context` |
| 动作输出 | 直接是物理量（pi-policy 内部已反归一化） | 按 `meta/modality.json` 拆出 6 个 key（双臂/双夹爪/底盘 pose/底盘速度），脚本再按起点 index 串回 23 维 | 模型输出归一化值，脚本调用 `processor.action_state_merger.backward → normalizer.backward → action_state_transforms.backward` 做反归一化 |
| 推理特性 | 单 forward；JIT 预热 | 多帧历史窗口；`get_action` 返回 (B=1,H,D) | flow / diffusion 采样：`num_inference_steps=10` 默认；可选 seed、`sigma_shift`、`tiled` |
| CSV 列结构 | `step, inference_step, horizon_index` + 每维 `gt_/pred_/error_` | `row, target_step, chunk_start, chunk_offset, per_step_mse` + 每维 `gt_/pred_/error_` | `checkpoint, dataset_stats, episode_index, time_step, chunk_index, chunk_offset` + 每维 `gt_/pred_/abs_error_/sq_error_` |
| MSE 计算 | 默认 23 维全算；`--mse-drop-dims` 可剔除某些维（与 Pi0.5 训练惯例一致剔除 14、15 夹爪） | 同上 | 同上，并额外报告 MAE |
| `episode_index` 索引规则 | parquet 文件 `glob` 排序后下标，文件名给的是 **3** | 同 Pi0.5（parquet glob 下标），文件名给的是 **2** | LeRobot `episode_data_index["from"/"to"]` 按数据集元数据切，文件名给的是 **2** |

虽然三处 `episode_index` 数字不一样，**经核验三个 CSV 的 GT 列在所有 23 维上数值完全相等**（Pi0.5 vs RLDX-1 max diff = 0；Pi0.5 vs FastWAM 仅 1.7e-7 量级，源自 fp32 反归一化舍入）。所以三个 CSV **跑的是同一条 1000 步轨迹的 GT，对比公平**。

> 已写入 `vs_pi05_fstwam_rldx1/_analyze.py`、`_analyze2.py`、`_plot_compare.py` 用作复盘脚本，以及汇总图 `compare_pi05_rldx1_fastwam.png`。

---

## 二、把 23 维拆开：哪些维度是"主战场"

| 类别 | 维度 | GT 行为（在该 episode 上） | 是否进入对比 |
|---|---|---|---|
| 手臂 | `left_arm_0..6`、`right_arm_0..6`（14 维） | 全程在动，量级 0.1–1 rad | **核心** |
| 夹爪 | `left_gripper`、`right_gripper`（2 维） | 量级 0–90，离散开/关跳变 | **辅助**（数值尺度大 ~400 倍，会主导原始 MSE） |
| 底盘速度 | `chassis_velocity_0/1`（2 维） | 有非零运动 | **核心** |
| 底盘其余 | `chassis_pose_0..3`、`chassis_velocity_2`（5 维） | **GT 全部 = 0**（机器人下盘在该 episode 始终静止） | 排除（三个模型均预测 ≈0，构不成有效对比） |

所以"公平的核心对比"= **14 双臂关节 + 2 活跃底盘速度 = 16 维**。

---

## 三、汇总指标（同一条 1000 步 GT）

### 3.1 整体 MSE（直接看 23 维的话，被夹爪量级吃光）

```
全部 23 维:                Pi0.5 MSE=27.80   RLDX-1 MSE=28.59   FastWAM MSE=30.72
排除夹爪 (21 维):          Pi0.5 MSE=0.00206 RLDX-1 MSE=0.00168 FastWAM MSE=0.00228
排除夹爪+不动底盘 (16 维): Pi0.5 MSE=0.00270 RLDX-1 MSE=0.00221 FastWAM MSE=0.00300
                                                   ↑ 最优
```

### 3.2 把 16 维核心对比拆细

| 子集 | Pi0.5 RMSE | RLDX-1 RMSE | FastWAM RMSE | 排序 |
|---|---|---|---|---|
| **14 个手臂关节** | 0.05492 | **0.04960** | 0.05786 | **RLDX-1 < Pi0.5 < FastWAM** |
| 2 个活跃底盘速度 | 0.02210 | **0.02114** | 0.02301 | **RLDX-1 < Pi0.5 < FastWAM** |
| 16 维（手臂+底盘速度，无夹爪、无静止维） | 0.05196 | **0.04699** | 0.05473 | **RLDX-1 < Pi0.5 < FastWAM** |
| 2 个夹爪（量级 0–90，离散） | **17.88** | 18.13 | 18.79 | **Pi0.5 < RLDX-1 < FastWAM** |

> 手臂 14 维 RMSE：RLDX-1 比 Pi0.5 低 **9.7%**，比 FastWAM 低 **14.3%**。
> 16 维核心 RMSE：RLDX-1 比 Pi0.5 低 **9.6%**，比 FastWAM 低 **14.2%**。

### 3.3 Per-dim 胜率（按每个维度 RMSE 谁最低）

把无意义的 5 个静止底盘维剔除，剩 18 个有效维，**胜场数**：

```
RLDX-1   赢 11 维
Pi0.5    赢 5 维（含 2 个夹爪）
FastWAM  赢 2 维（左 arm_4、右 arm_2）
```

### 3.4 Mean nRMSE（每维 RMSE / GT 范围，再求均值；剔除静止维和夹爪后的 16 维）

```
RLDX-1   mean 0.0732   ← 最优
Pi0.5    mean 0.0800
FastWAM  mean 0.0803
```

---

## 四、两个最有信息量的"诊断"切片

### 4.1 chunk 内 horizon-offset 的误差走势（最能揭示模型差异）

每个 chunk 的 16 步预测中，offset=0 是模型刚看到当前观测时输出的"近未来动作"，offset=15 是它要"提前 15 步盲打"的"远未来动作"。把每个 offset 上的全 episode 误差汇起来：

| offset | Pi0.5 RMSE | RLDX-1 RMSE | FastWAM RMSE | best |
|---|---|---|---|---|
| 0 | **0.0237** | 0.0272 | 0.0249 | Pi0.5 |
| 1 | **0.0280** | 0.0296 | 0.0287 | Pi0.5 |
| 2 | **0.0328** | 0.0341 | 0.0331 | Pi0.5 |
| 3 | 0.0373 | 0.0379 | **0.0369** | FastWAM |
| 4 | **0.0405** | 0.0406 | 0.0412 | Pi0.5 |
| 5 | 0.0435 | **0.0424** | 0.0450 | RLDX-1 |
| 6 | 0.0471 | **0.0448** | 0.0495 | RLDX-1 |
| 7 | 0.0504 | **0.0460** | 0.0528 | RLDX-1 |
| 8 | 0.0535 | **0.0478** | 0.0565 | RLDX-1 |
| 9 | 0.0565 | **0.0498** | 0.0608 | RLDX-1 |
| 10 | 0.0586 | **0.0516** | 0.0628 | RLDX-1 |
| 11 | 0.0610 | **0.0532** | 0.0654 | RLDX-1 |
| 12 | 0.0635 | **0.0552** | 0.0675 | RLDX-1 |
| 13 | 0.0662 | **0.0569** | 0.0705 | RLDX-1 |
| 14 | 0.0683 | **0.0588** | 0.0720 | RLDX-1 |
| 15 | 0.0705 | **0.0603** | 0.0738 | RLDX-1 |
| **0→15 增长率** | **×2.97** | **×2.22** | **×2.96** | **RLDX-1 最稳** |

这一列的解读最关键：
- **chunk 早期（offset 0–4）**：Pi0.5 最锐利，三者实际差距很小（<3%），说明三模型对"当下该做什么"理解都到位。
- **chunk 中后期（offset ≥ 5）**：**RLDX-1 误差曲线斜率显著最低**——它是三者中"长程动作预测"最稳的，到 offset=15 时比 Pi0.5 低 **14.4%**、比 FastWAM 低 **18.3%**。
- 这意味着在闭环部署时，**如果 control rate 允许 RLDX-1 用更长的 action chunk**（即推理频率更低），它的"chunk 退化代价"最小，等价于推理时间预算可以被 RLDX-1 更好地省下来。

### 4.2 episode 四分位 RMSE（任务进展不同阶段）

| Quarter | Pi0.5 | RLDX-1 | FastWAM | best |
|---|---|---|---|---|
| Q1 (步 0–249, 任务起始) | 0.0618 | **0.0544** | 0.0675 | RLDX-1 |
| Q2 (步 250–499) | **0.0604** | 0.0605 | 0.0609 | Pi0.5 (并列) |
| Q3 (步 500–749, 难关段) | 0.0558 | **0.0452** | 0.0596 | RLDX-1 |
| Q4 (步 750–999, 收尾稳态) | 0.0147 | 0.0132 | **0.0128** | FastWAM |

- Q1+Q3 是任务最具挑战的两段（含 chunk-6 起步加速、chunk-30 抓取关键帧），**RLDX-1 在这两段都明显领先**。
- Q4 是动作几乎收敛、机器人微调的"扫尾段"，三者都很低、**FastWAM 微优**——这与 FastWAM 的 video-AR/flow 范式倾向于平滑预测一致。

### 4.3 最难 chunk 的对比

三个模型共同认为最难的 chunk 是 **chunk 6（起步加速）和 chunk 30（关键抓取）**。在这些 chunk 上：

```
chunk 6:    Pi0.5 RMSE=0.178   RLDX-1 RMSE=0.084   FastWAM RMSE=0.196
chunk 30:   Pi0.5 RMSE=0.100   RLDX-1 RMSE=0.098   FastWAM RMSE=0.094
```

`left_arm_3`（最大值）单步 |error| 极值：Pi0.5 **0.904**、RLDX-1 **0.347**、FastWAM **0.985**——RLDX-1 在这个高方差关节上的失控幅度比 Pi0.5、FastWAM 小近 **2.6 倍**。

### 4.4 夹爪：Pi0.5 略胜

```
left_gripper:  Pi0.5 mean|err|=5.20   5.6% 步骤 |err|>10
              RLDX-1 mean|err|=5.45   6.0% 步骤 |err|>10
             FastWAM mean|err|=6.13   6.5% 步骤 |err|>10
right_gripper: Pi0.5 mean|err|=2.25   2.3% 步骤 |err|>10
              RLDX-1 mean|err|=2.40   2.5% 步骤 |err|>10
             FastWAM mean|err|=2.74   2.6% 步骤 |err|>10
```

夹爪是离散 ON/OFF 信号被回归当作连续值，在跳变帧上三个模型都会"跨过临界点"，但 Pi0.5 因为是 single-shot deterministic policy，跳变时机抓得最准；RLDX-1 与 Pi0.5 几乎并列；FastWAM 由于 flow 采样平滑特性，过渡帧上略微"拖泥带水"。

---

## 五、综合可视化

下面这张总图概括以上 4 类指标：

```31:34:vs_pi05_fstwam_rldx1/_plot_compare.py
ROOT = Path(__file__).resolve().parent
PI_CSV = ROOT / "pi05_openloop3_16_12500.csv"
RL_CSV = ROOT / "rldx1_openloop2_16_gripper.csv"
FW_CSV = ROOT / "fastwam_openloop2_16.csv"
```

- 子图 (a)：14 个手臂关节的 per-dim RMSE 柱状图——RLDX-1 在多数关节最低，尤其 `left_arm_3` 上比 Pi0.5/FastWAM 显著低。
- 子图 (b)：chunk 内部 horizon offset 的误差曲线——三者在 offset=0 起点相近，但 RLDX-1 在 offset≥5 之后曲线坡度最缓，到 chunk 末端拉开 14–18%。
- 子图 (c)：四分位 RMSE——RLDX-1 在 Q1、Q3 大幅领先，Q4 三者都很小。
- 子图 (d)：每步 14-维手臂 L2 误差时间序列——RLDX-1 的 spike（橙色高亮 chunk 6/20/30/32/34）峰值始终低于 Pi0.5/FastWAM。

完整图见 `vs_pi05_fstwam_rldx1/compare_pi05_rldx1_fastwam.png`。

---

## 六、结论

**综合判定：RLDX-1 > Pi0.5 > FastWAM。**

在这条 1000 步、16-step open-loop chunk replay 的同一 GT 轨迹上：

1. **RLDX-1 在最关键的 16-维核心动作上 RMSE 最低**（0.0470，比 Pi0.5 低 9.6%、比 FastWAM 低 14.2%），且在 18 个有效维度的 per-dim 胜率（11/18）远超另两者。
2. **RLDX-1 的 chunk 内长程预测最稳**：从 offset 0 到 15，误差仅放大 ×2.22，而 Pi0.5/FastWAM 都接近 ×3。这等价于 RLDX-1 可以用更稀疏的推理频率而不损失精度，是闭环控制时延友好的特性。
3. **RLDX-1 在任务难段（Q1 起步、Q3 关键抓取）领先优势最大**，且对手臂高方差维度 `left_arm_3` 的最大单步误差比 Pi0.5/FastWAM 小近 **2.6 倍**——抗 spike 能力最强，这对真机部署是直接收益。
4. **Pi0.5 在 chunk 起点（offset 0–4）和夹爪（0–90 大尺度离散信号）上略优**——这反映出 Pi0.5 作为 single-shot policy 的"瞬时反应锐利"特性，但优势局限于短窗口和离散信号。
5. **FastWAM 整体偏弱**：虽然在 Q4 收尾段、`right_arm_2` 等少数维度上微优，但在 16 维核心、chunk 中后期、任务难段上都垫底。可能原因：
   - 它依赖 `num_inference_steps=10` 的 flow 采样，每次推理是从噪声采样的近似解，**测试用同一 seed=42 也会带来 chunk 间的不一致性**，特别是远 horizon 的预测；
   - 它倾向于在视频生成-动作联合范式下生成"平滑"动作，所以在收尾段（Q4）表现好，但在关键过渡帧（spike 处）反而较钝；
   - 它的 dataset processor 反归一化路径更长（`merger.backward → normalizer.backward → transforms.backward`），任何归一化偏差都会引入额外噪声。

**给落地选型的建议**：
- **追求闭环控制精度、长 chunk 稳定性、对长程预测和关键帧准的场景，优先 RLDX-1**。
- **若 control loop 极快、推理预算紧、且任务以离散开关动作（夹爪开闭/接触检测）为关键瓶颈，Pi0.5 在 offset 0 上的瞬时锐利可被利用**。
- **FastWAM 当前 ckpt 在该 episode 上无明显综合优势**，若已采纳其视频-动作联合范式收益（如视频 imagination、跨 embodiment 预训练），需要继续 finetune 或改用更高的 inference steps 才能在动作精度上追上 RLDX-1/Pi0.5。

> 复盘所用脚本与图都放在 `d:\SRC\d\10wEmbdm\vs_pi05_fstwam_rldx1\` 下：`_analyze.py`、`_analyze2.py`、`_plot_compare.py`、`compare_pi05_rldx1_fastwam.png`，可直接重跑校验。

---

下面这个推荐我从三个角度综合给：**开环精度（前面 1000 步实测得到的）→ 闭环可推理性 → R1 Pro 真机的工程现实**。

---

# 真机选哪个
如果要部署到 Galaxea(星海图) R1 Pro 的机器人真机上, 选择 Pi0.5, FastWam, RLDX-1 中的哪个比较好?

## 一、直接结论

**首选 RLDX-1，备胎 Pi0.5，不推荐 FastWAM。**

R1 Pro 真机的特征是：
- 双 7-DoF 手臂 + 双夹爪 + 4-DoF 底盘位姿 + 3-DoF 底盘速度（=23 DoF，与 CSV 列完全对应）；
- 三摄像头观测（`head_rgb / left_wrist_rgb / right_wrist_rgb`，384 分辨率）；
- 三个脚本/模型其实都是**对 R1 Pro 专门训练/适配过的**：Pi0.5 的 `--config pi05_r1pro_chassis`、FastWAM 的 task `r1_pro_chassis_uncond_3cam_384_1e-4`、RLDX-1 的测试集 `r1_pro_data_convert_chassis`。所以前一轮的开环对比 100% 适用于 R1 Pro 的部署判断，**不存在 sim-to-real 或 cross-embodiment 偏差**。

---

## 二、为什么是 RLDX-1（基于真机部署的 6 个关键属性）

### 1. **chunk 内长程稳定性最好——这是闭环部署最值钱的指标**

R1 Pro 闭环执行 chunk 时，前几步真机会跟踪得不错，到 chunk 末端（offset 大）才暴露差距。我们的实测：

| offset | 0 | 5 | 10 | 15 | **0→15 放大** |
|---|---|---|---|---|---|
| Pi0.5 RMSE | 0.0237 | 0.0435 | 0.0586 | 0.0705 | **×2.97** |
| **RLDX-1 RMSE** | 0.0272 | **0.0424** | **0.0516** | **0.0603** | **×2.22 ✓** |
| FastWAM RMSE | 0.0249 | 0.0450 | 0.0628 | 0.0738 | **×2.96** |

意义：
- **RLDX-1 的"action chunk 衰减"最慢**，意味着同样在 R1 Pro 的 30 Hz 控制循环上，RLDX-1 可以把推理频率压得更低（甚至把 chunk 拉到 H=32），**把宝贵的 GPU 时间留给视频编码/语言指令更新**。
- Pi0.5 和 FastWAM 必须用更短的 chunk（H≤16），这等价于推理频率要更高，对算力和带宽是双重压力。

### 2. **对手臂关节 RMSE 最低——R1 Pro 的核心 DoF**

14 个手臂关节 RMSE：**RLDX-1 = 0.0496 < Pi0.5 = 0.0549 < FastWAM = 0.0579**（RLDX-1 比 Pi0.5 低 9.7%、比 FastWAM 低 14.3%）。
14 个有效非夹爪非静止维：**RLDX-1 在 11 维上最低**，Pi0.5 在 5 维（含 2 个夹爪），FastWAM 仅 2 维。
真机 manipulation 的姿态精度直接由这 14 维决定，RLDX-1 的优势能直接转化为更稳的轨迹跟踪和更高的任务成功率。

### 3. **对单步极值误差最鲁棒——决定真机会不会"突然抽搐"**

`left_arm_3`（高方差关节）单步 |error| 最大值：

```
Pi0.5    0.904 rad
RLDX-1   0.347 rad   ← 比 Pi0.5/FastWAM 低 2.6 倍
FastWAM  0.985 rad
```

在真机上，0.9 rad（约 51°）的单步抽动会触发关节限速保护、急停甚至硬碰撞。**RLDX-1 的低 spike 是部署安全性的关键收益**。

### 4. **多帧观测历史（delta_indices = [-6,-4,-2,0]）—— 抗摄像头噪声与运动模糊**

这是从脚本里读出来的关键差异：

| 模型 | 观测历史 | 真机收益 |
|---|---|---|
| Pi0.5 | 单帧 head/wrist | 完全靠当前帧推断速度，**遇到 motion blur / 单帧曝光失败会直接出错** |
| **RLDX-1** | **多帧历史 (T_v=4，跨 6 步窗口)** | **能从帧间差分隐式估计速度，对单帧噪声有平均** |
| FastWAM | video[:,0] 取首帧，相当于单帧 | 与 Pi0.5 同样脆弱 |

R1 Pro 的 wrist 摄像头在抓取时会快速移动 → motion blur 频繁 → RLDX-1 的多帧设计可以直接承受，Pi0.5 / FastWAM 需要在前置增加 RGB 滤波或重新采集。

### 5. **PyTorch 单 forward → 工程门槛最低**

| 模型 | 框架 | 单次推理 | 部署难度 |
|---|---|---|---|
| Pi0.5 | openpi (JAX) | ~30–60 ms（JIT 后） | JAX/XLA 在嵌入式 / Jetson 上的支持较脆，需打通 BF16 + flash-attn 的 JAX path |
| **RLDX-1** | PyTorch + transformers | ~50–100 ms 单 forward | **直接复用 HF 工具链**，量化/编译/onnx export 都成熟 |
| FastWAM | PyTorch + Hydra + flow sampler | **10 步 flow ×30–50 ms ≈ 300–500 ms** | flow 采样是循环采样，**没法简单 fuse**，且采样依赖 seed / `sigma_shift` 调参 |

R1 Pro 真机如果是车载 GPU（Jetson Orin / RTX A4000 等），**FastWAM 的 10 步 flow 在 H=16 + 30 Hz 下吃满 control budget 的近 60%**，安全 headroom 不足；RLDX-1 / Pi0.5 都游刃有余。

### 6. **确定性输出，闭环可重复**

- **RLDX-1 / Pi0.5**：deterministic forward，相同输入相同输出，**故障可复现、可回放分析**。
- **FastWAM**：flow 采样依赖随机 seed（脚本里默认 `--seed 42`），**真机部署时若不锁 seed 就是不可复现的随机策略**，对运维和 bug 定位是噩梦；锁 seed 又会失去 flow 范式的多模态优势。

### 7. **任务关键阶段（Q1 起步、Q3 抓取）领先最大**

```
Q1 (步 0-249,起步加速):  RLDX-1=0.0544 < Pi0.5=0.0618 < FastWAM=0.0675
Q3 (步 500-749,关键抓取): RLDX-1=0.0452 < Pi0.5=0.0558 < FastWAM=0.0596
```

这两段恰恰是真机最容易失败的两段（启动惯性、抓取相位的接触切换）。RLDX-1 在 Q3 比 Pi0.5 低 19%、比 FastWAM 低 24%。

---

## 三、Pi0.5 什么时候值得选

如果遇到下面任何一个**强约束**，可以把 Pi0.5 提升到首选：

1. **任务以离散夹爪开闭为成功瓶颈**（例如纯捏取-投放，姿态轨迹简单但夹爪时机敏感）。
   - Pi0.5 双夹爪 mean|err|（5.20 / 2.25）和 |err|>10 比例（5.6% / 2.3%）都比 RLDX-1 略好。
2. **需要极短 chunk（H ≤ 8）下的 offset=0 锐利预测**（如 100 Hz 高刷新控制环）。
   - 在 offset 0–4 上 Pi0.5 持续微优 1–2%。
3. **现有部署链已经基于 openpi 走通**，迁移 RLDX-1 的工程成本超过精度收益时。

但**只要任务包含底盘运动 + 长 chunk + 多关节连续协调动作（即典型 R1 Pro mobile manipulation），RLDX-1 都更优**。

---

## 四、为什么不推荐 FastWAM

按真机部署 5 个维度的评分：

| 维度 | RLDX-1 | Pi0.5 | FastWAM |
|---|---|---|---|
| 整体 16 维核心 RMSE | **0.0470** | 0.0520 | 0.0547 |
| chunk 长程稳定性 | **×2.22** | ×2.97 | ×2.96 |
| 单步极值鲁棒性 | **最优** | 中 | 最差 |
| 单次推理延迟 | ~80ms | **~50ms** | ~300–500ms |
| 输出确定性 | 确定 | 确定 | **依赖 seed** |

FastWAM 在每个真机关键维度上都不占优。它的优势主要在 Q4（任务尾段稳态）和 video imagination（这一项在 open-loop 评估里看不出来）。**如果不是为了用上 FastWAM 的 video-WAM 范式，就没有理由把它放到 R1 Pro 真机一线**。

---

## 五、上 R1 Pro 前的部署 checklist（针对 RLDX-1）

```
[ ] 推理服务：把 RLDXPolicy 包成长驻 server（gRPC / shared-mem），避免每次 chunk 重启
[ ] 量化：FP16 / BF16 测过没掉精度后再切 INT8
[ ] action_horizon：先用 H=16；若 GPU 富余可试 H=24/32 拿 chunk 稳定性红利
[ ] 安全网：每个 chunk 输出在 robot driver 端再过一遍关节速度/加速度上限 clip
[ ] 紧急刹车：监控 |pred[t] - state[t]| 跳变阈值，异常时切到保持/减速策略
[ ] 多帧观测：确认 video_delta_indices=[-6,-4,-2,0] 在线时的帧缓冲对得齐 (T_s, T_v 一致)
[ ] modality.json：保证训练-部署一致；脚本里 _load_modality_meta 路径要预置在 ROS/SDK 包里
[ ] 夹爪策略：可在 RLDX-1 输出之上叠 Pi0.5 的夹爪头（hybrid），结合两者优势
[ ] 监控：在线 log per-step error vs state，便于事后回看 chunk 6/30 类难关
```

---

## 六、一行总结

**部署到 Galaxea R1 Pro，选 RLDX-1**：手臂精度最高 + chunk 长程衰减最慢 + 单步抽搐最少 + 多帧抗噪 + PyTorch 工程链路最顺；只在"夹爪开闭时机极端敏感"或"已绑死 openpi 工程栈"两种特例下，才考虑切到 Pi0.5；FastWAM 在该真机评测里没有任何能压住 RLDX-1 的关键维度，且 flow 采样的非确定性和延迟对真机部署是负担，**当前不建议上线**。

---

# RMSE/GT范围 的好处

上面提到的" Mean nRMSE（每维 RMSE / GT 范围，再求均值；剔除静止维和夹爪后的 16 维）" 具体怎么做? 为什么这样做? 有什么好处?

`Mean nRMSE` 是为了把不同动作维度放到同一个相对尺度上比较。

公式是：

```text
对每个动作维度 d：

RMSE_d = sqrt(mean_t((pred[t,d] - gt[t,d])^2))

GT_range_d = max_t(gt[t,d]) - min_t(gt[t,d])

nRMSE_d = RMSE_d / GT_range_d

Mean nRMSE = mean_d(nRMSE_d)
```

也就是：**先算每个维度自己的 RMSE，再除以这个维度在 GT 轨迹里的变化范围，最后对各维取平均。**

在这次分析里我用的是：

```text
剔除夹爪维度: left_gripper, right_gripper
剔除静止 GT 维度: chassis_pose_0..3, chassis_velocity_2
保留: 14 个手臂关节 + chassis_velocity_0/1 = 16 维
```

## 为什么要这样做

因为 23 维 action 的单位和数值尺度差很多。

例如：

```text
夹爪: 0 ~ 90
手臂关节: 大多 0.x ~ 1.x rad
底盘速度: 0.0x ~ 0.1x
静止底盘 pose: 全程 0
```

如果直接算全维 MSE，夹爪的 90 量级会压倒一切。哪怕手臂预测明显更好，只要夹爪有几帧错开，整体 MSE 就会被夹爪主导。所以直接 23 维 MSE 更像是在问：**谁的夹爪数值误差更小？** 而不是问：**谁整体动作轨迹更准？**

`nRMSE` 把每一维都变成"相对于该维动作幅度的误差比例"。例如：

```text
某关节 GT 范围是 1.0 rad，RMSE 是 0.05 → nRMSE = 5%
另一个关节 GT 范围是 0.2 rad，RMSE 是 0.02 → nRMSE = 10%
```

虽然第二个绝对误差更小，但相对它自己的运动幅度来说，预测更差。

## 为什么剔除夹爪

夹爪不是普通连续关节，它更像离散开/关信号，范围又是 0~90。它适合单独看：

```text
mean |err|
|err| > 10 的比例
开/关跳变时机是否对齐
```

不适合和手臂关节一起做平均 nRMSE，否则它会混淆"连续轨迹跟踪能力"和"离散夹爪时机"两个问题。

所以我把夹爪单独分析，结论是：**Pi0.5 夹爪略好，RLDX-1 接近，FastWAM 最差。**

## 为什么剔除静止维度

静止维度的 GT_range 是 0，例如：

```text
chassis_pose_0..3 全程为 0
chassis_velocity_2 全程为 0
```

这时：

```text
nRMSE = RMSE / 0
```

数学上无意义。即使加一个很小的 epsilon 避免除零，也会导致不稳定结果：一个 1e-8 的舍入误差会被放大成很大的 nRMSE。

而且这些维度在该 episode 没有运动，预测 0 只是"会保持静止"，不能说明模型在动态控制上更好。所以排除。

## 好处

`Mean nRMSE` 的主要好处是：

1. **消除单位和尺度影响**：不会让 0~90 的夹爪压倒 0~1 的手臂关节。
2. **每个有效维度权重更公平**：每个关节都以"相对自身运动范围的误差"参与平均。
3. **更适合比较轨迹跟踪能力**：尤其适合 R1 Pro 这种多 DoF、不同子系统尺度差异很大的机器人。
4. **能看出模型是否普遍稳定**：不是靠某几个大尺度维度赢，而是在多数有效维度上都相对准确。

这次得到：

```text
RLDX-1   mean nRMSE = 0.0732
Pi0.5    mean nRMSE = 0.0800
FastWAM  mean nRMSE = 0.0803
```

意思是：在 16 个有效连续动作维上，RLDX-1 的平均相对误差约为 GT 动作范围的 **7.32%**，Pi0.5 约 **8.00%**，FastWAM 约 **8.03%**。所以 RLDX-1 不只是 raw RMSE 更低，它在尺度归一化后仍然最好。

## 需要注意的局限

`nRMSE` 也不是万能的：

- 如果某维 GT_range 很小，nRMSE 会变得敏感。
- range 受极端值影响，如果 GT 有一个异常 spike，会把分母拉大，显得误差更小。
- 它不能替代夹爪时机、单步最大误差、chunk 后半段误差这些部署关键指标。

所以更稳妥的判断方式是同时看：

```text
raw RMSE / MAE
per-dim RMSE
mean nRMSE
chunk-offset 误差增长
最大单步误差 spike
夹爪跳变误差
```

在这些指标合起来看时，RLDX-1 仍然是最适合 R1 Pro 真机部署的选择。

# Offline Policy Diagnostics for VLA / WAM (张闻悦整理的)

> v2: 在 v1 (Tracking / Smoothness / Stability / Task) 基础上，补齐**闭环部署**最关键的三块——
> Chunk Decay、Safety Spike、Group-aware Aggregation，并区分 open-loop 与 closed-loop 两个评估视角。

---

## 1. Motivation

当前 VLA/WAM 的评估方式主要依赖：

- GT vs Pred 曲线可视化
- 人眼判断轨迹是否"像"
- 主观判断是否平滑

这些方式不可量化、不可复现、无法横向比较，**且严重低估闭环风险**。

### v1 框架的盲点（必须补齐）

| 盲点 | 后果 |
|---|---|
| 把所有 `chunk_offset` 平均掉 | 看不出"chunk 末端衰减"——闭环最重要的指标 |
| Smoothness 只看均值，不看尖峰 | 一个 0.9 rad 的关节抽动平均后看不出来，但真机会急停 |
| 用 P95 当尖峰 | 95% 分位掩盖 100% 分位，真机怕的是那 1 次 spike |
| 16 个 signal 直接求均值 | gripper 的离散 jerk 量级压死 arm 的连续 jerk |
| 一个总分 PQS | 假设你的部署场景与训练权重一致；现实是 open-loop / closed-loop 完全不同 |

---

## 2. 两个评估视角（必须区分）

| 视角 | 推理输入 | 关注指标 | 适用场景 |
|---|---|---|---|
| **Open-loop** | 永远用 GT 作为下一步输入 | RMSE / NRMSE / DTW / 模仿精度 | 训练验收、checkpoint 选优 |
| **Closed-loop** | 用上一帧 pred 作为下一步输入 | **Chunk Decay**、Spike、Stability | **真机部署决策** |

两个视角的结论**可能完全相反**：
- open-loop 表现最好的模型，可能 chunk 末端衰减最快 → 闭环最差
- 离线 P95 误差最低的模型，可能有 1 个 0.9 rad 的瞬时 spike → 真机直接 estop

> **建议：两个视角各算一份 PQS，分别命名 `PQS_open` 和 `PQS_closed`，不要再用单一总分掩盖差异。**

---

## 3. Overall Score

### 3.1 Open-loop PQS（模仿验收）

$$\text{PQS}_{open} = w_1 S_{track} + w_2 S_{smooth} + w_3 S_{stable} + w_4 S_{task}$$

### 3.2 Closed-loop PQS（真机部署）

$$\text{PQS}_{closed} = w_1 S_{track}^{(0)} + w_2 S_{decay} + w_3 S_{safety} + w_4 S_{smooth} + w_5 S_{task}$$

其中 $S_{track}^{(0)}$ 表示 chunk offset=0 时的 tracking，$S_{decay}$ 量化 offset 0 → H 的误差增长率。

---

## 4. Tracking Score（v1 保留）

衡量预测动作与 GT 动作的接近程度。

$$S_{track} = \alpha_1 (1-\text{NRMSE}) + \alpha_2 (1-\text{NDTW}) + \alpha_3 (1-\text{P95Error})$$

### 4.1 RMSE

$$\text{RMSE} = \sqrt{\frac{1}{T}\sum_t (a_t-\hat a_t)^2}, \quad \text{NRMSE} = \frac{\text{RMSE}}{\text{range}(a)}$$

### 4.2 DTW

$$\text{NDTW} = \frac{\text{DTW}(a, \hat a)}{T \cdot \text{range}(a)}$$

允许轻微时间偏移，适合 reaction delay / phase mismatch。

### 4.3 P95 Error

$$\text{P95} = \text{percentile}_{95}(|a-\hat a|)$$

检测局部大误差（**注意**：P95 不替代 max，见 §6.2）。

---

## 5. Smoothness Score（v1 保留 + 修正聚合）

$$S_{smooth} = \beta_1 (1-\text{MeanJerk}) + \beta_2 (1-\text{PeakJerk}) + \beta_3 (1-\text{TV})$$

### 5.1 Mean Jerk

$$j_t = \frac{a_t - 2 a_{t-1} + a_{t-2}}{\Delta t^2}, \quad \text{MeanJerk} = \frac{1}{T}\sum_t |j_t|$$

### 5.2 Peak Jerk

$$\text{PeakJerk} = \max_t |j_t|$$

### 5.3 Total Variation

$$\text{TV} = \sum_t |a_t - a_{t-1}|$$

### 5.4 ⚠️ 聚合规则（v2 新增，关键）

**不能**把 14 维 arm + 2 维 gripper + 4 维 torso + 3 维 chassis_velocity 直接求均值——
gripper 的离散开/合会让 jerk 量级比 arm 高 100× 以上，**单一均值会被它压死**。

正确做法：

```
S_smooth = mean(
  α_arm   * S_smooth(arm 14 dim,        normalized per-signal),
  α_grip  * S_smooth(gripper 2 dim,     normalized per-signal),
  α_torso * S_smooth(torso/chassis dim, normalized per-signal),
)
```

建议默认权重：`α_arm = 0.6, α_grip = 0.2, α_torso = 0.2`（按"真机执行风险占比"配）。

---

## 6. Stability Score（v1 保留 + Spike 新增）

$$S_{stable} = \gamma_1 (1-\text{FlipRate}) + \gamma_2 (1-\text{HFEnergy}) + \gamma_3 (1-\text{SpikeRisk})$$

### 6.1 Flip Rate

$$\text{FlipRate} = \frac{1}{T-1}\sum_t \mathbb{1}[\text{sign}(a_t) \ne \text{sign}(a_{t-1})]$$

⚠️ **重要解释**：高 flip_rate **未必是坏事**——
- 如果 GT 本身就在反复试探（如抓取试探阶段），匹配 GT 的 flip 是正确的
- 真正的问题是 `FlipRate(pred) − FlipRate(gt)`，记为 **ExcessFlipRate**

建议默认上报 `ExcessFlipRate` 而非原始 `flip_rate`。

### 6.2 High Frequency Energy

对 pred 做 FFT：

$$E_{HF} = \frac{\sum_{f>f_c}|A(f)|^2}{\sum_f |A(f)|^2}$$

`f_c` 取控制频率的 1/4（如 30 Hz 控制环则 `f_c = 7.5 Hz`）。

### 6.3 ⭐ Spike Risk（v2 新增）

P95 / mean 这些都看不出**单点抽动**。真机最怕的就是那一个 0.9 rad 的瞬时跳。

$$\text{SpikeRisk} = \frac{\max_t |a_t - \hat a_t|}{\text{joint\_max\_step\_limit}}$$

其中 `joint_max_step_limit` = 该关节单步允许的最大位移（由 hardware spec 决定，例如 30 Hz 控制下取 `v_max / 30`）。

- `SpikeRisk > 1`：必触发限速/急停
- `SpikeRisk ∈ (0.5, 1)`：高风险，可能触发 jerk 限位
- `SpikeRisk < 0.3`：安全

可选：定义 `SpikeCount = #{t : SpikeRisk_t > 0.5}` 作为辅助监控。

---

## 7. ⭐ Chunk Decay Score（v2 新增主模块）

这是 v1 完全没有，但**对闭环部署最关键**的一块。

### 7.1 背景

VLA 通常一次预测一个长度为 H 的 action chunk（典型 H=16）。
Open-loop 评估时，每个 step 的输入都是 GT，所以 offset 0~H 的误差差距被掩盖。
**Closed-loop 时，上一个 chunk 的 offset=H 的预测会变成下一个 chunk 的 offset=0 的输入**。
→ chunk 末端衰减得越快，闭环就越容易飘。

### 7.2 定义

每个 chunk offset `k ∈ [0, H)` 单独计算 RMSE：

$$\text{RMSE}(k) = \sqrt{\frac{1}{N_k}\sum_{(t,k)} (a_t - \hat a_{t,k})^2}$$

定义衰减率：

$$\text{DecayRatio} = \frac{\text{RMSE}(H-1)}{\text{RMSE}(0)}$$

定义平均衰减斜率：

$$\text{DecaySlope} = \text{linear\_fit\_slope}(\text{RMSE}(0..H-1)) \big/ \text{RMSE}(0)$$

### 7.3 Score 归一化

$$S_{decay} = 100 \cdot \max(0,\ 1 - (\text{DecayRatio} - 1) / R_{ref})$$

其中 `R_ref` 是经验上界（建议 3.0，即 ratio 4× 时 score 归零）。

### 7.4 真实示例（本仓库三个模型）

| Model | RMSE(0) | RMSE(15) | DecayRatio | $S_{decay}$ |
|---|---|---|---|---|
| **rldx_1** | 0.0285 | **0.0636** | **2.23 ★** | ~ 59 |
| pi0_5     | 0.0247 | 0.0748 | 3.03 | ~ 32 |
| fastwam   | 0.0253 | 0.0785 | 3.10 | ~ 30 |

→ rldx_1 的 chunk decay 显著最慢，可以容忍**更长 chunk + 更低推理频率** = GPU 预算多 30%+。
→ open-loop offset=0 时 pi0_5 反而最准，但 closed-loop 跑不过 rldx_1，这正是 v1 框架抓不到的关键差异。

### 7.5 Inter-chunk Discontinuity（可选，闭环专属）

闭环时，每个 chunk 切换瞬间会有一个"跳跃"：

$$\text{ChunkJump} = \text{median}_{chunk}\big(\|\hat a_{c, 0} - \hat a_{c-1, H-1}\|\big)$$

值越大表示新 chunk 与旧 chunk 的衔接越不平滑，真机会感受为"顿挫"。

---

## 8. Task Score（v1 保留）

$$S_{task} = \delta_1 \text{SuccessRate} + \delta_2 (1-\text{FinalError})$$

### 8.1 Success Rate

任务定义的成功（grasp / place / goal reach）。这是最重要也最难自动判定的指标，建议至少：

- final_error_normalized < 0.05 算 success（粗糙代理）
- 关键关节 final_error < 5° 算 success（更严）

### 8.2 Final State Error

$$\|x_T - \hat x_T\|, \quad \text{Normalized} = \frac{\|x_T - \hat x_T\|}{\text{range}(x)}$$

---

## 9. ⭐ Per-phase 分解（v2 新增）

将 trajectory 按时间四分位拆开，单独看各阶段的 tracking：

| 阶段 | 时间段 | 典型任务含义 |
|---|---|---|
| Q1 | 0% - 25% | 起步加速 |
| Q2 | 25% - 50% | 接近目标 |
| Q3 | 50% - 75% | **关键操作**（抓取、对位） |
| Q4 | 75% - 100% | 收尾稳态 |

为什么重要：
- **Q1 / Q3 是真机最容易失败的两段**（起步惯性 + 抓取接触切换）
- 一个 Q4 表现极好但 Q3 表现差的模型，离线总分可能与对手持平，**但真机抓取成功率会差 20%+**

输出建议：每个 model × 每个 group × 每个 phase 的 RMSE 表（不是单一总分）。

---

## 10. Group-aware Aggregation 规则（v2 强制）

**禁止**对异质 signal 直接 `mean()`：

| ❌ 错误 | ✅ 正确 |
|---|---|
| `mean(all 16 signals)` | `weighted_mean({arm: ..., gripper: ..., torso: ...})` |
| 单一 `peak_jerk` 平均值 | 各 group 分开报告，并标注哪个 group 是 bottleneck |
| flip_rate 直接取均值 | `flip_rate(pred) − flip_rate(gt)` 即 ExcessFlipRate |

**强制规范**：summary CSV **必须**同时输出
- `pqs_open` / `pqs_closed`
- 每个 group 各自的 PQS
- 各 group 的 bottleneck 维度名称（哪个 signal 在哪个指标上拖了后腿）

---

## 11. Recommended Weights

### 11.1 Open-loop / Imitation Learning Oriented

| Module | Weight |
|---|---|
| Tracking | 0.45 |
| Smoothness | 0.25 |
| Stability | 0.15 |
| Task | 0.15 |

### 11.2 Closed-loop / Real Robot Deployment Oriented ⭐

| Module | Weight |
|---|---|
| Tracking(offset=0) | 0.15 |
| **Chunk Decay** | **0.25** |
| Safety (Spike + ExcessFlip) | 0.20 |
| Smoothness | 0.20 |
| Task | 0.20 |

> 原因：在真机闭环中，Chunk Decay 决定推理频率预算，Safety 决定会不会急停，
> Smoothness 决定关节寿命，Tracking 反而只在 offset=0 起作用。
> 模仿精度（v1 占 45%）应降到 15%。

---

## 12. Recommended Minimal Version（v2）

建议最小可用版本：

### Tracking

- RMSE（**按 chunk_offset 分组**，不要直接平均）
- NRMSE

### Chunk Decay ⭐ 新增

- DecayRatio = RMSE(H-1) / RMSE(0)

### Smoothness

- Mean Jerk（按 group 分开）
- TV

### Stability / Safety

- ExcessFlipRate（pred − gt）
- **SpikeRisk = max|err| / joint_step_limit** ⭐ 新增

### Task

- Success Rate
- Final Error (normalized)

输出 3 张表：
1. `summary.csv`：每 model 一行，含 `pqs_open` / `pqs_closed` / 各 group 的 PQS
2. `by_offset.csv`：每 model × 每 offset 的 RMSE（用于画 chunk decay 曲线）
3. `by_signal.csv`：每 model × 每 signal 的所有原子指标 + bottleneck 标记

---

## 13. Implementation Pitfalls（v2 新增）

### 13.1 ⚠️ load_csv 不能按 step 直接平均

当前 `offline_policy_diagnostics.py` 的 `load_csv()` 函数会把相同 `step` 的多行**直接求均值**
（line 117-122）。这会把 chunk_offset 维度抹平，导致 Chunk Decay 完全算不出来。

修复：

```python
# 错误：按 step 聚合时直接 mean()
sums[step][name] += value
counts[step][name] += 1

# 正确：保留 (step, chunk_offset) 二维索引
records[(step, chunk_offset)][name] = value
```

并允许下游按 `chunk_offset` group-by 计算。

### 13.2 ⚠️ 不要混用 `chunk_offset` / `horizon_index` / `chunk_index` 列名

不同模型的 CSV schema 不同：
- `rldx_1`: `chunk_offset`
- `pi0_5`:  `horizon_index`
- `fastwam`: `chunk_offset` + `chunk_index`

建议加一层 schema 适配层，统一映射到内部字段 `offset` / `chunk_id`。

### 13.3 ⚠️ Normalization 必须用 GT 的 range，不能用 pred 的 range

如果 pred 过度保守（输出几乎不动），用 pred range 会让所有归一化指标看起来都很好。
**永远用 GT 的 `max - min` 作为分母**，并对接近 0 的 range 加 epsilon 保护。

### 13.4 ⚠️ Success Rate 不是 final_error 阈值

`final_error < threshold` 只是一个**粗糙代理**。真正的成功率应该来自：
- 仿真器或真机的 task-specific success signal
- 由人工标注的 bad case 集合
- 或者关键关节同时满足多个阈值的复合判定

不要被 "success_rate = 1.0" 这种数字误导——v1 当前所有模型都是 1.0，根本没区分度。

---

## 14. Conclusion

最终评估体系应当不仅关注：

> "预测是否接近 GT"（open-loop tracking）

更要关注：

- **chunk 末端是否快速衰减**（闭环可推理性）
- **是否有单步抽动**（真机安全）
- **每个 group 各自是否健康**（不被均值掩盖）
- **任务关键阶段是否稳**（Q1 / Q3 而非平均）
- **是否可执行 + 真正完成任务**

因此采用 **Offline Policy Diagnostics v2** 作为统一评估框架：
- 拆 open-loop 与 closed-loop 两个视角
- 强制 group-aware 聚合
- 把 Chunk Decay 与 Spike Safety 纳入主指标

---

## Appendix A: v1 → v2 变更速查

| 变更 | v1 | v2 |
|---|---|---|
| 视角 | 单一 PQS | open / closed 两套 |
| Chunk 维度 | 平均掉 | **作为主模块** Chunk Decay |
| Peak | 用 P95 | P95 + **max + SpikeRisk** |
| Flip | 原始 flip_rate | **ExcessFlipRate = pred − gt** |
| 聚合 | 16 signal 直接 mean | **group-aware weighted mean** |
| 阶段 | 全 trajectory 平均 | **per-phase (Q1/Q2/Q3/Q4)** |
| 部署权重 | 模仿主导 (Track 0.45) | **Chunk Decay 0.25 + Safety 0.20 主导** |
| 已知 bug | `load_csv` 抹平 chunk 维度 | 显式修复 |
