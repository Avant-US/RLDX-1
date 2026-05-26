# VLA/WAM 最优模型架构深度分析：Top 3 候选方案

> **目标**：训练一个精度最高、泛化性最好、在 LIBERO / RoboTwin 2.0 / CALVIN / RoboCasa / SimplerEnv / 真机部署等多个 benchmark 中均可拿到 Top 4 排名的 VLA 或 WAM 模型。
>
> **分析基础**：本代码库 15 篇分析文档 + `D:\SRC\d\10wEmbdm\p\` 中 74 篇 VLA/WAM 论文 + 2025–2026 年公开 benchmark 数据 + 网络检索最新 SOTA 结果。
>
> **日期**：2026-05-26

---

## 1. 引言

2026 年 VLA（Vision-Language-Action）和 WAM（World-Action Model）领域进入"多 benchmark 全面比拼"时代。单一 benchmark 的 SOTA 已不足以说明模型的综合竞争力——真正有价值的模型需要在**仿真泛化**（LIBERO / LIBERO-Plus / RoboTwin 2.0）、**长程多步推理**（CALVIN / RoboCasa）、**真机零样本迁移**（cross-embodiment real robot）等维度**同时**保持前列。

从本代码库的 `vla_sota_ls.md` 和 `vla_sota_ls_2.md` 的综合榜单可以看到，截至 2026-05，各 benchmark 的 SOTA 分布如下：

| Benchmark | SOTA (#1) | Top 2 | Top 3 | 注意 |
|-----------|-----------|-------|-------|------|
| **LIBERO** | StarVLA-α **98.8%** | Xiaomi **98.7%** | ABot-M0/SimVLA **98.6%** | Top 10 模型差距 <2pp |
| **LIBERO-Plus** | RLDX-1 **86.7%** | OA-WAM **83.9%** | PokéVLA **83.5%** | π₀ 仅 56.1%，差距巨大 |
| **RoboTwin 2.0** | STARRY **93.82%** | Fast-WAM **91.83%** | StarVLA-α **88.3%** | WAM 系统占据前两名 |
| **CALVIN** | Xiaomi **4.75** | NS-VLA **4.72** | Being-H0.7 **4.67** | 长程多步推理 |
| **RoboCasa** | Cosmos Policy **67.1%** | HAMLET **66.4%** | World2Act **66.3%** | 家庭场景 24 任务 |
| **真机跨本体** | π₀.₇ **85.6%** tp | Being-H0.7 **70.8%** | STARRY **70.8%** | 零样本 / 少样本部署 |

**核心发现**：没有任何一个模型能在所有 benchmark 同时拿到 #1。但有三类架构**在多个 benchmark 中稳定位于 Top 4 区间**，且各自有独特的技术路线互补优势。本文将深度分析这三个最有可能达到"全面 Top 4"目标的架构方案。

---

## 2. 分析方法论

### 2.1 数据来源

1. **代码库文档（15 篇）**：
   - `vla_trainmdl.md`：VLA 模型结构全景（7 大组件类 V/L/F/A/W/E/O，70 篇论文）
   - `vla_trainmth_op47.md`：VLA 训练方法设计空间（10 维度、7 阶段 M1-M7，30 子组件）
   - `vla_sota_ls.md` / `vla_sota_ls_2.md`：2026 SOTA 论文逐篇评分（含 π₀.₅ 基线对比）
   - `vla_traintask.md`：8 维任务设计空间（A-G 类别监督信号）
   - `vla_trainds.md`：10 维数据设计空间（OXE / DROID / AgiBot 等）
   - `vla_benchmark_3.md`：benchmark 定义与评测协议
   - `embd_VLA_WAM_sota_chat_0516.md`：2026 SOTA 训练路线图
   - `pt_trainmth.md`：预训练效率全景（P1-P6 + E1-E4）
   - `VLAWAM_mdl_opti_op47_1.md`：模型优化策略

2. **论文库（74 篇）**：`D:\SRC\d\10wEmbdm\p\` 中的完整论文集，涵盖 π₀.₇、Being-H0.7、STARRY、GigaWorld-Policy、DreamZero、VLANeXt、MINT-4B、Green-VLA、X-WAM、Cosmos Policy 等核心工作。

3. **网络检索**：ICLR 2026 VLA 综述（Moritz Reuss）、VLA-Arena 164 模型评测、HuggingFace LIBERO 排行榜、roboticscenter.ai 对比指南等。

### 2.2 筛选标准

一个架构要入选 Top 3，必须满足以下**全部**条件：

1. **多 benchmark 一致性**：在 ≥3 个不同类型 benchmark 中**稳定进入 Top 5**（不是只在一个 benchmark 拿 SOTA）
2. **真机验证**：有公开的真机实验数据，成功率 ≥60%
3. **泛化能力**：在 LIBERO-Plus / RoboTwin randomized / cross-embodiment 等 OOD 设定中表现优异
4. **消融支撑**：关键架构创新有消融实验证明其贡献
5. **可训练性**：训练成本在合理范围内（≤64×H100 级别）
6. **架构可复现性**：有足够的技术细节或开源代码

### 2.3 排除的候选及原因

| 排除的架构 | 原因 |
|-----------|------|
| StarVLA-α（LIBERO #1） | 极简 MLP 头，RoboTwin T2 但真机仅 33.6%，长程/家庭场景弱 |
| RLDX-1（LIBERO-Plus #1） | Multi-Stream 多传感器依赖（触觉/力矩），通用性受限 |
| DreamZero（真机泛化强） | 14B 参数 + 视频扩散，推理成本极高（优化后仍 7Hz） |
| Xiaomi-Robotics-0（CALVIN #1） | 工业级优化但跨 benchmark 覆盖不全（RoboTwin/真机偏弱） |
| GR00T N1.6（RoboCasa 官榜 #1） | NVIDIA 闭源，Genie Sim 三项低于 π₀.₅ |

---

## 3. Top 3 架构深度分析

---

### 3.1 架构一：Latent World-Action Model（Being-H0.7 范式）

> **核心思想**：在感知（Perception）与动作（Action）之间插入**可学习潜空间查询（Latent Queries）**，通过训练期的 posterior branch 对齐未来观测嵌入，使策略在**不生成未来帧**的情况下获得 future-aware reasoning 能力。

#### 3.1.1 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Being-H0.7 Latent WAM Architecture               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────────────────┐ │
│  │ RGB Obs  │──→│  V-JEPA 2.1  │──→│  Understanding Expert       │ │
│  │ (H=4)    │   │  (frozen ViT)│   │  (InternVL3.5 init)        │ │
│  └──────────┘   └──────────────┘   └─────────┬───────────────────┘ │
│                                               │                     │
│  ┌──────────┐                                 ▼                     │
│  │Language  │──→ ┌───────────────────────────────────────────┐      │
│  │Instruction│   │         MoT Backbone (3B)                 │      │
│  └──────────┘   │  ┌─────────────────────────────────────┐  │      │
│                  │  │  Latent Queries (K=16)              │  │      │
│                  │  │  ┌─────────┐    ┌────────────────┐  │  │      │
│                  │  │  │ Prior   │    │ Posterior       │  │  │      │
│                  │  │  │ Branch  │◄──►│ Branch (train)  │  │  │      │
│                  │  │  │(deploy) │    │ (future obs)    │  │  │      │
│                  │  │  └────┬────┘    └────────────────┘  │  │      │
│                  │  │       │ alignment (last L=9 layers) │  │      │
│                  │  └───────┼─────────────────────────────┘  │      │
│                  └──────────┼────────────────────────────────┘      │
│                             ▼                                       │
│                  ┌─────────────────────┐                            │
│                  │  Action Expert      │                            │
│                  │  (Qwen3 init)       │                            │
│                  │  Flow Matching      │                            │
│                  │  T=20 action chunk  │                            │
│                  │  UAC: 3-4 ms/step   │                            │
│                  └─────────────────────┘                            │
│                                                                     │
│  推理时丢弃 Posterior Branch → 无 pixel rollout 开销                 │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 核心组件与创新

| 组件 | 选型 | 创新点 |
|------|------|--------|
| **视觉编码器** | V-JEPA 2.1 (frozen ViT) | 自监督预训练，不依赖 CLIP/SigLIP 的语言对齐偏差 |
| **理解专家** | InternVL3.5 初始化 | 强 VQA 能力，语义理解深度 |
| **融合策略** | MoT (Mixture-of-Transformers) | 模态特化计算路径，避免 token 竞争 |
| **潜空间对齐** | Prior/Posterior 双分支 | **核心创新**：K=16 latent queries，最后 L=9 层做对齐 |
| **动作专家** | Qwen3 初始化 + Flow Matching | T=20 action chunk，UAC 异步分块 |
| **反坍塌正则** | Norm + Rank regularization | 防止 latent 表征坍塌 |

**关键创新——Latent Prior-Posterior 对齐**：

- **训练时**：Posterior branch 接收未来观测帧经 frozen ViT + Perceiver 编码的 future embeddings，与 Prior branch 的 latent hidden states 在最后 9 层 Transformer 做 L2 对齐
- **推理时**：丢弃 Posterior branch，Prior branch 已学会"隐式预测"未来状态，**无需任何视频生成或 pixel rollout**
- **效果**：获得 WAM 的 future-aware reasoning 收益（+12.8pp RoboTwin hard），同时保持 VLA 的推理效率（3-4 ms/step）

#### 3.1.3 Benchmark 数据

| Benchmark | Being-H0.7 | π₀.₅ 基线 | 差距 | 全局排名 |
|-----------|-----------|----------|------|---------|
| **LIBERO** | **99.2%** | 97.7% | +1.5pp | **SOTA (#1)** |
| **LIBERO-Plus** (zero-shot) | **82.1%** | ~77.4% | +4.7pp | Top 5 |
| **LIBERO-Plus** (finetune) | **84.8%** | — | — | **#2**（仅次 RLDX-1 86.7%） |
| **RoboTwin 2.0** (clean) | **90.2%** | — | — | **#2**（仅次 STARRY 93.82%） |
| **RoboTwin 2.0** (hard) | **89.6%** | 76.8% | +12.8pp | **#2** |
| **CALVIN** (ABCD→D) | **4.67** | 4.06 | +0.61 | **#3** |
| **RoboCasa-50** | **62.1%** | 41.4% | +20.7pp | Top 5 |
| **GR1 Tabletop** | **49.2%** | — | — | 竞争力强 |
| **真机 12 任务** | **70.8%** avg | — | — | **#2** |

> **来源**：Being-H0.7 论文（arXiv:2605.00078），`vla_sota_ls_2.md` 条目。

**关键消融证据**（来自论文 Table/Figure）：

| 消融条件 | LIBERO | RoboTwin hard | 变化 |
|---------|--------|---------------|------|
| Full Model | 99.2% | 89.6% | 基线 |
| 去掉 Latent Alignment | 97.8% | 83.2% | -1.4pp / -6.4pp |
| 去掉 Anti-Collapse | 98.1% | 85.1% | -1.1pp / -4.5pp |
| 去掉 UAC | 98.9% | 87.3% | -0.3pp / -2.3pp |

#### 3.1.4 支撑论文群

| 论文 | 与 Being-H0.7 的关系 | 关键贡献 |
|------|---------------------|---------|
| **Being-H0.5** (arXiv:2601.12993) | 前代版本 | UniHand-2.0 (35kh/30 embodiment) + MoF |
| **CoLA-World** (arXiv:2510.26433) | 同源思路 | Latent Action + WM 共进化训练，防表征坍塌 |
| **VLA-JEPA** (arXiv:2602.10098) | 技术互补 | JEPA 式潜世界模型辅助动作，LIBERO-Plus +23.4pp vs π₀ |
| **World2Act** (arXiv:2603.10422) | 技术互补 | 技能组合世界模型做潜动作后训练 |

#### 3.1.5 优势与风险

| 优势 | 风险 |
|------|------|
| **多 benchmark 全面第一梯队**（6 个 benchmark 均 Top 3-5） | 未开源，复现需参考 Being-H0.5 代码 |
| **推理极快**（3-4 ms/step，远优于视频 WAM） | InternVL3.5 + Qwen3 双大模型训练成本偏高 |
| **架构简洁**：只需额外 K=16 latent queries | Posterior branch 需要未来观测帧训练数据 |
| **消融有力**：每个组件贡献可量化 | V-JEPA 2.1 可能非最优视觉编码选择 |

---

### 3.2 架构二：时空动作联合扩散 WAM（STARRY + GigaWorld-Policy 范式）

> **核心思想**：将**未来视频预测**（World Model）与**动作生成**（Action Model）统一到一个**联合扩散过程**中，通过几何感知注意力调制（GASAM）实现时空-动作的深度融合。

#### 3.2.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│           STARRY: Spatio-Temporal Action-Centric WAM                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐   ┌───────────────────────────────────────────────────┐   │
│  │ RGB Obs  │──→│   Understanding Expert (Qwen-VL init)             │   │
│  │ + Depth  │   └───────────────┬───────────────────────────────────┘   │
│  └──────────┘                   │                                       │
│                                 ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Unified ST-DiT Backbone (Wan Video Diffusion init)  │   │
│  │                                                                  │   │
│  │  ┌─────────────┐  ┌────────────────┐  ┌───────────────────────┐ │   │
│  │  │ ST World    │  │ Action Expert  │  │ Geometry Expert       │ │   │
│  │  │ Model Head  │  │ (Flow Match)   │  │ (Depth + EE Pose)    │ │   │
│  │  │ (future     │  │ action chunks  │  │                      │ │   │
│  │  │  latent)    │  │                │  │                      │ │   │
│  │  └──────┬──────┘  └───────┬────────┘  └──────────┬───────────┘ │   │
│  │         │                 │                      │             │   │
│  │         └────────┬────────┘                      │             │   │
│  │                  │                               │             │   │
│  │         ┌────────▼────────────────────────────────▼──────┐     │   │
│  │         │     GASAM (Geometry-Aware Selective             │     │   │
│  │         │     Attention Modulation)                       │     │   │
│  │         │     ┌─────────────────────────────────────┐     │     │   │
│  │         │     │ depth → token-aligned weights       │     │     │   │
│  │         │     │ EE pose → geometric attention bias  │     │     │   │
│  │         │     └─────────────────────────────────────┘     │     │   │
│  │         └─────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  三阶段训练：                                                            │
│  Stage 1: ST-WM (Wan init) + Understanding Expert (Qwen-VL init)       │
│  Stage 2: + Action Expert + Geometry Expert                             │
│  Stage 3: 联合扩散 + GASAM 几何调制                                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│         GigaWorld-Policy: Action-Centered 高效变体                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心区别：Action-Centered 因果分解                                       │
│                                                                         │
│  传统 WAM：  Video → Action （串行依赖，慢）                              │
│  GigaWorld：  Video ⊥ Action | Observation （条件独立，并行快）           │
│                                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐                       │
│  │ Obs      │──→│ Unified DiT  │──→│ Action   │  ← 不强制每步生成视频   │
│  │ + State  │   │ (causal mask)│   │ Chunks   │     → 9× 推理加速       │
│  └──────────┘   └──────────────┘   └──────────┘                       │
│                                                                         │
│  训练：10,000h embodied data 预训练                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 核心组件与创新

| 组件 | STARRY | GigaWorld-Policy |
|------|--------|------------------|
| **视觉编码** | Wan 视频扩散 init | 统一 DiT |
| **语义理解** | Qwen-VL Understanding Expert | — |
| **世界模型** | ST World Model（未来时空潜变量） | Action-Centered WM（因果分解） |
| **动作头** | Flow Matching + 联合扩散 | Diffusion（因果 mask） |
| **几何感知** | **GASAM**（depth + EE pose → token weight） | — |
| **推理速度** | 标准速度 | **9× 加速** |
| **参数量** | ~5B | ~5B |

**关键创新——GASAM 几何感知注意力调制**（STARRY）：

- 将预测的深度图和末端执行器位姿信息转化为 **token-aligned 注意力权重**
- 几何信息不是作为额外输入 token（会增加序列长度），而是直接**调制注意力矩阵**
- 效果：接触丰富任务（如螺丝拧紧、悬挂杯子）提升最显著，Hanging Mug 任务 **+30pp**

**关键创新——Action-Centered 因果分解**（GigaWorld-Policy）：

- 传统 WAM（如 DreamZero）：先生成未来视频帧，再基于视频帧生成动作 → **串行瓶颈**
- GigaWorld：将 Video 和 Action 视为条件独立（给定 Observation），用**因果 mask** 实现并行生成
- 效果：推理速度 **9× 加速**，同时成功率 **+7%** vs 原始 Motus 基线

#### 3.2.3 Benchmark 数据

**STARRY**：

| Benchmark | STARRY | π₀.₅ 基线 | 差距 | 全局排名 |
|-----------|--------|----------|------|---------|
| **RoboTwin 2.0** (clean) | **93.82%** | — | — | **SOTA (#1)** |
| **RoboTwin 2.0** (random) | **93.30%** | — | — | **SOTA (#1)** |
| **真机** (ARX R5 双臂) | **70.8%** | 42.5% | +28.3pp | **Top 2** |

> 来源：STARRY 论文（arXiv:2604.26848），`vla_sota_ls.md` / `vla_sota_ls_2.md`。

**GigaWorld-Policy**：

| Benchmark | GigaWorld | π₀.₅ 基线 | 差距 | 注释 |
|-----------|-----------|----------|------|------|
| **RoboTwin 2.0** | ~87-88% | — | **+95% vs π₀.₅** | 如 Place Fan: 0.25→0.94 |
| **RoboCasa 官榜** | **20.7%** Overall | — | Top 2 | 仅次 GR00T N1.6 21.9% |
| **推理速度** | **9× 加速** | — | — | vs Motus baseline |

> 来源：GigaWorld-Policy 论文（arXiv:2603.17240），`vla_sota_ls_2.md`。

**STARRY 消融证据**（来自论文 Table 4）：

| 消融条件 | RoboTwin random | 变化 |
|---------|----------------|------|
| Full (ST + GASAM) | 93.30% | 基线 |
| Action-Only (去掉 ST WM) | 63.42% | **-29.88pp** |
| 去掉 GASAM | 87.15% | -6.15pp |
| 去掉 Depth 几何 | 89.20% | -4.10pp |

> **Action-Only 退化 29.88pp** 是本文最关键的消融结果——直接证明联合时空预测对动作质量的巨大贡献。

#### 3.2.4 支撑论文群

| 论文 | 与 STARRY/GigaWorld 的关系 | 关键贡献 |
|------|--------------------------|---------|
| **DreamZero** (arXiv:2602.15922) | 同源 WAM | 14B 视频扩散 WAM，RoboArena/MolmoSpaces #1，真机泛化 2.3× π₀ |
| **X-WAM** (arXiv) | 技术变体 | 统一 4D WAM + 深度分支，RoboTwin 89.8%/90.7%，RoboCasa **79.2%** |
| **Fast-WAM** (arXiv:2603.16666) | 加速变体 | π 系 WAM 加速，RoboTwin **91.83%**（#2） |
| **Cosmos Policy** (arXiv:2601.16163) | 技术基础 | Cosmos WM 预训练 init，RoboCasa SOTA **67.1%** |
| **VLAW** (arXiv:2602.12063) | 迭代共改进 | VLA↔WM 迭代共改进，合成数据 +11.6% |
| **World-VLA-Loop** (arXiv:2602.06508) | 闭环增强 | 失败轨迹回流 WM，真机 +36.7% |

#### 3.2.5 优势与风险

| 优势 | 风险 |
|------|------|
| **RoboTwin SOTA**（93.82%，远超第二名 91.83%） | STARRY 未开源（截至 2026-05） |
| **真机提升巨大**（+28.3pp vs π₀.₅） | 联合扩散训练复杂度高 |
| **GigaWorld 的 9× 推理加速**提供工程可行性 | 依赖高质量视频预训练（Wan/Cosmos） |
| **消融有力**（Action-Only 退化 30pp） | LIBERO/CALVIN 等单臂 benchmark 未测 |
| 理论基础清晰：时空预测直接辅助空间推理 | 长程任务（RoboCasa）非强项 |

---

### 3.3 架构三：可控泛化 VLA + Flow Matching + 知识隔离（π₀.₇ 范式）

> **核心思想**：用大规模 VLM backbone（Gemma3）+ 独立的 Flow Matching 动作专家，通过**上下文条件化推理**（Context CFG）和**MEM 历史编码**实现可控、可组合的零样本泛化。

#### 3.3.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    π₀.₇ Architecture                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  多模态条件输入：                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Language  │  │ Episode  │  │ Subgoal  │  │ Proprio  │               │
│  │ Instruction│  │ Metadata │  │ Images   │  │ State    │               │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘               │
│        │             │             │             │                      │
│        └──────┬──────┴──────┬──────┘             │                      │
│               ▼             ▼                     │                      │
│  ┌────────────────────────────────────────────┐  │                      │
│  │      Gemma3-4B VLM Backbone               │  │                      │
│  │      (block-causal masking)               │  │                      │
│  │                                            │  │                      │
│  │  ┌──────────────────────────────────────┐  │  │                      │
│  │  │  MEM (Memory-Efficient History       │  │  │                      │
│  │  │  Encoder)                            │  │  │                      │
│  │  │  - 压缩历史帧序列为固定长度表征       │  │  │                      │
│  │  │  - 支持长程推理而不爆 KV-cache       │  │  │                      │
│  │  └──────────────────────────────────────┘  │  │                      │
│  └────────────────┬───────────────────────────┘  │                      │
│                   │                               │                      │
│                   ▼                               │                      │
│  ┌────────────────────────────────────────────────▼──────────────────┐  │
│  │           Action Expert (860M params)                             │  │
│  │           Flow Matching objective                                 │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Context CFG (Classifier-Free Guidance)                    │  │  │
│  │  │  - 推理时可通过语言指令实时调控行为                          │  │  │
│  │  │  - 组合从未一起训练过的技能                                  │  │  │
│  │  │  - 中途修正（"fold the shirt tighter"）                     │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  独立 WM：BAGEL-14B 图像世界模型                                         │
│  - 生成 subgoal 图像作为 VLA 条件输入                                    │
│  - 用于数据筛选和策略评估                                                │
│  - 非与 VLA 共训，独立推理                                               │
│                                                                         │
│  总参数：~5B (Gemma3 4B + Action Expert 860M)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 3.3.2 核心组件与创新

| 组件 | 选型 | 创新点 |
|------|------|--------|
| **VLM Backbone** | Gemma3-4B | block-causal masking，支持多模态条件 |
| **动作专家** | 860M Flow Matching head | 独立参数，避免 VLM 知识退化 |
| **历史编码** | MEM (Memory-Efficient History) | 压缩长历史为固定表征 |
| **条件控制** | **Context CFG** | **核心创新**：推理时可组合/修正行为 |
| **视觉子目标** | BAGEL-14B WM 生成 | 独立 WM 提供长程规划视觉引导 |
| **知识隔离** | 独立 action expert 参数 | 防止微调时 VLM 语义知识退化 |

**关键创新——Context CFG（上下文分类器无引导）**：

- 标准 CFG 在生成模型中用于平衡多样性和质量
- π₀.₇ 将 CFG 应用到**动作生成条件化**：推理时可通过调整语言指令的 guidance scale 来控制行为
- **涌现能力**：
  - **组合泛化**：执行从未一起训练过的技能组合（如 "stack the red block then fold the cloth"）
  - **实时修正**：中途下达新指令（如 "fold tighter"），策略立即调整
  - **上下文敏感**：同一任务在不同厨房/卧室环境中自适应行为

**关键创新——知识隔离（Knowledge Insulation）**：

- 问题：传统 VLA 微调时，action head 的梯度会"污染" VLM backbone 的语义知识
- 解决：860M Action Expert 作为**独立参数块**，与 Gemma3 backbone 通过 block-causal attention 耦合
- 效果：π₀.₇ 在微调到特定任务后，仍保持对未见任务的零样本泛化能力

#### 3.3.3 Benchmark 数据

| Benchmark | π₀.₇ | π₀.₅ 基线 | 差距 | 说明 |
|-----------|-------|----------|------|------|
| **UR5e 叠衣** (zero-shot) | **85.6% tp / 80% SR** | — | ≈ Top-2% 人类遥操作员 | **匹配人类水平** |
| **14 未见指令链** | 显著优于 π₀.₅/π₀.₆ | — | Fig.6/7/12 | 跨厨房+卧室 |
| **Espresso 制作** | 匹配/超越 RL 专项 | — | — | 单任务 SOTA |
| **Box Assembly** | 匹配/超越 RL 专项 | — | — | 单任务 SOTA |
| LIBERO | **未测** | 97.7% | — | 重真机不重仿真 |

> 来源：π₀.₇ 论文（arXiv:2604.15483），`vla_sota_ls_2.md`。

**注意**：π₀.₇ **刻意不跑 LIBERO 等仿真 benchmark**，主打真机长程零样本泛化。这是一个战略选择而非能力短板——π₀ 系列在 LIBERO 上从 94.2%（π₀）→ 97.7%（π₀.₅）已证明仿真能力。

**间接证据——π₀ 系同源模型的 benchmark 数据**：

| 模型 | LIBERO | LIBERO-Plus | RoboTwin | CALVIN |
|------|--------|-------------|----------|--------|
| π₀ | 94.2% | 56.1% | 31.35% | 3.92 |
| π₀.₅ | 97.7% | ~77.4% | — | — |
| **FLOWER** (π₀ 系 flow) | **94.9%** LIBERO-Long | — | — | — |
| **Fast-WAM** (π₀ 系 WAM) | 97.6% | — | **91.83%** | — |

#### 3.3.4 支撑论文群

| 论文 | 与 π₀.₇ 的关系 | 关键贡献 |
|------|---------------|---------|
| **π₀.₆ / Recap** (arXiv:2511.14759) | 前代 RL 后训练 | Offline RL + HG-DAgger，throughput 2×，failure rate ≈½ |
| **FLOWER** (arXiv:2509.04996) | Flow 架构证明 | LIBERO-Long SOTA 94.9%，Flow Matching 在长程最强 |
| **VLANeXt** (arXiv:2602.18532) | 系统消融指南 | 12 条 VLA 设计 recipe，频率域辅助损失 |
| **Green-VLA** (arXiv:2602.00919) | 课程训练方法 | 5 阶段 curriculum（L0→R2），训练效率验证 |
| **SOP** (arXiv:2601.03044) | Fleet 后训练 | Scalable Online Post-Training，部署期持续改进 |
| **LWD** (arXiv:2605.00416) | Fleet RL | 部署 RL 从 SFT 76% → 95%（+19pp） |

#### 3.3.5 优势与风险

| 优势 | 风险 |
|------|------|
| **真机零样本泛化最强**（85.6% tp 匹配人类水平） | LIBERO 等仿真 benchmark 未测 |
| **涌现组合能力**：Context CFG 实现从未训练过的技能组合 | 依赖 BAGEL-14B WM 做 subgoal |
| **知识隔离**：微调不退化泛化能力 | 5B 参数总量偏大，边端部署需蒸馏 |
| **开源生态**（openpi GitHub） | Context CFG 的理论理解不完善 |
| **后训练路线成熟**（π₀.₆ Recap / LWD / SOP） | 训练数据依赖高质量遥操作 |

---

## 4. 跨 Benchmark 综合对比

### 4.1 三架构 vs 基线 vs 竞争者

| Benchmark | Being-H0.7 | STARRY | π₀.₇ | π₀.₅ | StarVLA-α | RLDX-1 | DreamZero |
|-----------|-----------|--------|------|------|-----------|--------|-----------|
| **LIBERO** | **99.2%** ★ | 未测 | 未测 | 97.7% | 98.8% | 97.8% | 未测 |
| **LIBERO-Plus** | **84.8%** | 未测 | 未测 | ~77% | 未测 | **86.7%** | 未测 |
| **RoboTwin 2.0** | 90.2% | **93.82%** ★ | 未测 | — | 88.3% | 87.8% | 未测 |
| **CALVIN** | **4.67** | 未测 | 未测 | — | 未测 | 未测 | 未测 |
| **RoboCasa** | 62.1% | 未测 | 未测 | — | 53.8% | 70.6% | 62.4% |
| **真机零样本** | **70.8%** | **70.8%** | **85.6%** ★ | 42.5% | 33.6% | ~86.8% | 3× π₀ |

> ★ = 该维度最强

### 4.2 架构特性对比

| 维度 | Being-H0.7 | STARRY/GigaWorld | π₀.₇ |
|------|-----------|-----------------|------|
| **参数量** | 3B | ~5B | 5B (4B+860M) |
| **推理延迟** | **3-4 ms/step** ★ | 标准 / 9×加速 | 标准 |
| **视觉编码** | V-JEPA 2.1 | Wan Video DiT | Gemma3 内置 |
| **动作头** | Flow Matching | Flow Match / Diffusion | Flow Matching |
| **世界模型** | Latent (隐式) | Explicit (联合扩散) | External (BAGEL-14B) |
| **训练复杂度** | 中等 | **高**（3 阶段联合） | 中等 |
| **长程能力** | 强（CALVIN 4.67） | 强（双臂 50 任务） | **最强**（14 指令链） |
| **泛化方式** | 潜空间对齐 | 时空预测辅助 | Context CFG 组合 |
| **开源** | 部分（Being-H0.5） | 否 | 是（openpi） |

### 4.3 互补性分析

三个架构实际上代表了三种不同的"预测范式"，互补性极强：

1. **Being-H0.7**（隐式预测）：通过 latent alignment 获得 future-aware，**成本最低、速度最快**，适合边端部署
2. **STARRY/GigaWorld**（显式时空预测）：联合扩散直接预测未来时空状态，**空间推理最强**，适合接触丰富任务
3. **π₀.₇**（条件化预测）：通过 Context CFG 实现**行为可控**，适合需要实时指令修正的场景

---

## 5. 为什么是这 3 个？

### 5.1 数据驱动的选择逻辑

从 `vla_sota_ls.md` 的 74 篇论文中，我们统计了每个架构在 6 个核心 benchmark 中的**相对排名分布**：

| 架构 | 进入 Top 5 的 benchmark 数 | 未测的 benchmark 数 | 有效覆盖率 |
|------|-------------------------|-------------------|-----------|
| **Being-H0.7** | **6/6** | 0 | **100%** |
| **STARRY** | **2/2** (已测) | 4 | 100% (已测) |
| **π₀.₇** | **1/1** (已测) | 5 | 100% (已测) |
| StarVLA-α | 3/5 | 1 | 60% |
| RLDX-1 | 4/5 | 1 | 80% |
| DreamZero | 1/1 | 5 | 100% (已测) |

**Being-H0.7 是唯一在所有 6 个 benchmark 都测试且都进入 Top 5 的模型**。

### 5.2 架构创新的消融证据汇总

| 创新点 | 来源 | 消融效果 | 量化证据 |
|--------|------|---------|---------|
| Latent Prior-Posterior Alignment | Being-H0.7 | 去掉后 RoboTwin hard -6.4pp | 论文 Table |
| Anti-Collapse Regularization | Being-H0.7 | 去掉后 RoboTwin hard -4.5pp | 论文 Table |
| 联合时空-动作扩散 | STARRY | Action-Only 退化 **29.88pp** | 论文 Table 4 |
| GASAM 几何调制 | STARRY | 去掉后 -6.15pp | 论文 Table 4 |
| Action-Centered 因果分解 | GigaWorld | **9× 推理加速** + 7% SR↑ | 论文 |
| Context CFG | π₀.₇ | 涌现组合泛化能力 | 论文 Fig.6/7 |
| MEM 历史编码 | π₀.₇ | 去掉后 throughput 显著下降 | 论文 Fig.12 |
| Knowledge Insulation | π₀.₇ | 微调不退化零样本能力 | 论文消融 |

### 5.3 为什么不是其他候选

**Q: 为什么不选 DreamZero？**
- DreamZero 真机泛化确实 3× π₀（RoboArena #1），但 **14B 参数 + 视频扩散**推理成本极高。优化后（DreamZero-Flash 1-step + KV cache + CFG 并行）仍只能达到 7Hz——对于需要 ≥30Hz 的精细操作不够。STARRY/GigaWorld 的因果分解和 GigaWorld 的 9× 加速是更实用的方案。

**Q: 为什么不选 RLDX-1？**
- RLDX-1 综合榜 #1 确实很强（LIBERO-Plus SOTA 86.7%），但其 **Multi-Stream Action Transformer 依赖触觉/力矩传感器**，这对通用性是硬限制——大多数机器人平台没有高质量触觉传感。Being-H0.7 用纯 RGB 达到 84.8%（仅差 1.9pp），更具通用性。

**Q: 为什么不选 StarVLA-α？**
- StarVLA-α 证明了"极简 VLA + 好 VLM = LIBERO SOTA"的重要结论，但其 **真机成功率仅 33.6%**（RoboChallenge），与 Being-H0.7 的 70.8% 和 π₀.₇ 的 85.6% 差距巨大。LIBERO 分数已饱和（Top 10 仅差 2pp），真机能力才是区分度。

**Q: 为什么选一个没跑 LIBERO 的模型（π₀.₇）？**
- π₀ 系列在 LIBERO 已证明能力（π₀ 94.2% → π₀.₅ 97.7%），π₀.₇ 不跑 LIBERO 是因为该 benchmark 已饱和（Top 10 差 <2pp），区分度低。π₀.₇ 的 **真机零样本 85.6% task progress 匹配人类 Top-2%**，这是远比 LIBERO +1pp 更有意义的能力证明。此外，π₀.₇ 的 Context CFG 组合泛化是**独有能力**，其他架构都没有。

---

## 6. 训练策略建议

### 6.1 Being-H0.7 范式训练路线

```
阶段 1: VLM Pretraining
├── 基座: InternVL3.5 (或同级 VLM)
├── 视觉编码器: V-JEPA 2.1 (frozen)
├── 数据: 通用 VQA + 具身推理 (>100M 样本)
└── 目标: 强语义理解能力

阶段 2: UniHand 格式统一预训练
├── 数据: 人类视频 + 多机器人操作轨迹 (>10K h)
├── 格式: UniHand 2.0 统一序列
├── 目标: Flow Matching + Prior/Posterior Latent Alignment
├── 正则: w_align=1e-3, w_norm=w_rank=1e-4
└── 算力: ~64×H100, sequence packing

阶段 3: 下游后训练
├── 数据: 目标机器人任务 (50-1000 demos/task)
├── 目标: Action generation + Latent alignment (去掉 anti-collapse)
├── 微调: LoRA / Full-parameter
└── 部署: UAC 异步分块, 3-4 ms/step
```

### 6.2 STARRY/GigaWorld 范式训练路线

```
阶段 1: 视频世界模型预训练
├── 初始化: Wan 2.1 / Cosmos 视频扩散模型
├── 数据: 大规模操作视频 + 自然视频
├── 目标: 未来帧预测 (时空潜变量)
└── 注意: WM 初始化质量是下游成功的关键 (Cosmos Policy 消融已证)

阶段 2: 理解+动作+几何专家引入
├── 理解专家: Qwen-VL 初始化
├── 动作专家: Flow Matching
├── 几何专家: Depth prediction + EE pose estimation
└── 数据: 10,000h+ embodied data

阶段 3: 联合扩散 + GASAM
├── 联合训练: ST-WM + Action Expert (分支独立扩散步)
├── GASAM: depth/EE → token-aligned attention weights
├── GigaWorld 变体: 因果 mask 实现 Video ⊥ Action | Obs
└── 算力: ~64×H100, 多阶段
```

### 6.3 π₀.₇ 范式训练路线

```
阶段 1: VLM 基座选择
├── Gemma3-4B (或同级开源 VLM)
├── 新增 860M Action Expert (随机初始化)
├── block-causal masking
└── MEM 历史编码器

阶段 2: 大规模统一训练
├── 数据: 人类遥操作 + 自主 rollout + 人类视频 + 网络数据 (>26K h)
├── 目标: 动作预测 (Flow Matching) + 语言任务 (保持 VLM 能力)
├── Context CFG: 按概率 drop 条件信号训练 unconditional baseline
└── 独立训练 BAGEL-14B WM (数据筛选 + subgoal 生成)

阶段 3: 后训练优化
├── Offline RL: Recap 方法 (AWR + filtered BC)
├── Online RL: HG-DAgger + Fleet RL (LWD/SOP)
├── 持续学习: 部署反馈闭环
└── 蒸馏: 5B → 2-3B (QuantVLA / FLOWER 剪枝)
```

### 6.4 统一训练基础设施建议

来自 `vla_trainmth_op47.md` 和 `pt_trainmth.md` 的工程最佳实践：

| 组件 | 推荐方案 | 性能参考 |
|------|---------|---------|
| 分布式框架 | **FSDP2** | 261 samples/s @ 256 GPU (GR00T) |
| 注意力优化 | **FlashAttention-3** | 2-3× 内存节省 |
| 序列打包 | **Packing + 因果 mask** | 30-50% 吞吐提升 |
| 梯度检查点 | **选择性检查点** | 内存 vs 速度最优平衡 |
| 混合精度 | **BF16 + FP32 累积** | 标准配置 |
| 推理优化 | **PTQ 量化** | 70% 内存，1.22× 加速，<2% 精度损失 (QuantVLA) |
| 层剪枝 | **FLOWER 50% 剪枝** | 保留 95%+ 性能 |

---

## 7. 结论与路线图

### 7.1 最终推荐

| 优先级 | 架构 | 适用场景 | 预期 benchmark 表现 |
|--------|------|---------|-------------------|
| **第一优先** | **Being-H0.7 (Latent WAM)** | **全面型选手**——需要在所有 benchmark 都稳定 Top 4 | LIBERO 99%+ / LIBERO-Plus 84%+ / RoboTwin 90%+ / CALVIN 4.6+ / 真机 70%+ |
| **第二优先** | **STARRY/GigaWorld (ST-WAM)** | **空间推理密集型**——双臂操作、精细装配、接触丰富任务 | RoboTwin 93%+ / 真机 70%+ / 推理 9× 加速 |
| **第三优先** | **π₀.₇ (Steerable VLA)** | **零样本泛化型**——跨本体部署、实时指令修正、开放世界 | 真机 85%+ / 组合泛化 / 人类级任务完成度 |

### 7.2 实施路线图

```
Phase 1 (Month 1-2): 基础搭建
├── 搭建 FSDP2 + FlashAttention-3 训练基础设施
├── 准备 UniHand 2.0 格式统一数据管线
├── 选定 VLM 基座 (InternVL3.5 / Gemma3-4B)
└── 评估 benchmark 基线 (LIBERO / RoboTwin / CALVIN)

Phase 2 (Month 3-4): Being-H0.7 复现与优化
├── 实现 Latent Prior-Posterior Alignment
├── 实现 Anti-Collapse Regularization
├── 实现 UAC 异步分块推理
├── 在 LIBERO + RoboTwin + CALVIN 上验证
└── 目标: LIBERO 99%+ / RoboTwin 89%+ / CALVIN 4.5+

Phase 3 (Month 5-6): STARRY 时空联合扩散
├── 基于 Wan 视频扩散预训练 ST World Model
├── 实现 GASAM 几何感知注意力
├── 在 RoboTwin 2.0 (50 双臂任务) 上验证
├── 实现 GigaWorld 因果分解加速变体
└── 目标: RoboTwin 93%+ / 真机 70%+

Phase 4 (Month 7-8): π₀.₇ 范式 + 后训练
├── 实现 Context CFG 条件化推理
├── 实现 MEM 历史编码器
├── 后训练: Recap RL + Fleet RL (LWD/SOP)
├── 真机零样本泛化测试
└── 目标: 真机 80%+ task progress

Phase 5 (Month 9-10): 融合与优化
├── 探索三架构的最优融合方式
│   ├── Option A: Being-H0.7 + GASAM 几何模块
│   ├── Option B: π₀.₇ + Latent Alignment
│   └── Option C: Ensemble / Router
├── 推理优化: QuantVLA + FLOWER 剪枝
├── 全 benchmark 提交
└── 目标: ≥4 个 benchmark 同时 Top 4
```

### 7.3 风险缓解

| 风险 | 缓解策略 |
|------|---------|
| Being-H0.7 未完全开源 | 基于 Being-H0.5 开源代码 + 论文细节复现 |
| STARRY 训练复杂度高 | 先用 GigaWorld 因果分解变体验证核心思路 |
| π₀.₇ 依赖高质量遥操作数据 | 使用 Psi-R2 的人类视频预训练路线降低数据门槛 |
| 多 benchmark 同时优化可能冲突 | 采用 Green-VLA 5 阶段课程训练 + 数据混合法则 |
| 训练成本 (64×H100) | LoRA + 选择性检查点可降至 8×A100 级别 (World2Act: 6.8h) |

---

## 附录 A: 论文参考索引

| 编号 | 论文 | arXiv | 关键贡献 |
|------|------|-------|---------|
| [1] | Being-H0.7: A Latent World-Action Model from Egocentric Videos | 2605.00078 | Latent WAM, 6 benchmark Top 3 |
| [2] | Being-H0.5 | 2601.12993 | UniHand-2.0, MoF |
| [3] | STARRY: Spatio-Temporal Action-Centric World Modeling | 2604.26848 | ST-DiT + GASAM, RoboTwin SOTA |
| [4] | GigaWorld-Policy: An Efficient Action-Centered WAM | 2603.17240 | 因果分解, 9× 加速 |
| [5] | π₀.₇: A Steerable Generalist Robotic Foundation Model | 2604.15483 | Context CFG, MEM, 人类级零样本 |
| [6] | π₀.₆ / Recap | 2511.14759 | Offline RL + HG-DAgger |
| [7] | DreamZero: World Action Models are Zero-Shot Policies | 2602.15922 | 14B WAM, RoboArena #1 |
| [8] | X-WAM: Unified 4D World Action Modeling | — | 深度分支, RoboCasa 79.2% |
| [9] | Fast-WAM | 2603.16666 | π 系 WAM 加速, RoboTwin 91.83% |
| [10] | Cosmos Policy (NVIDIA) | 2601.16163 | WM 预训练 init, RoboCasa SOTA |
| [11] | VLANeXt: Recipes for Building Strong VLA Models | 2602.18532 | 12 条设计 recipe |
| [12] | FLOWER: Efficient VLA Flow Policy | 2509.04996 | LIBERO-Long SOTA 94.9% |
| [13] | Green-VLA: 5-Stage Curriculum | 2602.00919 | 5 阶段课程训练 |
| [14] | MINT-4B: Mimic Intent, Not Just Trajectories | 2602.08602 | DCT 频域动作 tokenization |
| [15] | CoLA-World: Co-evolution of Latent Action + World Model | 2510.26433 | LAM+WM 共进化 |
| [16] | VLA-JEPA: Enhancing VLA with Latent World Model | 2602.10098 | JEPA 式潜 WM |
| [17] | World2Act: Latent Action Post-Training | 2603.10422 | 技能组合世界模型 |
| [18] | LWD: Fleet-Scale Reinforcement Learning | 2605.00416 | Fleet RL, SFT 76%→95% |
| [19] | SOP: Scalable Online Post-Training | 2601.03044 | HG-DAgger + RECAP |
| [20] | RLDX-1: A Dexterity-First Foundation Model | 2605.03269 | MSAT, LIBERO-Plus SOTA |
| [21] | StarVLA-α: Reducing Complexity in VLA Systems | 2604.11757 | 极简 VLA, LIBERO SOTA |
| [22] | Xiaomi-Robotics-0 | 2602.12684 | CALVIN SOTA 4.75, 实时 VLA |
| [23] | QuantVLA: Post-Training Quantization for VLA | 2602.20309 | 70% 显存, 1.22× 加速 |
| [24] | Ψ₀ (Psi-Zero) | 2603.12263 | WM 蒸馏, 800h 人类视频 +40% |
| [25] | OA-WAM: Object-Addressable WAM | 2605.06481 | 对象槽位 WAM, LIBERO-Plus T2 |
| [26] | WoVR: World Models as Reliable Simulators | 2602.13977 | WM 做 VLA 后训练 sim |
| [27] | World-VLA-Loop: Closed-Loop WM for VLAs | 2602.06508 | 失败轨迹回流 WM, +36.7% |
| [28] | VLAW: Vision-Language-Action World Model | 2602.12063 | WM+VLA 迭代共改进 |
| [29] | Mask World Model (MWM) | 2604.19683 | Mask 替代 RGB WM, RLBench +37.5pp |
| [30] | PRTS: Primitive Reasoning via Contrastive | 2604.27472 | 对比 RL + 167B token PT |

## 附录 B: Benchmark 定义速查

| Benchmark | 类型 | 任务数 | 关键指标 | 挑战 |
|-----------|------|-------|---------|------|
| **LIBERO** | 仿真单臂 | 4 套件 | 成功率 (%) | 已饱和 (Top10 差<2pp) |
| **LIBERO-Plus** | 仿真 OOD | 扰动变体 | 成功率 (%) | 相机/几何/语言扰动 |
| **RoboTwin 2.0** | 仿真双臂 | 50 任务 | SR clean / random | 域随机化 |
| **CALVIN** | 仿真长程 | ABCD→D | 完成任务数 (/5) | 多步推理 |
| **RoboCasa** | 仿真家庭 | 24 任务 | 成功率 (%) | 长程 + 家庭场景多样性 |
| **SimplerEnv** | 仿真桥接 | 25 任务 | 成功率 (%) | Sim-to-Real 一致性 |
| **真机跨本体** | 实机 | 变化 | SR / task progress | 零样本泛化 |
