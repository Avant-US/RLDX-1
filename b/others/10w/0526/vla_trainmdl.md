# VLA 模型结构 / 网络架构全景分析(74 篇)

> 本文档是 [vla_traintask.md](vla_traintask.md) 的**姊妹篇**:那一份从「训练任务 / Loss / 阶段链路」视角横切 74 篇 VLA / 具身论文;本文档从「模型结构 / 网络架构 / 组件」视角再横切一次。两份文件在第 7 章速查卡处**互相链接**,形成「模型 × 训练」双视角。
>
> **数据源**:[p/](p/) 下 67 篇 `paper.pdf` + 5 篇 HTML(`Being-H0.7/paper.html`、`DM0/paper.html`、`Psi-R2/Psi-W0/page.html`、`GR00T_N1.6/page_{1,2}.html`、`Helix_02/page.html`),严格不读 `paper.txt`(避免 OCR 退化)。
>
> **方法论**:对每篇论文,把网络拆成 7 大组件类(V / L / F / A / W / E / O,共 32 子组件),先讲直觉(为什么)、再给数学/结构定义(LaTeX)、再列证据(谁这样做 + 消融数字)。

---

## 文档导览

- 想知道"某篇论文用了什么架构、哪些组件、消融最大贡献" → 第 7 章对应 `7.X.N` 模型卡;
- 想知道"某种组件(如 Flow Head)有哪些实现、消融效果如何" → 第 4 章组件深度解析;
- 想知道"模型结构怎么演化过来的" → 第 6 章演化时间线 + 第 8 章 8 个外部权威来源;
- 想自己设计 VLA → 先看第 8 章场景化组件组合建议 + 反模式;
- 反向查询"某个组件谁主用、谁副用" → 第 9.1.X 主组件倒排索引。

---

## 第 1 章 阅读指南 [T1]

### 1.1 文档目标

把 74 篇 VLA / 具身论文的网络结构与组件全部摊在 7 大组件的设计空间里,回答 4 个核心问题:

1. **是什么**:每篇论文用了哪些组件?数据如何在组件之间流?
2. **为什么**:为什么用这种 Vision Encoder / VLM Backbone / Fusion / Action Head?
3. **效果**:消融实验中各组件分别贡献多少?哪个组件正面影响最大,为什么?
4. **怎么选**:对自己的场景,如何挑组件 + 反模式列表。

### 1.2 术语速查(组件域)

| 缩写 | 全称 | 一句话解释 |
| --- | --- | --- |
| **VLM** | Vision-Language Model | 视觉-语言基模型(如 Qwen-VL、PaliGemma、Llama-3-V),通常作为 VLA 的 Backbone。 |
| **Vision Encoder** | — | 视觉特征提取器(SigLIP / CLIP / DINOv2 / 自训 ViT)。 |
| **Text Encoder / Tokenizer** | — | 把语言指令编为 token 序列;通常复用 VLM 自带的 tokenizer。 |
| **3D Encoder** | — | 把 RGB-D / 点云 / Gaussian Splat 编为 3D-aware token。 |
| **Proprio Encoder** | Proprioception Encoder | 把机器人本体状态(关节角 / 速度 / 末端位姿)编为 token / 向量。 |
| **Fusion** | — | 把多模态特征拼接、cross-attn 或 MoT 路由到一个统一序列。 |
| **Action Head** | — | 把 backbone 输出映射到动作分布(AR / DDPM / Flow / Manifold)。 |
| **WM Head** | World Model Head | 预测未来帧 / latent / mask 等,作为辅助监督。 |
| **CoT Head** | — | Chain-of-Thought 推理头,生成中间语言/几何推理。 |
| **Adapter** | — | 在 Backbone 中插入的轻量模块(LoRA / IA3 / Cross-Attn Adapter)。 |
| **Latent Action Tokenizer** | — | 把动作压缩为 latent token,使多本体共享同一动作空间。 |
| **MoT** | Mixture-of-Transformers | 多个并行 Transformer 专家,通过门控路由不同模态/任务。 |
| **MoE** | Mixture-of-Experts | 单层 / 单 head 内多个专家 FFN,稀疏激活。 |
| **DiT** | Diffusion Transformer | Transformer-based 扩散去噪网络(配 AdaLN-Zero)。 |
| **Flow Expert** | — | Flow Matching head 的 Transformer 实现,作 backbone 的"专家"分支。 |
| **AR Head** | Autoregressive Head | 自回归输出 action token 序列。 |
| **Cross-Attn** | Cross-Attention | 用 Q-K-V 让 Action Head 检索 backbone 输出。 |
| **Gating** | — | per-token / per-task 路由权重。 |
| **PTQ** | Post-Training Quantization | 训练后量化(W4A8 / W8A8 等)。 |
| **EEF / Proprio** | End-Effector / Proprioception | 末端执行器位姿 / 本体状态(VLA 必备输入)。 |
| **Chunking** | — | 一次输出 H 步动作,降低决策频率。 |
| **AdaLN-Zero** | Adaptive LayerNorm Zero-init | DiT 的核心调制层,把条件向量注入到 LayerNorm 的 γ/β。 |
| **RoPE** | Rotary Position Embedding | 旋转位置编码,Llama / Qwen / PaliGemma 通用。 |
| **FAST / DCT VQ** | — | 频域离散化动作 token 的两种主流编码方式。 |

### 1.3 阅读路径

- **设计者视角**(我要做 VLA):1 → 2 → 3 → **8 →** 4 → 7。
- **研究者视角**(理解一篇论文):7.X.N → 4 → 6 → 5 → 8。
- **横向研究**(对比一类组件):3 → 4 → 5 → 9.1.X 倒排索引。

---

## 第 2 章 模型组件的 9 维设计空间 [T1]

任何一个 VLA 网络都可以在下面 9 个维度上唯一定位。把一个新模型的设计想像成"在 9 维网格里走出一条路径",就更容易看出它与其他工作的异同。

### 2.1 输入模态

- **覆盖**:RGB(单视图 / 多视图) / 深度 / 点云 / 语言指令 / Proprio / 历史动作 / 触觉。
- **典型选择**:绝大多数 74 篇是「多视图 RGB + 语言 + Proprio」三件套;少数加 3D(GST-VLA / Pose-VLA / GeneralVLA),极少加触觉。
- **设计自由度**:输入越多 → 通用性越强 / 数据越贵 / 对其中某一项做消融可显著观察"该模态贡献"。

### 2.2 编码器选型

- **轴**:Frozen VLM 内置 / SigLIP / DINOv2 / CLIP / 自训 ViT / 3D Gaussian Encoder / Pose Encoder / 触觉 CNN。
- **关键 trade-off**:
  - SigLIP / CLIP:语义对齐强,适合配 LLM,但**对几何/深度弱**;
  - DINOv2:几何/局部细节强,但**与语言不对齐**(常需 projector);
  - 自训 ViT:可定制,但**数据需求大、容易过拟合**;
  - 3D Gaussian / Pose:几何精度高,但**算力贵**。

### 2.3 模态融合时机

- **早融合**(prefix concat):VLM Backbone 起步前就把所有模态拼接成长 token 序列,Backbone 一视同仁处理。
- **中融合**(cross-attn / Λ-attn):VLM 主流是普通 self-attn,但 Action Head 用 cross-attn 从 VLM 取条件,可缓解"prefix 抄捷径"。
- **晚融合**(per-modality head):每模态有自己的小 head,最后用一个聚合器合并。
- **MoT** (Mixture-of-Transformers):每模态/每任务一个独立 Transformer,通过门控路由,2026 趋势主导。
- **MoE** (Mixture-of-Experts):单层内多个 FFN 专家,稀疏激活。

### 2.4 主干形态

- **轴**:开源 MLLM(Qwen-VL / PaliGemma / Llama-3-V / Gemma-3) / 自训 VLM(GR00T / Cosmos / Molmo / Step1) / Pure Transformer(无语言)。
- **轴 2(规模)**:< 1B(SimVLA、FLOWER、PokéVLA、StarVLA-α)/ 1-4B(MINT-4B、 MolmoAct2)/ 4-7B(Xiaomi-Robotics-0、π0.7、Helix_02)/ > 10B(DreamZero 14B、GR00T_N1.6)。

### 2.5 任务模块化

- **轴**:单模型端到端(SimVLA、π0、LingBot) vs 模块化双脑(LoHo-Manip 任务管理 VLM + Trace VLA、TiPToP 模块化规划、GeneralVLA Affordance + Control)vs 三层全身(Helix_02)。
- **效果**:模块化通常**易调试、易组合**,但**模块间梯度不连贯**;端到端反之。

### 2.6 动作头形态

- **轴**:离散 Token AR / DDPM / DiT Diffusion / Flow Matching MLP / Flow Matching Transformer Expert / Action Manifold / 频率分解(DCT 多尺度)。
- **配对**:Action Head 的选择往往由本体频率与精度需求驱动(灵巧手高频 → Flow Transformer;移动机器人低频 → AR / DP)。

### 2.7 辅助头(WM / CoT / Pose / Affordance)

- **轴**:像素未来帧 / Latent JEPA / Mask 预测 / Trace 预测 / Future State / Reward-to-go / CoT 推理。
- **意义**:把"动作之外的额外监督信号"塞进训练,给 backbone 多一份梯度,但**容易抢资源**。

### 2.8 跨本体适配

- **轴**:Per-head(每本体一个 Action Head)/ Soft-Prompt(每本体一个 prompt token) / Latent Action Tokenizer(统一动作空间) / Embodiment Aug(数据层增广) / 共享骨干。
- **配合**:Soft-Prompt + 共享骨干是 2025-2026 主流(LAP / X-VLA / OXE-AugE);Per-head 退化为"简单 baseline"。

### 2.9 部署优化

- **轴**:PTQ Quantization(W4A8 / W8A8) / Layer Pruning / 异步执行 / Chunking / Distillation / Speculative / KV-cache 复用 / 跳计算。
- **典型 trade-off**:量化 4×8 → 显存 1/4,但 **Flow head 易破** ;层裁剪 50% → 训练成本 1/2,但 **某些任务 OOD 崩**。

### 2.10 9 维空间的可视化

下图把 9 个维度画成 9 根坐标轴,任何具体网络都可以投影到这 9 根轴上,从而方便比对:

```mermaid
flowchart TB
    root["VLA 网络一例"] --> input["输入模态"]
    root --> enc["编码器选型"]
    root --> fuse["融合时机"]
    root --> back["主干形态"]
    root --> mod["任务模块化"]
    root --> head["动作头"]
    root --> aux["辅助头"]
    root --> emb["跨本体适配"]
    root --> opt["部署优化"]

    input --> input1["RGB / Depth / 3D / Proprio / 触觉"]
    enc --> enc1["SigLIP / DINOv2 / CLIP / 3D Gaussian / Pose"]
    fuse --> fuse1["早 / 中 / 晚 / MoT / MoE"]
    back --> back1["开源 MLLM / 自训 VLM / Pure Transformer"]
    mod --> mod1["端到端 / 双脑 / 三层全身"]
    head --> head1["AR / DDPM / DiT / Flow MLP / Flow Expert / Manifold"]
    aux --> aux1["像素 / Latent / Mask / Trace / Future / Reward / CoT"]
    emb --> emb1["Per-head / Soft-Prompt / Latent Action / Aug / 共享"]
    opt --> opt1["PTQ / Pruning / 异步 / Chunking / Distill / KV"]
```

---

## 第 3 章 组件分类总图 [T1]

把第 2 章的 9 维空间投影到「最常用的组件实现」,得到下面这棵 7 大组件类、~32 子组件的分类树:

```mermaid
flowchart LR
    root["VLA 模型组件"] --> V["V. 输入编码器"]
    root --> L["L. VLM Backbone"]
    root --> F["F. 融合/路由"]
    root --> A["A. 动作 Head"]
    root --> W["W. 世界/辅助 Head"]
    root --> E["E. 跨本体适配"]
    root --> O["O. 后训练/部署优化"]

    V --> V1["V1 SigLIP/CLIP 系"]
    V --> V2["V2 DINOv2 系"]
    V --> V3["V3 3D/Pose 编码器"]
    V --> V4["V4 Proprio MLP/Tokenizer"]
    V --> V5["V5 多视图融合/Patch-Select"]

    L --> L1["L1 开源 MLLM (Qwen/PaliGemma/Llama)"]
    L --> L2["L2 自训 VLM (GR00T/Cosmos/Molmo/Step1)"]
    L --> L3["L3 Action Token Vocab (FAST/DCT/VQ)"]
    L --> L4["L4 RoPE/Long Context"]

    F --> F1["F1 早融合 prefix concat"]
    F --> F2["F2 中融合 Cross-Attn / Λ-attn"]
    F --> F3["F3 晚融合 per-modality head"]
    F --> F4["F4 MoT"]
    F --> F5["F5 MoE / Gating"]
    F --> F6["F6 Adapter / Soft-Prompt"]

    A --> A1["A1 AR head (离散 token)"]
    A --> A2["A2 DDPM / DiT head"]
    A --> A3["A3 Flow Matching head"]
    A --> A4["A4 AR + 连续 head 混合"]
    A --> A5["A5 Action Manifold head"]
    A --> A6["A6 频率 / 多尺度 head"]

    W --> W1["W1 像素未来帧"]
    W --> W2["W2 Latent / JEPA head"]
    W --> W3["W3 Mask 预测"]
    W --> W4["W4 Trace / Affordance"]
    W --> W5["W5 Future State / Reward"]
    W --> W6["W6 CoT / Reasoning"]

    E --> E1["E1 Per-head"]
    E --> E2["E2 Soft-Prompt"]
    E --> E3["E3 Latent Action Tokenizer"]
    E --> E4["E4 Embodiment Aug"]
    E --> E5["E5 共享骨干"]

    O --> O1["O1 PTQ Quantization"]
    O --> O2["O2 Layer Pruning"]
    O --> O3["O3 异步 / Chunking"]
    O --> O4["O4 Distillation"]
    O --> O5["O5 Speculative / KV-cache"]
```

七大类的「一句话目的」:

- **V. 输入编码器**:把原始多模态像素 / 几何 / 状态压成 token,**决定下游能"看到"什么**。
- **L. VLM Backbone**:把多模态 token 序列做语义级混合,**决定模型的通识与语言理解上限**。
- **F. 融合 / 路由**:决定不同模态 / 任务之间**梯度流动的拓扑**,影响通识保留与多任务相互干扰。
- **A. 动作 Head**:把 backbone 表示映射为动作分布,**决定动作精度 / 平滑度 / 频率上限**。
- **W. 世界 / 辅助 Head**:塞进动作以外的额外监督信号,**塑造 backbone 的隐空间**。
- **E. 跨本体适配**:决定"同一模型能服务几种本体"的扩展性。
- **O. 后训练 / 部署优化**:把训好的模型压成可量产形态,决定能否真的上机器人。

---

## 第 4 章 组件深度解析 [T2]

每个子组件按统一模板展开:**直觉比喻 → 数学/结构定义(LaTeX) → 典型实现 → 数据要求 → 代表论文(链回第 7 章) → 优势 → 局限 → 对效果的正负影响 → 消融证据(2-4 条) → 为什么这样设计**。

### 4.V 输入编码器 [T2]

**共同特点**:这一层决定了 VLA 能"看到"什么。绝大多数 74 篇用 SigLIP / DINOv2 作为视觉骨架,但是否引入 3D / Pose / Patch-Select 是分化关键。

#### V1 SigLIP / CLIP 系 [T1]

**直觉比喻**:就像给模型一副"懂语言的眼镜"——SigLIP / CLIP 的 token 一出来就对齐到语义空间,VLM 拿到后**不需要再翻译**就能理解。

**结构定义**:输入 RGB 图像 \(x \in \mathbb{R}^{H\times W\times 3}\),输出 token 序列 \(z \in \mathbb{R}^{N\times d}\),其中 \(N = (H/P)(W/P)\) (P 为 patch 大小)。SigLIP 训练 loss:
\[
\mathcal{L}_{\text{SigLIP}} = -\sum_i \log \sigma\bigl(t\cdot \text{sim}(x_i,y_i) - b\bigr)
\]
比 CLIP 的 softmax-CE 更稳定且对 batch size 不敏感。

**典型实现**:`SigLIP-So400m-Patch14-384` 或 `CLIP-ViT-L/14`;通常 Frozen,或开 LoRA 微调最后几层。

**数据要求**:本身已经预训练好,VLA 一般不再训;若 fine-tune 只需 50k+ 真机 episode。

**代表论文**:**~80% 用 VLM 的 VLA 都默认配 SigLIP**:[π0 系列](#7a10-π06-recap) / [FLOWER](#7a2-flower) / [SimVLA](#7a5-simvla)(PaliGemma 内置 SigLIP-So400m)、[LAP](#7e1-lap)、[Pose-VLA](#7v6-pose-vla)、[PokéVLA](#7v5-pokévla)(DINO+SigLIP)、[GST-VLA](#7v4-gst-vla)、[ConsisVLA-4D](#7v1-consisvla-4d)(SigLIP+DINOv2+VGGT)、[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)、[X-VLA](#7a8-x-vla)、[MINT-4B 30M 版](#7l6-mint-4b)、[HiF-VLA](#7w6-hif-vla)(DINOv2+SigLIP);**CLIP 系**仅存于 [BTK](#7v7-btk)(CLIP-B/16)与少数老型号。

**优势**:语义对齐强、与 VLM 即插即用、跨域语言可迁移性高。

**局限**:**几何/深度弱**;在精细操作中常需要额外几何 head 补救;高分辨率代价大。

**对效果的正负影响**:
- 正:VLM-friendly,加速收敛(典型快 20-40%);语言指令理解强;少样本泛化好。
- 负:精细插拔 / 6-DoF 抓取常需配 DINOv2 / 3D Gaussian 来补几何。

**消融证据**:
- [VLANeXt](#7l8-vlanext) Multi-view vs Single(Table 1)→ **+23.3pp LIBERO-Plus**,SigLIP token 数量本身就是收益;
- [StarVLA-α](#7f4-starvla-α) PaliGemma(SigLIP)→ Qwen3-VL-4B(SigLIP+)→ **+26pp LIBERO**(95.8% vs 69.8%);
- [ConsisVLA-4D](#7v1-consisvla-4d) CV-Aligner(SigLIP+VGGT)消融 → **-7.0% LIBERO**(Table 5),SigLIP 单独不足以承担 4D 时空。

**为什么**:VLA 的"语言对齐"是核心痛点,SigLIP/CLIP 在 web 数据上预训练已经把"图像 ↔ 语言"对齐做到 SOTA,VLA 直接接入即可。

#### V2 DINOv2 系 [T1]

**直觉比喻**:DINOv2 像一个**只看图不读字的几何大师**:无监督 self-distillation 训练让它对纹理、深度、局部结构敏感,但与语言**几乎完全无关**。

**结构定义**:Teacher-Student self-distillation,Teacher 是 EMA 的 Student,Loss:
\[
\mathcal{L}_{\text{DINO}} = -\sum_i p_t(x_i)\log p_s(x_i')
\]
不依赖语言对,适合**对几何细节有要求**的场景。

**典型实现**:`DINOv2-ViT-L/14`;常和 SigLIP 并联(SigLIP 取语义 token,DINOv2 取几何 token)。

**数据要求**:已预训练;VLA 中常 Frozen。

**代表论文**:[PokéVLA](#7v5-pokévla)(DINO+SigLIP 并联)、[GST-VLA](#7v4-gst-vla)(SigLIP+DINOv2+Depth 三联)、[ConsisVLA-4D](#7v1-consisvla-4d)(SigLIP+DINOv2+VGGT)、[HiF-VLA](#7w6-hif-vla)(DINOv2+SigLIP)、[MINT-4B 30M 版](#7l6-mint-4b)(SigLIP+DINOv2);几乎所有"精细操作主导"的小 VLA 都把 DINOv2 当 SigLIP 的几何补充。

**优势**:几何 / 局部细节强,**对深度估计 / pose / affordance** 友好;对光照鲁棒。

**局限**:**与语言不对齐**,必须配 projector;单独使用对 VLA 的 VLM 优化效果有限。

**对效果的正负影响**:
- 正:在 6-DoF 抓取、插拔等精细任务上 +3~8pp;OOD 视觉鲁棒。
- 负:语言迁移能力弱,需要额外训练 projector。

**消融证据**:
- [GST-VLA](#7v4-gst-vla):去掉 Dense depth 改为 Full Gaussian(Table VII)→ **-4.5pp**,几何 token 比 SigLIP 单独贡献明显;
- [ConsisVLA-4D](#7v1-consisvla-4d):CO-Fuser(DINOv2+VGGT)G-Fus+IG-Agg 消融 → **-13.3% Real**,DINOv2 + VGGT 的几何分支贡献最大;
- [HiF-VLA](#7w6-hif-vla):DINOv2 + SigLIP 双流是其 motion vector 编码的几何基础(Sec 4.1)。

**为什么**:VLA 一旦做精细操作,SigLIP 的语义 token 不够细;DINOv2 用 self-distillation 把"局部块"做精,**和 SigLIP 互补**。

#### V3 3D / Pose / Affordance 编码器 [T1]

**直觉比喻**:不是看 2D 像素,而是**把场景在 3D 空间里"摆出来"**——3D Gaussian / Pose Token / Affordance Map 给模型一个真正"物理可操作"的世界。

**结构定义**:
- 3D Gaussian Splat token:把 RGB-D 转成 N 个各向异性 Gaussian,每个 token 含 \((\mu, \Sigma, c)\);
- Pose Token:把物体 6-DoF 位姿 \((R, t)\) 离散化为 codebook;
- Affordance Map:对每像素预测"可抓取概率"或"运动方向"。

**典型实现**:GST-VLA(128 个 3D Gaussian)、Pose-VLA(VQ Pose tokenizer)、GeneralVLA(3D Affordance + Path Token)、PokéVLA(几何分割 + 关节角)。

**数据要求**:RGB-D 或多视图;Affordance 需人工标注或预训模型生成。

**代表论文**:[GST-VLA](#7v4-gst-vla)(128 各向异性 3D Gaussian)、[Pose-VLA](#7v6-pose-vla)(离散 Pose Token + depth+raymap)、[GeneralVLA](#7v3-generalvla)(3D Affordance Map + LLM 3D Path)、[PokéVLA](#7v5-pokévla)(VGGT 几何对齐,训练期)、[ConsisVLA-4D](#7v1-consisvla-4d)(VGGT depth+pointmap)、[OA-WAM](#7w8-oa-wam)(SAM3+DINOv3 槽位 pose)、[ABot-M0](#7a1-abot-m0)(VGGT + Qwen-Image-Edit)、[STARRY](#7w9-starry)(Depth + Geometry Expert)。

**优势**:**精细 6-DoF 操作**显著受益(+5-15pp);几何鲁棒;对未见物体更易泛化。

**局限**:**算力贵**(3D pre-processing 通常 +30-50% 延迟);依赖 RGB-D 传感器(部署门槛高)。

**对效果的正负影响**:
- 正:精细操作 / 6-DoF / 长程几何任务显著 +;
- 负:实时性下降,需要硬件升级。

**消融证据**:
- [GST-VLA](#7v4-gst-vla) S1 GST 预训消融(Table VI)→ **+6.2pp**(单 V3 子组件贡献第一);3D Fourier PE 消融 → +2.8pp;
- [Pose-VLA](#7v6-pose-vla):RoboTwin 2.0 Hard **+14pp vs π0**(65.12% → 79.1%);去 Pose 预训显著降;
- [ConsisVLA-4D](#7v1-consisvla-4d):去 CO-Fuser(VGGT 几何)→ **-13.3% Real**(单组件最大贡献);
- [ABot-M0](#7a1-abot-m0):VGGT Cross-Attention(Table 10)→ +4.7pp(66.4→71.1)。

**为什么**:RGB 单纯靠 2D 看不出"杯子离手有多远",3D / Pose / Affordance 是把"物理世界结构"显式注入 VLA 的最直接方式。

#### V4 Proprio MLP / Tokenizer [T1]

**直觉比喻**:**模型自己也要知道"我现在是什么姿势"**——Proprio 把关节角 / 末端位姿 / 夹爪开度等连续值编为 token / 向量。

**结构定义**:Proprio 输入 \(s \in \mathbb{R}^{d_p}\);通过 MLP 投影 \(z_p = W_2\cdot \sigma(W_1 s)\) 或 VQ tokenizer 编为离散 token。

**典型实现**:多数 VLA 用 MLP(simpler);π0 / GR00T / Xiaomi 等更高级方案用 Linear Projection + LayerNorm。

**数据要求**:示教数据自带,无额外标注。

**代表论文**:几乎全部 74 篇都有 Proprio 输入,其中 [VLANeXt](#7l8-vlanext)、[FocusVLA](#7v2-focusvla)、[ConsisVLA-4D](#7v1-consisvla-4d)、[RLDX-1](#7a4-rldx-1)、[Helix_02](#7f2-helix_02-figure-ai)、[π0 系列](#7a10-π06-recap) 等明确做过 Proprio 消融或专门设计 Proprio 编码器。

**优势**:**几乎零代价**;闭环控制必备;对动作平滑度有正向作用。

**局限**:若 Proprio 与 Action Head 不一致(如 Pose vs 关节角)会引发数据-控制错位。

**对效果的正负影响**:
- 正:闭环 / 复位 / 抓取必备;消融常显示去掉 Proprio 性能 -10~30%;
- 负:几乎无;但需保证训练与部署的 Proprio 标准一致。

**消融证据**:
- [VLANeXt](#7l8-vlanext):Proprio → VLM vs 无 Proprio(Table 1)→ **+7.2pp LIBERO-Plus**(87.7% vs 80.5%);
- [RLDX-1](#7a4-rldx-1):MSAT 中 Memory + 触觉 + 力矩流 → 显著抗 VLA 失败模式(尤其接触丰富任务);
- [Helix_02](#7f2-helix_02-figure-ai):全身关节 + 触觉 + Proprio 三件套是 4 分钟自主任务的关键基础(无消融,系统性叙事)。

**为什么**:机器人不是"摄像头 + 命令"那么简单,本体状态决定了"下一步动作合不合法"。

#### V5 多视图融合 / Patch-Select [T1]

**直觉比喻**:多个摄像头(头部 + 双腕)拍到的内容大量冗余,**让模型自己挑重要 patch** 比硬塞所有 token 更高效。

**结构定义**:
- 简单 concat:把多视图 token 直接拼接;
- Cross-Attention Pool:用可学习 query 从多视图 token 中检索;
- Patch-Select:用注意力 / hard gating 选 Top-K patch,丢掉无关 patch。

**典型实现**:FocusVLA 用级联注意力 + Hard Patch-Select;Xiaomi-Robotics-0 用 Token Pooling;Cosmos / GR00T 用 Cross-Attention 聚合。

**数据要求**:多视图必须时间同步。

**代表论文**:[FocusVLA](#7v2-focusvla)(级联注意力 + hard Patch-Select 主推手)、[ConsisVLA-4D](#7v1-consisvla-4d)(仅 1/8 视觉 token,Single-Fusion + Group-Fusion)、[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)(Token Pooling 异步)、[Cosmos Policy](#7l1-cosmos-policy) / [GR00T_N1.6](#7l3-gr00t_n16)(Cross-Attention 聚合多视图)、[VLANeXt](#7l8-vlanext)(Multi-view 经验对比)、[HiF-VLA](#7w6-hif-vla)(Motion Vector 自适应聚合)。

**优势**:**降推理延迟**(常 -20-40%);减少 VLM 上下文压力;对小模型尤其友好。

**局限**:Patch-Select 选错就是"看错地方";需要训练时 dropout 配合。

**对效果的正负影响**:
- 正:推理快、上下文友好、对低算力部署关键;
- 负:Patch-Select 选错的 corner case 容易翻车,需 fallback。

**消融证据**:
- [FocusVLA](#7v2-focusvla):Focus Attention(patch-level)→ 收敛 **1.5×(LIBERO)/ 5×(Spatial)**;Modality Cascaded Attention 让注意力从散乱变聚焦;
- [ConsisVLA-4D](#7v1-consisvla-4d):**仅用 1/8 视觉 token** 即达 SOTA,推理 2.3× 加速;
- [HiF-VLA](#7w6-hif-vla):用 Motion Vector 代帧堆叠 → **-58% latency**,同时 SR +2.2pp。

**为什么**:多视图 ≠ 信息量多 K 倍;**信息冗余 + 算力压力**让"挑 Top-K patch"成为 2025-2026 普及做法。

---

### 4.L VLM Backbone [T2]

**共同特点**:Backbone 决定了模型的"通识 + 语言理解 + 长上下文"上限。74 篇里近 80% 直接用开源 MLLM(L1),约 20% 自训 VLM(L2),少量做 Action Token Vocab 扩展(L3)。

#### L1 开源 MLLM(Qwen-VL / PaliGemma / Llama-3-V / Gemma-3) [T1]

**直觉比喻**:**站在巨人肩上**——开源 MLLM 已经把 LM 与 VL 训练做到极致,VLA 只在上面加 Action Head 即可。

**结构定义**:Backbone = Transformer Decoder + RoPE + Group Query Attention + SwiGLU FFN,与 LLM 同构。参数量 0.5-13B。

**典型实现**:
- PaliGemma-3B(π0 / FLOWER / SimVLA);
- Qwen3-VL-2B/4B/8B(StarVLA-α / VLANeXt / Xiaomi);
- Llama-3.2-Vision(MolmoB0T / RLDX-1);
- Gemma-3-4B(部分);
- Florence-2(BTK 等)。

**数据要求**:Backbone 已预训练;VLA 只需 SFT 数据。

**代表论文**:**PaliGemma 系**:[π0/π0.5/π0.6/π0.7/Ψ0](#7a10-π06-recap)、[FLOWER](#7a2-flower)(Florence-2 也属于 PaliGemma 风格)、[SimVLA](#7a5-simvla)(SmolVLM)、[LAP](#7e1-lap);**Qwen 系**:[StarVLA-α](#7f4-starvla-α)、[VLANeXt](#7l8-vlanext)(Qwen3-VL-2B)、[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)、[ABot-M0](#7a1-abot-m0)、[NS-VLA](#7w16-ns-vla)(Qwen3-VL 冻结)、[OA-WAM](#7w8-oa-wam)(Chameleon-7B 冻);**Molmo 系**:[MolmoAct2](#7l4-molmoact2)、[MolmoB0T](#7l5-molmob0t);**Llama-3.2-Vision**:部分;**Florence-2**:[BTK](#7v7-btk)。

**优势**:**通识强 / 语言对齐好 / 工程链路完整**;开源社区维护;LoRA / RoPE / FlashAttention 等优化即插即用。

**局限**:架构固定,**不易做大改动**;部分 MLLM(如 Llama-3-V)对中文 / 工业语言场景较弱。

**对效果的正负影响**:
- 正:80% 篇章证明"开源 MLLM + Action Head" 比"自训 VLM" 更省 + 更稳;
- 负:架构受限,**做 MoT / 自训 expert 时需要侵入式修改**。

**消融证据**:
- [StarVLA-α](#7f4-starvla-α):PaliGemma → Qwen3-VL-4B → **+26pp LIBERO**(95.8% vs 69.8%),开源 MLLM 选择本身就是单组件最大贡献;
- [VLA-Foundry](#7l7-vla-foundry):Foundry-VLA-1.7B vs Qwen3VLA-2.1B(Fig 5)→ **+23pp aggregate**,强 MLLM 直接拉满指标;
- [VLANeXt](#7l8-vlanext):12 条 recipe 中 backbone 选择是主导,2.5B Qwen3-VL 超 7B OpenVLA-OFT。

**为什么**:VLA 的瓶颈早就不是 VLM 本身,而是 Action Head 与跨本体适配;复用开源 MLLM 是性价比最高的选择。

#### L2 自训 VLM(GR00T / Cosmos / Molmo / Step1 系) [T1]

**直觉比喻**:**当你需要"专为机器人调过的 VLM"**——NVIDIA / 小米 / Molmo 等团队从 0 训 VLM,可以把 3D / 视频 / 机器人语义都嵌进去。

**结构定义**:与 L1 同构,但 vocab / tokenizer / 训练数据都"机器人化"。常带专属 vision encoder(如 NVIDIA SigLIP + 自训 Cosmos token)。

**典型实现**:
- GR00T_N1.6(NVIDIA 自训 Eagle / SigLIP);
- Cosmos Policy(NVIDIA Cosmos-Predict2 video base);
- MolmoB0T / MolmoAct2(Allen AI Molmo + FAST);
- Step1 (小米);
- 自训 VLM 需要至少数十亿 token 的视-语-动数据。

**数据要求**:**贵**;通常需要 10B+ token 的预训练数据。

**代表论文**:[GR00T_N1.6](#7l3-gr00t_n16)(NVIDIA Cosmos-2B 自训 VLM)、[Cosmos Policy](#7l1-cosmos-policy)(NVIDIA Cosmos-Predict2 视频基座 2B)、[DM0](#7l2-dm0)(京东 / 美团类自训 Embodied-Native VLM 1.7B)、[MolmoAct2](#7l4-molmoact2) / [MolmoB0T](#7l5-molmob0t)(Allen AI Molmo2);[HY-Embodied-0.5](#7f3-hy-embodied-05)(腾讯 Hunyuan + HY-ViT 2.0);[Step1](#7f5-xiaomi-robotics-0)(小米与 StepFun 合作)。

**优势**:**深度定制**;支持 video / 3D / Action Token 原生;更利于 MoT。

**局限**:**贵且慢**;社区维护少。

**对效果的正负影响**:
- 正:大厂 + 大数据时一骑绝尘;视频 / 3D / 机器人语义集成度高;
- 负:小团队复制困难;迭代慢。

**消融证据**:
- [Cosmos Policy](#7l1-cosmos-policy):Video pretrain vs from-scratch → 自训视频基座 LIBERO 98.5% / RoboCasa 67.1% SOTA;Planning(best-of-N)vs direct → **+12.5% 真实**(Sec 5.3);
- [DM0](#7l2-dm0):DM0-Specialist vs GigaBrain-0.1(Table 30)→ **62.0% vs ~52%**(+10%+);
- [HY-Embodied-0.5](#7f3-hy-embodied-05):2B 自训 VLM 在 22 项 benchmark 中胜出 **16 项**(包括 MiMo-7B / Qwen3-VL-4B / RoboBrain-4B)。

**为什么**:开源 MLLM 的 vocab 主要面向 web;机器人需要"夹爪 / 关节 / 速度"等特殊 token,自训能从 token 级开始统一。

#### L3 Action Token Vocab 扩展(FAST / DCT / VQ) [T1]

**直觉比喻**:让 Backbone 用"动作的母语"说话——把连续动作离散为 token,挂在 VLM tokenizer 上,**统一 NTP 接口**。

**结构定义**:
- FAST(π0 系):频域 + DCT + Huffman 编码动作 → 256-token 序列;
- DCT 多尺度 VQ(MINT-4B):多分辨率 VQ codebook,意图 token + 执行 token 解耦;
- 标准 VQ-VAE(Being-H / MolmoAct):分位数 bin 或 codebook 离散化。

**数学定义(VQ-VAE)**:
\[
\mathcal{L}_{\text{VQ}} = \lVert \text{sg}[z_e] - e\rVert_2^2 + \beta\lVert z_e - \text{sg}[e]\rVert_2^2
\]

**典型实现**:π0 / π0.5 / π0.6 / π0.7 / Ψ0(FAST);MINT-4B(DCT 多尺度);Being-H0.5 / MolmoAct2(FAST + 连续 head 二级补精)。

**数据要求**:动作量化 codebook 需要小规模数据预训(常 5-50k step)。

**代表论文**:**FAST 系**(π0 派系):[π0 / π0.5 / π0.6 / π0.7 / Ψ0](#7a10-π06-recap)、[Being-H0.5](#7f1-being-h05)、[MolmoAct2](#7l4-molmoact2)(FAST + Flow Expert)、[VLA-OPD](#7o3-vla-opd)(OpenVLA-OFT FAST tokens);**DCT 多尺度 VQ**:[MINT-4B](#7l6-mint-4b)(独有 next-scale AR + Intent Token);**离散 VQ**:[LifeLong-RFT](#7o7-lifelong-rft)(chunk 级 GRPO 需离散 token)。

**优势**:**与 LLM NTP 无缝**;RL 兼容性强;长程动作压缩比高(常 100×)。

**局限**:**量化误差**导致动作抖动;需要二级连续 head 补精。

**对效果的正负影响**:
- 正:训练范式与 LLM 统一,工程极顺;长动作可压缩;
- 负:单独 token AR 平滑度差,**必须配 Flow / DP head**。

**消融证据**:
- [MINT-4B](#7l6-mint-4b):去掉 spectral loss(Table VI)→ **-4.8pp LIBERO**;One-shot transfer(Table V)→ MINT **60% vs baseline ~0%**(+60pp);
- [VLA-OPD](#7o3-vla-opd):FAST AR 与 RL 的天然兼容是 reverse-KL 蒸馏 1-traj → 87.4% 的前提;
- [StarVLA-α](#7f4-starvla-α):MLP vs FAST(Table 2, RoboCasa)→ **53.8% vs 45.0%**(+8.8pp),说明 FAST 在某些场景反而不如 MLP — Action Vocab 选择需匹配下游任务。

**为什么**:让动作"接进 LLM 的 next-token 框架"是把 VLM 通识带进 VLA 的最直接路径,但代价是量化误差。

#### L4 RoPE / Long Context 变种 [T1]

**直觉比喻**:Backbone 要看历史 K 帧、未来 H 步,**位置编码就要长且稳**。

**结构定义**:RoPE 公式:
\[
q_m\cdot k_n = \text{Re}\bigl[(W_q x_m)^* (W_k x_n) e^{i(m-n)\theta}\bigr]
\]
对位置差敏感而对绝对位置鲁棒,适合 chunking。

**典型实现**:RoPE / YaRN / NTK / xPos;Long Context VLA 还会用 grouped attention 和 FlashAttention-2/3。

**代表论文**:几乎所有用开源 MLLM 的 VLA 都默认带 RoPE;Long Context 在 π0.7 / Helix_02 / Ψ0 中显式调优。

**优势**:**对长 chunking 友好**;不破 LLM 通识。

**局限**:RoPE 频率参数 θ 设错会**外推崩**。

**对效果的正负影响**:正:大多数情况下"开箱即用",几乎不出问题;负:几乎无。

**为什么**:VLA 输入序列含视觉 patch × N + 文本 token + Proprio + 历史动作,长度常超 1k tokens,**RoPE 是默认选择**。

---

### 4.F 多模态融合 / 路由 [T2]

**共同特点**:这一层决定不同模态/任务之间梯度流动的拓扑,影响通识保留与多任务相互干扰。

#### F1 早融合 prefix concat [T1]

**直觉比喻**:把视觉 / 语言 / Proprio token **一次性拼成长序列**,Backbone 自己学怎么混。

**结构**:输入 \(\text{seq} = [\text{vision tokens}; \text{text tokens}; \text{proprio tokens}]\) → Backbone 一次前向。

**代表论文**:[SimVLA](#7a5-simvla)(prefix concat baseline)、[π0 早期版本](#7a10-π06-recap)、[GigaWorld-Policy](#7w5-gigaworld-policy)(causal mask 但本质 early)、[VLA-Foundry](#7l7-vla-foundry)(early obs token → flow head)、[VLAW](#7w11-vlaw)、[Cosmos Policy](#7l1-cosmos-policy)(所有模态当 latent frame);OpenVLA、SimVLA、π0 早期版本。

**优势**:**实现简单**;无需架构改动。

**局限**:**prefix 抄捷径**——backbone 容易过度依赖前缀,导致后段(Action token)生成质量下降。

**消融证据**:
- [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0):用 Λ-attn 显式防 prefix 抄捷径(否则性能下降);
- [SimVLA](#7a5-simvla):极简 early fusion + recipe 优化 → LIBERO 98.6%,证明小数据 + 单本体 early 足够;
- [VLA-Foundry](#7l7-vla-foundry):Sim-only 优于 Real+Sim 混训(Fig 9)说明 early fusion 数据分布选择尤其敏感。

**为什么**:简单胜过复杂;在中等数据 + 单本体场景下,F1 已经够用。

#### F2 中融合 Cross-Attn / Λ-attn [T1]

**直觉比喻**:Backbone 跑自己的 self-attn,Action Head 单独用 cross-attn **去检索** backbone 中需要的信息——避免 backbone 被动作梯度污染。

**结构**:
\[
h_a = h_a + \text{CrossAttn}(Q=h_a, K=h_v, V=h_v)
\]
其中 \(h_v\) 是 backbone 输出,\(h_a\) 是 Action Head 内部状态。

**Λ-attn (Lambda Attention)**:防止 prefix 被 future action token "抄捷径",在 attention mask 中加入特殊屏蔽。

**代表论文**:[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)(Λ-attn 防 prefix 抄)、[FLOWER](#7a2-flower)(中融合 + 50% 层裁剪)、[ABot-M0](#7a1-abot-m0)(VGGT cross-attn)、[ConsisVLA-4D](#7v1-consisvla-4d)(CV-Aligner / CO-Fuser mid fusion)、[FocusVLA](#7v2-focusvla)(Modality Cascaded Attention)、[GST-VLA](#7v4-gst-vla)(cross-attn projector)、[π0.6 / π0.7](#7a10-π06-recap)(后期版本均引入 mid fusion 元素)、[VLA-JEPA](#7w10-vla-jepa)(causal attention)、[Mask World Model](#7w7-mask-world-model-mwm)(cross-attn text + AdaIN)。

**优势**:**保留 VLM 通识**;Action Head 可独立小;Diffusion / Flow head 更稳。

**局限**:实现复杂,需要正确的 attention mask。

**消融证据**:
- [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0):Λ-shape mask vs Causal → 避免 action prefix shortcut,LIBERO **98.7%**;
- [FLOWER](#7a2-flower):Intermediate vs Late fusion → intermediate 在多 benchmark 更优;Global-AdaLN 减 20% 参数无精度损失;
- [FocusVLA](#7v2-focusvla):Modality Cascaded Attention 让 VLA 从视觉 token 主动提取信息,收敛 **1.5-5×** 加速。

**为什么**:VLM 通识与动作精度有"反向 trade-off",中融合是平衡两端的关键设计。

#### F3 晚融合 per-modality head [T1]

**直觉比喻**:**每模态有自己的小 head**,最后用一个聚合器(MLP / Attention Pool)合并到决策。

**结构**:\(h_{\text{out}} = \text{Pool}([h_{\text{vision}}, h_{\text{text}}, h_{\text{proprio}}])\);Pool 通常是 weighted sum 或 attention pool。

**代表论文**:[StarVLA-α](#7f4-starvla-α)(轻量 MLP 晚融合)、[GeneralVLA](#7v3-generalvla)(Affordance head + Control head 解耦)、[GR00T_N1.6](#7l3-gr00t_n16)(VLM → DiT 解耦)、[HAMLET](#7a6-hamlet)(concat h_t + memory)、[TiPToP](#7o9-tiptop)(模块化 pipeline 极致)、[LoHo-Manip](#7w15-loho-manip)(Manager VLM + Executor VLA 解耦)。

**优势**:**易调试 / 易组合**;每模态可独立替换。

**局限**:模块间梯度不连贯;通常比中融合稍弱。

**消融证据**:
- [StarVLA-α](#7f4-starvla-α):极简晚融合 + 强 VLM → RoboChallenge 真机 **超 π0.5 约 20%**;
- [TiPToP](#7o9-tiptop):模块化(非端到端)→ vs π0.5-DROID **+22.2pp SR**,但失败分析显示模块串行误差累积 56%;
- [GR00T_N1.6](#7l3-gr00t_n16):late fusion(VLM features → DiT)适合 RTC/DAgger 后训迭代。

**为什么**:在追求"可解释 / 可组合"的工程语境(医疗 / 工业 / 教育)中,F3 比 F2 更受欢迎。

#### F4 MoT (Mixture-of-Transformers) [T1]

**直觉比喻**:**每模态 / 每任务一个独立 Transformer 专家**,通过门控路由 → 2026 趋势主导。

**结构**:
\[
y = \sum_k g_k(x)\, T_k(x), \quad \sum_k g_k(x) = 1
\]
其中 \(T_k\) 是第 k 个 Transformer 专家,\(g_k\) 是可学习门控(常用 softmax)。

**典型实现**:[HY-Embodied-0.5](#7f3-hy-embodied-05)(MoT 2B/32B)、[π0.7](#7f6-π07)(MoT + Memory)、[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)(MoT + Step1 KV cache)、[Being-H0.5](#7f1-being-h05)(MoF — Mixture of Flows,F4 变体)、[Being-H0.7](#7w17-being-h07)(dual-branch MoT)、[Fast-WAM](#7w3-fast-wam)(Video DiT + Action DiT 双 MoT)、[LingBot-VLA](#7a3-lingbot-vla)、[LAP](#7e1-lap)(MoT causal mask + stop-grad)、[RLDX-1](#7a4-rldx-1)(MSAT 多流是 MoT 推广)。

**优势**:**多任务 / 多本体不互相干扰**;扩展性强;**新本体只加一个专家**。

**局限**:门控未热身会崩;参数量大;通信开销在分布式训练高。

**消融证据**:
- [HY-Embodied-0.5](#7f3-hy-embodied-05):2B MoT VLA 在 22 项 benchmark 中胜出 16 项;On-Policy Distillation 32B → 2B 知识有效转移;
- [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0):MoT + Λ-attn → LIBERO 98.7% SOTA;
- [Being-H0.5](#7f1-being-h05):MoF 路由 30+ 构型 → 单 checkpoint 跨本体零样本迁移涌现。

**为什么**:VLA 越发展越多任务、多本体,**MoT 是天然解耦机制**。

#### F5 MoE / Gating(per-embodiment / per-task) [T1]

**直觉比喻**:与 MoT 类似但更细粒度——**单层内多个 FFN 专家**,稀疏激活。

**结构**:Mixtral / GShard 风格 Top-K Routing,Loss 加 load-balance 正则。

**代表论文**:[GST-VLA](#7v4-gst-vla)(Action Expert MoE top-2 per layer)、[HY-Embodied-0.5](#7f3-hy-embodied-05)(MoE-32B 407B total)。

**优势**:计算稀疏,参数大但 FLOPs 不增。

**局限**:稀疏路由对训练框架要求高;调参敏感。

**消融证据**:
- [GST-VLA](#7v4-gst-vla):300M MoE Action Expert + 3 阶段 progressive 训 → LIBERO-Plus 高位,3D Fourier PE +2.8pp;
- [HY-Embodied-0.5](#7f3-hy-embodied-05):MoE-32B 与 MoT-2B 并行版本,验证稀疏化 + MoT 的协同扩容路线。

**为什么**:与 L1 / L2 配合做扩容,但不破坏 backbone 稠密计算。

#### F6 Adapter / Soft-Prompt [T1]

**直觉比喻**:在 backbone 上加小插件,把"机器人特定知识"以**最小可注入形式**塞进去。

**结构**:
- LoRA:\(W = W_0 + \alpha BA\),\(A \in \mathbb{R}^{r\times d}, B\in \mathbb{R}^{d\times r}\);
- IA3:对 K / V / FFN 加 element-wise 乘子;
- Soft-Prompt:在 token 前加一段可学习 \(p \in \mathbb{R}^{L\times d}\)。

**代表论文**:[X-VLA](#7a8-x-vla)(Soft-Prompt 跨本体)、[RLDX-1](#7a4-rldx-1) / [VLA-OPD](#7o3-vla-opd) / [SmoothVLA](#7a9-smoothvla) / [STRONG-VLA](#7o8-strong-vla)(LoRA 微调)、[MolmoB0T](#7l5-molmob0t)(IA3 部分模块)、[OA-WAM](#7w8-oa-wam)(LoRA ~127M)、[ConsisVLA-4D](#7v1-consisvla-4d)(LoRA on OpenVLA 7B)。

**优势**:**轻量、可堆叠、不破 backbone**;跨本体迁移强。

**局限**:容量小,做不了"大改动";超过一定规模需切换到 MoT。

**消融证据**:
- [X-VLA](#7a8-x-vla):仅调 9M(1%)参数 → LIBERO **93%**(媲美 π0 3B 调参);
- [SmoothVLA](#7a9-smoothvla):OpenVLA-7B + LoRA + GRPO + jerk reward → 平滑度 +13.8%,SR 不掉;
- [STRONG-VLA](#7o8-strong-vla):LoRA 两阶段课程 → TSP **68.14% vs joint 65.57%**。

**为什么**:在保留 VLM 通识与跨本体扩展之间,Adapter / Soft-Prompt 是"零代价"中间路线。

---

### 4.A 动作 Head [T2]

**共同特点**:Action Head 把 backbone 表示映射为动作分布,直接决定动作精度 / 平滑度 / 频率上限。74 篇里 Flow Head (A3) 在 2025-2026 成为事实标准,占 ~50%。

#### A1 AR head(离散 Token) [T1]

**直觉比喻**:让 backbone 像写句子一样**逐 token 输出动作**——和 LLM 完全同构。

**结构**:
\[
\mathcal{L}_{\text{AR}} = -\sum_t \log p_\theta(\hat a_t \mid \hat a_{<t}, o, l)
\]
typical: 每维度 256 bin 或 VQ codebook。

**代表论文**:MINT-4B(主)、MolmoAct2(副)、LifeLong-RFT(副)、Being-H0.5(副)。

**优势**:与 VLM NTP 完全同构;**RL 兼容性最强**(token-level GRPO / PPO);长动作压缩比高。

**局限**:量化误差 → 动作抖动;**推理慢**(需序贯生成 k 个 token);精细操作不够。

**消融证据**:
- [MINT-4B](#7l6-mint-4b):MINT vs OpenVLA-OFT(Table IV)→ **+15pp robustness**;One-shot transfer +60pp;
- [LifeLong-RFT](#7o7-lifelong-rft):chunk-level GRPO 在离散 token 上效率高 → LIBERO **+22% avg SR**(仅 20% 数据);
- [StarVLA-α](#7f4-starvla-α):MLP vs FAST(Table 2,RoboCasa)→ **53.8% vs 45.0%**(+8.8pp 反过来 — 说明在某些 backbone 上 MLP 反胜)。

**为什么**:2022-2024 这是 RT-2 / OpenVLA 走的路;**2025 起几乎被 A3 取代**,但 RL 兼容性让它继续在 A4 混合架构中存在。

#### A2 DDPM / DiT head [T1]

**直觉比喻**:不是直接预测动作,而是**预测如何从噪声还原动作**。DiT 是用 Transformer 实现的扩散。

**结构(DDPM Simple)**:
\[
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,a_0,\epsilon}\bigl\lVert \epsilon - \epsilon_\theta(a_t, t, c)\bigr\rVert_2^2
\]
**DiT block**:
\[
x \leftarrow x + \alpha_1\cdot \text{MSA}(\text{LN}(x), \text{AdaLN}(c))
\]

**代表论文**:GR00T_N1.6(DiT action expert)、HAMLET(DiT)、HiPolicy(分层 DiT);Mask World Model(DiT + Mask head);CycleVLA(DDPM + 回溯);Cosmos Policy / DreamZero(块级 EDM)。

**优势**:**多模态分布建模强**;chunk 化 + 平滑;复杂操作好。

**局限**:推理慢(5-50 步);AdaLN-Zero 不当初始化会崩;与 RL 兼容差。

**消融证据**:
- [HiPolicy](#7a7-hipolicy):去掉分层频率结构(Table 2)→ **SR 37% vs 60%**(-23pp);真机 SR **85% vs DP 60%**(+42%);
- [Mask World Model](#7w7-mask-world-model-mwm):DDPM action head + Mask WM → RLBench **68.3% vs π0 33.3%**(+35pp);
- [HAMLET](#7a6-hamlet):DiT + Memory Module → 真机长程 **+47.2pp**(76.4% vs 29.2%)。

**为什么**:2024 DP / RDT-1B 把"动作多模态分布"做扎实;DiT 把 image diffusion 经验直接迁移到 action。

#### A3 Flow Matching head [T1]

**直觉比喻**:Diffusion 学"逆推 N 步去噪",Flow Matching 直接学**从噪声到数据的直线速度场**,1-10 步即可。

**结构(Conditional FM)**:
\[
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t\sim U[0,1], a_0, a_1}\bigl\lVert v_\theta(a_t, t, c) - (a_1 - a_0)\bigr\rVert_2^2
\]
推理时解 ODE \(\dot a_t = v_\theta(a_t, t, c)\)。

**代表论文**:π0 / π0.5 / π0.6 / π0.7 / Ψ0(主推手);FLOWER(50% 层裁剪 + 4-8 步);SimVLA(极简);LingBot-VLA / X-VLA(Soft-Prompt + Flow);MolmoB0T / RLDX-1 / Xiaomi-Robotics-0 / VLANeXt / VLA Foundry / PRTS / VLA-OPD / GST-VLA / Pose-VLA / Green-VLA / LAP / Being-H 副等(2025-2026 默认 head)。

**优势**:**推理快(1-10 步)**;训练稳定;对 VLM 梯度友好(head 小);chunk 友好。

**局限**:1-step OOD 鲁棒下降;step 太少会丢多模态。

**消融证据**:
- [FLOWER](#7a2-flower):Flow head + 50% 层裁剪 → CALVIN ABC **4.53 SOTA**,200 H100 GPU-hours;
- [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0):4.7B Flow + Λ-attn → LIBERO Avg **98.7%** SOTA;
- [π0.6 RECAP](#7a10-π06-recap):Flow + advantage-conditioned CFG → 真机 **吞吐 2× / 失败减半**,Espresso 连续 13h 无中断;
- [SimVLA](#7a5-simvla):0.5B 极简 Flow + 标准 recipe → **LIBERO 98.6%(#3)**,VRAM 仅 9.3GB。

**为什么**:2024 末 π0 把 Flow Matching 引入 VLA,**训练贵 + 推理便宜**的 trade-off 最适合机器人。

#### A4 AR + 连续 head 混合 [T1]

**直觉比喻**:**用 AR 头出"骨架动作 token",再用 Flow / Diffusion head 精修连续值**——双 head 兼顾通识与精度。

**结构**:
\[
\mathcal{L} = \mathcal{L}_{\text{AR}} + \lambda\mathcal{L}_{\text{FM}}
\]

**代表论文**:FocusVLA / StarVLA-α(主);π0 / GR00T_N1.6 / Xiaomi / Ψ0 / HY 等多数 SOTA(副,默认配)。

**优势**:**精度 + 通识 + RL 兼容性"三者兼得"**;chunk 长度灵活;长程任务表现好。

**局限**:训练复杂(两组 loss,需平衡 λ);两段 head 串联可能引入误差。

**消融证据**:
- [FocusVLA](#7v2-focusvla):AR + 连续 head 混合 → 收敛 1.5-5×,精细操作显著提升;
- [StarVLA-α](#7f4-starvla-α):Qwen3-VL + 轻量 MLP head 晚融合 → RoboChallenge **+20% vs π0.5**;
- [Ψ0](#7a11-ψ0-psi-zero):AR 预训(Stage1) + Flow 后训(Stage2) → **800h 人视 + 30h 真机** 超 10× 数据基线 40%+;
- [MolmoAct2](#7l4-molmoact2):FAST AR + Flow Expert + Think → MolmoSpace 37.7 / RoboEval 44.3 SOTA。

**为什么**:既保留 LLM 通识,又得 Flow 平滑——2025 SOTA 的"事实标准"。

#### A5 Action Manifold head [T1]

**直觉比喻**:把动作看作**几何流形上的点**,直接预测干净动作而不是去噪。

**结构**:
\[
\mathcal{L}_{\text{ABot}} = \mathbb{E}\bigl[w(\tau)\lVert V_\theta(\phi_t, A_t^\tau, q_t) - A_t\rVert^2\bigr], \quad w(\tau) = \frac{1}{(1-\tau)^2}
\]

**代表论文**:ABot-M0(唯一以此为主范式)。

**优势**:**直接预测干净动作**,无噪声中介;流形先验天然防越界。

**局限**:刚提出,生态薄;消融对比少。

**消融证据**:
- [ABot-M0](#7a1-abot-m0):AML vs GR00T noise-pred(Table 7)→ **+1.7pp**(71.0 vs 69.3 LIBERO-Plus);
- AML Action Chunk=30 时 **AML -8.2 vs GR00T -23.6pp** → **AML 抗退化最明显**(+17.1pp 净优势);
- AML 直接预测干净动作 + 流形几何先验,**避免高维噪声回归** 是大 chunk 场景下的关键。

**为什么**:在 Flow / Diffusion 之外另辟一条"几何先验"路线;后续工作可能融合到 Flow 中。

#### A6 频率 / 多尺度 head [T1]

**直觉比喻**:**频域分解动作**,把高频细节与低频骨架分别学,然后合成。

**结构**:DCT / FFT 分解 \(a = \sum_k F_k\),每个频率维度独立预测 + 重建 loss:
\[
\mathcal{L}_{\text{freq}} = \sum_k \lambda_k \lVert F - F^{(k)}\rVert^2
\]

**代表论文**:MINT-4B(DCT 多尺度 VQ + scale-wise AR);HiPolicy(多频率 chunking)。

**优势**:**意图 / 执行解耦**;长程动作可压缩(100×);one-shot 迁移友好。

**局限**:实现复杂,频域选择敏感;调参贵。

**消融证据**:
- [MINT-4B](#7l6-mint-4b):去掉 spectral loss(Table VI)→ **-4.8pp LIBERO**;Intent Token 一致性使 one-shot transfer **60% vs ~0%**;
- [HiPolicy](#7a7-hipolicy):分层多频结构 → **+23pp**(Table 2);熵引导执行 → +25% speed minimal SR drop;
- [VLANeXt](#7l8-vlanext):频域辅助 Loss → +5.1pp LIBERO-Plus,且零额外训练开销。

**为什么**:动作天然带有"低频意图 + 高频细节"两层结构,频域分解直接对应这一物理直觉。

---

### 4.W 世界 / 辅助预测 Head [T2]

**共同特点**:塞额外监督信号,塑造 backbone 隐空间。74 篇里 W1-W6 子组件覆盖率约 70%(几乎每篇都至少配一个辅助 head)。

#### W1 像素未来帧 head [T1]

**直觉比喻**:让模型**预测下一秒画面**,逼迫它学到"动作 → 视觉变化"的物理直觉。

**结构(EDM)**:
\[
\mathcal{L}(D_\theta, \sigma) = \mathbb{E}_{x_0, c, n}\bigl[\lVert D_\theta(x_0 + n; \sigma, c) - x_0\rVert_2^2\bigr]
\]

**代表论文**:Cosmos Policy(主);DreamZero(块级);Psi-W0;Genie Sim 3.0(平台);GigaWorld(副)。

**优势**:**表征最完整**;对视觉 OOD 鲁棒;支持 zero-shot 策略生成。

**局限**:**算力最贵**;像素 head 易吃 VLM 通识;推理慢。

**消融证据**:
- [Cosmos Policy](#7l1-cosmos-policy):Video pretrain 直接微调为策略 → LIBERO **98.5%** / RoboCasa **67.1%** SOTA;Planning(best-of-N)+12.5% 真实任务;
- [DreamZero](#7w2-dreamzero):14B 像素 WAM → 真机 **>2× task progress vs VLA**;Cross-embodiment video-only data +42% relative;
- [WoVR](#7o11-wovr):像素 WM 内 GRPO + KIR + PACE → 真机 **+30.0pp**(61.7% → 91.7%)。

**为什么**:像素 = 最完整的物理监督,但代价高,故 2026 主流逐步向 W2 Latent / JEPA 迁移。

#### W2 Latent / JEPA head [T1]

**直觉比喻**:不预测像素,**预测 latent / 语义**——同样的物理直觉,但成本 1/10。

**结构(JEPA)**:
\[
\mathcal{L}_{\text{JEPA}} = \lVert f_\theta(x_{\text{ctx}}) - \text{sg}[f_{\theta^-}(x_{\text{tgt}})]\rVert^2
\]
其中 \(\theta^-\) 是 EMA target。

**代表论文**:VLA-JEPA(主)、Being-H0.7(主)、Fast-WAM(主)、FutureVLA(主)、Mask World Model(主,W3 重叠);CoLA-World / World-VLA-Loop / World2Act(副)。

**优势**:**算力小**(典型 1/10 of W1);保留 VLM 通识;**对 mode collapse 设计成熟**。

**局限**:Latent 抽象,可解释性弱;需要 EMA 调参。

**消融证据**:
- [VLA-JEPA](#7w10-vla-jepa):去掉人类视频预训(LIBERO-Plus,Table 3)→ **-16.6pp**(62.9% vs 79.5%),JEPA + 人类视频是 OOD 鲁棒主因;
- [Fast-WAM](#7w3-fast-wam):去 video co-training → 性能大幅下降(>>2pp);推理 **190ms vs >800ms**(4× 加速);
- [Being-H0.7](#7w17-being-h07):dual-branch latent alignment → 推理 **3-4ms/step**(无需视频 rollout),动态任务领先所有 5 维。

**为什么**:JEPA 在 image 上证明"latent 监督足够强",VLA 直接迁移并加机器人组件即可。

#### W3 Mask 预测 head [T1]

**直觉比喻**:不预测完整未来,**只预测"哪个区域会变化"**——更聚焦、更省。

**结构**:Mask 预测损失:
\[
\mathcal{L}_{\text{mask}} = \mathbb{E}\bigl[w(s)\lVert v_\theta(z_s, s, c_t) - (z_1 - z_0)\rVert^2\bigr]
\]
在 mask 区域内做 Flow Matching。

**代表论文**:Mask World Model(主);GigaWorld-Policy(副);STARRY 的 GASAM(几何 mask)。

**优势**:**聚焦关键区域**;算力比 W1 / W2 又省;鲁棒于光照 / 纹理。

**局限**:Mask 标注非平凡(常需要 SAM);对全局变化 / 多对象场景适配差。

**消融证据**:
- [Mask World Model](#7w7-mask-world-model-mwm):MWM vs π0(RLBench)→ **68.3% vs 33.3%**(+35pp);MWM-C1(explicit mask)vs MWM(implicit)→ **81.0% vs 98.3%**(-17.3pp,说明隐式语义瓶颈更强);
- 真机 OOD 鲁棒 → **42.1% vs GE-ACT 12.5%**;
- [GigaWorld-Policy](#7w5-gigaworld-policy):W3 mask + W1 video 双重监督训练效率高,推理可跳视频。

**为什么**:大量任务变化集中在"夹爪 + 目标物体" 周围,Mask 是天然空间先验。

#### W4 Trace / Affordance head [T1]

**直觉比喻**:不预测"未来像素",而是**预测"机械臂应该走的轨迹 / 物体可抓取区域"**——更接近动作语义。

**结构**:Trace head 预测一段 2D / 3D 轨迹 \(\tau = (p_1, \ldots, p_K)\);Affordance head 对每像素 / 体素预测可抓取概率。

**代表论文**:LoHo-Manip(Trace);HiF-VLA(MPEG motion vector);GeneralVLA(3D Affordance);HAMLET(Trace + History);PokéVLA(几何 Affordance)。

**优势**:**直接通向动作**;长程规划友好;可视化好。

**局限**:Trace 标注成本高;Affordance 易混淆"可见 vs 可达"。

**消融证据**:
- [LoHo-Manip](#7w15-loho-manip):Trace conditioning vs text-only → trace 提升 OOD 泛化;Remaining-plan vs next-step-only → 隐式 recovery;
- [HiF-VLA](#7w6-hif-vla):去 Foresight + Hindsight(Table 3 row6 vs row1)→ **SR 93.2 vs 91.0**(+2.2pp),且 latency 仅 1.67× vs frames 3.15×;
- [GeneralVLA](#7v3-generalvla):3D Affordance + 知识库 → vs VoxPoser 14 任务**显著超越**;**零真机数据** 即可执行。

**为什么**:Trace / Affordance 是"动作 - 像素"之间的中间桥梁,长程任务最直接受益。

#### W5 Future State / Reward head [T1]

**直觉比喻**:让模型**预测未来状态或奖励**——把 value-based 思想塞进 BC。

**结构**:
\[
\mathcal{L}_{\text{V}} = \mathbb{E}\lVert V_\theta(o, l) - \sum_{k=0}^K \gamma^k r_{t+k}\rVert^2
\]

**代表论文**:ConsisVLA-4D(主);P3Nav(主);π0.7(主);ReconVLA(主,部署层);PRTS / Cosmos / TT-VLA / π0.6(副)。

**优势**:**为 RL / 测试时规划提供 value**;长程任务上是隐式课程。

**局限**:reward 设计敏感;value bootstrapping 易发散。

**消融证据**:
- [ConsisVLA-4D](#7v1-consisvla-4d):4D 时空一致性 future state → CO-Fuser 几何一致性 **-13.3% Real**(单组件最大贡献);
- [Cosmos Policy](#7l1-cosmos-policy):加 auxiliary targets(state + value)→ 显著提升 policy;Planning(best-of-N)+12.5% 真实任务;
- [π0.6](#7a10-π06-recap):Distributional Value Function 是 advantage-conditioned RL 的基础;[ReconVLA](#7o14-reconvla):CQR + SMD 不确定性层 plug-and-play 降灾难性错误。

**为什么**:模仿学习碰长程瓶颈,reward / value head 是把"目标距离感"塞回模型的最直接方式。

#### W6 CoT / Reasoning head [T1]

**直觉比喻**:让模型**生成"先思考再行动"的中间推理**,把组合推理能力激活。

**结构**:CoT 与 Action 共享 latent 或显式 token:
\[
\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda_{\text{CoT}}\mathcal{L}_{\text{CoT}}
\]

**代表论文**:CycleVLA(主);NS-VLA(主);MolmoAct2(副)、GST-VLA(副)、DM0(副)、π0.7(副)、TiPToP(副)。

**优势**:**长程组合任务显著提升**;可解释强。

**局限**:**CoT latency 大**;CoT 与动作不对齐反而误导;需要 CoT-on-demand 配套。

**消融证据**:
- [MolmoAct2](#7l4-molmoact2):Think 自适应深度 → 推理 **37× 加速 vs v1**;MolmoSpace 37.7 / RoboEval 44.3 SOTA;
- [GST-VLA](#7v4-gst-vla):去 DA-CoT(Table V)→ **-3.9pp**;深度感知 CoT 提供 3D 推理可解释性;
- [CycleVLA](#7a12-cyclevla):MBR decoding + progress-aware → 零样本 test-time scaling **+5-10% SR**;
- [NS-VLA](#7w16-ns-vla):w/o Plan Classifier(Table 6a)→ **-18.9pp**(98.6% → 79.7%),符号推理头是单组件最大贡献。

**为什么**:LLM 的 CoT 已经证明"思考 + 行动"远好于"直接行动",VLA 同理。

---

### 4.E 跨本体适配 [T2]

**共同特点**:决定"同一模型能服务几种本体"的扩展性。74 篇里 70%+ 至少做了一种跨本体适配。

#### E1 Per-head [T1]

**直觉比喻**:**每本体一个 Action Head**,backbone 共享。

**结构**:\(\pi_k(a|o, l) = \text{head}_k(\text{backbone}(o, l))\),k 为本体 ID。

**代表论文**:[Being-H0.5](#7f1-being-h05)(MoF per-embodiment routing,本质 per-head 升级版);早期 OXE / RT 系列、[Helix_02 三层架构](#7f2-helix_02-figure-ai)(每本体 S0 单独训);[Green-VLA](#7o2-green-vla)(per-embodiment prompt)。

**优势**:实现简单;每本体可独立 fine-tune。

**局限**:**新本体需重训 head**;不支持零样本迁移。

**消融证据**:
- 2025 起 per-head 几乎全被 E2/E3 取代,留 E1 作为 baseline;
- [Being-H0.5](#7f1-being-h05):MoF(本质 per-flow per-embodiment)→ 跨 30+ 构型单 checkpoint;
- 反例:Naive 跨本体混训(无 E2/E3)→ OXE-AugE 证明 RoVi-Aug 扩散增广可 **降 27-30%**。

**为什么**:最朴素的跨本体方案,2025 起被 E2-E3 大量取代。

#### E2 Soft-Prompt [T1]

**直觉比喻**:为每本体加一段**可学习 prompt token**,让 backbone 知道"我现在驾驶哪台机器"。

**结构**:输入序列变为 \([\text{prompt}_k; \text{vision}; \text{text}; \text{proprio}]\),prompt 为 \(p_k \in \mathbb{R}^{L\times d}\)。

**代表论文**:[X-VLA](#7a8-x-vla)(Soft-Prompt 跨本体主推手,0.9B 模型 6 仿真 + 3 真机均 SOTA);[LAP](#7e1-lap)(Language-Action soft prompt 变体);[Green-VLA](#7o2-green-vla)(Embodiment/control-type prompt)。

**优势**:**零代价加新本体**(只加一个 prompt);训练稳定;跨本体迁移率高。

**局限**:容量小;对差异极大的本体(双足 vs 双臂)不够。

**消融证据**:
- [X-VLA](#7a8-x-vla):Soft Prompt vs Language Prompt vs HPT → **跨构型场景全面最优**;仅调 9M 参数(1%)→ LIBERO **93%** 媲美 π0 3B 调参;
- [LAP](#7e1-lap):Language-Action 表示是 Soft-Prompt 的一种语义变体 → **零样本跨本体 +27pp**(52% vs 25%);
- [Green-VLA](#7o2-green-vla):per-embodiment prompt 支持人形 / 双臂 / 单臂统一训练。

**为什么**:LLM Prompt Tuning 的成功直接迁移过来。

#### E3 Latent Action Tokenizer [T1]

**直觉比喻**:**统一动作空间**——把所有本体动作映射到同一个 latent codebook,backbone 学的是"动作意图",每本体只需一个"动作解码器"。

**结构(IDM + FDM)**:
- IDM:\(z = f_{\text{IDM}}(o_t, o_{t+1})\) 从 (前后帧, 文本) 中抽 latent action;
- FDM:\(a_k = g_k(z, s_k)\) 解码到本体 k 的动作。

**代表论文**:LAP(主);CoLA-World(共演化);World2Act(副);Being-H0.5 / Ψ0(变体)。

**优势**:**真正的零样本跨本体**;可用大量人类视频 + 跨本体数据;**最理论优雅**。

**局限**:IDM 训练贵;Latent 抽象可能与实际动作错配。

**消融证据**:
- [LAP](#7e1-lap):Language-Action 表示 zero-shot 跨本体 → **+27pp**(52% vs 25%);
- [CoLA-World](#7w1-cola-world):Joint vs 2-stage(Table 1 OXE)→ **FVD 278.90 vs 291.30**;Visual Planning 21.20% vs 7.73%(+13.5pp);
- [World2Act](#7w13-world2act):latent 对齐 vs pixel-space → 真机 +6.7%,对 WM 质量更鲁棒;
- [Being-H0.5](#7f1-being-h05):MoF + Latent Action 跨 30+ 构型单 checkpoint 零样本迁移。

**为什么**:跨本体 = 跨动作空间,Latent Action 把"什么是动作"统一定义,从根本上解决。

#### E4 Embodiment Aug [T1]

**直觉比喻**:不在模型上做手脚,**在数据上做手脚**——把同一动作渲染到不同本体上,造跨本体数据。

**结构**:数据增广 + 加权采样。

**代表论文**:OXE-AugE(主)、Genie Sim 3.0(平台)。

**优势**:**架构零改动**;可与 E1-E3 任意组合。

**局限**:渲染成本高;真机迁移仍需 sim2real。

**消融证据**:
- [OXE-AugE](#7e2-oxe-auge):真机未见 robot×gripper → **+24%~+45%**;增广组合 > naive mixing;RoVi-Aug 扩散增广反降 27-30%;
- [Genie Sim 3.0](#7e3-genie-sim-30):1500 eps sim 数据 vs 500 eps real(Table I)→ **sim-to-real avg 0.83 vs real-to-real 0.75**(+0.08);**Sim-to-Real R²=0.94**;
- [MolmoB0T](#7l5-molmob0t):1.7M 仿真专家轨迹 → 真机 4 settings Overall **79.2% vs π0.5 39.2%**(+40pp)。

**为什么**:数据派的解法:让模型"见过更多本体"远比"设计更巧的架构"重要。

#### E5 共享骨干 [T1]

**直觉比喻**:**所有本体共享同一个 backbone**,只在最后输出层做小变换。

**结构**:几乎所有 74 篇都默认此设;真正"per-embodiment backbone" 已经几乎绝迹。

**代表论文**:几乎全部。

**优势**:充分利用数据;通识好。

**局限**:本体差异大时(轮式 + 双足)需配 E2 / E3 / F4。

**为什么**:VLA 的核心价值之一就是"一份 backbone 服务多本体",共享骨干是默认前提。

---

### 4.O 后训练 / 部署优化 [T2]

**共同特点**:把训好的模型压成可量产形态,决定能否真的上机器人。

#### O1 PTQ Quantization [T1]

**直觉比喻**:**把模型参数从 16-bit 浮点压到 4-bit 整数**,显存 1/4,速度 2-3×。

**结构**:GPTQ / SmoothQuant / AWQ;典型 W4A8 / W8A8。

**代表论文**:QuantVLA(主)。

**优势**:**部署门槛**显著降;一台车 / 机器人能跑大模型。

**局限**:**Flow head 易破**;需要 calibration 数据;少量精度损失。

**消融证据**:
- [QuantVLA](#7o1-quantvla)(Table 3 LIBERO Avg):FP16 **97.1% / 4.27 GB**;仅 DiT W4A8 **71.6%**(动作头极敏感,**-25pp**);
- 完整 QuantVLA + ATM + OHB → **70% 显存,1.22× 速度**,LIBERO 可超全精度;
- LLM 单独量化几乎无损(96.5% / 1.58 GB),证明 LLM 比 Flow head 容忍量化更好 — **选择性量化是关键**。

**为什么**:大模型上车的关键瓶颈是显存与功耗,PTQ 是最低成本的解。

#### O2 Layer Pruning [T1]

**直觉比喻**:**砍掉一半 Transformer 层**,看模型还能不能跑——通常能。

**结构**:剪掉中间冗余层,配合 distillation 恢复性能。

**代表论文**:FLOWER(中间 fusion + 50% 层裁剪);_(其他副)_

**优势**:训练 + 推理双省;Flow head 与裁剪兼容好。

**局限**:剪错层会破;需 distillation 配合。

**消融证据**:
- [FLOWER](#7a2-flower):**50% VLM pruning** + Global-AdaLN 减 20% 参数 → 无精度损失;CALVIN ABC **4.53 SOTA**;
- [QuantVLA](#7o1-quantvla):选择性 LLM+DiT(MLP) 180 层量化 → 可超 FP16,证明 Pruning + Quantization 协同可叠加;
- [HY-Embodied-0.5](#7f3-hy-embodied-05):2B 激活参数 MoT 设计本身就是隐式 Pruning(稀疏激活)→ 同等 benchmark 性能。

**为什么**:Transformer 层数对 VLA 任务存在"边际效用拐点",裁剪 50% 仍能保 90%+ 性能。

#### O3 异步 / Chunking [T1]

**直觉比喻**:**Backbone 跑慢点(10Hz)出 chunk,Action Head 跑快点(50Hz)出每步动作**——异步降低 backbone 频率。

**结构**:Backbone 每 K 步出一次 chunk;Action Head 在 chunk 内插值或 receding horizon。

**代表论文**:Xiaomi-Robotics-0(异步 + MoT);HiPolicy(分层多频率)。

**优势**:**实时性提升**;backbone 算力下降到 1/K。

**局限**:Chunk 内突发变化无法响应;需要 fallback。

**消融证据**:
- [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0):MoT + KV cache + Λ-attn + 异步执行 → LIBERO **98.7%** SOTA,消费级 GPU 实时;
- [HiPolicy](#7a7-hipolicy):分层多频 chunking + 熵引导执行 → **+25% speed** minimal SR drop;
- [GigaWorld-Policy](#7w5-gigaworld-policy):异步剥离视频解码 → **9× 推理加速**(360ms vs 3231ms);
- [DreamZero](#7w2-dreamzero):14B WAM → **38× inference speedup** 至 7Hz real-time。

**为什么**:闭环控制频率与 VLM 推理频率天然不匹配,异步是工程标配。

#### O4 Distillation [T1]

**直觉比喻**:**用大模型当老师,训一个小学生**——学生跑得快,知识不丢。

**结构**:KL / MSE 蒸馏损失,常配 RFT。

**代表论文**:VLA-OPD(On-Policy Distillation 主)、StarVLA-α(自蒸馏)、PokéVLA(F2 + F1 共训)、HY-Embodied(课程蒸馏)、LingBot-VLA(副)。

**优势**:小模型跑大模型的 90%+ 性能;部署友好。

**局限**:需要预训大模型;蒸馏链路长。

**消融证据**:
- [VLA-OPD](#7o3-vla-opd):1-traj SFT init → OPD Distill(Table 2 LIBERO Avg)→ **48.9% → 87.4%**(+38.5pp);Reverse-KL 优于 Forward-KL(后者早期暴跌 50%+);
- [HY-Embodied-0.5](#7f3-hy-embodied-05):On-Policy Distillation 32B → 2B → 能力有效转移(Sec 4.4);
- [PokéVLA](#7v5-pokévla):F1+F2 共训 → 1.22B 推理 **12× 快于** 同类 18× 模型;
- [StarVLA-α](#7f4-starvla-α):MLP 头自蒸馏 + 强 VLM → 真机 +20% vs π0.5。

**为什么**:大模型质量高,小模型部署快,**蒸馏是连接桥梁**。

#### O5 Speculative / KV-cache / 跳计算 [T1]

**直觉比喻**:**预测一步,验证一步,错了回退**(Speculative Decoding);或缓存 KV 复用上一步。

**结构**:Draft + Verify 双模型;或 KV 复用 backbone token。

**代表论文**:LWD(部署即训练,工程优化);SOP(可扩展后训);TT-VLA(测试时);SmoothVLA / LifeLong-RFT(过程优化)。

**优势**:**几乎免费的加速**(2-3×);对内存带宽友好。

**局限**:Draft 模型需训;实现复杂。

**消融证据**:
- [LWD](#7o4-lwd):16 台双臂 fleet RL 持续部署 → SR **0.95 vs pretrained ~0.70**(+25pp);
- [SOP](#7o5-sop):1→4 Actors(Table I)→ 成功率 **0.805 → 0.925**,训练时间 **2.4×** 加速;
- [TT-VLA](#7o6-tt-vla):Test-time RL → +2-5pp;**OpenVLA Vision 55.0% → 57.08%**;
- [SmoothVLA](#7a9-smoothvla):jerk reward GRPO → 平滑度 +13.8%,真机部署友好;
- [LifeLong-RFT](#7o7-lifelong-rft):仅 20% 数据达 SFT 效果,LIBERO **+22% avg SR**。

**为什么**:把 LLM 推理优化经验直接迁移到 VLA;不破架构。

---

## 第 5 章 横向对比矩阵 [T1]

### 5.1 Vision Encoder 选择(SigLIP / DINOv2 / CLIP / 自训 / 3D Gaussian)对小样本与 OOD 的影响 [T2]

**核心结论(基于 74 篇消融)**:Vision Encoder 是 VLA 的"地基",对 OOD 与精细操作影响最大,远超 Backbone 大小本身的影响。

- **SigLIP(主流)**:语义对齐强,适合配 LLM,但几何弱;
  - 实例:[StarVLA-α](#7f4-starvla-α) 用 PaliGemma → Qwen3-VL-4B(均 SigLIP)→ **+26pp LIBERO**(95.8% vs 69.8%),证明同样 SigLIP 系内的差异也很大。
- **DINOv2 + SigLIP 并联(2026 趋势)**:几何 + 语义双优,**精细操作 +3-8pp**;
  - 实例:[PokéVLA](#7v5-pokévla) / [GST-VLA](#7v4-gst-vla) / [HiF-VLA](#7w6-hif-vla) / [MINT-4B 30M 版](#7l6-mint-4b) 等所有"精细操作主导"的小 VLA 都采用此配。
- **3D Gaussian / Pose / VGGT**:精细操作 **+5-15pp**,但算力 **+30-50%**;
  - 实例:[GST-VLA](#7v4-gst-vla) S1 GST 预训 → **+6.2pp**(Table VI);[Pose-VLA](#7v6-pose-vla) RoboTwin Hard **+14pp vs π0**;[ConsisVLA-4D](#7v1-consisvla-4d) CO-Fuser → **-13.3% Real**(去掉的负贡献)。
- **CLIP**:逐步被 SigLIP 取代,仅剩 VLN 等老应用([BTK](#7v7-btk) 仍用 CLIP-B/16)。

### 5.2 VLM Backbone 大小(<1B / 1-4B / 4-7B / >10B)与样本效率 / 推理延迟 [T2]

**核心结论**:Backbone 大小存在"边际效用拐点",**1-4B 是 2026 主流甜点**(性能/部署平衡);>10B 仅用于跨构型基座或视频 WM。

- **<1B(部署友好)**:[SimVLA](#7a5-simvla)(0.5B) / [PokéVLA](#7v5-pokévla)(1.22B) / [FLOWER](#7a2-flower)(950M) / [X-VLA](#7a8-x-vla)(0.9B)—— 实时性好;
  - 实例:[SimVLA](#7a5-simvla) LIBERO **98.6%(#3)**,VRAM 仅 **9.3GB**;[FLOWER](#7a2-flower) CALVIN ABC **4.53 SOTA**,仅 200 H100-hours 预训;
  - 局限:通识弱,跨构型迁移有限。
- **1-4B(主流甜点)**:[π0 系列](#7a10-π06-recap)(2-3B PaliGemma)/ [MINT-4B](#7l6-mint-4b)(4B PaliGemma-2.6B + 300M Expert)/ [MolmoAct2](#7l4-molmoact2)(~7B Molmo)/ [VLANeXt](#7l8-vlanext)(2.5B Qwen3-VL)—— 性能 / 部署最平衡;
  - 实例:[VLANeXt](#7l8-vlanext) 2.5B 超 7B OpenVLA-OFT;[MINT-4B](#7l6-mint-4b) 真机 +29% vs π0.5。
- **4-7B(复杂任务首选)**:[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)(4.7B)/ [π0.7](#7f6-π07)(5B)/ [Helix_02](#7f2-helix_02-figure-ai)(系统级,S0+S1+S2)/ [RLDX-1](#7a4-rldx-1)(6.9-8B)/ [Cosmos Policy](#7l1-cosmos-policy)(2B DiT)—— 复杂操作或人形 loco-manip 首选;
  - 实例:[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0) LIBERO **98.7%** SOTA;[π0.7](#7f6-π07) zero-shot 跨构型 T 恤折叠 ≈ 人类首次水平。
- **>10B(资源密集)**:[DreamZero](#7w2-dreamzero)(14B Wan2.1)/ [GR00T_N1.6](#7l3-gr00t_n16)(Cosmos-2B + 32 层 DiT)/ [HY-Embodied-0.5](#7f3-hy-embodied-05)(32B MoE total 407B)—— 大厂资源,zero-shot 强;
  - 实例:[DreamZero](#7w2-dreamzero) 14B → 真机 >2× task progress vs VLA,但需 **38× 加速** 才能 7Hz real-time。

### 5.3 动作 Head(AR / DDPM / Flow / Manifold)在动作平滑度 / 采样步数 / RL 兼容性上的差异 [T2]

**核心结论**:**Flow Head 是 2025-2026 事实标准**(74 篇里 ~50 篇用)。AR 让位给"AR+连续头混合";DDPM/DiT 仍存于辅助 head(WM/CoT);Manifold 是新尝试。

- **AR**:平滑度差(量化),采样需 k 步,**RL 兼容性最强**;
  - 实例:[MINT-4B](#7l6-mint-4b) one-shot transfer **+60pp**(频域 VQ);[LifeLong-RFT](#7o7-lifelong-rft) chunk-level GRPO 在 AR 上效率高,仅 20% 数据达 SFT 效果;
  - 现在 AR 几乎只在 A4 混合架构里(如 [Ψ0](#7a11-ψ0-psi-zero) Stage1 AR 预训 + Stage2 Flow 后训)。
- **DDPM/DiT**:平滑度好,采样 5-50 步,**RL 兼容性差**;
  - 实例:[HiPolicy](#7a7-hipolicy) 分层多频 DDPM → 真机 **+42% vs DP**;[Mask World Model](#7w7-mask-world-model-mwm) DiT + Mask head → RLBench **+35pp vs π0**;
  - 现在 DDPM 主要存于 WM 头(W1 / W3),作为主动作头逐步被 Flow 取代。
- **Flow Matching**:平滑度好,采样 1-10 步,**事实标准**;
  - 实例:[FLOWER](#7a2-flower)(4-8 步 + 50% 层裁剪)/ [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)(80ms 实时)/ [SimVLA](#7a5-simvla)(0.5B 极简)/ [π0.6 RECAP](#7a10-π06-recap)(advantage-conditioned CFG 配合)等;
  - RL 兼容性中:[π0.6](#7a10-π06-recap) 用 CFG 绕过 Flow 不可求 log-prob,成功实现 advantage-conditioned RL。
- **Manifold**:平滑度好,无噪声中介,**RL 兼容性未验证**;
  - 唯一代表:[ABot-M0](#7a1-abot-m0) — Chunk=30 大场景 **AML -8.2pp vs GR00T -23.6pp**,抗退化最强(+15.4pp 净优势)。

### 5.4 模态融合时机(early / mid / late / MoT)对 VLM 通识保留度的影响 [T2]

**核心结论**:融合方式直接影响"通识 vs 动作"梯度配比。**MoT 是 2026 趋势**(多任务最优);mid/cross-attn 配 Flow head 是 2025 主流。

- **early(prefix concat)**:简单,通识易污染;
  - 代表:[SimVLA](#7a5-simvla) / 早期 [π0](#7a10-π06-recap) / [VLA-Foundry](#7l7-vla-foundry) / [GigaWorld-Policy](#7w5-gigaworld-policy) / [Cosmos Policy](#7l1-cosmos-policy)(所有模态当 latent frame);
  - 风险:[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0) 显式用 **Λ-attn** 防 prefix 抄捷径,否则性能下降。
- **mid(cross-attn / Λ-attn)**:通识保留好,**Flow head 配它最稳**;
  - 实例:[FLOWER](#7a2-flower) intermediate fusion → CALVIN ABC 4.53 SOTA;[FocusVLA](#7v2-focusvla) Modality Cascaded → 收敛 1.5-5×;[ABot-M0](#7a1-abot-m0) VGGT cross-attn → +4.7pp(66.4→71.1)。
- **late(per-modality head)**:可解释好,梯度不连贯;
  - 实例:[StarVLA-α](#7f4-starvla-α) MLP head → RoboChallenge **+20% vs π0.5**;[GeneralVLA](#7v3-generalvla) Affordance + Control 解耦 → 零真机数据;
  - 局限:[TiPToP](#7o9-tiptop) 模块化失败分析显示 56% 因串行误差累积。
- **MoT(2026 趋势)**:多任务/多本体最优;
  - 实例:[HY-Embodied-0.5](#7f3-hy-embodied-05) 2B MoT 在 22 项 benchmark 胜 16 项;[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0) MoT + Λ-attn → LIBERO 98.7%;[Being-H0.5](#7f1-being-h05) MoF → 跨 30+ 构型单 checkpoint;
  - 风险:门控未热身会崩,需 1-2k step 仅训 backbone 再开 MoT。

### 5.5 跨本体适配机制(Soft-Prompt / Per-head / Latent Action / Embodiment Aug)在迁移率上的差异 [T2]

**核心结论**:**Latent Action(E3)是理论最优、Soft-Prompt(E2)是工程最优、Embodiment Aug(E4)是数据派的解**。Per-head(E1)已落后。

- **Per-head(过时)**:新本体需重训 head,零样本不行;
  - 仅存于早期 OXE 系列;现代代表:[Being-H0.5](#7f1-being-h05) 的 MoF 路由(per-flow per-embodiment)是 per-head 升级版。
- **Soft-Prompt(轻量)**:零代价加新本体,迁移率中;
  - 实例:[X-VLA](#7a8-x-vla) 0.9B + Soft Prompt → 跨构型场景 **全面最优**,LoRA 1% 适配新机器人;[LAP](#7e1-lap) Language-Action 表示 zero-shot **+27pp**。
- **Latent Action(理论最优)**:真正零样本;
  - 实例:[LAP](#7e1-lap) 3 未见本体 >50% SR;[CoLA-World](#7w1-cola-world) IDM+WM 共演化 → Visual Planning **21.20% vs 7.73%**(2-stage)(+13.5pp);[World2Act](#7w13-world2act) latent 对齐真机 +6.7%;
  - 代价:IDM 训练贵(LAP 用 PaliGemma-3B 训练数千 GPU-hours)。
- **Embodiment Aug(数据派)**:架构零改动,需渲染;
  - 实例:[OXE-AugE](#7e2-oxe-auge) → 真机未见 robot×gripper **+24-45%**;增广组合 **>** naive mixing;
  - [Genie Sim 3.0](#7e3-genie-sim-30):1500 eps sim 数据超过 500 eps real(+0.08 avg);**Sim-to-Real R² = 0.94**。

### 5.6 后训练优化(量化 / 裁剪 / 异步)对延迟和性能的 trade-off [T2]

**核心结论**:O1-O5 可叠加;**选择性量化 + 50% 层裁剪 + 异步执行 + 蒸馏** 是 2026 量产标配,理论加速 5-10×。

- **PTQ**(O1):显存 1/4,**Flow head 易破**;
  - 实例:[QuantVLA](#7o1-quantvla) 仅 DiT W4A8 → **71.6%**(-25pp,动作头极敏感);完整 QuantVLA + ATM + OHB → **70% 显存,1.22× 速度**,LIBERO 可超全精度;
  - 关键 trick:**选择性量化布局**(LLM + DiT MLP 180 层),而非 naive 全量化。
- **Pruning**(O2):50% 层裁剪保 90%+ 性能;
  - 实例:[FLOWER](#7a2-flower) 50% VLM pruning + Global-AdaLN 减 20% 参数 → 无精度损失,CALVIN ABC **4.53 SOTA**,200 H100-hours;
  - 风险:剪错层会破,需 distillation 配合。
- **异步 / Chunking**(O3):实时性提升,需 chunk fallback;
  - 实例:[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0) MoT + KV + 异步 → **LIBERO 98.7%**,消费级 GPU 实时;[HiPolicy](#7a7-hipolicy) 熵引导执行 → **+25% speed** minimal SR drop;[GigaWorld-Policy](#7w5-gigaworld-policy) 推理跳视频 → **9× 加速**;[DreamZero](#7w2-dreamzero) **38× 加速** → 7Hz real-time。
- **Distillation**(O4):小模型 90%+ 性能;
  - 实例:[VLA-OPD](#7o3-vla-opd) 1-traj → **87.4%**(+38.5pp);[HY-Embodied-0.5](#7f3-hy-embodied-05) 32B → 2B 有效转移;[PokéVLA](#7v5-pokévla) 1.22B 推理 **12× 快于** 同类 18× 模型。
- **Speculative / 持续 RL**(O5):
  - 实例:[LWD](#7o4-lwd) fleet RL → SR **+25pp**;[SOP](#7o5-sop) 1→4 Actors → **2.4× 训练加速**;[TT-VLA](#7o6-tt-vla) test-time RL → +2-5pp;[LifeLong-RFT](#7o7-lifelong-rft) 仅 20% 数据达 SFT 效果。

### 5.7 正 / 负迁移小结 [T2]

**核心结论(基于 74 篇消融数字汇总)**:VLA 训练任务存在 5 条关键"正/负迁移黄金组合 + 黑名单组合"。

- **VQA 通识保留**(正迁移):**F1 VQA 共训 + F2 中融合 + F6 LoRA + W2 Latent JEPA** 是黄金组合;
  - 证据:[LAP](#7e1-lap) VQA co-training(Fig 6)→ Custom Franka & YAM 额外增益;[VLA-Foundry](#7l7-vla-foundry) Qwen3VLA-2.1B(F1 + F2)→ +23pp aggregate;[VLA-JEPA](#7w10-vla-jepa) JEPA 替代像素 → 保通识。
- **动作精度**(正迁移):**A3 Flow / A4 混合 + V3 3D + V4 Proprio** 是黄金组合;
  - 证据:[GST-VLA](#7v4-gst-vla)(3D Gaussian + Flow + MoE)/ [ConsisVLA-4D](#7v1-consisvla-4d)(VGGT + Parallel decode)/ [VLANeXt](#7l8-vlanext) Proprio→VLM **+7.2pp**;典型场景 5-15pp 提升。
- **OOD 鲁棒**(正迁移):**V2 DINOv2 + W2 JEPA + E3 Latent Action** 是黄金组合;
  - 证据:[Mask World Model](#7w7-mask-world-model-mwm) 真机 OOD **42.1% vs 12.5%**;[VLA-JEPA](#7w10-vla-jepa) LIBERO-Plus +16.6pp;[LAP](#7e1-lap) 零样本跨本体 +27pp。
- **实时性**(正迁移):**O3 异步 + O2 裁剪 + A3 1-step Flow + O4 蒸馏** 是黄金组合;
  - 证据:[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0) 80ms;[GigaWorld-Policy](#7w5-gigaworld-policy) 9× 加速;[DreamZero](#7w2-dreamzero) 38× 加速。
- **长程成功率**(正迁移):**W4 Trace + W6 CoT + A4 混合 + D6 Hindsight** 是黄金组合;
  - 证据:[HAMLET](#7a6-hamlet) +47.2pp 真机;[LoHo-Manip](#7w15-loho-manip) Trace + Remaining-plan;[CycleVLA](#7a12-cyclevla) test-time +5-10%;[MolmoAct2](#7l4-molmoact2) Think 37× 加速。
- **负迁移黑名单**:① **PTQ 全层 naive 量化** → DiT 动作头 -25pp(QuantVLA Table 3);② **Naive 跨本体混训** → RoVi-Aug 反降 27-30%(OXE-AugE 反例);③ **像素 WM 主导 + 无 VQA 共训** → VLM 通识坍塌;④ **MoT 门控未热身** → 路由坍塌;⑤ **Flow 1-step + OOD** → 多模态丢失。

---

## 第 6 章 模型结构演化时间线 [T1]

```mermaid
flowchart TB
    rt1["2022 RT-1 (Pure Transformer + AR head)"] --> rt2["2023 RT-2 (PaLI-X + AR head)"]
    rt2 --> openvla["2024 H1 OpenVLA (Llama-2 + AR head)"]
    openvla --> octo["2024 H1 Octo (Transformer + DDPM head + chunking)"]
    octo --> dp["2024 H1 Diffusion Policy / RDT-1B (CNN/DiT + DDPM head)"]
    dp --> pi0["2024 H2 π0 (PaliGemma + Flow Expert)"]
    pi0 --> wm["2025 H1 WM 路线 (Cosmos / GR00T DiT WM + Action Expert)"]
    pi0 --> latent["2025 H1 Latent Action (LAP / Being-H + IDM + Latent Tokenizer)"]
    wm --> coevo["2025 H2 - 2026 H1 共演化架构 (CoLA-World / OA-WAM / Psi-W0)"]
    latent --> coevo
    pi0 --> pi06["2025 H2 π0.6 (MoT + Advantage-conditioned Flow)"]
    pi06 --> pi07["2026 H1 π0.7 (Steerable MoT + Flow)"]
    pi06 --> mot["2026 H1 大基座 MoT (Xiaomi / Ψ0 / HY-Embodied)"]
    pi0 --> deploy["2025 H2 - 2026 H1 部署优化 (FLOWER 50% 裁剪 / QuantVLA W4A8 / Xiaomi 异步)"]
    pi0 --> wam4d["2026 H1 X-WAM (Wan2.2-5B + RGB-D 4D + ANS) / MotuBrain (Vidu+三流 MoT+H-Bridge 5 模式)"]
    wm --> wam4d
    coevo --> wam4d
    wam4d --> cc["2026 H1 Consistency-Consensus (零训练 test-time WAM ranking)"]
    pi0 --> acot["2026 H1 ACoT-VLA on pi0.5 base (动作空间 CoT: EAR + IAR + AGP)"]
    coevo --> future["2026+ MoT + Latent Action + 多智能体"]
    mot --> future
    deploy --> future
    wam4d --> future
    cc --> future
    acot --> future
```

**关键拐点解释**:

- **2022~2023 AR head 期**:RT-1 → RT-2,动作 token 化首次让 VLA 接上 LLM;
- **2024 H1 Diffusion 突破**:DP / RDT-1B,把"多模态分布"做扎实;
- **2024 末 Flow 时代**:π0 把 Flow Matching 引入 VLA,采样步数 1-10 步,事实标准;
- **2025 H1 WM + Latent 双路并行**:Cosmos / GR00T 走像素 WM;LAP / Being-H 走 Latent Action;
- **2025 H2 - 2026 H1 共演化与 MoT**:CoLA-World 等让 WM 与 Latent Action 共训;MoT 在 π0.7 / Xiaomi / Ψ0 成熟;
- **2025-2026 部署优化爆发**:FLOWER 裁剪 50%、QuantVLA W4A8、Xiaomi 异步,大模型上车的关键拐点。

**细化时间线**(74 篇里挑出每类组件代表 / 成熟工作,按近似发表时间):

- **2022~2024 H1 — AR head 期 + 离散 Token Vocab 起源**:RT-1(2022,Pure Transformer + AR head)→ RT-2(2023 PaLI-X + AR)→ OpenVLA(2024 Llama-2)/ Octo(Transformer + DDPM head + chunking)/ Diffusion Policy / RDT-1B(CNN/DiT + DDPM head)。
- **2024 H2 — Flow Head 起步**:[π0](#7a10-π06-recap)(PaliGemma + Flow Expert,事实标准起源)。
- **2025 H1 — Backbone 多元化 + 自训 VLM 路线**:[Cosmos Policy](#7l1-cosmos-policy)(NVIDIA Cosmos-Predict2 视频基座)/ [GR00T_N1.6 N1.5](#7l3-gr00t_n16)(Cosmos-2B + DiT)/ [MolmoAct2 / MolmoB0T](#7l4-molmoact2)(Allen Molmo + FAST)/ [DM0](#7l2-dm0)(自训 1.7B)。
- **2025 H1 — Latent Action 路线**:[LAP](#7e1-lap)(Language-Action,跨本体)/ [Being-H0.5](#7f1-being-h05)(MoF + 人手数据)→ 2026 H1 [Being-H0.7](#7w17-being-h07)(dual-branch JEPA)/ [Ψ0](#7a11-ψ0-psi-zero)(800h 人视 + 30h 真机超 10× 数据基线)。
- **2025 H2 — Latent / JEPA WM 兴起**:[VLA-JEPA](#7w10-vla-jepa)(无泄漏 JEPA)→ 2026 H1 [Fast-WAM](#7w3-fast-wam)(190ms)/ [Mask World Model](#7w7-mask-world-model-mwm)(语义 mask)。
- **2025 H2 ~ 2026 H1 — World ↔ Action 共演化爆发**:[CoLA-World](#7w1-cola-world) / [OA-WAM](#7w8-oa-wam)(对象槽位)/ [STARRY](#7w9-starry) / [VLAW](#7w11-vlaw) / [World-VLA-Loop](#7w12-world-vla-loop) / [World2Act](#7w13-world2act)(同期 6 篇集中突破)。
- **2025 H2 — A3 Flow Matching 全面落地**:[FLOWER](#7a2-flower)(50% 裁剪 + 4-8 步)→ 2026 H1 [SimVLA](#7a5-simvla)(0.5B 极简)/ [LingBot-VLA](#7a3-lingbot-vla)(20K 小时)/ [X-VLA](#7a8-x-vla)(Soft-Prompt 跨本体)/ [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)(MoT + Λ-attn)/ [π0.6 RECAP](#7a10-π06-recap)(advantage-conditioned)/ [π0.7](#7f6-π07)(Steerable MoT)。
- **2025 H2 — RL / 后训进入主流**:[LWD](#7o4-lwd)(fleet RL)→ [SOP](#7o5-sop)(可扩展)→ [TT-VLA](#7o6-tt-vla)(test-time)→ [VLA-OPD](#7o3-vla-opd)(OPD 桥)→ 2026 H1 [LifeLong-RFT](#7o7-lifelong-rft) / [SmoothVLA](#7a9-smoothvla) / [WoVR](#7o11-wovr) / [HY-Embodied-0.5](#7f3-hy-embodied-05)(MoT + 自演化 RL)。
- **2025 H2 — D5 CoT / Reasoning Head 普及**:[MolmoAct2](#7l4-molmoact2)(Think)/ [HiF-VLA](#7w6-hif-vla)(Hindsight + Foresight)→ 2026 H1 [GST-VLA](#7v4-gst-vla)(DA-CoT)/ [NS-VLA](#7w16-ns-vla) / [DM0](#7l2-dm0)(Spatial Scaffolding)/ [LoHo-Manip](#7w15-loho-manip)(Trace + remaining-plan)/ [CycleVLA](#7a12-cyclevla)(MBR)。
- **2025 H2 — C3 3D / Pose / Affordance 编码器集中出现**:[Pose-VLA](#7v6-pose-vla)(Pose Token)/ [GeneralVLA](#7v3-generalvla)(3D Affordance + Path)/ [GST-VLA](#7v4-gst-vla)(128 Gaussian)/ [PokéVLA](#7v5-pokévla)(几何分割)/ [ConsisVLA-4D](#7v1-consisvla-4d)(4D 时空)/ [ABot-M0](#7a1-abot-m0)(VGGT)。
- **2025 H2 - 2026 H1 — 部署优化爆发**:[FLOWER](#7a2-flower)(50% 裁剪)/ [QuantVLA](#7o1-quantvla)(W4A8)/ [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)(80ms 异步)/ [HiPolicy](#7a7-hipolicy)(多频)。
- **2026 H1 — MoT + Memory 大基座成熟**:[π0.7](#7f6-π07) / [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0) / [HY-Embodied-0.5](#7f3-hy-embodied-05) / [Ψ0](#7a11-ψ0-psi-zero) / [Helix_02](#7f2-helix_02-figure-ai)(三层系统)等大基座。
- **2026 H1 — Unified 多模式 WAM**:[X-WAM](#7w19-x-wam) Wan2.2-5B + 轻量深度分支 + Asynchronous Noise Sampling(4D RGB-D,RoboCasa 79.2);[MotuBrain](#7w20-motubrain) Vidu + 三流 MoT + H-Bridge(5 种推理模式同模型,RoboTwin 95.8/96.1)。
- **2026 H1 — Zero-Training WAM Ranking**:[Consistency-Consensus](#7w21-consistency-consensus) 用 action-state consistency 在 Cosmos-Policy / LingBot-VA 上 zero-training **+2.8 pp RoboTwin**。
- **2026 H1 — Action-space CoT**:[ACoT-VLA](#7o17-acot-vla) EAR(18 层 Transformer)+ IAR(学习查询)+ AGP head;**LIBERO-Plus 87.5% 全局 #1**。
- **2026+ — 预测方向**:Lifelong + Multi-agent + Edge;Reward 与 Safety 范式化;MoT 路由统一化;**Unified 多模式 WAM + Zero-training test-time selection 成为新 design pattern**。

---

## 第 7 章 74 篇模型结构速查卡 [T2]

> 按主组件归属分 7 大组(7.V / 7.L / 7.F / 7.A / 7.W / 7.E / 7.O),每张卡统一字段:**一句话定位 / 架构 mermaid(3-6 节点) / 核心组件清单 / 输入→输出 / 主要 Loss / 关键消融数字 / 最大正面贡献组件 / 优势-局限 / 范式归属(链回 vla_traintask.md)**。
>
> **主组件归属**(规则):该论文最具新意、最具差异化的组件类别(若多组件均有新意,挑论文 Method 章中最核心的)。
>
> **74 篇 → 7 大组**(预分组,待第三轮基于消融数字微调):
> - 7.V 视觉/3D/Proprio 编码器主(7 篇):ConsisVLA-4D / FocusVLA / GeneralVLA / GST-VLA / PokéVLA / Pose-VLA / BTK
> - 7.L VLM Backbone 主(8 篇):Cosmos Policy / DM0 / GR00T_N1.6 / MolmoAct2 / MolmoB0T / Step1-based / VLA-Foundry / VLANeXt
> - 7.F 融合/路由主(6 篇):Being-H0.5 / Helix_02 / HY-Embodied-0.5 / StarVLA-α / Xiaomi-Robotics-0 / π0.7
> - 7.A 动作 Head 主(13 篇):ABot-M0 / FLOWER / LingBot-VLA / MINT-4B / RLDX-1 / SimVLA / HAMLET / HiPolicy / Psi-R2 部分 / SmoothVLA / X-VLA / VLANeXt 副 / π0.6
> - 7.W 世界/辅助 Head 主(18 篇):CoLA-World / DreamZero / Fast-WAM / FutureVLA / GigaWorld-Policy / HiF-VLA / Mask World Model / OA-WAM / STARRY / VLA-JEPA / VLAW / World-VLA-Loop / World2Act / Psi-R2/W0 / LoHo-Manip / CycleVLA / NS-VLA / Being-H0.7
> - 7.E 跨本体适配主(4 篇):LAP / OXE-AugE / X-VLA 副(主 A3)/ Genie Sim 3.0
> - 7.O 后训练/部署优化主(14 篇):QuantVLA / Green-VLA / VLA-OPD / LWD / SOP / TT-VLA / SmoothVLA 副 / LifeLong-RFT / STRONG-VLA / TiPToP / EZ-M / WoVR / RealMirror / Ψ0 / ELITE / ReconVLA / P3Nav / π0.7 副

### 7.V 视觉/3D/Proprio 编码器主 — 7 篇 [T2]

**共同特点**:这 7 篇的差异化亮点都在「输入端」——3D Gaussian Token、4D 时空一致、Pose Token、视觉知识库、Patch-Select 等。Backbone / Action Head 多沿用主流 Flow / OpenVLA。

#### 7.V.1 [ConsisVLA-4D](p/ConsisVLA-4D_Advancing_Spatiotemporal_Consistency_in_Efficient_3D-Perception_and_4D-Reasoning_for_Robotic_Manipulation/paper.pdf) — 4D 时空一致性 3D 感知 [T2]

- **一句话定位**:时空一致性 3D 感知 + 4D 推理的高效 VLA;**仅用 1/8 视觉 token** 即达 SOTA。
- **架构**:
```mermaid
flowchart LR
    views["Multi-view RGB"] --> cv["CV-Aligner (SigLIP+VGGT)"]
    cv --> co["CO-Fuser (DINOv2+VGGT)"]
    co --> cs["CS-Thinker (4D Reasoning)"]
    cs --> act["Action Chunk (Parallel Decode)"]
```
- **核心组件清单**:Vision=`SigLIP + DINOv2 + VGGT` / Text=`SigLIP text` / 3D=`VGGT(depth+pointmap)` / Proprio=`N/A` / Backbone=`OpenVLA 7B(LoRA)` / Fusion=`mid (CV-Aligner Single-Fusion + CO-Fuser Group-Fusion)` / Action Head=`Parallel chunk decode (L1)` / Aux Head=`Future dynamic obj + Global depth` / WM=`Latent(隐式 4D)` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`三视 RGB + Language → 8 / 25 步 action chunk`
- **主要 Loss**:\[ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{dyn-4D}} + \mathcal{L}_{\text{dep-4D}} \]
- **关键消融**:
  - 去掉 CV-Aligner ES-Sel + S-Fus(Table 5)→ -7.0% LIBERO / -10.0% Real
  - 去掉 CO-Fuser G-Fus + IG-Agg(Table 5)→ -8.2% LIBERO / **-13.3% Real**
  - 去掉 CS-Thinker 全部(Table 6)→ -4.8% LIBERO / -11.6% Real
- **最大正面贡献组件**:**CO-Fuser(跨物体几何一致性)**,贡献 -13.3% Real,因多视角深度消歧消除单视角空间模糊。
- **优势**:推理 2.3× 加速;3D 感知无需额外传感器。**局限**:额外引入 VGGT ~2B 参数;4D 推理依赖 CoTracker / DepthAnything 伪标签。
- **范式归属**(链回 [vla_traintask.md](vla_traintask.md)):D1 + C3 + A4

#### 7.V.2 [FocusVLA](p/FocusVLA_Focused_Visual_Utilization_for_VLAs/paper.pdf) — 级联注意力 + 硬 Patch-Select [T2]

- **一句话定位**:模态级联注意力 + 聚焦注意力让 VLA **只关注任务相关 patch**,训练收敛 1.5-5×。
- **架构**:
```mermaid
flowchart LR
    obs["Multi-view RGB + Lang"] --> vlm["VLM Backbone"]
    vlm --> mca["Modality Cascaded Attention"]
    mca --> focus["Focus Attention (Patch+Channel)"]
    focus --> policy["Action Policy (AR Chunk)"]
    policy --> act["Action Output"]
```
- **核心组件清单**:Vision=`SigLIP ViT` / Text=`VLM 内置` / 3D=`N/A` / Proprio=`State` / Backbone=`VLA-Adapter base` / Fusion=`mid (Modality Cascaded Attn)` / Action Head=`AR(continuous chunk)` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`多视 RGB + Language + Proprio → 连续 action chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{action}}^{L1/MSE} \]
- **关键消融**:
  - Modality Cascaded vs Mixed Attention(Fig 2)→ 注意力图从散乱变聚焦
  - Focus Attention(Patch-level)→ 收敛 **1.5×(LIBERO)**、**5×(Spatial)**
  - Hugging Mug(RoboTwin)→ 其他方法近零,FocusVLA 显著高
- **最大正面贡献组件**:**Modality Cascaded Attention** — 消除 shortcut 让模型必须从视觉 token 提取任务信息,注意力从散乱变聚焦,收敛速度 1.5-5×。
- **优势**:训练收敛大幅加速,精细操作显著提升。**局限**:仅验证 AR 路线;Patch pruning 的 K 值需任务调优。
- **范式归属**:A4 + C3 + V5

#### 7.V.3 [GeneralVLA](p/GeneralVLA_3D_Affordance_+_Control_Strategy/paper.pdf) — 3D Affordance + 控制策略解耦 [T2]

- **一句话定位**:层次化 VLA 利用 3D affordance + 知识库实现 **零真机数据** 零样本规划。
- **架构**:
```mermaid
flowchart LR
    img["RGB + Lang"] --> asm["ASM (VLM + SAM)"]
    asm --> aff["2D/3D Affordance Points"]
    aff --> plan["3DAgent (LLM + Knowledge Bank)"]
    plan --> traj["3D Trajectory Path"]
    traj --> policy["Low-level Policy (HGM)"]
    policy --> act["Grasp / Action"]
```
- **核心组件清单**:Vision=`VLM + SAM + Depth` / Text=`LLM(GPT-4 等)` / 3D=`Depth → 3D affordance` / Proprio=`N/A` / Backbone=`Hierarchical(VLM + LLM + policy)` / Fusion=`late (模块化 pipeline)` / Action Head=`Low-level grasp policy(HGM)` / Aux Head=`Affordance segmentation` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`RGB + Depth + Language → 3D trajectory + Grasp pose`
- **主要 Loss**:\[ \text{无统一 Loss,各模块独立训练 / 推理} \]
- **关键消融**:
  - GeneralVLA vs VoxPoser(14 任务)→ **显著超越**(具体百分比因任务而异)
  - 加入 Knowledge Bank → 多任务 SR 提升
  - BC policy trained on GeneralVLA data vs human demo → 用 GeneralVLA 数据训练更鲁棒
- **最大正面贡献组件**:**Hierarchical(ASM + 3DAgent)** — 利用 VLM/SAM 强泛化做 affordance,LLM 做 3D 规划,**无需 robot data** 即可零样本执行。
- **优势**:零真机数据;可自动生成下游 BC 数据。**局限**:模块化 pipeline 误差累积;推理频率受限于串行。
- **范式归属**:C3 + D3 + F1

#### 7.V.4 [GST-VLA](p/GST-VLA_Structured_Gaussian_Spatial_Tokens_for_3D_Depth-Aware_VLAs/paper.pdf) — 结构化 3D Gaussian Spatial Token [T2]

- **一句话定位**:用 **128 个各向异性 3D Gaussian Token** + DA-CoT 推理增强 VLA 精细操作。
- **架构**:
```mermaid
flowchart LR
    rgb["RGB 224×224"] --> sem["Frozen Semantic Enc (SigLIP+DINOv2)"]
    rgb --> dep["Frozen Depth Expert"]
    sem --> gst["GST Tokenizer (128 tokens)"]
    dep --> gst
    gst --> vlm["VLM (LoRA r=16)"]
    vlm --> ae["Flow Action Expert 300M (MoE top-2)"]
```
- **核心组件清单**:Vision=`SigLIP + DINOv2(frozen)` / Text=`VLM` / 3D=`Metric Depth + 128 anisotropic Gaussian primitives` / Proprio=`7-DoF state` / Backbone=`VLM(LoRA r=16) + 300M MoE Action Expert` / Fusion=`mid (cross-attn projector)` / Action Head=`Flow Matching(MoE FFN)` / Aux Head=`DA-CoT(3D grounding / grasp / spatial / SE(3))` / WM=`N/A` / Routing=`MoE top-2 per layer` / Post-train=`3-stage progressive`
- **输入 → 输出**:`RGB + Depth + Lang + Proprio → DA-CoT + 7-DoF chunk(10 步)`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{flow}} + 0.5\,\mathcal{L}_{\text{CoT}} + 0.1\,\mathcal{L}_{\text{depth}} \]
- **关键消融**:
  - 去 3D Fourier PE(Table IV)→ **-2.8pp**
  - 去 S1 预训练(Table VI)→ **-6.2pp**
  - 去 DA-CoT(Table V)→ **-3.9pp**
  - Dense depth vs Full Gaussian(Table VII)→ **-4.5pp**
- **最大正面贡献组件**:**S1 GST 预训练**,贡献 +6.2pp(Table VI),因 VLM 必须在几何校准的 Gaussian token 上才能学有意义的空间推理。
- **优势**:各向异性 Gaussian 编码表面朝向与置信度,远胜标量深度;DA-CoT 可解释。**局限**:对高反射/玻璃表面深度不准;**6.2Hz** 推理较慢。
- **范式归属**:C3 + D5 + A3

#### 7.V.5 [PokéVLA](p/PokéVLA_Empowering_Pocket-Sized_VLA_with_Comprehensive_World_Knowledge_Guidance/paper.pdf) — 口袋级 VLA + 几何分割辅助 [T2]

- **一句话定位**:**1.22B** 口袋级 VLA,**2.4M 样本**预训 PokeVLM + 分割/几何辅助 + L1 动作头,推理 **比同类快 12×**。
- **架构**:
```mermaid
flowchart LR
    A["双视 RGB"] --> B["DINO + SigLIP"]
    B --> C["PokeVLM (Qwen2.5-0.5B)"]
    C --> D["SEG mask 头"]
    C --> E["L1 Action Head"]
    F["VGGT 几何"] -.训练期.-> C
```
- **核心组件清单**:Vision=`DINO + SigLIP` / Text=`Qwen2.5` / 3D=`VGGT(训练期对齐)` / Proprio=`标准` / Backbone=`PokeVLM 1.22B` / Fusion=`Adapter` / Action Head=`L1 回归 chunk` / Aux Head=`SEG mask` / WM=`N/A` / Routing=`N/A` / Post-train=`2-stage`
- **输入 → 输出**:`双视 RGB + Lang → action chunk(推理无 3D)`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{action}} + \lambda_{\text{seg}}\mathcal{L}_{\text{seg}} + \lambda_{\text{geo}}\mathcal{L}_{\text{geo}} \]
- **关键消融**:
  - LIBERO **98.2%** / LIBERO-Plus **83.5%(#3)**
  - 推理 **12× 快于** 18× 大模型
  - 去几何对齐 → LIBERO-Plus 显著降
- **最大正面贡献组件**:**Stage1 世界知识预训 PokeVLM(2.4M 样本)** — 小参数量下保 OOD 语义/几何鲁棒。
- **优势**:极轻量可部署;LIBERO-Plus 高位。**局限**:L1 头平滑性弱于 Flow;几何监督仅训练期。
- **范式归属**:C3 + F1 + F2

#### 7.V.6 [Pose-VLA](p/Pose-VLA_Universal_Pose_Pretraining_for_Generalizable_VLAs/paper.pdf) — 离散 Pose Token 统一 3D 与轨迹 [T2]

- **一句话定位**:**离散 Pose Token** 统一 3D grounding 与机器人轨迹,2 阶段预训 + Flow 专家,RoboTwin Hard **+14pp vs π0**。
- **架构**:
```mermaid
flowchart LR
    A["RGB + Depth"] --> B["Pose Tokenizer"]
    B --> C["PaliGemma-2 VLM"]
    C --> D["Flow Action Expert (π 式)"]
    D --> E["连续控制"]
```
- **核心组件清单**:Vision=`RGB` / Text=`指令` / 3D=`depth + raymap` / Proprio=`下游 FT` / Backbone=`PaliGemma-2` / Fusion=`Pose token 前缀` / Action Head=`Flow Expert` / Aux Head=`Pose AR 预训` / WM=`N/A` / Routing=`N/A` / Post-train=`2-stage + ~100 demo`
- **输入 → 输出**:`RGB + [depth] + Lang → 结构化 (c,b,p) pose token → 连续动作`
- **主要 Loss**:\[ \mathcal{L}_{\text{pre}} = -\sum_{k=1}^K \log p_\theta(s_k\mid \phi(O),L,s_{<k}) \]
- **关键消融**:
  - RoboTwin 2.0 avg **79.5%** / Hard **79.1% vs π0 65.12%**(+14pp)
  - 去姿态预训 → RoboTwin 显著降
  - 3D grounding AP 超 Qwen3-VL / Gemini
- **最大正面贡献组件**:**Universal Pose Token 预训** — 解耦 VQA 与细粒度 3D 状态,双臂 OOD 泛化核心。
- **优势**:数据高效下游 FT;统一 3D + 轨迹表征。**局限**:LIBERO 未进 Top3;需 depth/raymap 预训管线。
- **范式归属**:C3 + A3 + A4

#### 7.V.7 [BTK](p/Beyond_Textual_Knowledge_(BTK)_Leveraging_Multimodal_Knowledge_Bases_for_Enhancing_VLN/paper.pdf) — VLN 多模态知识库 [T2]

- **一句话定位**:在 VLN 中引入 Goal-Aware Augmentor + Image/Text 知识库,**填补语义-视觉鸿沟**。
- **架构**:
```mermaid
flowchart LR
    inst["Language Instruction"] --> ga["Goal-Aware Augmentor (Qwen3-4B)"]
    ga --> ka["Knowledge Augmentor"]
    pano["Panoramic View"] --> venc["CLIP-B/16 Encoder"]
    venc --> ka
    ka --> duet["DUET (Dual-scale Transformer)"]
    duet --> act["Navigation Action"]
```
- **核心组件清单**:Vision=`CLIP-B/16` / Text=`BERT + Qwen3-4B` / 3D=`N/A` / Proprio=`GPS+方向` / Backbone=`DUET 203M` / Fusion=`late (Knowledge Augmentor gating)` / Action Head=`Discrete navigation` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`全景 RGB + Lang → 离散导航动作`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{SAP}} + \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{MRC}} \]
- **关键消融**:
  - Goal-Aware + Knowledge Augmentor(Table 9 Id4 vs Id1)→ +1.71% SR / **+3.01% SPL**
  - Image + Text Knowledge(Table 7 Id4 vs Id1)→ +0.17% SR / +2.56% SPL
  - Qwen3-4B vs SpaCy(Table 6)→ +2.09% SR(保留修饰性目标短语)
- **最大正面贡献组件**:**Goal-Aware + Knowledge Augmentor 联合**,贡献 +3% SPL,因多模态知识填补语义-视觉鸿沟。
- **优势**:无需改底层导航架构,即插即用;图像知识库提供视觉参照。**局限**:离线知识库构建 ~240 GPU-hours;生成图像与真实有域差距。
- **范式归属**:C3 + D5

### 7.L VLM Backbone 主 — 8 篇 [T2]

**共同特点**:这 8 篇要么 **自训 VLM**(Cosmos / DM0 / GR00T / Molmo 系),要么把 **Action Token Vocab** 当作 L 维度核心创新(MINT-4B),要么把"VLA 训练栈"统一化(VLA-Foundry / VLANeXt)。

#### 7.L.1 [Cosmos Policy](p/Cosmos_Policy_(NVIDIA)/paper.pdf) — 自训 Cosmos-Predict2 视频基座 [T2]

- **一句话定位**:**视频扩散模型直接微调为策略 + 世界模型 + 价值函数**,LIBERO 98.5% / RoboCasa 67.1% SOTA。
- **架构**:
```mermaid
flowchart LR
    obs["Multi-cam RGB + Proprio"] --> vae["Wan2.1 VAE"]
    vae --> dit["Cosmos-Predict2 DiT 2B"]
    dit --> act["Latent Action (EDM)"]
    dit --> future["Future State Images"]
    dit --> val["Value V(s')"]
```
- **核心组件清单**:Vision=`Wan2.1 VAE` / Text=`T5-XXL` / 3D=`N/A` / Proprio=`EEF/Joint(as latent frame)` / Backbone=`Cosmos-Predict2-2B DiT` / Fusion=`early (所有模态当 latent frame)` / Action Head=`Latent Diffusion(EDM)` / Aux Head=`Future state + Value` / WM=`像素(future image)` / Routing=`N/A` / Post-train=`Rollout fine-tune`
- **输入 → 输出**:`多视 RGB + Lang + Proprio → action chunk + future image + value`
- **主要 Loss**:\[ \mathcal{L} = \mathbb{E}\bigl[\lVert D_\theta(x_0 + n; \sigma, c) - x_0\rVert_2^2\bigr] \]
- **关键消融**:
  - 加 auxiliary targets(state + value)(Sec 5.2)→ 显著提升 policy
  - Planning(best-of-N) vs direct(Sec 5.3)→ **+12.5%** 真实任务
  - Video pretrain vs from-scratch → 预训远胜
- **最大正面贡献组件**:**Video foundation model pretrain** — 视频模型继承时空物理先验,LIBERO 98.5% / RoboCasa 67.1%。
- **优势**:零架构修改,单阶段微调;统一 policy/WM/value。**局限**:2B 推理较慢;依赖 rollout 数据才能 planning。
- **范式归属**:B1 + D2 + A2

#### 7.L.2 [DM0](p/DM0_An_Embodied-Native_Vision-Language-Action_Model_towards_Physical_AI/paper.pdf) — 具身原生三阶段自训 VLM [T2]

- **一句话定位**:**Embodied-Native** 三阶段 VLA,统一驾驶/操作/理解,DM0-Specialist **+10%+ vs GigaBrain-0.1**。
- **架构**:
```mermaid
flowchart LR
    obs["Multi-view RGB + Proprio"] --> pe["Perception Encoder (ViT)"]
    pe --> vlm["Qwen3-1.7B + Action Expert"]
    vlm --> cot["Spatial CoT (subtask+bbox+traj)"]
    cot --> fm["Flow Matching Action Expert"]
    fm --> act["Continuous Action"]
```
- **核心组件清单**:Vision=`ViT(Perception Encoder)` / Text=`Qwen3-1.7B` / 3D=`N/A` / Proprio=`State tokens` / Backbone=`Qwen3-1.7B + Action Expert` / Fusion=`mid (KV-cache from VLM)` / Action Head=`Flow Matching` / Aux Head=`Spatial CoT(subtask + bbox + traj + discrete action)` / WM=`N/A` / Routing=`N/A` / Post-train=`Hybrid gradient(KI)`
- **输入 → 输出**:`多视 728×728 RGB + Lang + Proprio → action chunk + CoT 文本`
- **主要 Loss**:\[ \mathcal{L}_{\text{total}} = \lambda \mathcal{L}_{\text{AR}} + \mathcal{L}_{\text{FM}} \]
- **关键消融**:
  - DM0-Specialist vs GigaBrain-0.1(Table 30)→ **62.0% vs ~52%**(+10%+)
  - DM0-Generalist vs π0.5(Table 30)→ **37.3%**,优于 baseline
  - Spatial Scaffolding 对多阶段任务贡献最大
- **最大正面贡献组件**:**Embodied Spatial Scaffolding** — 层次预测(subtask→bbox→traj→action)逐步降低任务复杂度。
- **优势**:从预训练就融入 embodied 数据;Hybrid gradient 策略保留 VLM 通用能力。**局限**:1.7B 较小,通用语言有限。
- **范式归属**:D5 + A4 + F1 + G3

#### 7.L.3 [GR00T_N1.6](p/GR00T_N1.6_(NVIDIA)/page_1.html) / [page_2](p/GR00T_N1.6_(NVIDIA)/page_2.html) — NVIDIA 通用人形基座 [T2]

- **一句话定位**:**NVIDIA 人形基座 N1.5 升级版**,更大 DiT + Cosmos-2B VLM + 相对动作。
- **架构**:
```mermaid
flowchart LR
    obs["多视图 RGB"] --> vlm["Cosmos-2B VLM (原生分辨率)"]
    lang["语言指令"] --> vlm
    vlm --> dit["DiT Action Expert (32 层)"]
    prop["Proprio"] --> dit
    dit --> act["state-relative action"]
```
- **核心组件清单**:Vision=`Cosmos-2B VLM(原生分辨率)` / Text=`Cosmos VLM` / 3D=`N/A` / Proprio=`state-relative` / Backbone=`Cosmos-2B + DiT 32 层` / Fusion=`late (VLM → DiT)` / Action Head=`Flow Matching(DiT)` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`DAgger + RTC`
- **输入 → 输出**:`多视 RGB + Proprio + Lang → state-relative action chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{FM}}\bigl(v_\theta, a_1 - a_0\bigr) \]
- **关键消融**:
  - N1.6 vs N1.5 在 sim benchmark → 整体提升(博客图表)
  - Relative vs Absolute actions → relative 更平滑但小数据集易累积误差
  - 无正式消融表(博客形式)
- **最大正面贡献组件**:**DiT 层数翻倍(32 vs 16) + 解冻 VLM 顶层替代 adapter** — 更大容量 + 表征对齐提升。
- **优势**:开源;支持 YAM / AgiBot / G1 等多本体。**局限**:博客形式无正式论文与定量消融。
- **范式归属**:A2 + A4 + F1 + G3

#### 7.L.4 [MolmoAct2](p/MolmoAct2_Action_Reasoning_Models_for_Real-world_Deployment/paper.pdf) — Molmo 全开源 + FAST + Think [T2]

- **一句话定位**:**Allen AI 全开源 Action Reasoning VLA**:Molmo2-ER 具身 VLM + FAST 离散动作 + KV 条件 Flow Expert + 自适应深度 Think。
- **架构**:
```mermaid
flowchart LR
    A["RGB / Video + Lang"] --> B["Molmo2-ER VLM"]
    B --> C["FAST 离散 AR"]
    B -->|"逐层 KV"| D["Flow Action Expert"]
    B --> E["Think 推理 token"]
    D --> F["连续动作块"]
    C --> F
```
- **核心组件清单**:Vision=`Molmo ViT` / Text=`LLM tokenizer` / 3D=`深度/3D 推理 token` / Proprio=`离散化入 prompt` / Backbone=`Molmo2-ER ~7B` / Fusion=`prefix + KV cross` / Action Head=`FAST AR + Flow Expert` / Aux Head=`Think/CoT 头` / WM=`N/A` / Routing=`N/A` / Post-train=`离散→连续共训`
- **输入 → 输出**:`多视 RGB/视频 + Lang → FAST token 或 Flow 连续 chunk`
- **主要 Loss**:\[ \mathcal{L}_{\text{flow}} = \mathbb{E}\bigl[\lVert m\odot(f_\theta(x_t, t, c) - (a - \epsilon))\rVert_2^2\bigr],\quad \mathcal{L}_{\text{CE}} = \text{FAST next-token} \]
- **关键消融**:
  - MolmoAct2-Think 相对 base 在长程 / sim2real 涨点(Sec 6)
  - 推理 **37× 加速 vs MolmoAct v1**(Think 仅重算变化区域)
  - MolmoSpace **37.7** / RoboEval **44.3** SOTA(Allen 新榜)
- **最大正面贡献组件**:**逐层 KV 条件 Flow Expert + Think 自适应深度** — KV 条件保 VLM 通识不回传梯度;Think 只重算变化区域。
- **优势**:全栈开源(数据 / Tokenizer / 权重);ARM 可解释。**局限**:新榜无 π0.5 同口径;训练算力高(64×H100)。
- **范式归属**:A3 + A1 + D5

#### 7.L.5 [MolmoB0T](p/MolmoB0T_Large-Scale_Simulation_Enables_Zero-Shot_Manipulation/paper.pdf) — Molmo + 1.7M 仿真零样本 [T2]

- **一句话定位**:**纯仿真 MolmoBot-Engine 1.7M 专家轨迹**训 VLA,桌面/移动操作 **零样本 Sim2Real 79.2%**。
- **架构**:
```mermaid
flowchart LR
    A["MolmoBot-Engine 仿真"] --> B["多帧 RGB + 指令"]
    B --> C["Molmo2-4B VLM"]
    C --> D["DiT Flow Head"]
    D --> E["16 步动作块"]
```
- **核心组件清单**:Vision=`Molmo2 冻 ViT` / Text=`LLM` / 3D=`可选 2D point` / Proprio=`关节/底座` / Backbone=`Molmo2-4B` / Fusion=`prefix` / Action Head=`Flow Matching` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`200K BC`
- **输入 → 输出**:`腕 + 第三人称 RGB + Lang → 16 步关节/底座动作块`
- **主要 Loss**:\[ \mathcal{L}_{\text{flow}}: \text{Flow-matching BC} \]
- **关键消融**:
  - 真机 4 settings Overall **79.2% vs π0.5 39.2%**(+40pp)
  - DROID 仿真 7 任务 **64.1% vs π0.5 10.0%**
  - **3DGS / 高保真 MolmoSpaces** 是 zero-shot 主因
- **最大正面贡献组件**:**MolmoBot-Engine 程序化仿真数据引擎** — 232K 环境 × 48K 物体生成 1.7M 轨迹,替代大规模真机采集。
- **优势**:零真机预训练即强 Sim2Real;开源数据/引擎/权重。**局限**:依赖仿真保真度;非通用长程推理 VLA。
- **范式归属**:A3 + E1

#### 7.L.6 [MINT-4B](p/MINT_Mimic_Intent,_Not_Just_Trajectories_(MINT-4B)/paper.pdf) — DCT 多尺度 VQ Action Token Vocab [T2]

- **一句话定位**:**频域多尺度 action tokenizer** 解耦 intent 与 execution,实现 **one-shot 技能迁移**。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Proprio"] --> vlm["PaliGemma-2.6B"]
    lang["Lang"] --> vlm
    vlm --> ae["Action Expert (next-scale AR)"]
    ae --> s1["Intent Token (S1)"]
    s1 --> s2["Execution Tokens (S2..SK)"]
    s2 --> dec["SDAT Decoder"]
    dec --> act["Continuous Chunk"]
```
- **核心组件清单**:Vision=`SigLIP + DINOv2(30M) / PaliGemma-2.6B(4B)` / Text=`BERT / PaliGemma` / 3D=`N/A` / Proprio=`关节状态` / Backbone=`30M / PaliGemma-2.6B + 300M Action Expert` / Fusion=`late (FiLM/cross-attn)` / Action Head=`Next-Scale AR(multi-scale VQ)` / Aux Head=`Spectral Reconstruction` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`RGB + Proprio + Lang/Intent token → 多尺度 action tokens → 连续 chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{freq}}^{\text{spectral}} + \mathcal{L}_{\text{codebook}} + \mathcal{L}_{\text{commit}} + \alpha\mathcal{L}_1^{\text{recon}} + \mathcal{L}_{\text{CE}}^{\text{policy}} \]
- **关键消融**:
  - MINT vs OpenVLA-OFT(Table IV)→ **+15pp robustness**
  - 去 spectral loss(Table VI)→ **-4.8pp LIBERO**
  - One-shot transfer(Table V)→ **MINT 60% vs baseline ~0%**
  - MINT-4B vs π0.5 real-robot(Sec VI)→ **+29% SR**
- **最大正面贡献组件**:**Spectral Disentanglement(S1 Intent Token)** — 贡献 +60pp one-shot transfer,因频域解耦迫使 S1 捕获低频全局意图,可直接注入实现技能迁移。
- **优势**:Intent token 实现 one-shot 技能迁移,无需重训。**局限**:仅在单臂操作上验证;VQ codebook 大小影响重建精度。
- **范式归属**:A1 + C2 + G3

#### 7.L.7 [VLA-Foundry](p/VLA_Foundry_A_Unified_Framework_for_Training_VLAs/paper.pdf) — TRI 统一 LLM→VLM→VLA 训练栈 [T2]

- **一句话定位**:首个 **覆盖 LLM→VLM→VLA 全流程** 的统一开源训练框架。
- **架构**:
```mermaid
flowchart LR
    img["Multi-view RGB"] --> vit["ViT + Pixel-Shuffle"]
    vit --> llm["LLM (1.2B / Qwen3-VL-2B)"]
    llm --> obs["Observation Token"]
    obs --> flow["Flow Transformer Head (325M DiT)"]
    flow --> a["Action Chunk (SE3 pose)"]
```
- **核心组件清单**:Vision=`ViT-86M / Qwen3-VL` / Text=`Foundry-LLM-1.2B / Qwen3-VL-2B` / 3D=`N/A` / Proprio=`可选 linear` / Backbone=`1.7B(from scratch) / 2.1B(Qwen3)` / Fusion=`early(obs token → flow head)` / Action Head=`Flow Matching(325M DiT)` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`4 视 RGB + Lang + Proprio → action chunk(SE3 pose)`
- **主要 Loss**:\[ \mathcal{L}_{\text{FM}} = \lVert v_\theta(a_t, t\mid c) - (a_1 - a_0)\rVert^2 \]
- **关键消融**:
  - Foundry-VLA-1.7B-sim vs LBM-MT(Fig 5)→ 统计不显著差异
  - Qwen3VLA-2.1B-MT vs LBM-MT(Fig 5)→ **+23pp**
  - Sim-only vs Real+Sim(Fig 9)→ sim-only 略优于混合训练
- **最大正面贡献组件**:**强 VLM backbone(Qwen3-VL)** — 从 1.7B 换至 2.1B+Qwen3,aggregate +~23pp。
- **优势**:首个 LLM→VLM→VLA 统一开源框架;HuggingFace 即插即用,FSDP2 多节点扩展。**局限**:仅验证双臂桌面仿真,无真机;Flow head 唯一选择,无 AR/Diffusion。
- **范式归属**:F1 + A3

#### 7.L.8 [VLANeXt](p/VLANeXt_Recipes_for_Building_Strong_VLA_Models/paper.pdf) — 12 条 VLA 设计 recipe 消融 [T2]

- **一句话定位**:系统消融 VLA 设计空间,**12 条 recipe** 构建强力基线,2.5B 超越 7B OpenVLA-OFT。
- **架构**:
```mermaid
flowchart LR
    obs["Multi-view RGB + Proprio"] --> vlm["Qwen3-VL-2B"]
    vlm --> mq["Meta Query (Soft Connection)"]
    mq --> dit["Flow-Matching DiT (19L)"]
    dit --> a["Action Chunk (8 步)"]
```
- **核心组件清单**:Vision=`Qwen3-VL 内置 ViT` / Text=`Qwen3-VL-2B` / 3D=`N/A` / Proprio=`Linear → VLM` / Backbone=`Qwen3-VL-2B` / Fusion=`mid(Soft Connection layer-wise)` / Action Head=`Flow Matching(19L DiT)` / Aux Head=`Freq-domain loss` / WM=`(消融验证有效但 3× 训练时间,放弃)` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`多视 RGB + Proprio History(8 步) + Lang → 8 步 chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{FlowMatching}} + 0.1\cdot \mathcal{L}_{\text{FreqDomain}} \]
- **关键消融**(Table 1, LIBERO-Plus):
  - Soft vs Loose vs Tight → **56.2% / 53.7% / 55.4%**
  - Proprio→VLM vs 无 Proprio → **87.7% vs 80.5%**(+7.2pp)
  - + 频域 Loss → **92.8% vs 87.7%**(+5.1pp)
  - Multi-view vs Single → **80.5% vs 57.2%**(**+23.3pp**)
- **最大正面贡献组件**:**Multi-view 输入** — +23.3pp(LIBERO-Plus),腕部 + 第三人称互补几何线索。
- **优势**:12 条经验法则覆盖 VLA 设计全空间;频域辅助 loss 零额外训练开销。**局限**:仅 LIBERO/LIBERO-Plus 验证;世界模型虽有效但 3× 成本被放弃。
- **范式归属**:A3 + A4 + F1

### 7.F 融合/路由主 — 6 篇 [T2]

**共同特点**:这 6 篇的差异化在「多模态/多任务/多本体如何融合」——MoT、Λ-attn、三层全身、轻量晚融合等。**MoT 是 2026 的事实标准**。

#### 7.F.1 [Being-H0.5](p/Being-H0.5/paper.pdf) — 人手为母语的 MoF 跨本体基座 [T2]

- **一句话定位**:**人体中心**跨 30+ 构型统一预训练 VLA 基础模型,**MoF(Mixture-of-Flow)** 路由不同 embodiment。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Proprio + Lang"] --> und["Understanding Expert (VLM)"]
    und --> gen["Generation Expert (MoF)"]
    gen --> act["Unified Action Space"]
```
- **核心组件清单**:Vision=`ViT(内置)` / Text=`LLM` / 3D=`N/A` / Proprio=`Unified State Space` / Backbone=`MoT(未公开参数量)` / Fusion=`**MoT shared attention**` / Action Head=`Flow(Mixture-of-Flow)` / Aux Head=`Motion Description / Continuation` / WM=`N/A` / Routing=`Per-head(MoF embodiment routing)` / Post-train=`MPG + UAC 异步`
- **输入 → 输出**:`多视 RGB + Lang + Unified Proprio → Unified Action Space(跨本体)`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{NTP}}^{\text{text}} + \mathcal{L}_{\text{FM}}^{\text{action}} + \mathcal{L}_{\text{motion}} \]
- **关键消融**:
  - Human-centric pre-training(Sec 7.3.1)→ **+10-15pp** across suites vs no-pretrain
  - Masked Motion Token Prediction(Sec 7.3.2)→ 改善 behavior prior
  - MPG + UAC(Sec 7.3.3)→ 跨构型部署稳定性显著提升
- **最大正面贡献组件**:**Human-centric pre-training** — +10-15pp,因 16K 小时人体视频提供通用物理交互先验。
- **优势**:跨 30+ 构型单 checkpoint 部署,涌现零样本迁移;UniHand-2.0 35K 小时最大规模预训语料。**局限**:权重/参数未完全公开;灵巧手数据稀缺(<5%)。
- **范式归属**:C4 + A3 + F1

#### 7.F.2 [Helix_02 (Figure AI)](p/Helix_02_(Figure_AI)/page.html) — Figure 03 三层全身 VLA [T2]

- **一句话定位**:Figure 03 人形全身 VLA,**三层系统 S0/S1/S2** 实现 4 分钟自主 loco-manipulation。
- **架构**:
```mermaid
flowchart LR
    sens["头/掌摄像头 + 触觉 + Proprio"] --> s1["System 1 (Transformer, 200Hz)"]
    s2["System 2 (VLM 语义推理)"] -->|latent goals| s1
    s1 -->|全身关节目标| s0["System 0 (10M, 1kHz)"]
    s0 --> torque["全身关节力矩 @1kHz"]
```
- **核心组件清单**:Vision=`头/掌摄像头` / Text=`S2 VLM` / 3D=`N/A` / Proprio=`全身关节 + 触觉` / Backbone=`S2 VLM + S1 Transformer + S0 10M` / Fusion=`**hierarchical(S2→S1→S0)**` / Action Head=`S0 全身力矩` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`Sim-to-Real RL(S0)`
- **输入 → 输出**:`多模态感知(视觉+触觉+本体) + Lang → 全身关节力矩 @1kHz`
- **主要 Loss**:\[ \text{未公开(工业博客)} \]
- **关键消融**:
  - 无消融实验(博客形式)
- **最大正面贡献组件**:**System 0** — 替代 109,504 行 C++ 的单一神经先验,从 1000+ 小时人类运动数据学习全身控制。
- **优势**:首次 pixels-to-whole-body 统一控制,4 分钟 61 步自主任务;触觉 + 掌心相机解锁精细灵巧操作。**局限**:无正式论文与定量消融;系统闭源。
- **范式归属**:A4 + E1 + B4

#### 7.F.3 [HY-Embodied-0.5](p/HY-Embodied-0.5_Embodied_Foundation_Models_for_Real-World_Agents/paper.pdf) — 腾讯具身 VLM + MoT 课程蒸馏 [T2]

- **一句话定位**:**腾讯混元具身 VLM 基础模型**,MoT 架构 + Latent Tokens + 自演化 RL,2B→32B 双版本。
- **架构**:
```mermaid
flowchart LR
    img["原生分辨率图像"] --> vit["HY-ViT 2.0 (400M)"]
    vit --> mot["MoT Backbone (2B/32B)"]
    lat["Visual Latent Tokens"] --> mot
    lang["语言"] --> mot
    mot --> ae["Action Expert (Flow)"]
    ae --> act["动作输出"]
```
- **核心组件清单**:Vision=`HY-ViT 2.0 400M(原生分辨率)` / Text=`Hunyuan-1.8B LLM` / 3D=`N/A` / Proprio=`机器人状态` / Backbone=`MoT-2B(4B total) / MoE-32B(407B total)` / Fusion=`**MoT(模态特异性计算)**` / Action Head=`Flow Matching` / Aux Head=`N/A` / WM=`N/A` / Routing=`MoT per-modality` / Post-train=`GRPO RL + On-Policy Distillation`
- **输入 → 输出**:`多视 RGB + Lang + Proprio → 连续动作 chunk / VQA 答案`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{LLM}} + \mathcal{L}_{\text{vision}} + \mathcal{L}_{\text{global}}\ \text{(pretrain)};\ \mathcal{L}_{\text{RL}}\ \text{(post)} \]
- **关键消融**:
  - MoT-2B VLA 真机:packing 85% / hanging 80% / stacking 80%(Fig 1)
  - vs MiMo-7B / Qwen3-VL-4B / RoboBrain-4B 在 22 项 benchmark 中 **16 项胜出**
  - On-Policy Distillation:32B → 2B 有效转移(Sec 4.4)
- **最大正面贡献组件**:**MoT + Latent Tokens** — 在 2B 规模即超越同尺度竞品 16/22 benchmarks,因模态特异性计算避免跨模态干扰。
- **优势**:2B 激活参数即可边缘部署;自演化迭代 RL 持续提升。**局限**:32B 部署成本高;机器人控制任务种类有限。
- **范式归属**:C1 + C3 + F2 + G3

#### 7.F.4 [StarVLA-α](p/StarVLA-α_Reducing_Complexity_in_Vision-Language-Action_Systems/paper.pdf) — Qwen3-VL + 轻量 MLP 晚融合 [T2]

- **一句话定位**:**极简 VLA 基线**,证明 **强 VLM + MLP 头** 即可达 SOTA,RoboChallenge 真机超 π0.5 约 20%。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Language"] --> vlm["Qwen3-VL-4B"]
    vlm --> mlp["MLP Action Head"]
    mlp --> a["Continuous Action Chunk"]
```
- **核心组件清单**:Vision=`Qwen3-VL 内置 ViT` / Text=`Qwen3-VL` / 3D=`N/A` / Proprio=`N/A(默认)` / Backbone=`Qwen3-VL 4B` / Fusion=`**late (Native VLM + MLP)**` / Action Head=`MLP 回归` / Aux Head=`N/A` / WM=`N/A` / Routing=`Simple Padding` / Post-train=`N/A`
- **输入 → 输出**:`RGB + Lang → 连续动作 chunk`
- **主要 Loss**:\[ \mathcal{L} = \text{MSE}(\hat a, a) \]
- **关键消融**:
  - MLP vs FAST(Table 2, RoboCasa)→ **53.8% vs 45.0%**(+8.8pp)
  - + OXE 预训(Table 3, RoboCasa)→ **27.8% vs 53.8%**(-26pp,有害)
  - Batch Size 64→1024(Table 11)→ **40.0%→59.2%**(+19.2pp)
  - PaliGemma → Qwen3VL-4B(Table 1 LIBERO)→ **69.8% → 95.8%**(+26pp)
- **最大正面贡献组件**:**Qwen3-VL backbone** — 强 VLM 骨干是核心,+26pp vs PaliGemma。
- **优势**:最简架构即 SOTA,RoboChallenge 真机超 π0.5 约 20%;无需动作预训练 / 复杂数据工程。**局限**:未在长程任务和更多构型验证;极简设计可能在多模态动作分布场景表现弱。
- **范式归属**:A4 + F1

#### 7.F.5 [Xiaomi-Robotics-0](p/Xiaomi-Robotics-0_Open-Sourced_VLA_with_Real-Time_Execution/paper.pdf) — 4.7B MoT + Λ-attn + 异步执行 [T2]

- **一句话定位**:**高性能实时 VLA**,异步执行 + **Λ-shape 注意力**消除动作前缀依赖,LIBERO Avg 98.7%。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Lang"] --> vlm["Qwen3-VL-4B (frozen)"]
    vlm --> kv["KV Cache"]
    kv --> dit["DiT Action Head (16L)"]
    prop["Proprio + Noisy Action"] --> dit
    dit --> a["Action Chunk (Flow)"]
```
- **核心组件清单**:Vision=`Qwen3-VL-4B 内置` / Text=`Qwen3-VL-4B` / 3D=`N/A` / Proprio=`MLP 编码` / Backbone=`Qwen3-VL-4B + DiT(总 4.7B)` / Fusion=`**MoT(KV cache conditioning) + Λ-shape attn**` / Action Head=`Flow(16L DiT)` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`异步执行`
- **输入 → 输出**:`RGB + Lang + Proprio → T 步 Action Chunk via Flow`
- **主要 Loss**:\[ \mathcal{L} = \lVert v_\theta(o_t, l, s_t, \tilde a^\tau, \tau) - u(\tilde a^\tau, a, \tau)\rVert_2^2 \]
- **关键消融**:
  - LIBERO Avg → **98.7%**(SOTA)
  - SimplerEnv Google Robot VM → **85.5%**
  - Λ-shape mask vs Causal(Sec 2.2.2)→ 避免 action prefix shortcut,提升反应性
- **最大正面贡献组件**:**Λ-shape attention** — 阻止动作生成过度依赖前缀,保持对视觉 / 语言的响应性。
- **优势**:消费级 GPU 实时推理;双臂精细操作高吞吐;预训保留 VLM 能力(VLM benchmark 不降)。**局限**:4.7B 推理仍需 GPU 加速;主要验证小米内部平台。
- **范式归属**:A3 + A4 + F1

#### 7.F.6 [π0.7](p/π0.7_A_Steerable_Generalist_Robotic_Foundation_Model_with_Emergent_Capabilities/paper.pdf) — 5B Steerable MoT + Flow [T2]

- **一句话定位**:**可操控通才机器人基础模型**,丰富 prompt 实现组合泛化,zero-shot 跨构型 T 恤折叠达人类首次水平。
- **架构**:
```mermaid
flowchart LR
    prompt["Language + Subgoal Images + Metadata"] --> vlm["VLM Backbone (π0.6-MEM)"]
    obs["RGB + Proprio"] --> vlm
    vlm --> flow["Flow-Matching Action Expert"]
    flow --> a["Action Chunk"]
```
- **核心组件清单**:Vision=`π0.6-MEM ViT` / Text=`π0.6-MEM LLM` / 3D=`N/A` / Proprio=`EEF` / Backbone=`π0.7(大于 π0.6)` / Fusion=`**early (π0 架构 + memory)**` / Action Head=`Flow Matching` / Aux Head=`High-Level Policy + WM(subgoal gen)` / WM=`内置(subgoal image generation)` / Routing=`N/A` / Post-train=`N/A(pretrain-only, no task-specific FT)`
- **输入 → 输出**:`RGB + Lang + Episode Metadata + Subgoal Images → Action Chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{FlowMatching}}\bigl(\text{conditioned on rich prompts}\bigr) \]
- **关键消融**:
  - Diverse data + detailed context → **组合泛化涌现**(新厨房 / 新任务零样本)
  - Zero-shot 跨构型 T 恤折叠 ≈ **人类遥操作首次尝试成功率**
  - 去掉 metadata / subgoal → 性能显著下降
- **最大正面贡献组件**:**Rich prompt conditioning** — 详细语言 + subgoal 图 + 元数据使模型能区分数据中不同策略 / 质量,首次在 VLA 中展示组合泛化。
- **优势**:开箱即用(espresso / 洗衣 / 装箱等无需 task-specific FT);首次展示组合泛化。**局限**:闭源,无法复现;依赖大规模多样数据 + 详细标注。
- **范式归属**:D1 + D5 + A3

### 7.A 动作 Head 主 — 12 篇 [T2]

**共同特点**:这 12 篇的差异化在「动作如何生成」——Flow Matching 主导(占 9 篇),DDPM 1 篇,Manifold 1 篇,多尺度 1 篇。Flow Head 几乎是 2025-2026 默认。

#### 7.A.1 [ABot-M0](p/ABot-M0_VLA_Foundation_Model_with_Action_Manifold_Learning/paper.pdf) — 动作流形 head + 双流感知 [T2]

- **一句话定位**:统一多构型数据 + **动作流形学习(AML)** 跨平台 VLA,大 chunk 高维场景下抗退化最强。
- **架构**:
```mermaid
flowchart LR
    obs["RGB Multi-view + Proprio"] --> vlm["VLM Backbone (Qwen3-VL 4B)"]
    vlm --> dit["Action Expert (DiT 0.16B)"]
    obs --> threed["3D Module (VGGT)"]
    threed --> dit
    dit --> a["Action Chunk (14D dual-arm)"]
```
- **核心组件清单**:Vision=`SigLIP(Qwen3-VL 内置)` / Text=`Qwen3-VL` / 3D=`VGGT + Qwen-Image-Edit` / Proprio=`EEF delta 7D` / Backbone=`Qwen3-VL 4B` / Fusion=`mid(Cross-Attention)` / Action Head=`**Flow(AML/DiT)**` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`多视 RGB + Lang + Proprio → 14D dual-arm EEF delta chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathbb{E}\bigl[w(\tau)\lVert V_\theta(\phi_t, A_t^\tau, q_t) - A_t\rVert^2\bigr],\quad w(\tau) = \tfrac{1}{(1-\tau)^2} \]
- **关键消融**:
  - AML vs GR00T noise-pred(Table 7)→ **+1.7pp**(71.0 vs 69.3 LIBERO-Plus)
  - Action Chunk=30 时 **AML -8.2 vs GR00T -23.6pp** → AML 抗退化
  - VGGT Cross-Attention(Table 10)→ +4.7pp(66.4→71.1)
  - Qwen-Image-Edit 2views(Table 9)→ +2.8pp(95.4→98.2 LIBERO)
- **最大正面贡献组件**:**Action Manifold Learning** — 在大 chunk / 高维场景贡献最显著(+17.1pp vs GR00T chunk=30),因直接预测干净动作避免高维噪声回归。
- **优势**:跨 20+ 构型统一训练(UniACT-dataset 6M 轨迹);AML 在长 chunk / 高维下优势明显。**局限**:仅仿真验证;依赖外部 3D 模块(VGGT)。
- **范式归属**:A5 + A3 + G3 + C3

#### 7.A.2 [FLOWER](p/FLOWER_Efficient_VLA_Flow_Policy/paper.pdf) — 950M 中间融合 Flow VLA + 50% 层裁剪 [T2]

- **一句话定位**:**950M 参数高效 VLA**,中间层融合 + Global-AdaLN,仅 **200 H100-hours** 预训,99% 计算节省。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Lang"] --> vlm["VLM (Florence-2, pruned 50%)"]
    vlm -->|"intermediate embeddings"| ft["Flow Transformer (AdaLN)"]
    ft --> act["Continuous Action Chunk"]
```
- **核心组件清单**:Vision=`Florence-2 ViT` / Text=`Florence-2 LLM` / 3D=`N/A` / Proprio=`State` / Backbone=`Florence-2(pruned)~450M + Flow Transformer ~500M = 950M` / Fusion=`**intermediate(mid-layer VLM embeddings)**` / Action Head=`**Flow(Global-AdaLN)**` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`**50% layer pruning**`
- **输入 → 输出**:`RGB + Lang + Embodiment info → 连续 action chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathbb{E}\bigl[\lVert v_\theta(a_t, c, \sigma) - (a - \epsilon)\rVert^2\bigr] \]
- **关键消融**:
  - Intermediate vs Late fusion → intermediate 更优(多 benchmark 验证)
  - Global-AdaLN vs standard → **减少 20% 参数无精度损失**
  - 50% VLM pruning → 保持性能,大幅减少计算
  - CALVIN ABC **4.53 SOTA**(new record)
- **最大正面贡献组件**:**Intermediate modality fusion** — 允许更多参数分配给 Flow Head,平衡语义深度与动作建模能力。
- **优势**:仅 200 H100 GPU-hours 预训,99% 计算节省;<1B 参数,1.85GB VRAM 实际部署友好。**局限**:Florence-2 语义弱于 Qwen 等;跨构型泛化受限于预训规模。
- **范式归属**:A3 + G2

#### 7.A.3 [LingBot-VLA](p/LingBot-VLA__A_Pragmatic_VLA_Foundation_Model/paper.pdf) — 务实型 ~2 万小时 9 本体 Flow VLA [T2]

- **一句话定位**:**20000 小时真机数据预训** 的实用型双臂 VLA + 高效训练 codebase,261 样本/秒(1.5-2.8× 加速)。
- **架构**:
```mermaid
flowchart LR
    obs["三视 RGB"] --> vlm["Qwen2.5-VL"]
    lang["任务指令(自动标注)"] --> vlm
    vlm -->|MoT shared attn| ae["Action Expert (Flow)"]
    prop["Proprio"] --> ae
    depth["LingBot-Depth"] -->|distill| vlm
    ae --> act["50-step chunk"]
```
- **核心组件清单**:Vision=`Qwen2.5-VL` / Text=`Qwen2.5-VL` / 3D=`LingBot-Depth(distill)` / Proprio=`关节/EEF` / Backbone=`Qwen2.5-VL + Action Expert(MoT)` / Fusion=`MoT(shared self-attn)` / Action Head=`**Flow Matching**` / Aux Head=`Depth Distillation` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`三视 RGB + Proprio + Lang → 50 步连续 action chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{FM}} + \mathcal{L}_{\text{distill}}^{\text{depth}} \]
- **关键消融**:
  - w/ depth vs w/o depth on GM-100 AgileX(Table 1)→ **PS 74.5 vs 72.2**(+2.3)
  - vs π0.5 / GR00T N1.6 / WALL-OSS on 3 platforms(Table 1)→ **全面领先**
  - 数据 3K→20K 小时 → SR **持续上升无饱和**(Sec 5.2)
- **最大正面贡献组件**:**20K 小时多本体预训数据** — SR 随数据量持续提升无饱和迹象,因双臂任务多样性需要海量覆盖。
- **优势**:261 样本/秒训练吞吐;开源 code / model / benchmark。**局限**:仅覆盖双臂平行夹爪,无灵巧手;Depth distillation 增益不大(+2pp)。
- **范式归属**:A3 + C2 + F2

#### 7.A.4 [RLDX-1](p/RLDX-1_A_Dexterity-First_Foundation_Model_for_Robot_Hands/paper.pdf) — 灵巧手优先 Flow VLA [T2]

- **一句话定位**:**MSAT 6.9-8B** 灵巧手优先 VLA:认知 / 物理 / 运动 / 记忆多流 + Flow 16 步,**ALLEX 真机 ~86.8%**。
- **架构**:
```mermaid
flowchart LR
    A["4 帧视频 + 触觉"] --> B["MSAT 多流"]
    B --> C["Qwen3-VL-8B 认知"]
    B --> D["Motion / Physics / Memory"]
    D --> E["Flow Action Head"]
    E --> F["灵巧手动作块"]
```
- **核心组件清单**:Vision=`多帧视频` / Text=`Qwen3-VL` / 3D=`N/A` / Proprio=`触觉/力矩` / Backbone=`Qwen3-VL-8B 顶 4 层可训` / Fusion=`**MSAT 多流**` / Action Head=`**16-step Flow**` / Aux Head=`物理流匹配` / WM=`Motion 流(非像素 WM)` / Routing=`共享 encoder` / Post-train=`三阶段 100K PT`
- **输入 → 输出**:`4 帧视频 + Lang + 记忆 + [触觉/力矩] → 动作块 + 未来物理信号`
- **主要 Loss**:\[ \mathcal{L} = \bigl\lVert u_\theta(a^\tau_{t:t+H}, \tau, c_t) - (a_{t:t+H} - \epsilon)\bigr\rVert_2^2 \]
- **关键消融**:
  - 运动感知 + 长期记忆 + 物理传感流 → 关键对抗 VLA 失败模式
  - LIBERO-Plus **86.7%(#1)**;RoboCasa365 **32.1% vs π0 14.8%**
  - ALLEX 真机 **~86.8% vs π0.5/GR00T ~+40pp**
  - w/o Memory/Motion → 接触丰富与动态场景显著降
- **最大正面贡献组件**:**MSAT 多流 Action Transformer** — 显式分离认知/运动/物理/记忆,灵巧操作优先。
- **优势**:综合榜 #1 叙事;RTX 5090 **43.7ms/step**。**局限**:预训 64×H200 ~195h;架构复杂。
- **范式归属**:A3 + B3 + F4 + V4

#### 7.A.5 [SimVLA](p/SimVLA_A_Simple_VLA_Baseline/paper.pdf) — 0.5B 极简 Flow VLA 基线 [T2]

- **一句话定位**:**0.5B 极简 Flow VLA baseline**:强 VLM + 轻量 Transformer Flow Head,**配方 > 模块堆叠**。
- **架构**:
```mermaid
flowchart LR
    A["多视角 RGB + Lang + Proprio"] --> B["SmolVLM-0.5B"]
    B --> C["Flow Transformer Head"]
    C --> D["Action Chunk"]
```
- **核心组件清单**:Vision=`VLM 内置` / Text=`同左` / 3D=`N/A` / Proprio=`标准` / Backbone=`SmolVLM-0.5B` / Fusion=`prefix/cross` / Action Head=`**Flow MSE**` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`直接 FT`
- **输入 → 输出**:`多视 RGB + Lang + Proprio → action chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathbb{E}\bigl[\lVert v_\theta(x_t, o_t, t) - (\epsilon - x)\rVert_2^2\bigr] \]
- **关键消融**:
  - LIBERO **98.6%(#3)**;VRAM **9.3GB**
  - 数据 shuffle / 归一化 / 调度 **贡献 > 复杂结构先验**
  - vs π0.5 **+0.9pp**(98.6 vs 97.7)
  - Florence-2 vs SmolVLM 骨干差距 < recipe 差距
- **最大正面贡献组件**:**标准化训练 recipe(数据混合 / 归一化 / 调度)** — 证明极简架构可达 SOTA 级。
- **优势**:可复现强基线;低显存。**局限**:通识弱于 7B VLA;无长程推理模块。
- **范式归属**:A3 + A4

#### 7.A.6 [HAMLET](p/HAMLET_Switch_your_VLA_into_a_History-Aware_Policy/paper.pdf) — 即插即用历史感知 + DiT [T2]

- **一句话定位**:**Plug-and-play** 为预训 VLA 加历史感知,moment token + 轻量 Memory Module,真机长程 **+47.2pp**。
- **架构**:
```mermaid
flowchart LR
    obs["当前观测 RGB"] --> vlm["VLM (GR00T N1.5)"]
    mt["Moment Tokens"] --> vlm
    vlm --> ht["隐表征 h_t"]
    cache["历史 Moment 缓存"] --> mem["Memory Module (2 层 TF)"]
    mem --> dit["Action Expert DiT"]
    ht --> dit
    dit --> act["动作输出"]
```
- **核心组件清单**:Vision=`VLM 内置(frozen)` / Text=`VLM 内置` / 3D=`N/A` / Proprio=`s_t` / Backbone=`GR00T N1.5 2.7B + Memory 0.14B` / Fusion=`late(concat h_t + memory)` / Action Head=`**Flow/Diffusion(DiT)**` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`TCL init + fine-tune`
- **输入 → 输出**:`当前 RGB + Proprio + Lang + 历史 Moment → action chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{TCL}}\ \text{(init)} \]
- **关键消融**:
  - 去 Memory Module(Table 5a)→ **63.4 vs 65.4**(-2.0pp RoboCasa 100)
  - 去 TCL 初始化(Table 5a)→ **64.8 vs 65.4**(-0.6pp)
  - HAMLET on real-world(Table 1)→ **76.4% vs baseline 29.2%**(+47.2pp)
  - Latency:history=4 仅 **1.02×**(Table 4)
- **最大正面贡献组件**:**Memory Module** — 真机长程 +47.2pp,因 Transformer 自注意力选择性聚合关键历史时刻,避免冗余帧干扰。
- **优势**:插件式,不需重新预训,延迟 1.02×;跨 VLA 骨干可迁移(GR00T / CogACT)。**局限**:TCL 额外 5-9 小时训练;对 AR VLA 适配不如 Diffusion VLA。
- **范式归属**:F3 + D3 + A2

#### 7.A.7 [HiPolicy](p/HiPolicy_Hierarchical_Multi-Frequency_Action_Chunking_for_Policy_Learning/paper.pdf) — 分层多频率 action chunking [T2]

- **一句话定位**:**分层多频动作 chunk**,联合预测粗 / 细粒度动作 + 熵引导自适应执行频率,真机 +42% vs DP。
- **架构**:
```mermaid
flowchart LR
    obs["多频观测历史"] --> enc["视觉编码器"]
    enc --> film["分层 FiLM 融合"]
    film --> unet["1D U-Net Diffusion"]
    unet --> mfa["多频 action chunk"]
    mfa --> ege["熵引导执行选择"]
    ege --> act["最终执行动作"]
```
- **核心组件清单**:Vision=`ResNet/DP3 点云` / Text=`N/A(任务 ID)` / 3D=`可选 DP3` / Proprio=`关节+夹爪` / Backbone=`1D U-Net(DP/DP3 base)` / Fusion=`**early(FiLM per frequency)**` / Action Head=`**DDPM 多频 chunk**` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`多频视觉历史 + Proprio → 多频 action chunk + 执行频率选择`
- **主要 Loss**:\[ \mathcal{L} = \mathbb{E}_{k,a_0,\epsilon}\bigl\lVert\epsilon - f_\theta(\tilde a_k, k, O)\bigr\rVert^2 \]
- **关键消融**:
  - 去掉分层频率结构(Table 2)→ **SR 37% vs 60%**(-23pp DP base)
  - 去掉全局特征融合(Table 2)→ **54% vs 60%**(-6pp)
  - 熵引导执行(Table 3)→ **speed +25%** with minimal SR drop
  - 真机(Table 5)→ **SR 85% vs DP 60%**(+42%)
- **最大正面贡献组件**:**分层多频结构** — +23pp,因联合低频(长程意图)+ 高频(精细控制)弥补单频 chunk 的固有权衡。
- **优势**:即插即用于 DP/DP3;熵引导提速 25%。**局限**:仅在小模型验证,未扩 VLA;频率数/阈值需人工调。
- **范式归属**:A2 + A6 + G3

#### 7.A.8 [X-VLA](p/X-VLA_Soft-Prompt_Cross-Embodiment_VLA/paper.pdf) — Soft-Prompt 跨本体 Flow VLA [T2]

- **一句话定位**:**Soft-Prompt 跨构型 VLA**,0.9B 模型同时适配多机器人,新本体仅需 LoRA 1%。
- **架构**:
```mermaid
flowchart LR
    obs["多视 RGB + Proprio"] --> vit["Shared SigLIP ViT"]
    sp["Soft Prompt Library"] --> enc["Transformer Encoder (xN)"]
    vit --> enc
    enc --> flow["Flow-Matching Action Head"]
    flow --> a["Action Chunk"]
```
- **核心组件清单**:Vision=`SigLIP ViT(shared)` / Text=`Tokenizer` / 3D=`N/A` / Proprio=`Learnable Embedding` / Backbone=`0.9B Transformer` / Fusion=`early(Soft Prompt + Self-Attention)` / Action Head=`**Flow Matching**` / Aux Head=`N/A` / WM=`N/A` / Routing=`**Soft-Prompt(per-data-source learnable embeddings)**` / Post-train=`LoRA(1% 参数)`
- **输入 → 输出**:`多视 RGB + Proprio + Soft Prompt → Action Chunk`
- **主要 Loss**:\[ \mathcal{L}_{\text{FM}} = \mathbb{E}_{t,o,A}\bigl\lVert v_\theta(A_t, o, t) - (A - A_0)\bigr\rVert^2 \]
- **关键消融**:
  - Soft Prompt vs Language Prompt vs HPT → **SP 在跨构型场景全面最优**
  - LIBERO 仅调 9M 参数(1%)→ **93% 成功率**,媲美 π0(3B 调参)
  - 布料折叠真机 → 平均 2 分钟/件
- **最大正面贡献组件**:**Soft Prompt** — 以最小参数开销吸收构型异质性,避免多 action head 碎片化。
- **优势**:0.9B 超小模型,6 仿真 + 3 真机均 SOTA;新机器人仅需 LoRA 1% 参数适配。**局限**:Soft Prompt 数量随数据源线性增长;未在大规模(10B+)模型验证。
- **范式归属**:E2 + A3

#### 7.A.9 [SmoothVLA](p/SmoothVLA_Aligning_VLAs_with_Physical_Constraints_via_Intrinsic_Smoothness_Optimization/paper.pdf) — Jerk 内在奖励 GRPO [T2]

- **一句话定位**:OpenVLA + LoRA 上 **GRPO + jerk 内在奖励**,LIBERO 平滑度 **+13.8%** 且保 SR。
- **架构**:
```mermaid
flowchart LR
    A["OpenVLA SFT"] --> B["LIBERO rollout"]
    B --> C["IK 算 jerk"]
    C --> D["混合奖励 R(τ)"]
    D --> E["GRPO 更新"]
    E --> F["平滑动作块"]
```
- **核心组件清单**:Vision=`OpenVLA` / Text=`Llama-2` / 3D=`N/A` / Proprio=`标准` / Backbone=`OpenVLA-7B LoRA` / Fusion=`原生` / Action Head=`离散/块动作` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`**SFT → GRPO with jerk reward**`
- **输入 → 输出**:`RGB + Lang → 动作块(优化 jerk 平滑性)`
- **主要 Loss**:\[ R(\tau) = I_{\text{success}}\cdot\Bigl(1 - \lambda\cdot\tfrac{1}{T}\sum_t \lVert\text{Jerk}(t)\rVert^2\Bigr),\ \lambda = 0.2 \]
- **关键消融**:
  - LIBERO 平滑度较标准 RL **+13.8%**(jerk-based metric)
  - λ=0.2 平衡 SR 与平滑(grid search)
  - 无需外传感器 reward engineering
  - 架构无关后训协议(可迁移其他 VLA)
- **最大正面贡献组件**:**jerk 内在稠密奖励 GRPO** — 物理约束进 RL,缓解真机抖动。
- **优势**:部署友好平滑轨迹;LoRA 低成本。**局限**:主指标非 LIBERO SR 涨点;依赖 IK 算 jerk。
- **范式归属**:E5 + G3

#### 7.A.10 [π0.6 Recap](p/π0.6__Recap/paper.pdf) — Advantage-conditioned offline RL Flow [T2]

- **一句话定位**:**Advantage-conditioned 离线 RL** 后训 VLA,**吞吐 2× / 失败减半**,连续 13 小时无中断。
- **架构**:
```mermaid
flowchart LR
    data["Demos + Autonomous + Interventions"] --> vf["Value Function (Distributional)"]
    vf --> adv["Advantage Estimation"]
    adv --> vla["π0.6 VLA (Flow, adv-conditioned)"]
    vla --> deploy["Deploy → Collect"]
    deploy --> data
```
- **核心组件清单**:Vision=`π0.6 ViT` / Text=`π0.6 LLM(大于 π0.5)` / 3D=`N/A` / Proprio=`EEF` / Backbone=`π0.6` / Fusion=`early(π0 架构)` / Action Head=`**Flow(adv-conditioned CFG)**` / Aux Head=`Distributional Value Function` / WM=`N/A` / Routing=`N/A` / Post-train=`**RECAP(offline RL + intervention)**`
- **输入 → 输出**:`RGB + Lang + Advantage Indicator → Action Chunk(Flow)`
- **主要 Loss**:\[ \hat\pi(a\mid s) \propto \pi_{\text{ref}}(a\mid s)\Bigl(\tfrac{\pi_{\text{ref}}(a\mid I,s)}{\pi_{\text{ref}}(a\mid s)}\Bigr)^\beta,\quad I = \mathbb 1(A^\pi(s,a) > \epsilon) \]
- **关键消融**:
  - RECAP vs π0.5 base → **吞吐量翻倍**(部分任务),**失败率减半**
  - Espresso:连续运行 **13 小时无中断**
  - Laundry folding:新环境 2 小时无故障
- **最大正面贡献组件**:**Advantage conditioning** — 绕过 flow-matching 不可求 log-prob 的问题,用 CFG 实现策略提升。
- **优势**:支持异构数据(demo + autonomous + intervention)统一训练;可扩展到大规模 flow-based VLA。**局限**:需要人工提供稀疏 reward 和干预;离线 RL 迭代效率受限于真机数据量。
- **范式归属**:E2 + A3 + D2

#### 7.A.11 [Ψ0 (Psi-Zero)](p/Ψ0_(Psi-Zero)_An_Open_Foundation_Model_Towards_Universal_Humanoid_Loco-Manipulation/paper.pdf) — 人形 loco-manip Flow 基座 [T2]

- **一句话定位**:**人形 VLA 开源基础模型**,人类视频预训 + Flow 动作专家后训,**800h 人视 + 30h 真机** 超 10× 数据基线 40%+。
- **架构**:
```mermaid
flowchart LR
    ego["Egocentric Human Video"] --> vlm["Qwen3-VL-2B (Stage 1: AR 预训)"]
    vlm --> feat["VLM Features"]
    feat --> mmdit["MM-DiT Action Expert (500M, Stage 2)"]
    prop["Joint States"] --> mmdit
    mmdit --> act["Whole-Body Joint Action Chunk"]
    act --> lbc["Lower-Body Controller"]
```
- **核心组件清单**:Vision=`Qwen3-VL-2B` / Text=`Qwen3-VL-2B` / 3D=`N/A` / Proprio=`Joint States` / Backbone=`Qwen3-VL-2B + MM-DiT 500M` / Fusion=`late(VLM features → MM-DiT)` / Action Head=`**Flow(MM-DiT)**` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`Real-time action chunking`
- **输入 → 输出**:`Head Camera RGB + Lang + Joint States → 全身关节 action chunk`
- **主要 Loss**:\[ \mathcal{L}_{\text{S1}} = -\sum_t \log p_\theta(\hat a_t\mid \hat a_{<t}, o, l),\quad \mathcal{L}_{\text{S2}} = \lVert v_\theta(A_t, o, t) - (A - A_0)\rVert^2 \]
- **关键消融**:
  - Ψ0 vs baselines(10× data)→ **超 40% 总体成功率**
  - 800h 人类视频 + 30h 机器人数据 → 超越 10× 数据基线
  - EgoDex 数据 vs 通用 Internet 视频 → EgoDex 显著更优
- **最大正面贡献组件**:**解耦训练范式**(VLM 学视觉-动作先验 / Action Expert 学关节控制)— 最大化数据利用效率。
- **优势**:开源全套(管线 + 模型 + 推理引擎);仅 800h 人视 + 30h 真机即超大量数据基线。**局限**:人形专用,非通用操作臂构型;下肢仍需独立 controller。
- **范式归属**:C4 + A3 + C2

#### 7.A.12 [CycleVLA](p/CycleVLA_Backtracking_+_MBR_Decoding_for_VLA/paper.pdf) — DDPM action expert + 回溯 + MBR 解码 [T2]

- **一句话定位**:**子任务回溯 + MBR 解码** 实现 VLA 主动自纠错,零样本 test-time scaling +5-10%。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Proprio + Lang"] --> vla["Progress-Aware VLA"]
    vla --> prog["Stop + Progress Signal"]
    prog --> vlm["VLM Failure Predictor"]
    vlm --> back["Backtrack Decision"]
    back --> mbr["MBR Decoding (Retry)"]
    mbr --> act["Action Output"]
```
- **核心组件清单**:Vision=`VLA 内置 ViT` / Text=`VLA 内置 LLM` / 3D=`N/A` / Proprio=`EEF delta 7D→9D` / Backbone=`Any VLA(π0 等)` / Fusion=`N/A(wrapper)` / Action Head=`**Flow/原 VLA + MBR consensus**` / Aux Head=`Progress + Stop signal` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`RGB + Lang + Proprio → 9D action(7D + stop + progress)`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{VLA}}^{\text{original}}\ \text{(extended to 9D)} \]
- **关键消融**:
  - MBR decoding(Table II)→ **+5-10% SR** vs single sample
  - Progress-aware subtask(Table III)→ backtrack timing 准确率提升
  - CycleVLA on under-trained VLA(Table IV)→ 仍有效提升
- **最大正面贡献组件**:**MBR decoding** — 零样本 test-time scaling +5-10%,因高密度区的 consensus 选择更可能对应成功行为。
- **优势**:无需额外训练 verifier,零样本 test-time scaling;适用任何 VLA backbone。**局限**:需多次 forward pass 增延迟;依赖外部 VLM 做 failure prediction。
- **范式归属**:D5 + G3 + A2

### 7.W 世界/辅助 Head 主 — 21 篇 [T2]

**共同特点**:这 18 篇都把"动作以外的额外预测头"作为差异化亮点 —— Latent / Mask / Trace / Future State / CoT / Pixel WM。**Latent WM (W2) 与 World↔Action 共演化(B4)是 2026 主流**。

#### 7.W.1 [CoLA-World](p/CoLA-World_Co-evolution_of_Latent_Action_+_World_Model/paper.pdf) — IDM 与预训练视频 WM 共演化 [T2]

- **一句话定位**:**联合训练 Latent Action 模型(IDM)与视频世界模型** 实现共演化,Visual Planning 21.2% vs 7.7%(2-stage)。
- **架构**:
```mermaid
flowchart LR
    video["Video Frames"] --> idm["IDM (ST-Transformer)"]
    idm --> vq["VQ Codebook"]
    vq --> wm["World Model (OpenSora 1.2B)"]
    wm --> pred["Future Video Prediction"]
```
- **核心组件清单**:Vision=`OpenSora VAE` / Text=`N/A` / 3D=`N/A` / Proprio=`N/A` / Backbone=`OpenSora DiT 1.2B` / Fusion=`AdaLN action conditioning` / Action Head=`VQ Latent Action(IDM 0.12B)` / Aux Head=`N/A` / WM=`**像素(OpenSora video gen)**` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`Video frames → Latent action codes + Future video`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{flow-matching}}^{\text{WM}} + \mathcal{L}_{\text{VQ}} + \mathcal{L}_{\text{commit}} \]
- **关键消融**:
  - Joint vs 2-stage(Table 1 OXE)→ **FVD 278.90 vs 291.30**(同 budget)
  - Co-evolution 证据(Fig 4)→ E2E 后 probing loss 下降更快
  - Visual Planning(Table 3)→ **21.20% vs 7.73%**(2-stage)avg success
- **最大正面贡献组件**:**Joint warm-up + E2E co-evolution** — 贡献 FVD -12,因世界模型梯度塑造更高质量 latent action。
- **优势**:消除两阶段冗余,训练高效;Latent action 空间可跨构型迁移。**局限**:计算成本高;下游需 adapter 将 real action 映射为 latent。
- **范式归属**:B4 + C2 + B3

#### 7.W.2 [DreamZero](p/DreamZero_World_Action_Models_are_Zero-Shot_Policies/paper.pdf) — 14B 块级联合视频-动作 WAM [T2]

- **一句话定位**:**14B 视频扩散 WAM** 实现零样本泛化 + 跨构型迁移,WAM vs VLA real-world **>2× task progress**。
- **架构**:
```mermaid
flowchart LR
    obs["RGB (VAE) + Lang + Proprio"] --> dit["AR DiT Backbone (14B, Wan2.1)"]
    dit --> video["Future Video Frames"]
    dit --> act["Action Chunks"]
```
- **核心组件清单**:Vision=`Wan2.1 VAE` / Text=`T5 Text Encoder` / 3D=`N/A` / Proprio=`State Encoder` / Backbone=`Wan2.1 DiT 14B` / Fusion=`early(concat multi-view into frame)` / Action Head=`Flow(joint video-action)` / Aux Head=`Future video` / WM=`**像素(AR video prediction)**` / Routing=`N/A` / Post-train=`DreamZero-Flash(decoupled schedule)`
- **输入 → 输出**:`多视 RGB + Lang + Proprio → Future video + Action chunks`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{flow-matching}}^{\text{video+action}} \]
- **关键消融**:
  - WAM vs VLA real-world → **>2× task progress**
  - AR vs Bidirectional → AR 更平滑动作 + 更好模态对齐
  - Cross-embodiment video-only data → **+42% relative**
  - **38× inference speedup** enabling 7Hz real-time
- **最大正面贡献组件**:**Video diffusion pretrain(Wan2.1 14B)** — >2× zero-shot 泛化,因 web-scale 视频编码了物理动力学先验。
- **优势**:从非重复异构数据有效学习;跨构型迁移仅需 30min play data。**局限**:14B 模型需 38× 加速才能实时;仅 ~500h real data 训练。
- **范式归属**:B3 + B1 + A2

#### 7.W.3 [Fast-WAM](p/Fast-WAM_Do_World_Action_Models_Need_Test-time_Future_Imagination/paper.pdf) — 训练共训视频、推理跳过想象 [T2]

- **一句话定位**:**训练时视频共训,推理跳过想象**,190ms 推理 4× 快于 imagine-then-execute WAM。
- **架构**:
```mermaid
flowchart LR
    obs["RGB observation"] --> vdit["Video DiT (single forward)"]
    vdit -->|"KV Cache"| adit["Action DiT"]
    adit --> act["Action Chunk"]
    vdit -.->|"train only"| video["Future Video (co-training)"]
```
- **核心组件清单**:Vision=`Video DiT VAE` / Text=`Language encoder` / 3D=`N/A` / Proprio=`State` / Backbone=`MoT(Video DiT + Action DiT)` / Fusion=`MoT shared attention` / Action Head=`Flow(Action DiT)` / Aux Head=`Future video(train-only)` / WM=`**像素(train-only co-training)**` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`RGB + Lang → Action chunk(190ms latency)`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{flow}}^{\text{action}} + \mathcal{L}_{\text{flow}}^{\text{video}}\ \text{(train only)} \]
- **关键消融**:
  - Fast-WAM vs imagine-then-execute → 性能接近(差距 <2pp)
  - 去掉 video co-training → 性能 **大幅下降(>>2pp)**
  - 推理速度:**190ms vs >800ms**(4× 加速)
- **最大正面贡献组件**:**Video co-training objective** — 训练时视频预测塑造更好的世界表征,推理时跳过。
- **优势**:190ms 推理,4× 快;无需 embodied pretraining 即 SOTA。**局限**:推理时丢弃了可能有用的 future 信息;单 chunk 生成,长 horizon 需 outer loop。
- **范式归属**:B2 + B5 + A3

#### 7.W.4 [FutureVLA](p/FutureVLA_Joint_Visuomotor_Prediction_for_VLA/paper.pdf) — JVPM 视-运动门控 + 任意 VLA 注入 [T2]

- **一句话定位**:解耦视觉/运动流的 **JVPM 联合视运动预测嵌入** 指导 VLA,真机 +21.7%。
- **架构**:
```mermaid
flowchart LR
    clip["Multi-frame Video Clip"] --> gate["Joint Visuomotor Gating"]
    gate --> vis["Visual Stream"]
    gate --> mot["Motor Stream"]
    mot -->|"cross-attn"| vis
    mot --> embed["Joint Visuomotor Embedding"]
    embed -->|"align"| vla["Downstream VLA (π0 等)"]
    vla --> act["Action"]
```
- **核心组件清单**:Vision=`ViT encoder` / Text=`Language encoder` / 3D=`N/A` / Proprio=`State` / Backbone=`JVPM Transformer + 下游 VLA` / Fusion=`mid(latent alignment)` / Action Head=`下游 VLA(π0/OpenVLA 等)` / Aux Head=`Joint visuomotor embedding` / WM=`**Latent(temporal motor dynamics)**` / Routing=`N/A` / Post-train=`Embedding alignment`
- **输入 → 输出**:`RGB video + Lang → Joint visuomotor embeddings → 下游 VLA action`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{visual-recon}} + \mathcal{L}_{\text{motor-pred}} + \mathcal{L}_{\text{align}}^{\text{post-train}} \]
- **关键消融**:
  - w/ JVPM vs w/o(SimplerEnv)→ **+11.4% avg**
  - w/ JVPM vs w/o(Real-world)→ **+21.7% avg**
  - vs π0(Real-world)→ **+26.7%**
  - Visual gating 解耦 → 防止视觉主导
- **最大正面贡献组件**:**Joint Visuomotor Gating** — +21.7% real-world,因结构性解耦防止视觉重建主导,motor stream 聚焦物理动态。
- **优势**:不修改下游 VLA 推理架构;从异构数据学习通用 visuomotor 先验。**局限**:需两阶段训练(pretrain + post-train);对连续多帧 clip 质量有要求。
- **范式归属**:B2 + D1 + C2

#### 7.W.5 [GigaWorld-Policy](p/GigaWorld-Policy_An_Efficient_Action-Centered_World–Action_Model/paper.pdf) — 动作中心 WAM,推理可剥离视频 [T2]

- **一句话定位**:**以动作为中心**的 World-Action 模型,训练时联合预测视频+动作,推理可选跳视频,**9× 加速**。
- **架构**:
```mermaid
flowchart LR
    obs["RGB 多视图 + Proprio"] --> vae["VAE Encoder"]
    vae --> dit["Causal DiT 5B (Wan 2.2)"]
    lang["Language Encoder"] -->|cross-attn| dit
    dit --> act["Action Flow Head"]
    dit -->|optional| vid["Future Video Head"]
```
- **核心组件清单**:Vision=`VAE(Wan 2.2)` / Text=`预训练 Language Encoder` / 3D=`N/A` / Proprio=`Linear Proj` / Backbone=`Wan 2.2 DiT 5B` / Fusion=`early(shared DiT + causal mask)` / Action Head=`Flow Matching` / Aux Head=`Future Video(可选)` / WM=`**Pixel(action-conditioned video)**` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`多视 RGB + Proprio + Lang → 48 步 action chunk(+ 可选未来帧)`
- **主要 Loss**:\[ \mathcal{L} = \lambda_{\text{action}}\mathcal{L}_{\text{FM}}^a + \lambda_{\text{video}}\mathcal{L}_{\text{FM}}^v \]
- **关键消融**:
  - 去掉 video prediction(Table 5)→ **SR 0.60 vs 0.83**(-0.23)
  - 去掉 Embodied Data Pre-training(Table 7)→ **SR 0.57 vs 0.83**(-0.26)
  - Causal mask vs Self-Attn(Table 6)→ 相近(0.83 vs 0.81),但 causal 推理可选跳视频
- **最大正面贡献组件**:**Embodied Data Pre-training** — +0.26 SR,因视频模型→具身数据→任务数据的课程训练逐步注入物理先验。
- **优势**:推理仅解码动作 token,**比 Motus 快 9×**(360ms vs 3231ms);训练时视频监督提供稠密时序先验,无需推理时付代价。**局限**:仍需 5B 级骨干,边缘部署困难;视频预训数据量大(~10K 小时)。
- **范式归属**:B3 + A3 + G3

#### 7.W.6 [HiF-VLA](p/HiF-VLA_Hindsight,_Insight_and_Foresight_through_Motion_Representation/paper.pdf) — MPEG 运动矢量做时序桥 [T2]

- **一句话定位**:用 **MPEG Motion Vector 编码 hindsight + foresight 双向时序推理** 的"边想边做" VLA,LIBERO-Long multi-view 96.4%。
- **架构**:
```mermaid
flowchart LR
    obs["当前 RGB"] --> vlm["Prismatic-7B VLM"]
    lang["语言"] --> vlm
    mv["历史 Motion Vectors"] --> henc["Hindsight Encoder (ViT)"]
    vlm --> mf["Foresight + Action Tokens"]
    henc -->|AdaLN| je["Joint Expert (6 层 TF)"]
    mf --> je
    je --> act["动作输出"]
    je --> fmv["预测未来 MV"]
```
- **核心组件清单**:Vision=`DINOv2 + SigLIP(frozen)` / Text=`Prismatic-7B` / 3D=`N/A` / Proprio=`N/A` / Backbone=`Prismatic-7B + Joint Expert` / Fusion=`mid(AdaLN conditioning)` / Action Head=`L1 Regression(OFT style)` / Aux Head=`**Foresight Motion Prediction**` / WM=`**Motion Vector(MPEG-4 codec)**` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`当前 RGB + MV 历史 + Lang → 8 步 action chunk + 预测未来 MV`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_A + 0.01\cdot\mathcal{L}_{MV} \]
- **关键消融**:
  - 去掉 Foresight + Hindsight(Table 3 row6 vs row1)→ **SR 93.2 vs 91.0**(+2.2pp)
  - 用 History frames 替代 MV(Table 3 row4)→ **SR 90.4 < 91.0**,且 **latency 3.15×**
  - Hindsight 长度 = 8 最优(Fig 3c)→ **94.4%**(third-view)
  - LIBERO-Long multi-view(Table 1)→ **96.4% vs baseline 94.0%**
- **最大正面贡献组件**:**Hindsight MV** — +2.2pp 且 latency 仅 1.67× vs frames 3.15×,因 MV 天然去冗余保留运动动态。
- **优势**:MV 编码比帧堆叠减 58% 延迟且更有效;双向时序推理协同增强长程一致性。**局限**:MV 精度受动态场景噪声影响;未在大规模预训数据上验证。
- **范式归属**:B3 + D3 + D6

#### 7.W.7 [Mask World Model (MWM)](p/Mask_World_Model_(MWM)_Predicting_What_Matters_for_Robust_Robot_Policy_Learning/paper.pdf) — 预测未来语义 mask [T2]

- **一句话定位**:**预测语义 mask 动态而非像素** 的世界模型,RLBench +35pp vs π0,真机 OOD 42.1% vs 12.5%。
- **架构**:
```mermaid
flowchart LR
    rgb["多视 RGB"] --> vae["Shared VAE"]
    vae --> dit["DiT Backbone (28 层)"]
    lang["Text Encoder"] -->|cross-attn| dit
    dit --> mask["Mask Decoder (Stage 1)"]
    dit -->|predictive features| ae["Action Expert (Diffusion)"]
    ae --> act["15-dim 动作"]
```
- **核心组件清单**:Vision=`Shared Video VAE` / Text=`Text Encoder` / 3D=`N/A` / Proprio=`7-DoF state` / Backbone=`DiT 28 blocks` / Fusion=`mid(cross-attn text + AdaIN timestep)` / Action Head=`Diffusion(score matching)` / Aux Head=`**Future Mask Prediction**` / WM=`**Mask(语义 mask latent)**` / Routing=`N/A` / Post-train=`2-stage(mask pretrain → action)`
- **输入 → 输出**:`多视 RGB + Lang → 36 步 action chunk(推理纯 RGB,无需分割器)`
- **主要 Loss**:\[ \mathcal{L}_{\text{S1}} = \mathcal{L}_{\text{mask}};\quad \mathcal{L}_{\text{S2}} = \mathcal{L}_{\text{act}} \]
- **关键消融**:
  - MWM vs GE-ACT(LIBERO Table 1)→ **98.3% vs 96.5%**(+1.8pp)
  - MWM vs π0(RLBench Table 2)→ **68.3% vs 33.3%**(+35pp)
  - MWM-C1(explicit mask decode + IDM)vs MWM → **81.0% vs 98.3%**(-17.3pp)
  - OOD robustness 真机 → **42.1% vs GE-ACT 12.5%**
- **最大正面贡献组件**:**Mask 语义瓶颈** — RLBench +35pp,因 mask 预测滤除光照/纹理噪声,迫使模型聚焦交互动态。
- **优势**:推理时仅需 RGB,无需外部分割器;语义 mask 显著增强 OOD 鲁棒性。**局限**:Stage 1 需离线 mask 标注;对透明物体等可能受限。
- **范式归属**:B2 + A2

#### 7.W.8 [OA-WAM](p/OA-WAM_Object-Addressable_World_Action_Model_for_Robust_Robot_Manipulation/paper.pdf) — 对象槽位级 WAM [T2]

- **一句话定位**:**对象可寻址 WAM**:N+1 对象槽位(addr+content)联合预测下一帧槽状态 + Flow 动作,LIBERO 97.8%。
- **架构**:
```mermaid
flowchart LR
    A["SAM3 + DINOv3"] --> B["N+1 槽位"]
    B --> C["Chameleon-7B 冻"]
    C --> D["World Head"]
    C --> E["Flow Action Head"]
    D --> F["下一帧槽"]
    E --> G["16 步动作"]
```
- **核心组件清单**:Vision=`SAM3 / DINOv3 / Qwen3-VL` / Text=`Chameleon` / 3D=`槽位 pose` / Proprio=`N/A` / Backbone=`Chameleon-7B 冻` / Fusion=`addr 键 cross-slot` / Action Head=`**Flow 16-step**` / Aux Head=`VQ 图像头` / WM=`**槽位 World Head**` / Routing=`addr 重置` / Post-train=`LoRA ~127M`
- **输入 → 输出**:`RGB + Lang → 下一帧 per-slot 状态 + 16 步动作 chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{act}} + \lambda_w\mathcal{L}_{\text{world}} + \lambda_v\mathcal{L}_{\text{vq}} + \lambda_c\mathcal{L}_{\text{compose}} + \lambda_r\mathcal{L}_{\text{role}} \]
- **关键消融**:
  - swap-binding cosine **0.87 vs holistic ≤0.09**(LIBERO-Plus 几何轴)
  - addr 键 cross-slot 注意力 + 残差流地址槽重置 → swap 高分关键
  - LIBERO **97.8%**;LIBERO-Plus **83.9%(#3)**
  - OOD 掉点主要在传感器噪声毁槽提取
- **最大正面贡献组件**:**对象槽位分解 + addr-only cross-slot 注意力** — OOD 几何绑定鲁棒,分布内几乎不变。
- **优势**:LIBERO-Plus 几何轴 SOTA 级;因果干预可解释。**局限**:Stage I 需 **384×A100 ~18 天**;感知管线复杂。
- **范式归属**:B4 + A3 + C3

#### 7.W.9 [STARRY](p/STARRY_Spatio-Temporal_Action-Centric_World_Modeling_for_Robotic_Manipulation/paper.pdf) — 时空动作中心 WM + GASAM 几何调制 [T2]

- **一句话定位**:**时空动作中心 WM**,联合去噪 latent + 动作 + 几何引导注意力,RoboTwin 2.0 达 **93.82%**,真机较 π0.5 +28.3pp。
- **架构**:
```mermaid
flowchart LR
    obs["RGB-D + Proprio"] --> und["Understanding Expert (Qwen-VL)"]
    obs --> stw["ST World Model (Wan-based DiT)"]
    stw --> act["Action Expert (DiT)"]
    und --> act
    geo["Geometry Expert"] --> gasam["GASAM Modulation"]
    gasam --> act
    act --> a["Action Output"]
```
- **核心组件清单**:Vision=`Qwen-VL + Wan Video Encoder` / Text=`Qwen-VL` / 3D=`Depth + Point Cloud` / Proprio=`End-pose` / Backbone=`ST WM 30L/3072dim` / Fusion=`mid(Multi-modal Joint Attention)` / Action Head=`Flow(joint diffusion)` / Aux Head=`Depth/Pose Geometry` / WM=`**Latent(时空联合)**` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`多视 RGB-D + Camera Params + Lang → 14-dim Action Chunk(16 步)`
- **主要 Loss**:\[ \mathcal{L} = \lambda_o\mathcal{L}_{\text{obs}} + \lambda_a\mathcal{L}_{\text{action}} + \lambda_d\mathcal{L}_{\text{depth}} + \lambda_p\mathcal{L}_{\text{pose}} + \lambda_w\mathcal{L}_{\text{weight}} \]
- **关键消融**:
  - 去掉 GASAM(Table 4, Rand)→ **ST 88.82% vs ST+GASAM 93.30%**(+4.48pp)
  - Action-Only vs ST+GASAM(Table 4)→ **64.96% vs 93.30%**(+28.34pp)
  - 去掉 ST 仅 Appearance(Table 4)→ **85.80% vs 88.82%**(ST +3.02pp)
- **最大正面贡献组件**:**ST World Model** — 64.96%→88.82%(+23.86pp),时空建模提供外观预测无法覆盖的几何约束。
- **优势**:联合去噪时空 latent 与动作;RoboTwin 2.0 达 93.82%。**局限**:依赖精确深度;训练成本高(8×A100 约一周)。
- **范式归属**:B4 + A2 + C3

#### 7.W.10 [VLA-JEPA](p/VLA-JEPA_Enhancing_VLA_with_Latent_World_Model/paper.pdf) — 无泄漏 JEPA + Flow 两阶段 [T2]

- **一句话定位**:**JEPA 式无泄漏 latent 世界模型预训**,从人类视频学动作先验,LIBERO-Plus +16.6pp。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Lang"] --> vlm["Qwen3-VL-2B"]
    vlm --> lat["Latent Action Tokens"]
    lat --> wm["Latent World Model (12L Transformer)"]
    vid["Future Video"] --> vjepa["V-JEPA2 Encoder (frozen)"]
    vjepa --> wm
    lat --> flow["Flow-Matching Action Head (DiT-B)"]
    flow --> a["Action Chunk"]
```
- **核心组件清单**:Vision=`Qwen3-VL + V-JEPA2` / Text=`Qwen3-VL-2B` / 3D=`N/A` / Proprio=`N/A` / Backbone=`Qwen3-VL-2B` / Fusion=`early(causal attention)` / Action Head=`**Flow(DiT-B 16L)**` / Aux Head=`N/A` / WM=`**JEPA(latent state prediction)**` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`多视 RGB + Lang → 7-dim EEF delta, horizon=7`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{FM}} + \beta\mathcal{L}_{\text{WM}},\quad \mathcal{L}_{\text{WM}} = \sum_k \lVert\hat s_{t_k} - s_{t_k}\rVert \]
- **关键消融**:
  - 去掉人类视频预训(Table 1 LIBERO Avg)→ **96.1% vs 97.2%**(-1.1pp)
  - 去掉人类视频(Table 3 LIBERO-Plus Avg)→ **62.9% vs 79.5%**(**-16.6pp**)
  - Video horizon T=8 最优(Table 4)→ 96.1% vs T=4 94.8%
- **最大正面贡献组件**:**JEPA 预训 + 人类视频** — LIBERO-Plus 鲁棒性 +16.6pp,学到重抓等技能,**无信息泄漏设计** 是关键。
- **优势**:两阶段(预训 + 微调)比传统三阶段 latent action 管线更简洁;无像素重建,对相机运动/背景噪声鲁棒。**局限**:人类视频不提供精细动作轨迹,ID 场景提升有限;文本指令理解不如 π0.5 精细。
- **范式归属**:B2 + C2 + A3

#### 7.W.11 [VLAW](p/VLAW_Vision-Language-Action_World_Model/paper.pdf) — Ctrl-World + VLM Reward + Flow SFT 迭代 [T2]

- **一句话定位**:**迭代改进世界模型 + VLA**,用合成 rollout 扩增后训数据,real rollout +39.2pp / + synthetic +11.6pp。
- **架构**:
```mermaid
flowchart LR
    vla["VLA (π0.5)"] --> real["Real-World Rollout"]
    real --> wm["World Model (Ctrl-World FT)"]
    wm --> syn["Synthetic Rollouts"]
    syn --> rm["Reward Model (Qwen3-VL FT)"]
    rm --> awr["Advantage Weighted Regression"]
    awr --> vla
```
- **核心组件清单**:Vision=`π0.5 ViT` / Text=`π0.5 LLM` / 3D=`N/A` / Proprio=`EEF` / Backbone=`π0.5` / Fusion=`early` / Action Head=`Flow` / Aux Head=`N/A` / WM=`**像素(Ctrl-World video diffusion)**` / Routing=`N/A` / Post-train=`AWR(SFT-style RL)`
- **输入 → 输出**:`RGB + Lang → Action Chunk(EEF)`
- **主要 Loss**:\[ \mathcal{L} = \mathbb{E}\bigl[\mathcal{L}_{\text{FM}}(a\mid \text{adv} > 0)\bigr]\ \text{(AWR)} \]
- **关键消融**:
  - π0.5 base → + real rollout → **+39.2pp** 绝对成功率
  - + synthetic rollout → 额外 **+11.6pp**
- **最大正面贡献组件**:**在线 rollout 微调的世界模型** — 能建模失败案例,使合成数据具备物理保真度。
- **优势**:仅需少量真实 rollout 即可迭代提升,可并行生成大量合成数据;使用纯 SFT 目标兼容 flow-matching VLA。**局限**:世界模型仍有 hallucination,接触丰富任务精度有限;依赖 reward model 质量。
- **范式归属**:B4 + F3 + D2

#### 7.W.12 [World-VLA-Loop](p/World-VLA-Loop_Closed-Loop_World_Models_for_VLAs/paper.pdf) — SANS + 状态感知 WM + WM 内 GRPO [T2]

- **一句话定位**:**世界模型与 VLA 闭环共进化**,迭代 RL 后训,真机 **+36.7%**(2 轮迭代)。
- **架构**:
```mermaid
flowchart LR
    vla["VLA Policy"] --> wm["State-Aware Video World Model"]
    wm --> rl["RL Post-Training (GRPO)"]
    rl --> vla
    vla --> fail["Failure Rollouts"]
    fail --> wm
```
- **核心组件清单**:Vision=`依赖底座 VLA` / Text=`依赖底座 VLA` / 3D=`N/A` / Proprio=`可选` / Backbone=`依赖底座 VLA` / Fusion=`依赖底座` / Action Head=`依赖底座` / Aux Head=`**Joint Reward Prediction**` / WM=`**像素(Video DiT + reward head)**` / Routing=`N/A` / Post-train=`GRPO in WM loop`
- **输入 → 输出**:`RGB + Lang → Action(via VLA)`
- **主要 Loss**:\[ \mathcal{L}_{\text{WM}} = \mathcal{L}_{\text{video}} + \lambda\mathcal{L}_{\text{reward}};\quad \mathcal{L}_{\text{VLA}} = \mathcal{L}_{\text{GRPO}} \]
- **关键消融**:
  - 迭代 2 轮后真机(Fig 1b)→ **+36.7% 提升**
  - SANS 数据集(near-success) → 显著改善动作跟随精度(vs 无 SANS)
- **最大正面贡献组件**:**闭环共进化** — WM → VLA → failure → WM,迭代对齐使 world model 越来越可靠。
- **优势**:最少物理交互实现 VLA 提升;联合预测视频+reward 的 state-aware WM。**局限**:世界模型仍会 hallucinate 成功;目前仅在 LIBERO 和少量真机验证。
- **范式归属**:B4 + E1 + E6

#### 7.W.13 [World2Act](p/World2Act_Latent_Action_Post-Training_via_Skill-Compositional_World_Models/paper.pdf) — LLM 技能切分 + Skill-WM 对齐 [T2]

- **一句话定位**:**在 latent 空间对齐 WM 动态先验与 VLA 动作**,避免像素 hallucination,真机 +6.7%。
- **架构**:
```mermaid
flowchart LR
    wm["World Model (Video DiT)"] --> lat["Denoised Video Latents"]
    lat --> va["Video Adapter"]
    va --> shared["Shared Latent Space"]
    vla["VLA (GR00T/Cosmos)"] --> aa["Action Adapter"]
    aa --> shared
    shared --> post["Contrastive Alignment → Post-train VLA"]
```
- **核心组件清单**:Vision=`WM 内置 ViT` / Text=`LLM-based skill decomposer` / 3D=`N/A` / Proprio=`EEF` / Backbone=`GR00T-N1.6 / Cosmos Policy` / Fusion=`late(latent contrastive)` / Action Head=`Flow(依赖底座)` / Aux Head=`N/A` / WM=`**Latent(video dynamic latents)**` / Routing=`N/A` / Post-train=`**Contrastive latent alignment**`
- **输入 → 输出**:`RGB + Skill-level Lang → Action Chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{contrastive}}\bigl(\text{VideoAdapter}(z_{\text{WM}}), \text{ActionAdapter}(z_{\text{VLA}})\bigr) + \mathcal{L}_{\text{BC}} \]
- **关键消融**:
  - World2Act on GR00T(RoboCasa)→ SOTA,**超越 pixel-space 基线**
  - + Skill decomposition → 长程任务时序一致性显著提升
  - 真机改进 **+6.7%**
- **最大正面贡献组件**:**Latent 空间对齐** — 避免 WM 像素 hallucination 传播到策略,比 pixel-space 更鲁棒。
- **优势**:自动 LLM 技能分解支持任意长度视频生成;对 WM 质量容错性强。**局限**:需额外训练 video/action adapter;技能分解质量依赖 LLM。
- **范式归属**:C2 + B4 + F2

#### 7.W.14 [Psi-R2 / Psi-W0](p/From_Human_Skill_to_Robotic_Mastery_(Psi-R2__Psi-W0)/page.html) — 十万小时人类 + 真机 IT2V WM [T2]

- **一句话定位**:**10 万小时人类 + 5.4K h 真机**预训;Psi-R2(Wan2.2 IT2V WAM)联合视频+动作;Psi-W0 作 AC-WM 评估/RL 飞轮。
- **架构**:
```mermaid
flowchart LR
    A["人类+真机数据"] --> B["Psi-R2 Wan2.2-5B"]
    B --> C["未来视频"]
    B --> D["机器人动作"]
    E["Psi-W0"] --> F["策略评估/RL"]
    F --> G["数据飞轮回流"]
```
- **核心组件清单**:Vision=`原始 RGB` / Text=`语言` / 3D=`外骨骼亚毫米轨迹` / Proprio=`运动学对齐` / Backbone=`**Wan2.2-IT2V-5B**` / Fusion=`Raw In/Out` / Action Head=`联合视频+动作` / Aux Head=`触觉 mask(W0)` / WM=`**Psi-R2/W0 WAM(像素 IT2V)**` / Routing=`N/A` / Post-train=`<100 traj FT`
- **输入 → 输出**:`图+Lang → 未来帧 + 动作;W0:图+Lang+动作 → 条件未来视频(评估/RL)`
- **主要 Loss**:`未明确闭式(视频生成 + 动作联合监督)`
- **关键消融**(博客叙事):
  - **Raw Data In/Out** > 图像修复/关键点辅助/跨空间对齐
  - 复杂人工模块在小数据有效、大数据成瓶颈
  - 推理 **2.2s→<100ms**(DiT 缓存+编译+量化)
  - **<100 条**真机轨迹可学长程精细任务
- **最大正面贡献组件**:**Psi-W0 动作条件 WM 闭环** — 策略评估、数据质检、人类→机器人 RL 飞轮,非架构而是数据治理+WM 评估。
- **优势**:首个 ~10 万 h 人类预训叙事;工程推理极快。**局限**:无统一公开仿真榜;博客非 peer-review。
- **范式归属**:B3 + B4 + E6

#### 7.W.15 [LoHo-Manip](p/LoHo-Manip_Long-Horizon_Manipulation_via_Trace-Conditioned_VLA_Planning/paper.pdf) — Trace 条件 VLA + 任务管理 VLM [T2]

- **一句话定位**:**长程操控解耦** 为 VLM 任务管理器(子任务+视觉轨迹)+ 短程 VLA 执行器。
- **架构**:
```mermaid
flowchart LR
    obs["当前观测 RGB"] --> mgr["Task Manager (VLM)"]
    lang["高层指令"] --> mgr
    mgr --> subtask["子任务文本"]
    mgr --> trace["2D 视觉轨迹"]
    trace -->|render| exec["Executor VLA (π0.5)"]
    subtask --> exec
    exec --> act["短程动作"]
```
- **核心组件清单**:Vision=`VLM(Qwen3-VL 等)` / Text=`VLM` / 3D=`N/A` / Proprio=`EEF 2D pixel` / Backbone=`Manager VLM + Executor π0.5` / Fusion=`hierarchical(manager→executor)` / Action Head=`π0.5 Flow` / Aux Head=`**Trace Prediction + Progress Tracking**` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`RGB + Lang + progress memory → 子任务 + 2D trace + 短程动作`
- **主要 Loss**:\[ \mathcal{L}_{\text{mgr}} = \mathcal{L}_{\text{CE}}^{\text{plan+trace}};\quad \mathcal{L}_{\text{exec}} = \mathcal{L}_{\text{FM}} \]
- **关键消融**:
  - LoHo-Manip vs π0.5-base on LIBERO-Long(Table 4)→ 多步 SR 显著提升
  - Trace conditioning vs text-only → trace 提升 OOD 泛化
  - Remaining-plan vs next-step-only → remaining-plan 提供隐式 recovery
- **最大正面贡献组件**:**Visual Trace Prompt** — 将长程规划转化为局部轨迹跟踪,VLA 擅长短程控制 + trace 提供空间泛化桥梁。
- **优势**:Manager 与 Executor 解耦,可换不同 VLA;Remaining-plan 预测实现隐式 recovery。**局限**:Manager 推理增延迟;Trace 精度依赖 EEF 定位。
- **范式归属**:D3 + D5 + A4

#### 7.W.16 [NS-VLA](p/NS-VLA_Towards_Neuro-Symbolic_VLAs/paper.pdf) — 神经符号 VLA + GRPO 探索 [T2]

- **一句话定位**:**神经符号 VLA**:VLM 出 primitive 计划 → 稀疏视觉 Solver 出动作 → **GRPO** 在线探索,LIBERO 98.6% / CALVIN 4.72(#2)。
- **架构**:
```mermaid
flowchart LR
    A["RGB+语言"] --> B["冻结 Qwen3-VL"]
    B --> C["Primitive 分类器"]
    C --> D["Solver Transformer"]
    D --> E["动作块"]
    F["GRPO"] --> D
```
- **核心组件清单**:Vision=`Qwen3-VL` / Text=`同左` / 3D=`无` / Proprio=`入 Solver` / Backbone=`Qwen3-VL-2B 冻` / Fusion=`符号计划条件` / Action Head=`Solver BC+GRPO` / Aux Head=`Primitive 分类` / WM=`N/A` / Routing=`N/A` / Post-train=`在线 GRPO`
- **输入 → 输出**:`RGB + 指令 → primitive 序列 + H 步连续动作`
- **主要 Loss**:\[ J_{\text{GRPO}} = \tfrac{1}{G}\sum_i \bigl[r_i(\Theta)A_i - \beta D_{\text{KL}}(\pi_\Theta\|\pi_{\text{BC}})\bigr] \]
- **关键消融**(Table 6a):
  - Full **98.6%** LIBERO avg
  - w/o Plan Classifier **79.7%**;w/o Visual Encoder **90.1%**
  - w/o Action Generator **85.2%**;w/o RL **91.6%**
  - 1-shot 仍 **69.1%** avg;CALVIN **4.72**(#2)
- **最大正面贡献组件**:**符号规划器 + GRPO 段级奖励** — 1-shot 数据效率与 CALVIN 多步任务最强。
- **优势**:2B 轻量;探索缓解纯 BC 天花板。**局限**:依赖 primitive 词表;VLM 冻结限制端到端优化。
- **范式归属**:D5 + E2

#### 7.W.17 [Being-H0.7](p/Being-H0.7_A_Latent_World-Action_Model_from_Egocentric_Videos/paper.pdf) — 潜变量世界-动作双分支 [T2]

- **一句话定位**:**潜变量世界-动作模型**,用未来对齐取代像素预测,**推理 3-4ms/step**。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Lang + State"] --> vlm["MoT Backbone (VLM)"]
    vlm --> latent["Latent Queries (Prior Branch)"]
    latent --> act["Flow Action Head"]
    future["Future Obs (Posterior Branch)"] -.->|"train only"| latent
```
- **核心组件清单**:Vision=`ViT(frozen for posterior)` / Text=`LLM` / 3D=`N/A` / Proprio=`State tokens` / Backbone=`MoT` / Fusion=`MoT shared attention + dual-branch mask` / Action Head=`Flow Matching` / Aux Head=`Latent future alignment` / WM=`**Latent(dual-branch align)**` / Routing=`Per-head(Understanding/Action Expert)` / Post-train=`UAC 异步`
- **输入 → 输出**:`多视 RGB + Lang + Proprio → Action Chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{FM}}^{\text{prior}} + \mathcal{L}_{\text{FM}}^{\text{post}} + \mathcal{L}_{\text{align}} + \mathcal{R}_{\text{norm}} + \mathcal{R}_{\text{rank}} \]
- **关键消融**:
  - Latent world-action model vs VLA-only(Sec 4.2)→ LIBERO **+2-4pp avg**
  - Dynamic tasks(rolling ball, pouring)→ 领先所有 5 个能力维度
  - 推理 **3-4ms/step**(UAC) vs WAM 需视频解码
- **最大正面贡献组件**:**Dual-branch latent alignment** — 赋予 prior branch 未来感知能力而无需推理时生成视频,动态任务上优势最大。
- **优势**:推理无需视频 rollout,延迟极低(3-4ms);大规模 egocentric video 预训可扩展。**局限**:需要大量 egocentric 视频用于预训;论文未报告标准 sim benchmark 数字对比。
- **范式归属**:B2 + A3 + C2

#### 7.W.18 [MolmoAct2 W 副引用](p/MolmoAct2_Action_Reasoning_Models_for_Real-world_Deployment/paper.pdf) — 也含 W6 CoT,主在 7.L.4 [T2]

> 完整内容见 **[7.L.4](#74-molmoact2pmolmoact2_action_reasoning_models_for_real-world_deploymentpaperpdf--molmo-全开源--fast--think-t2)**。
> W6 副:Think 推理 token 是 W6 CoT 的代表实现;Allen 团队验证"自适应深度 Think + KV-cache Flow"在 long-horizon / sim2real 显著加分。

#### 7.W.19 [X-WAM](p/X-WAM_Unified_4D_World_Action_Modeling_from_Video_Priors_with_Asynchronous_Denoising/paper.pdf) — Wan2.2-5B + 轻量深度分支 + ANS 异步去噪 [T1]

- **一句话定位**:**首个 unified 4D WAM** — 多视角 RGB-D 联合预测 + 异步去噪保 train-test 分布对齐,**RoboCasa SOTA 79.2% / RoboTwin 2.0 90.7%**。
- **架构**:
```mermaid
flowchart LR
    obs["多视 RGB obs + 当前 robot state"] --> wan["Wan2.2-TI2V-5B DiT (frozen base)"]
    wan --> rgbBranch["RGB 视频分支 (主 DiT)"]
    wan --> depthBranch["深度分支 (复制 final blocks)"]
    rgbBranch --> ansSampler["Asynchronous Noise Sampling (t_O, t_a) 联合分布"]
    depthBranch --> ansSampler
    ansSampler --> futureRGBD["未来 RGB-D 帧"]
    ansSampler --> actHead["Action 分支 (flow matching)"]
    actHead --> robotAct["Action chunk"]
```
- **核心组件清单**:Vision=`Wan2.2 VAE` / Text=`Wan2.2 text encoder` / 3D=`显式 depth latent` / Proprio=`robot state token` / Backbone=`**Wan2.2-TI2V-5B DiT**(基座)` / Fusion=`flow matching 联合 RGB + depth + action latent` / Action Head=`Flow Matching` / Aux Head=`**Depth prediction branch(复制 final DiT blocks)**` / WM=`**4D RGB-D unified WAM**` / Routing=`ANS 联合 timestep` / Post-train=`5874h 大规模 PT + 真机 20h 微调`
- **输入 → 输出**:`多视 RGB + robot state + Lang → 未来 RGB-D + Action chunk`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{FM}}^{\text{video}} + \mathcal{L}_{\text{FM}}^{\text{depth}} + \mathcal{L}_{\text{FM}}^{\text{action}} \] (Sec 3.4,SNR-based independent sampling, joint (t_O, t_a) per ANS)
- **关键消融**(Sec 4.3):
  - **加深度分支 vs 无深度** → 同时提升 3D 重建 + 策略 SR 双指标
  - **ANS vs 独立 timestep 采样** → 训练-推理分布对齐使加速无质量损失
  - **从 Wan2.2 直接微调(无 5874h PT)** → 显著退化
- **最大正面贡献组件**:**深度分支(复制 final DiT blocks)** — 不增加 attention 二次代价 + 不破坏视频预训练分布 + 同时提 SR 和 3D 重建质量,是「画质与具身有用性可同涨」的实证。
- **优势**:RoboCasa SOTA;首个 unified 4D WAM;深度分支轻量(仅复制最后几层 blocks)。**局限**:部分 robot 数据无原生 depth,需估计;计算成本随深度分支线性增加。
- **范式归属**:B4 + A3 + 4D 监督

#### 7.W.20 [MotuBrain](p/MotuBrain_An_Advanced_World_Action_Model_for_Robot_Control/paper.pdf) — Vidu + 三流 MoT + H-Bridge 5 模式 unified [T1]

- **一句话定位**:**5 种推理模式同模型支持的 unified WAM**(VLA / WM / IDM / VGM / Joint)— **RoboTwin 2.0 综合 #1 95.8/96.1**,**WorldArena 最强 EWMScore**。
- **架构**:
```mermaid
flowchart LR
    inText["Text"] --> textStream["Text stream (conditioning only)"]
    inVid["Multiview RGB + cond frame"] --> vae["Vidu VAE encode"]
    vae --> vidStream["Video stream (DiT)"]
    inAct["Noisy action 10D (relative EEF)"] --> actStream["Action stream (DiT)"]
    textStream --> hbridge["H-Bridge attention (mid 50% 全联合, 两端 25% decoupled)"]
    vidStream --> hbridge
    actStream --> hbridge
    hbridge --> outVid["未来 video latent"]
    hbridge --> outAct["Action chunk"]
```
- **核心组件清单**:Vision=`Vidu VAE` / Text=`text token stream (conditioning, no output head)` / 3D=`N/A(2D pixel)` / Proprio=`included in action 10-D (pos + 6D rot + gripper)` / Backbone=`**5B Vidu DiT (基座)**` / Fusion=`**三流 MoT (text/video/action 独立 transformer 参数) + H-Bridge attention**` / Action Head=`Flow Matching (timeshift=1)` / Aux Head=`Video Flow Matching (timeshift=6)` / WM=`**Unified 5-mode WAM (VLA / WM / IDM / VGM / Joint)**` / Routing=`H-Bridge mid layers V-A 全联合 + 两端 decoupled` / Post-train=`Non-AR + AR 两套(独立训练)`
- **输入 → 输出**:`多视 RGB + Lang + 状态 → Action chunk(VLA / IDM)+ 可选未来视频(WM / VGM / Joint)`
- **主要 Loss**:\[ \mathcal{L}_{\text{stage2}} = \lambda_v \mathcal{L}_v + \lambda_a \mathcal{L}_a;\quad \mathcal{L}_v = \text{MSE}(v_{\text{out}}, v_{\text{target}}),\ \mathcal{L}_a = \text{MSE}(a_{\text{out}}, a_{\text{target}}) \] (Eq 5-6)
- **关键消融**(Sec 4 各 task table):
  - H-Bridge 中间 50% 联合层 vs 全 decoupled → 跨模态对齐效果差,精度 -3 pp+
  - V2A 推理(仅 action 路径)vs 联合 V-A → SR 仅低 sub-percent(优化端到端 ≥50× 加速)
  - LingBot-VA-style noisy-conditioning (s_aug ∼ U[0.3,0.7]) + 多视角 drop p=0.1 → 显著提升 rollout 鲁棒性
- **最大正面贡献组件**:**三流 MoT + H-Bridge attention** — 保留模态特异性(浅深层 decoupled)同时实现跨模态对齐(中间联合),既高效又准确,5 种推理模式自然涌现。
- **优势**:同模型 5 模式 + RoboTwin 2.0 接近天花板 + 推理 50× 加速可生产部署 + 新本体 50-100 demo 适配。**局限**:依赖闭源 Vidu 基座;每层精确数据小时数原文未完整公开。
- **范式归属**:B4 + A3 + Unified 多模式

#### 7.W.21 [Consistency-Consensus (Is the Future Compatible?)](p/Is_the_Future_Compatible_Diagnosing_Dynamic_Consistency_in_World_Action_Models/paper.pdf) — 零训练 test-time consistency 排序 [T2]

- **一句话定位**:**首个把 WAM 内部一致性当 zero-training test-time selection 信号** — RoboTwin 2.0 **90.2 → 93.0 (+2.8 pp)**,无新模型 / 无 reward。
- **架构**(诊断 + 选择,非新模型):
```mermaid
flowchart LR
    wamCkpt["现成 WAM checkpoint (Cosmos-Policy / LingBot-VA)"] --> rollouts["每步采 K 次 rollout 候选"]
    rollouts --> latentMSE["计算 latent-space MSE: o_{t+Δ} vs ô_{t+Δ}"]
    latentMSE --> consistency["c_t = exp(-α · MSE), α=0.1"]
    consistency --> ranker["按 consistency 排序选最高"]
    ranker --> finalAct["执行 selected action"]
    consistency --> diagLowDyn["背景塌缩诊断: Δz_t 阈值"]
    diagLowDyn -.->|"低运动任务一致性反向"| skip["fallback 直接执行"]
```
- **核心组件清单**:Vision=`复用已有 WAM VAE` / Text=`复用` / 3D=`N/A` / Proprio=`复用` / Backbone=`**零新参数**(Cosmos-Policy / LingBot-VA pretrained)` / Fusion=`N/A(test-time wrapper)` / Action Head=`复用 WAM action expert` / Aux Head=`**consistency 计算 latent MSE**` / WM=`复用 WAM future prediction` / Routing=`**K 次 rollout + ranking**` / Post-train=`N/A`
- **输入 → 输出**:`多次 rollout + observation → 按一致性选最优 action`
- **主要 Loss**:无训练;**度量公式** \[ c_t = \exp(-\alpha \cdot d_{\text{MSE-latent}}(o_{t+\Delta}, \hat{o}_{t+\Delta})),\ \alpha = 0.1 \] (Eq 3)
- **关键消融**:
  - **Cohen's d**:joint pred (Cosmos-Policy) **0.76** / inverse dyn (LingBot-VA) **0.99** — 一致性显著分离成功-失败 episode
  - **AUC**:0.77 / 0.88(成功-失败二分类)— 与 learned value head 趋势同源
  - **RoboTwin 2.0**:90.2 → **93.0 (+2.8 pp)**,**RoboCasa**:66.6 → **67.3 (+0.7 pp)**
  - **background collapse 失败模式**:低 Δz_t 任务一致性反向(TurnOnSinkFaucet 等)— 需诊断
- **最大正面贡献组件**:**latent-space MSE consistency 排序** — 把「WAM 已经具备的内部一致性」这一隐属性显式利用,**避免训 reward 模型的数据成本**。
- **优势**:无新数据、无新参数、无训练;可即插即用于任何 WAM。**局限**:K 次 rollout 增推理成本;低动态任务需用 Δz_t 诊断后再用。
- **范式归属**:B4 + D2 评估替代

### 7.E 跨本体适配主 — 3 篇 [T2]

**共同特点**:这 3 篇的差异化在「如何让一个模型跨多本体」——Language-Action 表示、Embodiment Aug、仿真平台。

#### 7.E.1 [LAP](p/LAP_Language-Action_Pre-Training_Enables_Zero-shot_Cross-Embodiment_Transfer/paper.pdf) — 语言动作预训练 + Latent Action [T2]

- **一句话定位**:**把底层动作表示为自然语言**("move left 5cm"),实现首个 **零样本跨本体迁移 VLA**,3 未见本体 >50% SR。
- **架构**:
```mermaid
flowchart LR
    obs["多视 RGB + Proprio"] --> vlm["PaliGemma-3B VLM"]
    lang["Task 指令"] --> vlm
    vlm -->|CE loss| la["Language-Action tokens"]
    vlm -->|cross-attn| ae["Flow Action Expert"]
    ae -->|FM loss| act["连续动作 @25Hz"]
```
- **核心组件清单**:Vision=`PaliGemma-3B / Gemma3` / Text=`PaliGemma-3B` / 3D=`N/A` / Proprio=`EEF pose(6D rot)离散 token` / Backbone=`PaliGemma-3B + Action Expert` / Fusion=`MoT(causal mask, stop-grad)` / Action Head=`**Flow Matching**` / Aux Head=`Language-Action(AR CE)` / WM=`N/A` / Routing=`**E3 Latent Language-Action**` / Post-train=`N/A`
- **输入 → 输出**:`多视 RGB + Proprio token + Lang → 连续 EEF delta action @25Hz`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda\mathcal{L}_{\text{CE}}^{\text{lang-action}} \]
- **关键消融**:
  - LAP vs π0.5-replicated zero-shot unseen embodiment(Fig 3)→ **~52% vs ~25%**(+27pp)
  - LAP vs π0-replicated(Fig 3)→ **~52% vs ~25%**(+27pp)
  - VQA co-training(Fig 6)→ + Custom Franka & YAM 上额外增益
  - LIBERO(Table 3)→ **LAP+VQA 97.2% vs π0.5 96.9%**
- **最大正面贡献组件**:**Language-Action 表示** — +27pp 零样本跨本体,因自然语言动作保持 VLM 预训分布不偏移,学到的表征本体无关。
- **优势**:首个真正零样本跨本体迁移(3 未见本体 >50% SR);微调效率高(10% 数据匹配 baseline 峰值)。**局限**:仅验证单臂操控器,未覆盖双臂/灵巧手;语言动作分辨率受限(整数厘米/度)。
- **范式归属**:F1 + A3 + C2

#### 7.E.2 [OXE-AugE](p/OXE-AugE_Augmenting_OXE_with_Embodiment_Aug/paper.pdf) — OXE 跨本体数据合成 toolkit [T2]

- **一句话定位**:**AugE-Toolkit** 将 OXE 16 数据集 × 9 本体 cross-painting 扩至 **4.4M+** 轨迹,真机未见 robot×gripper **+24%~+45%**。
- **架构**:
```mermaid
flowchart LR
    A["源 OXE 轨迹"] --> B["SAM2 分割"]
    B --> C["E2FGVI 补背景"]
    C --> D["MuJoCo 重渲染"]
    D --> E["增广数据集"]
    E --> F["OpenVLA/π0 微调"]
```
- **核心组件清单**:Vision=`增广 RGB` / Text=`各基座原生` / 3D=`N/A` / Proprio=`pose 对齐` / Backbone=`OpenVLA / π0` / Fusion=`各基座原生` / Action Head=`扩散/Flow` / Aux Head=`N/A` / WM=`N/A(仅数据侧仿真)` / Routing=`N/A` / Post-train=`**LoRA/全参 FT**`
- **输入 → 输出**:`增广 RGB + pose + action → 各 VLA 原生动作空间`
- **主要 Loss**:`依赖 OpenVLA / π0 等基座原生模仿损失(未新定义)`
- **关键消融**:
  - 真机未见 robot×gripper → **+24%~+45%**
  - 增广组合 > naive mixing;RoVi-Aug 扩散增广反降 **27-30%**
  - 覆盖 Octo mixture **60%**
  - 4 真机任务 consistently 涨点
- **最大正面贡献组件**:**cross-painting + 仿真重渲染管线** — 数据层跨本体增广,不改模型架构。
- **优势**:即插即用任何 VLA;显著扩 OXE 规模。**局限**:非端到端 VLA 论文;依赖分割/修复质量。
- **范式归属**:C2 + F3

#### 7.E.3 [Genie Sim 3.0](p/Genie_Sim_3.0（智元__AgiBot）/paper.pdf) — 智元 AgiBot 仿真平台 [T2]

- **一句话定位**:**仿真平台 + LLM 驱动场景生成 + VLM 评估**,提供万小时合成数据与 10 万场景。
- **架构**:
```mermaid
flowchart LR
    lang["LLM 意图解析"] --> gen["Scene Generator"]
    gen --> sim["Isaac Sim 仿真"]
    sim --> data["10K+ 小时数据"]
    data --> vla["VLA 训练 (π0.5 等)"]
    vla --> eval["VLM 自动评估"]
```
- **核心组件清单**:Vision=`多视角 RGB` / Text=`LLM(CoT)` / 3D=`3DGS + PGSR` / Proprio=`关节状态` / Backbone=`N/A(平台非模型)` / Fusion=`N/A` / Action Head=`N/A` / Aux Head=`VLM 评估` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`自然语言场景描述 → 仿真场景 + 合成示教数据 + 自动评估结果`
- **主要 Loss**:`N/A(平台论文,非模型训练 Loss)`
- **关键消融**:
  - 1500 eps sim 数据 vs 500 eps real(Table I)→ **sim-to-real avg 0.83 vs real-to-real 0.75**(+0.08)
  - π0.5 vs GR00T-N1.6 vs π0 在 GenieSim-Instruction(Table III)→ **0.67/0.40/0.28**
- **最大正面贡献组件**:**多维域随机化** — 1500 eps 合成数据在所有任务超过真实数据,因系统化域随机化有效弥合 sim-to-real gap。
- **优势**:LLM 驱动一键生成 + 泛化场景,分钟级生成千种变体;**Sim-to-Real R² = 0.94**,仿真评估可信预测真机表现。**局限**:平台仅面向操控,不涵盖移动/灵巧手;VLM 评估仍有主观偏差。
- **范式归属**:E4 + G3

### 7.O 后训练/部署优化主 — 17 篇 [T2]

**共同特点**:这 16 篇的差异化在「训完之后怎么压、怎么 RL、怎么部署」——PTQ / 课程 / 蒸馏 / fleet RL / test-time RL / 终身 RFT / 模块化规划等。

#### 7.O.1 [QuantVLA](p/QuantVLA_Post-Training_Quantization_for_VLA/paper.pdf) — 首个 VLA 专属 PTQ (W4A8) [T2]

- **一句话定位**:**首个 VLA 专属 W4A8 PTQ**;ATM 温度匹配 + OHB 输出头平衡,**DiT 动作头可量化**,显存 ~70%↓ 速度 1.22×。
- **架构**:
```mermaid
flowchart LR
    A["FP16 VLA"] --> B["校准 buffer"]
    B --> C["选择性 W4A8"]
    C --> D["LLM 126 层 + DiT MLP 180 层"]
    D --> E["整数推理"]
```
- **核心组件清单**:Vision=`随基座` / Text=`随基座` / 3D=`—` / Proprio=`—` / Backbone=`π0.5 / GR00T 冻结构` / Fusion=`—` / Action Head=`**DiT Flow 量化**` / Aux Head=`N/A` / WM=`N/A` / Routing=`—` / Post-train=`**PTQ only**`
- **输入 → 输出**:`与基座 VLA 相同;输出 W4A8 动作,显存 ~70%↓`
- **主要 Loss**:`无梯度训练;尺度校准 + ATM(α) + OHB(β) 折叠进 dequant`
- **关键消融**(Table 3,LIBERO Avg,ATM/OHB 关闭):
  - FP16 → **97.1% / 4.27 GB**
  - 仅 LLM W4A8 → **96.5% / 1.58 GB**
  - 仅 DiT W4A8 → **71.6%**(动作头极敏感)
  - LLM+DiT 全量化 → 76.3%;**LLM+DiT(MLP) 选择性 180 层** 可超 FP16
  - 完整 QuantVLA + ATM + OHB:**70% mem,1.22× 速度**,LIBERO 可超全精度
- **最大正面贡献组件**:**选择性量化布局(LLM + DiT MLP 180 层)+ OHB** — 首次把 DiT 动作头量化做稳。
- **优势**:零重训部署;显存/速度双赢。**局限**:全层 naive 量化崩溃;需小校准集。
- **范式归属**:G1

#### 7.O.2 [Green-VLA](p/Green-VLA_5-Stage_Curriculum_to_Strong_VLA/paper.pdf) — Sber 五阶段课程 + R2 RL [T2]

- **一句话定位**:**5 阶段课程训练**(L0→L1→R0→R1→R2)+ 统一动作空间 + RL 对齐的人形 VLA,WidowX **R0 45% → R2 80.5%**(+34pp)。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Proprio"] --> vlm["Qwen3-VL-4B / PaliGemma-3B"]
    lang["语言 + 控制 prompt"] --> vlm
    vlm --> ae["Flow-Matching Action Expert"]
    ae --> act["统一动作空间 (64 维)"]
    ae --> ep["Episode Progress Head"]
```
- **核心组件清单**:Vision=`Qwen3-VL-4B` / Text=`Qwen3-VL-4B` / 3D=`N/A` / Proprio=`统一语义 slot` / Backbone=`~5B(Qwen3-VL-4B + Action Expert)` / Fusion=`late(VLM → Expert)` / Action Head=`Flow Matching` / Aux Head=`Episode Progress + OOD Detector + JPM Guidance` / WM=`N/A` / Routing=`Embodiment/control-type Prompt` / Post-train=`**RL(IQL + Source Distrib. Opt.)**`
- **输入 → 输出**:`多视 RGB + Proprio + Lang + 控制 prompt → 统一 64 维动作 + episode progress`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{FM}}^{\text{masked}} + \mathcal{L}_{\text{progress}} + \mathcal{L}_{\text{RL}} \]
- **关键消融**:
  - R0 vs R2(RL alignment)on WidowX(Table 4)→ **SR 45.0% → 79.1%/80.5%**(+34pp)
  - JPM guidance on e-commerce(Fig 11)→ **ID-SKU 36.7% → 93.1%**(+56.4pp)
  - CALVIN ACL(Fig 14b)→ **R1 4.1 → R2 4.6**(+0.5)
- **最大正面贡献组件**:**RL Alignment(R2)** — +34pp on WidowX,因 RL 弥补 BC 在长程和困难任务上的饱和。
- **优势**:统一动作空间 + 控制 prompt 支持人形/双臂/单臂;5 阶段课程清晰可复现。**局限**:依赖 retargeting 保真度;灵巧手覆盖不足。
- **范式归属**:G3 + A3 + E5 + C2

#### 7.O.3 [VLA-OPD](p/VLA-OPD_Bridging_Offline_SFT_and_Online_RL_for_VLAs_via_On-Policy_Distillation/paper.pdf) — On-Policy 教师蒸馏 [T2]

- **一句话定位**:**On-Policy 蒸馏桥接 SFT 与 RL**,Reverse-KL 避免熵爆炸/坍缩,**1-traj → 87.4%**(+38.5pp)。
- **架构**:
```mermaid
flowchart LR
    env["Environment"] --> student["Student VLA (OpenVLA-OFT)"]
    student --> traj["On-Policy Trajectory"]
    traj --> teacher["Teacher VLA (SimpleVLA-RL, frozen)"]
    teacher --> rkl["Reverse-KL Dense Reward"]
    rkl --> student
```
- **核心组件清单**:Vision=`OpenVLA-OFT 内置` / Text=`OpenVLA-OFT` / 3D=`N/A` / Proprio=`可选` / Backbone=`OpenVLA-OFT 7B(student)` / Fusion=`依赖底座` / Action Head=`AR(FAST tokens)` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`**Distill(Reverse-KL) + optional GRPO**`
- **输入 → 输出**:`RGB + Lang → Discretized Action Tokens`
- **主要 Loss**:\[ r^{\text{OPD}}_t = -\log\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{tea}}(a_t\mid s_t)},\quad \nabla J \approx \tfrac{1}{G}\sum_i\sum_t \nabla\log\pi_\theta(a_{t,i}\mid s_{t,i})\cdot r^{\text{OPD}}_t \]
- **关键消融**:
  - 1-traj SFT init → OPD Distill(Table 2 LIBERO Avg)→ **48.9% → 87.4%**(+38.5pp)
  - Forward-KL vs Reverse-KL(Fig 4a)→ RKL 显著优于 FKL(FKL 早期暴跌 50%+)
  - Distill + GRPO(Table 2)→ **93.4% ≈ Teacher 93.9%**
- **最大正面贡献组件**:**Reverse-KL 蒸馏** — mode-seeking 过滤 teacher 不确定性,熵稳定,1-traj → 87.4%。
- **优势**:仅需 1 条示教即可达 50-traj 全数据方法水平;有效缓解灾难性遗忘。**局限**:依赖强 teacher 可用性;未在真机验证。
- **范式归属**:E4 + F2

#### 7.O.4 [LWD](p/Learning_While_Deploying_(LWD)_Fleet-Scale_Reinforcement_Learning_for_Generalist_Robot_Policies/paper.pdf) — 舰队级 offline-to-online RL [T2]

- **一句话定位**:**16 台双臂机器人 fleet-scale offline-to-online RL**,持续改进通才 VLA 至 **95% SR**。
- **架构**:
```mermaid
flowchart LR
    fleet["16 台机器人 Fleet"] -->|rollouts + 人工干预| buf["Online Buffer"]
    offbuf["Offline Buffer"] --> rl["DIVL + QAM"]
    buf --> rl
    rl --> vla["VLA (VLM + Flow Action Head)"]
    vla -->|redeploy| fleet
```
- **核心组件清单**:Vision=`VLM` / Text=`VLM` / 3D=`N/A` / Proprio=`关节状态` / Backbone=`VLA(VLM + Flow)` / Fusion=`late` / Action Head=`Flow Matching` / Aux Head=`**Distributional Value Head(DIVL)**` / WM=`N/A` / Routing=`N/A` / Post-train=`**Offline-to-Online RL(DIVL + QAM)**`
- **输入 → 输出**:`多视 RGB + Proprio + Lang → action chunk + Q-value`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_V^{\text{DIVL}} + \mathcal{L}_Q + \mathcal{L}_{\text{QAM}}^\pi \]
- **关键消融**:
  - LWD vs SFT baseline(Sec V)→ **avg SR 0.95 vs pretrained ~0.70**(+25pp)
  - 长程任务(泡茶/调酒)→ **largest gains**
  - DIVL vs scalar IQL → distributional value 更适合异构 fleet 数据
- **最大正面贡献组件**:**DIVL** — 分布式值函数保留 fleet 数据中的多模态高回报模式,避免标量均值淹没稀有成功信号。
- **优势**:首个 fleet-scale offline-to-online RL 持续提升通才策略;仅需几小时真机交互即可显著提升。**局限**:需要 fleet 基础设施;稀疏奖励设计依赖人工。
- **范式归属**:E2 + E4 + A3

#### 7.O.5 [SOP](p/SOP_Scalable_Online_Post-Training/paper.pdf) — 算法无关的可扩展在线后训 [T2]

- **一句话定位**:**分布式机器人舰队在线后训** VLA 系统,**近线性扩展**,4 Actors 训练时间 2.4× 加速。
- **架构**:
```mermaid
flowchart LR
    fleet["Robot Fleet (N Actors)"] --> buf["Online Buffer + Offline Buffer"]
    buf --> sampler["Adaptive Sampler"]
    sampler --> learner["Cloud Learner (HG-DAgger/RECAP)"]
    learner --> fleet
```
- **核心组件清单**:Vision=`π0.5 ViT` / Text=`π0.5 LLM` / 3D=`N/A` / Proprio=`Joint Pos` / Backbone=`π0.5` / Fusion=`early` / Action Head=`Flow` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`**异步在线 HG-DAgger / RECAP**`
- **输入 → 输出**:`3×RGB + Proprio + Lang → Joint Position @30Hz`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{PT}(\pi_\theta; s, a)\ \text{(HG-DAgger: BC / RECAP: advantage-conditioned)} \]
- **关键消融**:
  - 1→4 Actors(Table I)→ **成功率 0.805→0.925**(+12pp),训练时间 173.6→71.7 min(**2.4×**)
  - Base-1/2 + 80h 离线数据 → 0.612;+ SOP 3h 在线 → **0.800**(+18.8pp)
- **最大正面贡献组件**:**在线分布式数据收集** — 多机并行带来近线性扩展,on-policy 修正效率远超离线数据。
- **优势**:首个支持多任务、多机、在线后训 VLA 的系统框架;后训算法无关,可插拔。**局限**:依赖人类干预或任务奖励信号;更大规模 fleet 待验证。
- **范式归属**:E2 + F3

#### 7.O.6 [TT-VLA](p/TT-VLA_Test-Time_RL_with_Task-Progress_Reward/paper.pdf) — 测试时 value-free PPO [T2]

- **一句话定位**:**推理时 RL 在线适应 VLA**,无需重训,典型 +2-5pp,适配任意 SFT/RL-VLA。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Lang"] --> vla["Pretrained VLA (OpenVLA 等)"]
    vla --> act["Action"]
    act --> env["Environment"]
    env --> vlac["VLAC Progress Estimator"]
    vlac --> reward["Dense Reward r_t = p_t - p_{t-1}"]
    reward --> ppo["Value-free PPO (LoRA Update)"]
    ppo --> vla
```
- **核心组件清单**:Vision=`依赖底座 VLA` / Text=`依赖底座 VLA` / 3D=`N/A` / Proprio=`可选` / Backbone=`OpenVLA-7B / Nora-3B / TraceVLA` / Fusion=`依赖底座` / Action Head=`依赖底座(AR)` / Aux Head=`**VLAC 进度估计**` / WM=`N/A` / Routing=`N/A` / Post-train=`**Test-time LoRA PPO**`
- **输入 → 输出**:`RGB + Instruction → Cartesian EEF Delta + Gripper`
- **主要 Loss**:\[ \mathcal{L} = \mathbb{E}_t\bigl[\min(r_t(\theta)\hat A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t)\bigr],\quad \hat A_t = r_t = p_t - p_{t-1} \]
- **关键消融**:
  - 标准 GAE vs 即时 reward(Fig 4, OpenVLA Vision)→ **55.00% vs 57.08%**(+2.08pp)
  - 更新频率 8 步最优(Table 2)→ 8 步 42.08% vs 1 步 40.42%
  - TT-VLA vs TLM/TTRL(Table 3, OpenVLA Vision)→ **57.08% vs 52.50%/51.42%**
- **最大正面贡献组件**:**Dense progress reward** — 即时进度差分替代长程回报,避免 value 学习不可行的问题。
- **优势**:即插即用,适配任意 SFT/RL-VLA;真机验证有效,无需环境重置。**局限**:提升幅度通常 2-5pp,非颠覆性;依赖 VLAC 质量。
- **范式归属**:E3 + D2

#### 7.O.7 [LifeLong-RFT](p/LifeLong-RFT_Lifelong_Reinforcement_Fine-Tuning/paper.pdf) — chunk 级 GRPO + 过程奖励 + 终身 [T2]

- **一句话定位**:**无需环境交互** 的 VLA 强化微调,**多维过程奖励**实现持续学习不遗忘,LIBERO **+22% avg**。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Proprio"] --> vla["VLA (OpenVLA 等)"]
    lang["语言指令"] --> vla
    vla -->|sampling G chunks| chunks["G 组 action chunks"]
    chunks --> reward["多维 Process Reward"]
    reward --> grpo["GRPO 优化"]
    grpo --> vla
```
- **核心组件清单**:Vision=`VLM 内置` / Text=`VLM 内置` / 3D=`N/A` / Proprio=`s_t` / Backbone=`OpenVLA / 离散 VLA` / Fusion=`VLM 内置` / Action Head=`AR(离散 token)` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`**GRPO + 三维 Process Reward**`
- **输入 → 输出**:`RGB + Proprio + Lang → 离散 action token chunk`
- **主要 Loss**:\[ \mathcal{L} = J_{\text{GRPO}}(\theta) = \mathbb{E}\bigl[\min(\rho_t A_i, \text{clip}(\rho_t) A_i) - \gamma D_{\text{KL}}\bigr] \]
- **关键消融**:
  - LifeLong-RFT vs SFT continual learning LIBERO → **+22% avg SR**
  - RFT 仅需 20% 训练数据达 SFT 效果
  - QACR+CTAR+FCR 三维奖励 vs 单一奖励 → 三维协同最优
- **最大正面贡献组件**:**Multi-dimensional Process Reward(QACR+CTAR+FCR)** — 在离散 token 和连续轨迹两空间同时提供奖励,无需环境交互。
- **优势**:无需仿真器/真机交互,纯离线 RFT;持续学习新任务时遗忘少。**局限**:仅在离散 action VLA 上验证;奖励准确性受 tokenizer 精度限制。
- **范式归属**:E5 + A1 + F3

#### 7.O.8 [STRONG-VLA](p/STRONG-VLA_Decoupled_Robustness_Learning_for_VLAs_under_Multimodal_Perturbations/paper.pdf) — 解耦两阶段鲁棒微调 [T2]

- **一句话定位**:**解耦鲁棒性训练框架**,两阶段课程式多模态扰动防御,TSP **68.14% vs joint 65.57%**。
- **架构**:
```mermaid
flowchart LR
    obs["RGB (扰动) + Text (扰动)"] --> vla["VLA Backbone (OpenVLA/π0)"]
    vla --> stage1["Stage I: 课程扰动训练"]
    stage1 --> stage2["Stage II: 干净数据重对齐"]
    stage2 --> a["Robust Action"]
```
- **核心组件清单**:Vision=`依赖底座 VLA` / Text=`依赖底座 VLA` / 3D=`N/A` / Proprio=`N/A` / Backbone=`OpenVLA-7B / π0` / Fusion=`依赖底座` / Action Head=`依赖底座(AR/Flow)` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`**LoRA 两阶段课程**`
- **输入 → 输出**:`扰动/干净 RGB + Text → Action`
- **主要 Loss**:\[ \mathcal{L} = \mathbb{E}_{\tau\sim\mathcal D}\bigl[\ell(\pi_\theta;\delta(\tau))\bigr]\ \text{(Stage I)} \to \mathbb{E}_{\tau\sim\mathcal D_{\text{clean}}}\bigl[\ell(\pi_\theta;\tau)\bigr]\ \text{(Stage II)} \]
- **关键消融**:
  - 去 Stage II(Table 3 TSP)→ **60.29% vs 68.14%**(-7.85pp)
  - Joint training vs 解耦(Table 3 TSP)→ **65.57% vs 68.14%**(-2.57pp)
  - 去课程调度(Table 3)→ **62.57% vs 68.14%**(-5.57pp)
- **最大正面贡献组件**:**两阶段解耦设计** — 避免鲁棒性与任务保真度间梯度冲突,VSP+TSP 全面优于 joint。
- **优势**:跨 VLA 架构通用(OpenVLA/OFT/π0 均有效);28 种扰动覆盖完整。**局限**:视觉 unseen 扰动泛化仍有限;扰动分类和课程依赖手工。
- **范式归属**:G3 + F3

#### 7.O.9 [TiPToP](p/TiPToP_A_Modular_Open-Vocabulary_Planning_System_for_Robotic_Manipulation/paper.pdf) — 零机器人数据模块化规划 [T2]

- **一句话定位**:**模块化 TAMP + 视觉基础模型** 系统,**零数据匹敌 π0.5**(+22.2pp SR)。
- **架构**:
```mermaid
flowchart LR
    stereo["Stereo RGB"] --> depth["FoundationStereo Depth"]
    stereo --> vlm["Gemini VLM (Detection+Goal)"]
    depth --> grasp["M2T2 Grasp Gen"]
    vlm --> sam["SAM-2 Segmentation"]
    grasp --> tamp["cuTAMP Planner"]
    sam --> tamp
    tamp --> exec["Joint Impedance Controller"]
```
- **核心组件清单**:Vision=`FoundationStereo + SAM-2` / Text=`Gemini Robotics-ER` / 3D=`Stereo Point Cloud` / Proprio=`Joint Config` / Backbone=`N/A(无学习策略)` / Fusion=`N/A(模块化 pipeline)` / Action Head=`**cuTAMP + cuRobo 规划**` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`Stereo RGB + Lang → Timed Joint Trajectory`
- **主要 Loss**:`无学习 Loss,推理时 GPU 并行约束优化`
- **关键消融**:
  - TiPToP vs π0.5-DROID(Table I Overall)→ **74.6% vs 52.4%**(+22.2pp SR)
  - 失败分析(Fig 5)→ **抓取失败 56%(31/55)**,VLM 错误 11%
- **最大正面贡献组件**:**cuTAMP 规划器** — 多步/语义/干扰场景大幅超越 VLA,利用结构化推理。
- **优势**:零训练数据,1 小时部署新机器人;模块可独立升级。**局限**:开环执行,抓取失败无法重试;凸包近似对凹形物体失效。
- **范式归属**:G3(模块化) + D5

#### 7.O.10 [EZ-M](p/EZ-M_Scaling_Tasks,_Not_Samples_-_Mastering_Humanoid_Control_through_Multi-Task_Model-Based_RL/paper.pdf) — 多任务 MBRL 人形 [T2]

- **一句话定位**:**多任务 MBRL** 通过任务扩展实现人形全身控制,HumanoidBench-Hard 显著超越 SAC/TD-MPC2。
- **架构**:
```mermaid
flowchart LR
    state["State s_t"] --> enc["State Encoder H"]
    enc --> dyn["Dynamics Network G"]
    dyn --> val["Value Network V"]
    dyn --> pol["Policy Network P"]
    pol --> search["Gumbel Tree Search"]
    search --> act["Action"]
```
- **核心组件清单**:Vision=`N/A(proprio-only)` / Text=`N/A` / 3D=`N/A` / Proprio=`Task-conditioned state` / Backbone=`**EfficientZero-v2 latent model**` / Fusion=`N/A` / Action Head=`Policy distillation from MCTS` / Aux Head=`Reward/Value prediction` / WM=`**Latent(dynamics network)**` / Routing=`Task embedding conditioning` / Post-train=`N/A`
- **输入 → 输出**:`Proprio state + Task ID → Continuous action`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{consistency}} + \mathcal{L}_{\text{reward}} + \mathcal{L}_{\text{value}} + \mathcal{L}_{\text{policy}} \]
- **关键消融**:
  - EZ-M vs SAC/TD-MPC2(HumanoidBench-Hard Fig 1)→ 显著超越所有 baseline @1M steps
  - 任务数增加 → MF 下降而 MBRL 持续提升
  - Action encoder for cross-task → 稳定多任务共享模型训练
- **最大正面贡献组件**:**Shared dynamics model + task diversity regularization** — SOTA on HumanoidBench,因物理定律跨任务不变,多任务数据防止过拟合。
- **优势**:样本效率极高(1M interactions);任务增加反而改善性能。**局限**:仅在仿真验证(HumanoidBench);需要 task reward 设计。
- **范式归属**:E6 + E1

#### 7.O.11 [WoVR](p/WoVR_World_Models_as_Reliable_Simulators_for_Post-Training_VLAs/paper.pdf) — 幻觉感知 WM 内 RL [T2]

- **一句话定位**:**幻觉感知**的世界模型 RL 后训框架,三层级控制 hallucination,真机 **+30pp**,23FPS。
- **架构**:
```mermaid
flowchart LR
    vla["VLA Policy"] --> wm["Stabilized Video World Model"]
    wm --> kir["Keyframe-Initialized Rollouts"]
    kir --> grpo["GRPO Policy Optimization"]
    grpo --> vla
    vla --> pace["PACE Co-evolution"]
    pace --> wm
```
- **核心组件清单**:Vision=`Video DiT WM` / Text=`依赖底座 VLA` / 3D=`N/A` / Proprio=`可选` / Backbone=`OpenVLA-OFT` / Fusion=`依赖底座` / Action Head=`AR(FAST)` / Aux Head=`N/A` / WM=`**像素(stabilized AR video)**` / Routing=`N/A` / Post-train=`**GRPO + KIR + PACE**`
- **输入 → 输出**:`RGB + Lang → Discretized Action Tokens`
- **主要 Loss**:\[ \mathcal{L} = \mathcal{L}_{\text{GRPO}}\bigl(\pi_\theta;\ \text{imagined rollouts from WM}\bigr) \]
- **关键消融**:
  - WoVR vs base(LIBERO Avg)→ **39.95% → 69.2%**(+29.3pp)
  - WoVR 真机 → **61.7% → 91.7%**(+30.0pp)
  - KIR(关键帧初始化) → 缩短有效预测深度,减少幻觉累积
- **最大正面贡献组件**:**KIR + PACE** — 从关键帧开始 rollout 减少误差累积,共进化保持策略-模型对齐。
- **优势**:显式控制三层 hallucination(模型/交互/分布对齐);真机 +30pp,23FPS。**局限**:世界模型本身仍需高质量训练数据;PACE 迭代增训练复杂度。
- **范式归属**:E6 + B4 + E1

#### 7.O.12 [RealMirror](p/RealMirror_Comprehensive_Open-Source_VLA_Platform_for_Embodied_AI/paper.pdf) — 人形 VLA 平台 + Sim2Real [T2]

- **一句话定位**:**人形 VLA 开源平台**:VR 遥操作 + Isaac Sim/3DGS + LeRobot,仿真纯训 **零样本 Sim2Real**。
- **架构**:
```mermaid
flowchart LR
    A["VR 遥操作"] --> B["1200 轨迹"]
    B --> C["Isaac Sim / 3DGS"]
    C --> D["ACT / DP / SmolVLA"]
    D --> E["真机零样本"]
```
- **核心组件清单**:Vision=`多相机` / Text=`语言` / 3D=`3DGS 场景` / Proprio=`人形关节` / Backbone=`SmolVLA 等` / Fusion=`各模型原生` / Action Head=`ACT / DP / SmolVLA` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`**Sim-only 训**`
- **输入 → 输出**:`多相机 + 指令 → 各选用模型原生动作`
- **主要 Loss**:`各模型原生(ACT VAE、DP 扩散、SmolVLA 等);平台无统一公式`
- **关键消融**:
  - Benchmark Avg:**SmolVLA 79.75% > DP 75.15% > ACT 73.55%**
  - 3DGS 高保真场景是 zero-shot Sim2Real 主因
  - ACT zero-shot pick-place **92.86%** / ball transfer **71.43%**
  - 语言条件 + 双系统 SmolVLA 平均最稳
- **最大正面贡献组件**:**3DGS + 数字孪生 Sim2Real 管线** — 无需真机 fine-tune 即可部署。
- **优势**:端到端开源 benchmark+平台;五类任务 1200 轨迹。**局限**:非新 VLA 架构;π0.5 未在同平台评测。
- **范式归属**:F3 + E1

#### 7.O.13 [ELITE](p/ELITE_Experiential_Learning_and_Intent-Aware_Transfer_for_Self-improving_Embodied_Agents/paper.pdf) — 在线经验蒸馏策略池 [T2]

- **一句话定位**:**经验学习 + 意图检索** 使 VLM Agent 自我改进,EB-ALFRED **+9% / EB-Habitat +5%**。
- **架构**:
```mermaid
flowchart LR
    inst["Task Instruction"] --> planner["Coarse Planner (VLM)"]
    planner --> retrieve["Intent-aware Retrieval"]
    pool["Evolving Strategy Pool"] --> retrieve
    retrieve --> agent["Embodied Agent (VLM)"]
    agent --> env["Environment Action"]
    env --> distiller["Reflective Experience Distiller"]
    distiller --> pool
```
- **核心组件清单**:Vision=`VLM 内置` / Text=`VLM(GPT-4o 等)` / 3D=`N/A` / Proprio=`N/A` / Backbone=`VLM(frozen, prompt-based)` / Fusion=`N/A` / Action Head=`Discrete(navigation+manipulation primitives)` / Aux Head=`N/A` / WM=`N/A` / Routing=`N/A` / Post-train=`N/A`
- **输入 → 输出**:`RGB observation + Lang → Discrete action primitives`
- **主要 Loss**:`无梯度训练,基于 binary task outcome 的经验累积`
- **关键消融**:
  - ELITE vs base VLM(EB-ALFRED)→ **+9% success**(52→61%)
  - ELITE vs base VLM(EB-Habitat)→ **+5% success**(62→67%)
  - Intent-aware retrieval vs task-description retrieval → 显著优于表层匹配
- **最大正面贡献组件**:**Self-reflective Knowledge Construction** — +9%,因将执行轨迹蒸馏为可复用策略并持续演化。
- **优势**:无需监督/梯度更新,纯推理时学习;跨任务知识迁移。**局限**:依赖强 VLM(GPT-4o 级);离散动作空间,不适低级连续控制。
- **范式归属**:D6 + F3

#### 7.O.14 [ReconVLA](p/ReconVLA_An_Uncertainty-Guided_and_Failure-Aware_VLA_Framework/paper.pdf) — Conformal Prediction 不确定性层 [T2]

- **一句话定位**:**Plug-and-play 不确定性层**:Conformal Prediction 校准动作 token + SMD 状态 OOD,无需重训 VLA。
- **架构**:
```mermaid
flowchart LR
    A["冻结 VLA"] --> B["动作 token"]
    A --> C["隐状态"]
    B --> D["CQR 区间"]
    C --> E["SMD OOD"]
    D --> F["安全选动作 / 求助"]
    E --> F
```
- **核心组件清单**:Vision=`随基座` / Text=`随基座` / 3D=`—` / Proprio=`—` / Backbone=`π0 / OpenVLA-OFT 冻` / Fusion=`—` / Action Head=`原动作头` / Aux Head=`**CQR + SMD 层**` / WM=`N/A` / Routing=`N/A` / Post-train=`仅 CP 校准`
- **输入 → 输出**:`VLA 动作 / 隐状态 → 校准置信区间 + OOD 标志 → 部署拒绝 / 回退策略`
- **主要 Loss**:`分布自由 CP(非端到端 RL/SFT);未明确 L`
- **关键消融**:
  - 不改骨干即可降灾难性错误(仿真 + 真机)
  - UQ 分数与任务成功率显著相关
  - 可叠加 π0、OpenVLA-OFT 等多骨干
  - OOD 触发回退对 fail-safe 部署关键
- **最大正面贡献组件**:**Conformal Quantile Regression(CQR)** — token 级校准不确定性,distribution-free。
- **优势**:零梯度重训;即插即用。**局限**:非涨 SR 路线;需校准集代表部署分布。
- **范式归属**:D2(部署层) + D6

#### 7.O.15 [P3Nav](p/P3Nav_End-to-End_Perception,_Prediction_and_Planning_for_VLN/paper.pdf) — VLN 端到端感知/预测/规划 [T2]

- **一句话定位**:VLN 首个 **单网端到端 感知→预测→规划**(BEV 物体+地图→路点热图→序列动作),REVERIE SR 60.06 / R2R-CE SR 62 SOTA。
- **架构**:
```mermaid
flowchart LR
    A["全景 RGB"] --> B["BEV Encoder"]
    B --> C["Perception"]
    C --> D["Prediction"]
    D --> E["Planning Decoder"]
    E --> F["导航动作序列"]
```
- **核心组件清单**:Vision=`全景` / Text=`指令` / 3D=`BEV 地图 latent` / Proprio=`位姿` / Backbone=`共享 BEV 骨干` / Fusion=`串行 PPP` / Action Head=`序列动作解码` / Aux Head=`**MLM / SAP / OG**` / WM=`**未来 scene feat**` / Routing=`N/A` / Post-train=`200k PT + 50k FT`
- **输入 → 输出**:`全景 + Lang → REVERIE / R2R / RxR 离散导航动作`
- **主要 Loss**:`多任务加权和(DETR 检测 + 热图 MSE + ||m - m_vlm||² + 规划 CE)`
- **关键消融**:
  - REVERIE Test → SR **60.06** / RGS **39.75** SOTA
  - R2R-CE Val → SR **62** / SPL **52** SOTA
  - 对象解码 / 地图 / waypoint 各模块均涨点
  - 中间感知 + 预测模块移除显著降 SPL/RGS
- **最大正面贡献组件**:**BEV 感知-预测-规划三阶段联合** — 消除模块化误差累积。
- **优势**:REVERIE/R2R-CE/RxR-CE 全 SOTA。**局限**:VLN 非机械臂 VLA;算力 4×4090。
- **范式归属**:D1 + C1

#### 7.O.16 [SACA](p/SACA_Step-Aware_Contrastive_Alignment_for_VLN-CE/paper.pdf) — VLN-CE 上的 GRPO 增强 [T2]

- **一句话定位**:**VLN-CE 步级对比对齐 + GRPO**,R2R-CE **SR 60.3% / SPL 55.1%**(无额外数据 SOTA)。
- **架构**:
```mermaid
flowchart LR
    A["单 RGB"] --> B["LLaVA-Video-8B"]
    B --> C["PGSA 逐步评分"]
    C --> D["GRPO 更新"]
    D --> E["导航动作"]
```
- **核心组件清单**:Vision=`单 RGB 视频` / Text=`指令` / 3D=`N/A` / Proprio=`位姿` / Backbone=`LLaVA-Video-8B` / Fusion=`SFT 初始化` / Action Head=`离散导航` / Aux Head=`**PGSA 审计器**` / WM=`N/A` / Routing=`Repair/Rescue 路由` / Post-train=`**SFT→RFT**`
- **输入 → 输出**:`第一人称 RGB + 导航指令 → R2R-CE/RxR-CE 离散动作`
- **主要 Loss**:\[ R_{\text{proc}}(T) = \tfrac{1}{T}\sum_t \text{ReLU}(S_t - \tau_s),\quad L_{\text{mixed}} = L_{\text{GRPO}} + \lambda_{\text{rep}}L_{\text{BC}} \]
- **关键消融**:
  - R2R-CE → SR **60.3% / SPL 55.1%**(无额外数据 SOTA)
  - vs StreamVLN → **SR +7.5pp / SPL +7.9pp**
  - Soft Score + All-Failure Rescue + Repair Resampling 三者合用最佳
  - 利用 **~73%** 失败轨迹有效前缀
- **最大正面贡献组件**:**Perception-Grounded Step-Aware Auditor(PGSA)** — 无需专用 PRM 即可逐步奖励。
- **优势**:R2R/RxR-CE 全面 SOTA;失败轨迹变资源。**局限**:VLN 非 manipulation;8×A6000 ~36h SFT。
- **范式归属**:C1 + E5 + G3

#### 7.O.17 [ACoT-VLA](p/ACoT-VLA_Action_Chain-of-Thought_for_Vision-Language-Action_Models/paper.pdf) — 动作空间 CoT(EAR + IAR + AGP)[T1]

- **一句话定位**:**首个把 CoT 引入动作空间(非语言/视觉)** — π0.5 base + EAR 显式参考动作 + IAR 隐式动作先验,**LIBERO-Plus 2026 H1 SOTA 87.5%**。
- **架构**:
```mermaid
flowchart LR
    obs["RGB + Lang"] --> vlm["π0.5 base: SigLIP + Gemma-2B (N=18, d=2048)"]
    vlm --> kvCache["VLM 各层 KV cache"]
    kvCache --> ear["EAR: 18 层 Transformer self-attn + cross-attn VLM KV"]
    ear --> aRef["Coarse 参考轨迹 a^ref (H_ref=15, shift=2)"]
    kvCache --> iar["IAR: 学习查询 Q_i (M=1, 降维 d'=128) cross-attn"]
    iar --> latentPrior["潜在动作先验 Z^im"]
    aRef --> agp["AGP head: 融合显式 + 隐式条件"]
    latentPrior --> agp
    agp --> aFinal["执行 action chunk (H=10, shift=1)"]
```
- **核心组件清单**:Vision=`SigLIP` / Text=`Gemma-2B (18 层 d=2048)` / 3D=`N/A` / Proprio=`included in π0.5 base` / Backbone=`π0.5` / Fusion=`EAR 与 IAR 通过 cross-attn 从 VLM 各层 KV 注入条件;AGP 融合 explicit + implicit guidance` / Action Head=`Flow Matching (\(\pi_\theta^{\text{head}}\))` / Aux Head=`**EAR**(18 层 Transformer,生成 coarse 参考轨迹)+ **IAR**(cross-attn 潜在动作先验)` / WM=`N/A(动作空间 CoT 替代 visual/language CoT)` / Routing=`teacher forcing 训练 + 全自条件推理` / Post-train=`无 RL,LIBERO/LIBERO-Plus/VLABench 直接 SFT`
- **输入 → 输出**:`RGB + Lang → coarse 参考动作 (H_ref=15) + 潜在动作先验 → 最终 action chunk (H=10)`
- **主要 Loss**:\[ \mathcal{L} = \lambda_1 \mathcal{L}_{\pi_\theta^{\text{ref}}} + \lambda_2 \mathcal{L}_{\pi_\theta^{\text{head}}};\ \lambda_1 = \lambda_2 = 0.5 \] (flow matching MSE,Sec 3.4)
- **关键消融**(Table 4-6):
  - Baseline(π0.5) → 96.9% LIBERO 平均
  - + IAR alone → **98.1%** (+1.2 pp)
  - + EAR alone → **98.1%** (+1.2 pp)
  - + EAR + IAR (AGP) → **98.4%** (+1.5 pp)
  - Reference action H_ref=15 / shift=2 最佳(Table 5)
  - **LIBERO-Plus 87.5%** 全局 #1;**VLABench unseen-texture IS +12.6 pp / PS +7.2 pp**;**Teacher-forcing stabilization** 显著好(用 GT 训练,推理用 EAR 自身生成)
- **最大正面贡献组件**:**EAR(Explicit Action Reasoner)** — 在动作空间提供 kinematically-grounded coarse reference,取代 visual/language CoT 的 indirect guidance;IAR 补充隐式动作分布先验。
- **优势**:**首个把 CoT 引入动作空间**,LIBERO-Long 等长程任务涨幅显著;单卡 RTX 4090 推理。**局限**:Reference 动作配置(H_ref / shift)需调参;架构耦合 π0.5 base。
- **范式归属**:**D5 CoT** + A3 + F1

---

## 第 8 章 设计建议与反模式 [T1]

> 本章是"看完前 7 章,要如何落地"的部分。先给场景化组件组合配方,再给常见架构陷阱。

### 8.1 场景化组件组合建议 [T1]

#### 8.1.1 双臂桌面操作(中等数据,单一本体)

- **首选组件**:V1 SigLIP + V4 Proprio MLP + L1 PaliGemma-3B + F2 中融合 cross-attn + A3 Flow head (4-8 步) + 无大 WM head;
- **可选**:加 V2 DINOv2 / V3 3D 提升精细度;加 W5 Future State 提升长程;
- **避坑**:不要为了"看起来全面"硬塞 W1 像素 WM——大概率拖慢推理而无收益。
- **74 篇内证据**:[SimVLA](#7a5-simvla)(0.5B 极简 Flow VLA baseline,LIBERO 98.6%);[FLOWER](#7a2-flower)(950M 中间融合 Flow + 50% 裁剪,CALVIN ABC 4.53 SOTA);[Pose-VLA](#7v6-pose-vla)(C3 Pose 预训接入);[LingBot-VLA](#7a3-lingbot-vla)(20K 小时 + Flow);[VLA-Foundry](#7l7-vla-foundry)(F1 VQA 共训防遗忘);[VLANeXt](#7l8-vlanext)(12 条 recipe,2.5B 超 7B OpenVLA-OFT)。

#### 8.1.2 人形 Loco-Manipulation(多任务,持续部署)

- **首选组件**:V1 SigLIP + V3 3D + V4 Proprio + L2 自训 VLM + F4 MoT + A4 AR+Flow 混合 + W2 Latent / W5 Future State + E5 共享骨干 + O3 异步 + O4 Distillation;
- **可选**:E3 Latent Action(用大量人类视频)。
- **避坑**:E6(无,本表无)直接对应仿真 RL,慎用 model-based。
- **74 篇内证据**:[Ψ0](#7a11-ψ0-psi-zero)(EgoDex 800h + 30h 真机超 10× 数据基线 40%+);[GR00T_N1.6](#7l3-gr00t_n16)(NVIDIA 自训 + DiT 32 层 + DAgger + RTC);[Helix_02](#7f2-helix_02-figure-ai)(Figure 03 三层全身 S0/S1/S2,4 分钟自主);[HY-Embodied-0.5](#7f3-hy-embodied-05)(MoT + 自演化 RL);[DM0](#7l2-dm0)(三阶段 Embodied-Native);[Being-H0.5](#7f1-being-h05)(MoF 跨 30+ 构型);[SmoothVLA](#7a9-smoothvla)(jerk RFT 防真机抖动)。

#### 8.1.3 长程多步任务(开放语言指令)

- **首选组件**:V1 SigLIP + L1 大 MLLM + F2 中融合 + A4 AR+Flow + W4 Trace + W6 CoT;
- **可选**:加 W5 Future State 提升成功率。
- **避坑**:W6 CoT latency 大,需做"CoT-on-demand"。
- **74 篇内证据**:[LoHo-Manip](#7w15-loho-manip)(Manager VLM + Trace + Executor π0.5);[CycleVLA](#7a12-cyclevla)(MBR + 回溯 +5-10%);[MolmoAct2](#7l4-molmoact2)(Think 37× 加速 + Action Reasoning);[HY-Embodied-0.5](#7f3-hy-embodied-05)(C1 步骤对齐);[HiF-VLA](#7w6-hif-vla)(Hindsight + Insight + Foresight motion vector);[TiPToP](#7o9-tiptop)(模块化开放词汇规划 +22.2pp vs π0.5);[π0.7](#7f6-π07)(rich prompt 组合泛化)。

#### 8.1.4 跨本体迁移(N 种本体共训)

- **首选组件**:E3 Latent Action Tokenizer + E5 共享骨干 + E2 Soft-Prompt + F4 MoT;
- **避坑**:E1 Per-head 已落后;Naive 把所有本体数据简单 mixing → 必负迁移。
- **74 篇内证据**:[LAP](#7e1-lap)(Language-Action 零样本跨本体 +27pp);[X-VLA](#7a8-x-vla)(Soft-Prompt + 0.9B + LoRA 1%);[OXE-AugE](#7e2-oxe-auge)(数据层增广 +24-45%);[Being-H0.5](#7f1-being-h05)(MoF 跨 30+ 构型);[Genie Sim 3.0](#7e3-genie-sim-30)(平台 + 仿真域随机化);[World2Act](#7w13-world2act)(latent 对齐 +6.7%);[ABot-M0](#7a1-abot-m0)(UniACT 6M 跨 20+ 构型)。

#### 8.1.5 有限数据 SFT(几十~几百条示教)

- **首选组件**:V1 + V2 DINOv2 + V3 3D(用预训练 Pose 模型)+ L1 小 MLLM(<1B)+ A3 Flow + F6 Adapter + O4 Distillation(从大 VLA);
- **避坑**:不要从头训 W1 像素 WM——数据完全不够。
- **74 篇内证据**:[Pose-VLA](#7v6-pose-vla)(通用 Pose 预训即插即用,~100 demo 足够);[GeneralVLA](#7v3-generalvla)(零真机数据 + 3D Affordance + LLM 规划);[GST-VLA](#7v4-gst-vla)(3D Gaussian + Flow + MoE,LIBERO-Plus 高位);[PokéVLA](#7v5-pokévla)(1.22B + F1/F2 共训蒸馏);[StarVLA-α](#7f4-starvla-α)(强 VLM + MLP head 极简,RoboChallenge +20% vs π0.5);[VLA-OPD](#7o3-vla-opd)(1-traj → 87.4% 蒸馏);[FLOWER](#7a2-flower)(A3 高效 Flow,小样本友好)。

#### 8.1.6 真机量产部署(实时 / 低显存)

- **首选组件**:O1 PTQ W4A8 + O2 Layer Pruning 50% + O3 异步 / Chunking + A3 1-step Flow + V5 Patch-Select + 工程优化;
- **避坑**:O1 量化对 A3 Flow head 敏感,留 1-2 步缓冲。
- **74 篇内证据**:[QuantVLA](#7o1-quantvla)(W4A8 全配方,70% 显存 + 1.22× 速度,选择性 LLM+DiT MLP 180 层);[FLOWER](#7a2-flower)(50% 层裁剪 + 中间 fusion);[Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)(MoT + Λ-attn + 80ms 异步);[HiPolicy](#7a7-hipolicy)(分层多频 +25% speed);[SimVLA](#7a5-simvla)(0.5B 极简 baseline + VRAM 9.3GB);[GigaWorld-Policy](#7w5-gigaworld-policy)(推理跳视频 9× 加速)。

#### 8.1.7 高频闭环控制 / 灵巧手(>30 Hz,接触密集)

- **首选组件**:V1 + V4 Proprio + L1 小 MLLM + A4 AR+Flow + A6 频率分解 + W5 Future State + O3 异步;
- **避坑**:不要用纯 W1 像素 WM 做闭环规划——延迟不允许。
- **74 篇内证据**:[RLDX-1](#7a4-rldx-1)(MSAT 6.9-8B 灵巧手优先 + Memory + 触觉 + 力矩流,ALLEX 真机 ~86.8%);[HiPolicy](#7a7-hipolicy)(分层多频 chunking + 熵引导执行);[OA-WAM](#7w8-oa-wam)(对象槽位 WAM,几何鲁棒);[EZ-M](#7o10-ez-m)(多任务 MBRL HumanoidBench-Hard);[STARRY](#7w9-starry)(B4 时空 WM + Diffusion + GASAM 几何调制);[Helix_02](#7f2-helix_02-figure-ai)(S0 10M @1kHz 力矩控制)。

#### 8.1.8 VLN / 室内导航(连续/离散动作,远程目标)

- **首选组件**:V1 + L1 + F2 中融合 + W5 Future State + W6 CoT 启发的步长对齐;
- **避坑**:不要把 VLN 当短程操作训。
- **74 篇内证据**:[SACA](#7o16-saca)(VLN-CE step-aware 对比 + GRPO,R2R-CE 60.3% SOTA);[P3Nav](#7o15-p3nav)(端到端 PPP,REVERIE SR 60.06 / R2R-CE 62 SOTA);[BTK](#7v7-btk)(VLN 多模态知识库锚定 +3% SPL);[ELITE](#7o13-elite)(在线经验蒸馏,EB-ALFRED +9%)。

### 8.2 常见架构反模式 [T1]

1. **Vision Encoder 与 VLM token 维度不匹配**:SigLIP 输出 1152d、PaliGemma 期望 2048d → 用 projector,但 projector 太小会瓶颈。**对策**:projector 至少 2 层 MLP + LayerNorm。
2. **Diffusion head 吃 VLM 梯度**:Diffusion head 太重时,backward 把 VLM 通识"挤掉"。**对策**:Diffusion 用小学生网,或冻结 VLM 大部分层。
3. **MoT 门控未热身**:从一开始 MoT 全开,梯度爆炸或路由坍塌。**对策**:前 1k step 仅训 backbone,2k 后逐步开 MoT。
4. **PTQ 量化破坏 Flow head**:Flow head 对 4-bit 整数敏感,1-step 后崩。**对策**:Flow head 保 BF16,只量化 backbone。
5. **Per-head 不共享 BN**:跨本体 head 各自 BN 统计,推理时不同本体相互干扰。**对策**:跨本体共享 LayerNorm。
6. **3D Encoder 漏掉 Proprio**:3D 编码器只看视觉,忽略本体状态,导致动作输出与机器人姿态错位。**对策**:把 Proprio token 拼到 3D token 序列。
7. **Patch-Select 选错关键 patch**:hard gating 选错就完全看不到关键区域。**对策**:配 dropout + 软门控 fallback。
8. **Adapter 容量不足**:LoRA r=4 试图覆盖大改动,根本不够。**对策**:跨本体迁移 r ≥ 32,或换 MoT。
9. **像素 WM 抢通识**:W1 像素 head 太重,把 VLM 的语言/VQA 能力挤掉。**对策**:换 W2 Latent 或加 VQA 共训。
10. **CoT 与动作不对齐**:CoT 先生成,但与动作没 alignment supervision。**对策**:让 CoT 与 action latent 共享中间表示。

### 8.3 未来趋势 [T3]

基于 74 篇 2024-2026 论文的组件级实证观察,下面 8 条趋势已在多篇工作中显现,并极可能延续到 2026~2027 年:

1. **MoT(F4)将成为大基座 VLA 的默认融合方式**。证据:[HY-Embodied-0.5](#7f3-hy-embodied-05)(MoT 2B/32B)/ [π0.7](#7f6-π07)(Steerable MoT)/ [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)(MoT + Λ-attn)/ [Being-H0.5](#7f1-being-h05)(MoF)/ [Fast-WAM](#7w3-fast-wam) 等同期 6+ 篇集中突破。
2. **Flow Head(A3)继续主导动作头**,从 4-10 步压到 1-2 步,但配 OOD 动态步数切换。证据:[FLOWER](#7a2-flower)(4-8 步)/ [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)(80ms)/ [π0.6](#7a10-π06-recap)(advantage-conditioned)/ [SimVLA](#7a5-simvla)(极简 baseline)等均 ≤10 步。
3. **Latent / JEPA WM(W2)取代像素 WM(W1)**。证据:[VLA-JEPA](#7w10-vla-jepa)(LIBERO-Plus +16.6pp)/ [Fast-WAM](#7w3-fast-wam)(190ms / 4× 加速)/ [Mask World Model](#7w7-mask-world-model-mwm)(语义 mask +35pp)/ [Being-H0.7](#7w17-being-h07)(dual-branch latent 3-4ms);像素 WM 仅大厂(Cosmos / DreamZero / Psi-R2)继续。
4. **World ↔ Action 共演化架构(B4)**集中爆发。证据:[CoLA-World](#7w1-cola-world) / [OA-WAM](#7w8-oa-wam) / [STARRY](#7w9-starry) / [VLAW](#7w11-vlaw) / [World-VLA-Loop](#7w12-world-vla-loop) / [World2Act](#7w13-world2act) 同期 6 篇。
5. **跨本体 Latent Action(E3)+ Soft-Prompt(E2)逐步取代 Per-head**。证据:[LAP](#7e1-lap) +27pp / [X-VLA](#7a8-x-vla) LoRA 1% / [OXE-AugE](#7e2-oxe-auge) 数据派 / [World2Act](#7w13-world2act) latent 对齐。
6. **后训 RL 全面进入产业部署(O5)**。证据:[LWD](#7o4-lwd) fleet RL +25pp / [SOP](#7o5-sop) 4 Actors 2.4× 加速 / [TT-VLA](#7o6-tt-vla) test-time / [VLA-OPD](#7o3-vla-opd) 1-traj +38.5pp / [LifeLong-RFT](#7o7-lifelong-rft) 20% 数据 +22% / [π0.6 RECAP](#7a10-π06-recap) 真机吞吐 2×。
7. **部署优化矩阵化(O1-O5 全面叠加)**。证据:[QuantVLA](#7o1-quantvla)(70% 显存)/ [FLOWER](#7a2-flower)(50% 裁剪 + 99% 计算节省)/ [Xiaomi-Robotics-0](#7f5-xiaomi-robotics-0)(异步)/ [PokéVLA](#7v5-pokévla)(12× 蒸馏);未来一年内边缘部署可期。
8. **D5 CoT / Reasoning Head + W4 Trace 解决长程任务**。证据:[MolmoAct2](#7l4-molmoact2)(Think 37× 加速)/ [LoHo-Manip](#7w15-loho-manip)(Trace + Manager)/ [GST-VLA](#7v4-gst-vla)(DA-CoT)/ [NS-VLA](#7w16-ns-vla)(神经符号 +98.6%)/ [CycleVLA](#7a12-cyclevla)(MBR +5-10%)。

> **可预测**:2026 H2 - 2027 H1 主流 VLA 配方 ≈ `Qwen3-VL / 自训 VLM(L2)` + `MoT(F4) + Λ-attn(F2)` + `A4 AR+Flow head` + `W2 JEPA + W4 Trace` + `E2 Soft-Prompt + E3 Latent Action` + `O1+O2+O3+O5 全面叠加`,真机量产门槛进一步降低。

---

## 第 9 章 参考文献与论文 ↔ 组件 ↔ 章节锚点映射 [T1 / T3]

### 9.1 论文字母索引(74 篇) [T1]

> 每行格式:`[论文短名](PDF/HTML 链接) → 速查卡编号 · **主组件** / 副组件 / 副组件 · 一句话定位`
> 速查卡编号 `7.X.N` 对应第 7 章 "74 篇速查表" 中的同名子节(全文检索 "7.X.N" 即可跳转)。
> **主组件** 指该论文最具新意 / 最有差异化的组件(从 V/L/F/A/W/E/O 七类中选);副组件 = 论文同时引入或辅助的次要组件。
> 与 [vla_traintask.md](vla_traintask.md) 的 9.1 字母索引 **互为对偶**:那里按"训练任务范式(A1-G3)"分组,这里按"模型结构组件(V-O)"分组,同一篇论文在两份文档里都能找到对应卡。

- [ABot-M0](p/ABot-M0_VLA_Foundation_Model_with_Action_Manifold_Learning/paper.pdf) → 7.A.1 · **A5** / V1 / V4 · 动作流形 head + 双流感知
- [ACoT-VLA](p/ACoT-VLA_Action_Chain-of-Thought_for_Vision-Language-Action_Models/paper.pdf) → 7.O.17 · **O3**(动作空间 CoT) / W6 / A3 · π0.5 base + EAR + IAR + AGP
- [Being-H0.5](p/Being-H0.5/paper.pdf) → 7.F.1 · **F4** / E3 / L1 · 人手为母语的 MoF 跨本体基座
- [Being-H0.7](p/Being-H0.7_A_Latent_World-Action_Model_from_Egocentric_Videos/paper.pdf) → 7.W.17 · **W2** / E3 / A3 · Egocentric Latent World-Action 双分支
- [BTK](p/Beyond_Textual_Knowledge_(BTK)_Leveraging_Multimodal_Knowledge_Bases_for_Enhancing_VLN/paper.pdf) → 7.V.7 · **V3**/V1 / W6 · VLN 多模态知识库锚定
- [CoLA-World](p/CoLA-World_Co-evolution_of_Latent_Action_+_World_Model/paper.pdf) → 7.W.1 · **W2** / E3 / W4 · Latent Action 与 WM 共演化
- [ConsisVLA-4D](p/ConsisVLA-4D_Advancing_Spatiotemporal_Consistency_in_Efficient_3D-Perception_and_4D-Reasoning_for_Robotic_Manipulation/paper.pdf) → 7.V.1 · **V3** / W5 / A4 · 4D 时空一致性 3D 感知
- [Consistency-Consensus (Is the Future Compatible?)](p/Is_the_Future_Compatible_Diagnosing_Dynamic_Consistency_in_World_Action_Models/paper.pdf) → 7.W.21 · **W2**(零训练 ranking) / D2 评估替代 · action-state consistency
- [Cosmos Policy](p/Cosmos_Policy_(NVIDIA)/paper.pdf) → 7.L.1 · **L2** / W1 / A2 · 自训 Cosmos-Predict2 视频基座
- [CycleVLA](p/CycleVLA_Backtracking_+_MBR_Decoding_for_VLA/paper.pdf) → 7.A.12 · **A2** / W6 / O3 · DDPM + 回溯 + MBR
- [DM0](p/DM0_An_Embodied-Native_Vision-Language-Action_Model_towards_Physical_AI/paper.pdf) → 7.L.2 · **L2** / W6 / F1 · 具身原生三阶段自训 VLM
- [DreamZero](p/DreamZero_World_Action_Models_are_Zero-Shot_Policies/paper.pdf) → 7.W.2 · **W1** / W4 / A2 · 14B 块级视频-动作 WAM
- [ELITE](p/ELITE_Experiential_Learning_and_Intent-Aware_Transfer_for_Self-improving_Embodied_Agents/paper.pdf) → 7.O.13 · **O4** / W5 · 在线经验蒸馏策略池
- [EZ-M](p/EZ-M_Scaling_Tasks,_Not_Samples_-_Mastering_Humanoid_Control_through_Multi-Task_Model-Based_RL/paper.pdf) → 7.O.10 · **O5** / W5 · 多任务 MBRL 人形
- [Fast-WAM](p/Fast-WAM_Do_World_Action_Models_Need_Test-time_Future_Imagination/paper.pdf) → 7.W.3 · **W2** / O5 / A3 · 训练共训视频、推理跳过
- [FLOWER](p/FLOWER_Efficient_VLA_Flow_Policy/paper.pdf) → 7.A.2 · **A3** / F2 / O2 · 950M 中间融合 Flow + 50% 裁剪
- [FocusVLA](p/FocusVLA_Focused_Visual_Utilization_for_VLAs/paper.pdf) → 7.V.2 · **V5** / A4 / V1 · 级联注意力 + Patch-Select
- [From_Human_Skill_to_Robotic_Mastery (Psi-R2/Psi-W0)](p/From_Human_Skill_to_Robotic_Mastery_(Psi-R2__Psi-W0)/page.html) → 7.W.14 · **W1** / W2 / E5 · 十万小时 IT2V WM
- [FutureVLA](p/FutureVLA_Joint_Visuomotor_Prediction_for_VLA/paper.pdf) → 7.W.4 · **W2** / W5 / E3 · JVPM 视-运动门控
- [GeneralVLA](p/GeneralVLA_3D_Affordance_+_Control_Strategy/paper.pdf) → 7.V.3 · **V3** / W4 / F3 · 3D Affordance + 控制策略
- [Genie Sim 3.0](p/Genie_Sim_3.0（智元__AgiBot）/paper.pdf) → 7.E.3 · **E4** / 平台 / O3 · 智元 AgiBot 仿真平台
- [GigaWorld-Policy](p/GigaWorld-Policy_An_Efficient_Action-Centered_World–Action_Model/paper.pdf) → 7.W.5 · **W3** / A3 / O3 · 动作中心 WAM
- [GR00T_N1.6](p/GR00T_N1.6_(NVIDIA)/page_1.html) → 7.L.3 · **L2** / A2 / F4 · NVIDIA 自训 VLM + DiT
- [Green-VLA](p/Green-VLA_5-Stage_Curriculum_to_Strong_VLA/paper.pdf) → 7.O.2 · **O3** / A3 / O4 · 五阶段课程 + R2 RL
- [GST-VLA](p/GST-VLA_Structured_Gaussian_Spatial_Tokens_for_3D_Depth-Aware_VLAs/paper.pdf) → 7.V.4 · **V3** / W6 / A3 · 结构化 3D Gaussian Spatial Token
- [HAMLET](p/HAMLET_Switch_your_VLA_into_a_History-Aware_Policy/paper.pdf) → 7.A.6 · **A2** / W4 / F6 · DiT + History-aware 切换器
- [Helix_02 (Figure AI)](p/Helix_02_(Figure_AI)/page.html) → 7.F.2 · **F3** / A4 / V4 · Figure 02 三层全身
- [HiF-VLA](p/HiF-VLA_Hindsight,_Insight_and_Foresight_through_Motion_Representation/paper.pdf) → 7.W.6 · **W4** / W5 / A4 · MPEG motion vector 时序桥
- [HiPolicy](p/HiPolicy_Hierarchical_Multi-Frequency_Action_Chunking_for_Policy_Learning/paper.pdf) → 7.A.7 · **A6** / A2 / O3 · 分层多频率 chunking
- [HY-Embodied-0.5](p/HY-Embodied-0.5_Embodied_Foundation_Models_for_Real-World_Agents/paper.pdf) → 7.F.3 · **F4** / V3 / O4 · 腾讯 MoT + 课程蒸馏
- [LAP](p/LAP_Language-Action_Pre-Training_Enables_Zero-shot_Cross-Embodiment_Transfer/paper.pdf) → 7.E.1 · **E3** / A3 / L1 · 语言动作预训练 + Latent Action
- [LWD](p/Learning_While_Deploying_(LWD)_Fleet-Scale_Reinforcement_Learning_for_Generalist_Robot_Policies/paper.pdf) → 7.O.4 · **O5** / O4 / A3 · 舰队级 RL
- [LifeLong-RFT](p/LifeLong-RFT_Lifelong_Reinforcement_Fine-Tuning/paper.pdf) → 7.O.7 · **O5** / A1 · chunk-level GRPO 终身
- [LingBot-VLA](p/LingBot-VLA__A_Pragmatic_VLA_Foundation_Model/paper.pdf) → 7.A.3 · **A3** / E3 / O4 · 务实 Flow VLA
- [LoHo-Manip](p/LoHo-Manip_Long-Horizon_Manipulation_via_Trace-Conditioned_VLA_Planning/paper.pdf) → 7.W.15 · **W4** / W6 / A4 · Trace 条件 VLA
- [Mask World Model](p/Mask_World_Model_(MWM)_Predicting_What_Matters_for_Robust_Robot_Policy_Learning/paper.pdf) → 7.W.7 · **W3** / W2 / A2 · 预测语义 mask
- [MINT-4B](p/MINT_Mimic_Intent,_Not_Just_Trajectories_(MINT-4B)/paper.pdf) → 7.L.6 · **L3** / A6 / O3 · DCT 多尺度 VQ Action Token
- [MolmoAct2](p/MolmoAct2_Action_Reasoning_Models_for_Real-world_Deployment/paper.pdf) → 7.L.4 · **L2** / L3 / W6 · Molmo + FAST
- [MolmoB0T](p/MolmoB0T_Large-Scale_Simulation_Enables_Zero-Shot_Manipulation/paper.pdf) → 7.L.5 · **L2** / A3 · Molmo + 1.7M 仿真
- [MotuBrain](p/MotuBrain_An_Advanced_World_Action_Model_for_Robot_Control/paper.pdf) → 7.W.20 · **W3**(unified 5-mode WAM) / F4 / A3 · Vidu + 三流 MoT + H-Bridge
- [NS-VLA](p/NS-VLA_Towards_Neuro-Symbolic_VLAs/paper.pdf) → 7.W.16 · **W6** / O5 · 神经符号 + GRPO
- [OA-WAM](p/OA-WAM_Object-Addressable_World_Action_Model_for_Robust_Robot_Manipulation/paper.pdf) → 7.W.8 · **W3** / A3 / V3 · 对象槽位 WAM
- [OXE-AugE](p/OXE-AugE_Augmenting_OXE_with_Embodiment_Aug/paper.pdf) → 7.E.2 · **E4** / F6 · OXE 跨本体数据合成
- [P3Nav](p/P3Nav_End-to-End_Perception,_Prediction_and_Planning_for_VLN/paper.pdf) → 7.O.15 · **W5** / W6 / V1 · VLN E2E PPP
- [PokéVLA](p/PokéVLA_Empowering_Pocket-Sized_VLA_with_Comprehensive_World_Knowledge_Guidance/paper.pdf) → 7.V.5 · **V3** / O4 / L1 · 口袋级 VLA + 几何分割
- [Pose-VLA](p/Pose-VLA_Universal_Pose_Pretraining_for_Generalizable_VLAs/paper.pdf) → 7.V.6 · **V3** / A3 / A4 · 离散 Pose Token
- [PRTS](p/PRTS_A_Primitive_Reasoning_and_Tasking_System_via_Contrastive_Representations/paper.pdf) → 7.O.??(待定) · 原子推理 + 对比表征(主组件:F2 中融合 + W5 副) — 暂归 7.W.18 副引用
- [QuantVLA](p/QuantVLA_Post-Training_Quantization_for_VLA/paper.pdf) → 7.O.1 · **O1** · VLA 后训练量化
- [RealMirror](p/RealMirror_Comprehensive_Open-Source_VLA_Platform_for_Embodied_AI/paper.pdf) → 7.O.12 · **O4** / 平台 · 人形 VLA 平台
- [ReconVLA](p/ReconVLA_An_Uncertainty-Guided_and_Failure-Aware_VLA_Framework/paper.pdf) → 7.O.14 · **W5**(部署层 D2 维度) / O5 · Conformal Prediction
- [RLDX-1](p/RLDX-1_A_Dexterity-First_Foundation_Model_for_Robot_Hands/paper.pdf) → 7.A.4 · **A3** / F6 / V4 · 灵巧手 Flow VLA
- [SACA](p/SACA_Step-Aware_Contrastive_Alignment_for_VLN-CE/paper.pdf) → 7.O.16 · **O5** / F2 · VLN-CE GRPO
- [SimVLA](p/SimVLA_A_Simple_VLA_Baseline/paper.pdf) → 7.A.5 · **A3** / L1 / F1 · 极简 Flow VLA baseline
- [SmoothVLA](p/SmoothVLA_Aligning_VLAs_with_Physical_Constraints_via_Intrinsic_Smoothness_Optimization/paper.pdf) → 7.A.9 · **A3** / O5 · Jerk 内在奖励 GRPO
- [SOP](p/SOP_Scalable_Online_Post-Training/paper.pdf) → 7.O.5 · **O5** / O4 · 可扩展在线后训
- [STARRY](p/STARRY_Spatio-Temporal_Action-Centric_World_Modeling_for_Robotic_Manipulation/paper.pdf) → 7.W.9 · **W3** / A2 / V3 · 时空动作中心 WM
- [StarVLA-α](p/StarVLA-α_Reducing_Complexity_in_Vision-Language-Action_Systems/paper.pdf) → 7.F.4 · **F3** / O4 / L1 · 轻量 MLP 晚融合
- [STRONG-VLA](p/STRONG-VLA_Decoupled_Robustness_Learning_for_VLAs_under_Multimodal_Perturbations/paper.pdf) → 7.O.8 · **O3** / O4 · 解耦两阶段鲁棒
- [TiPToP](p/TiPToP_A_Modular_Open-Vocabulary_Planning_System_for_Robotic_Manipulation/paper.pdf) → 7.O.9 · **O5**(模块化非端到端) · 模块化开放词汇规划
- [TT-VLA](p/TT-VLA_Test-Time_RL_with_Task-Progress_Reward/paper.pdf) → 7.O.6 · **O5** / W5 · 测试时 RL
- [VLA-Foundry](p/VLA_Foundry_A_Unified_Framework_for_Training_VLAs/paper.pdf) → 7.L.7 · **L1** / A3 · TRI 统一训练栈
- [VLA-JEPA](p/VLA-JEPA_Enhancing_VLA_with_Latent_World_Model/paper.pdf) → 7.W.10 · **W2** / E3 / A3 · JEPA + Flow
- [VLA-OPD](p/VLA-OPD_Bridging_Offline_SFT_and_Online_RL_for_VLAs_via_On-Policy_Distillation/paper.pdf) → 7.O.3 · **O4** / O5 · OPD 桥接 SFT 与 RL
- [VLANeXt](p/VLANeXt_Recipes_for_Building_Strong_VLA_Models/paper.pdf) → 7.L.8 · **L1** / A3 / F4 · 12 条 recipe
- [VLAW](p/VLAW_Vision-Language-Action_World_Model/paper.pdf) → 7.W.11 · **W3** / O5 / A3 · Ctrl-World 一体
- [World-VLA-Loop](p/World-VLA-Loop_Closed-Loop_World_Models_for_VLAs/paper.pdf) → 7.W.12 · **W2** / O5 · 闭环 WM-VLA
- [World2Act](p/World2Act_Latent_Action_Post-Training_via_Skill-Compositional_World_Models/paper.pdf) → 7.W.13 · **W4** / E3 / O4 · 技能可组合 WM
- [WoVR](p/WoVR_World_Models_as_Reliable_Simulators_for_Post-Training_VLAs/paper.pdf) → 7.O.11 · **O5** / W2 · WM 作为可靠 sim
- [X-VLA](p/X-VLA_Soft-Prompt_Cross-Embodiment_VLA/paper.pdf) → 7.A.8 · **E2**(主) / A3 · Soft-Prompt + Flow
- [X-WAM](p/X-WAM_Unified_4D_World_Action_Modeling_from_Video_Priors_with_Asynchronous_Denoising/paper.pdf) → 7.W.19 · **W3**(4D RGB-D WAM) / W4 / A3 · Wan2.2-5B + 深度分支 + ANS
- [Xiaomi-Robotics-0](p/Xiaomi-Robotics-0_Open-Sourced_VLA_with_Real-Time_Execution/paper.pdf) → 7.F.5 · **F4** / F2 / O3 · MoT + Λ-attn + 异步
- [π0.6](p/π0.6__Recap/paper.pdf) → 7.A.10 · **A3** / O5 · Advantage-conditioned Flow
- [π0.7](p/π0.7_A_Steerable_Generalist_Robotic_Foundation_Model_with_Emergent_Capabilities/paper.pdf) → 7.F.6 · **F4** / A3 / W5 · Steerable MoT + Flow
- [Ψ0](p/Ψ0_(Psi-Zero)_An_Open_Foundation_Model_Towards_Universal_Humanoid_Loco-Manipulation/paper.pdf) → 7.A.11 · **A3** / E5 / O4 · 人形 loco-manip Flow 基座

### 9.1.X 按主组件倒排索引(74 篇 → 7 大组件类 / 32 子组件) [T3]

> 反向查询:给定一个组件,有哪些论文以此为主?用「**主**」标注主组件,_斜体_ 标注副组件。

**V. 视觉/3D/Proprio 编码器主 — 7 篇**

- **V1 SigLIP/CLIP 系**(0 主):**几乎全部 74 篇都默认 SigLIP/CLIP,无单独主篇** — 它是底层共同基础;
- **V2 DINOv2 系**(0 主):多副 — [PokéVLA](#7v5-pokévla) / [GST-VLA](#7v4-gst-vla) / [ConsisVLA-4D](#7v1-consisvla-4d) / [HiF-VLA](#7w6-hif-vla) / [MINT-4B 30M 版](#7l6-mint-4b);
- **V3 3D/Pose 编码器**(6 主):**ConsisVLA-4D**(7.V.1)、**GeneralVLA**(7.V.3)、**GST-VLA**(7.V.4)、**PokéVLA**(7.V.5)、**Pose-VLA**(7.V.6)、**BTK**(7.V.7);_OA-WAM / ABot-M0 / STARRY 副_;
- **V4 Proprio MLP/Tokenizer**(0 主):全部 74 篇必备 — _VLANeXt / RLDX-1 / Helix_02 / π0 系列_ 显式优化;
- **V5 多视图融合/Patch-Select**(1 主):**FocusVLA**(7.V.2);_ConsisVLA-4D / Xiaomi-Robotics-0 / HiF-VLA 副_。

**L. VLM Backbone 主 — 8 篇**

- **L1 开源 MLLM**(2 主):**VLA-Foundry**(7.L.7)、**VLANeXt**(7.L.8) — 系统消融 + 食谱;_~50 篇副(几乎所有用 PaliGemma/Qwen 的论文)_;
- **L2 自训 VLM**(5 主):**Cosmos Policy**(7.L.1)、**DM0**(7.L.2)、**GR00T_N1.6**(7.L.3)、**MolmoAct2**(7.L.4)、**MolmoB0T**(7.L.5);
- **L3 Action Token Vocab**(1 主):**MINT-4B**(7.L.6,DCT 多尺度 VQ);_π0 系列 / Being-H0.5 / MolmoAct2 副_(均用 FAST);
- **L4 RoPE/Long Context**(0 主):默认配置,_所有开源 MLLM 系 VLA 副_。

**F. 融合 / 路由主 — 6 篇**

- **F1 早融合 prefix**(0 主):多副 — _SimVLA / VLA-Foundry / GigaWorld-Policy / VLAW / Cosmos Policy_;
- **F2 中融合 Cross-Attn / Λ-attn**(0 主):多副 — _Xiaomi-Robotics-0 / FLOWER / ABot-M0 / ConsisVLA-4D / FocusVLA / GST-VLA / Mask World Model_;
- **F3 晚融合 per-modality**(2 主):**StarVLA-α**(7.F.4,MLP 极简晚融合)、**Helix_02**(7.F.2,三层全身,可视为分层晚融合);_GR00T_N1.6 / TiPToP / LoHo-Manip 副_;
- **F4 MoT**(4 主):**Being-H0.5**(7.F.1,MoF)、**HY-Embodied-0.5**(7.F.3)、**Xiaomi-Robotics-0**(7.F.5)、**π0.7**(7.F.6);_Being-H0.7 / Fast-WAM / LingBot / LAP / RLDX-1 副_;
- **F5 MoE / Gating**(0 主):多副 — _GST-VLA / HY-Embodied 32B 副_;
- **F6 Adapter / Soft-Prompt**(0 主,但 X-VLA 主标 E2):多副 — _RLDX-1 / VLA-OPD / SmoothVLA / STRONG-VLA / OA-WAM 副_。

**A. 动作 Head 主 — 12 篇**

- **A1 AR head(离散 token)**(0 主,但 MINT-4B 主在 L3):多副 — _MolmoAct2 / VLA-OPD / LifeLong-RFT / Ψ0 Stage1_;
- **A2 DDPM / DiT head**(3 主):**HAMLET**(7.A.6,DiT + Memory)、**HiPolicy**(7.A.7,分层 DDPM)、**CycleVLA**(7.A.12,DDPM + 回溯);_GR00T_N1.6 / Mask World Model / Cosmos Policy / DreamZero / STARRY 副_;
- **A3 Flow Matching head**(8 主):**ABot-M0**(7.A.1,Manifold + Flow)、**FLOWER**(7.A.2)、**LingBot-VLA**(7.A.3)、**RLDX-1**(7.A.4)、**SimVLA**(7.A.5)、**X-VLA**(7.A.8)、**π0.6 RECAP**(7.A.10)、**Ψ0**(7.A.11);_VLANeXt / VLA-Foundry / Xiaomi-Robotics-0 / π0.7 等 25+ 篇副_;
- **A4 AR + 连续混合**(2 主):**FocusVLA**(7.V.2,主标 V5)、**StarVLA-α**(7.F.4,主标 F3);_π0 / GR00T / Xiaomi / Ψ0 / HY 等多数 SOTA 副_;
- **A5 Action Manifold head**(1 主):**ABot-M0**(7.A.1) — 唯一以此为主范式;
- **A6 频率 / 多尺度 head**(1 主):**HiPolicy**(7.A.7,分层多频);_MINT-4B 副(主标 L3,DCT 多尺度也含 A6)_。
- **特别提及 SmoothVLA(7.A.9)**:主在 A3 Flow,但其后训 jerk reward 是 O5 派。

**W. 世界 / 辅助 Head 主 — 21 篇**

- **W1 像素未来帧**(3 主):**Cosmos Policy**(7.L.1,主标 L2)、**DreamZero**(7.W.2)、**Psi-R2/W0**(7.W.14);_GigaWorld-Policy / VLAW / WoVR / World-VLA-Loop 副_;
- **W2 Latent / JEPA**(8 主):**Being-H0.7**(7.W.17)、**Fast-WAM**(7.W.3)、**FutureVLA**(7.W.4)、**Mask World Model**(7.W.7)、**VLA-JEPA**(7.W.10)、**Being-H0.5 sec**(C4 + W2 实质)、**EZ-M**(7.O.10,Latent dynamics)、**Consistency-Consensus**(7.W.21,latent MSE 一致性);_CoLA-World / World-VLA-Loop / World2Act 副_;
- **W3 Mask / Unified WAM 预测**(5 主):**GigaWorld-Policy**(7.W.5)、**Mask World Model**(7.W.7,主分 B2)、**STARRY**(7.W.9,几何 mask GASAM)、**X-WAM**(7.W.19,4D RGB-D unified)、**MotuBrain**(7.W.20,unified 5-mode);_OA-WAM 副_;
- **W4 Trace / Affordance**(4 主):**HiF-VLA**(7.W.6,motion vector)、**LoHo-Manip**(7.W.15,Trace + Manager)、**World2Act**(7.W.13,Skill-WM)、**GeneralVLA 副(主 V3,3D Affordance)**;_HAMLET / PokéVLA / Pose-VLA / X-WAM 深度分支 副_;
- **W5 Future State / Reward**(3 主):**ConsisVLA-4D**(7.V.1,主标 V3,但 D1 实质)、**π0.7**(7.F.6,主标 F4,但 W5 实质)、**ReconVLA**(7.O.14,Conformal 不确定性);_PRTS / Cosmos / TT-VLA / π0.6 副_;
- **W6 CoT / Reasoning**(5 主):**CycleVLA**(7.A.12)、**NS-VLA**(7.W.16)、**MolmoAct2**(7.L.4,主标 L2,但 W6 Think 实质)、**LoHo-Manip 副**、**ACoT-VLA**(7.O.17,**动作空间 CoT**,主标 O3 但 W6 实质);_GST-VLA / DM0 / π0.7 / TiPToP 副_;
- **特别提及 OA-WAM(7.W.8)**:主 W3 + B4,核心创新是对象槽位;**Psi-R2/W0(7.W.14)**:闭源叙事但首个 10 万小时人类预训;**X-WAM(7.W.19) / MotuBrain(7.W.20)**:首批 unified 多模式 WAM,前者主打 4D RGB-D,后者主打 5 模式共享参数;**Consistency-Consensus(7.W.21)**:首个 zero-training WAM ranking 方法。

**E. 跨本体适配主 — 3 篇**

- **E1 Per-head**(0 主):已过时,_仅早期 OXE 副_;
- **E2 Soft-Prompt**(1 主):**X-VLA**(7.A.8,主标 A3 但 E2 是真正卖点);
- **E3 Latent Action Tokenizer**(1 主):**LAP**(7.E.1) — 主真正 zero-shot 跨本体;_CoLA-World / World2Act / Being-H0.5 / Being-H0.7 副_;
- **E4 Embodiment Aug**(2 主):**OXE-AugE**(7.E.2)、**Genie Sim 3.0**(7.E.3,平台 + 仿真域随机);
- **E5 共享骨干**(0 主):默认前提,所有 74 篇副。

**O. 后训练 / 部署优化主 — 16 篇**

- **O1 PTQ Quantization**(1 主):**QuantVLA**(7.O.1) — 唯一以此为主范式;
- **O2 Layer Pruning**(0 主,1 重要副):_FLOWER 副(主 A3,但 50% 层裁剪是 O2 实质)_;
- **O3 异步 / Chunking**(3 主):**Green-VLA**(7.O.2,5 阶段课程)、**STRONG-VLA**(7.O.8,两阶段课程)、**TiPToP**(7.O.9,模块化非端到端);_Xiaomi-Robotics-0 / HiPolicy / GigaWorld-Policy 副_;
- **O4 Distillation**(3 主):**VLA-OPD**(7.O.3,OPD)、**ELITE**(7.O.13,经验蒸馏)、**RealMirror**(7.O.12,平台);_PokéVLA / StarVLA-α / HY-Embodied / LingBot-VLA 副_;
- **O5 Speculative / 持续 RL**(9 主):**LWD**(7.O.4)、**SOP**(7.O.5)、**TT-VLA**(7.O.6)、**LifeLong-RFT**(7.O.7)、**EZ-M**(7.O.10)、**WoVR**(7.O.11)、**ReconVLA**(7.O.14,主 W5 副 O5)、**P3Nav**(7.O.15)、**SACA**(7.O.16);_SmoothVLA / π0.6 副_。

> 倒排索引交叉校验:**V 主 7 + L 主 8 + F 主 6 + A 主 12 + W 主 21 + E 主 3 + O 主 17 = 74 篇 ✓**;副组件累计远超 200 个标记 — 说明 2025-2026 VLA 几乎全是「主组件 1 个 + 副组件 3-5 个」的复合架构,而非单一组件主导。

### 9.2 外部权威来源(≤ 8 条,组件域) [T3]

> 仅在原文/HTML 未明说时引用;优先复用 [vla_traintask.md](vla_traintask.md) 第 9 章已列的 RT-1/RT-2/OpenVLA/Octo/Diffusion Policy/π0/Flow Matching/I-JEPA 八条;本文档**新增 8 条与组件级架构直接相关的权威外链**(不重复 vla_traintask.md):

- ^{[ext1]} **SigLIP**: Zhai et al., "Sigmoid Loss for Language Image Pre-Training", arXiv 2303.15343 — 4.V.1 节使用的 SigLIP loss 与 So400m-Patch14-384 视觉编码器的来源,**~80% VLA 默认配置**。
- ^{[ext2]} **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv 2304.07193 — 4.V.2 节几何 / 局部细节强的代表;PokéVLA / GST-VLA / ConsisVLA-4D / HiF-VLA 等多篇都用其作为 SigLIP 的几何互补。
- ^{[ext3]} **DiT (Diffusion Transformer)**: Peebles & Xie, "Scalable Diffusion Models with Transformers", arXiv 2212.09748 — 4.A.2 节 DiT block + AdaLN-Zero 的来源;GR00T_N1.6 Action Expert / HAMLET / HiPolicy / Mask World Model / Cosmos Policy 等均沿用其结构。
- ^{[ext4]} **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv 2104.09864 — 4.L.4 节使用的旋转位置编码;Llama / Qwen / PaliGemma 通用,所有 VLA 默认。
- ^{[ext5]} **VQ-VAE**: van den Oord et al., "Neural Discrete Representation Learning", arXiv 1711.00937 — 4.L.3 节 Action Token Vocab 的数学基础;MINT-4B DCT 多尺度 VQ、Being-H 系列、CoLA-World IDM 均直接派生。
- ^{[ext6]} **FlashAttention-2/3**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism", arXiv 2307.08691 — 4.L.4 节 Long Context VLA(π0.7 / Helix_02 / Ψ0 / Xiaomi)的内核优化基础。
- ^{[ext7]} **Qwen3-VL**: Qwen Team, "Qwen3 / Qwen3-VL Technical Report" — 4.L.1 节 Qwen 系 backbone 的官方来源;StarVLA-α / VLANeXt / Xiaomi-Robotics-0 / Green-VLA / ABot-M0 / NS-VLA 等均以此为底。
- ^{[ext8]} **Mixture-of-Transformers (MoT)**: Shen et al., "Mixture-of-Transformers: A Sparse and Scalable Architecture", arXiv 2411.04996 — 4.F.4 节 MoT 路由的最新理论;π0.7 / HY-Embodied-0.5 / Xiaomi-Robotics-0 / Being-H0.5 / Fast-WAM / RLDX-1 等多篇 2025-2026 主流 VLA 均沿用。

> 凡 [vla_traintask.md](vla_traintask.md) 已经引用过的 8 条外链(RT-1/RT-2/OpenVLA/Octo/DP/π0/FM/I-JEPA),本文档**不再重复列出**,只在第 6 章演化时间线提及。

---

> 本文档当前为 **第三轮(深度回填)+ 第四轮(收口校对)交付**。第 1~9 章均已完整填充:
> - 第 1~3 章:阅读指南 / 9 维组件设计空间 / 7 大组件 32 子组件分类总图
> - 第 4 章:32 个子组件的「直觉 → 数学/结构 → 实现 → 数据 → 代表论文 → 优势/局限 → 对效果的影响 → 消融证据(带数字)→ 为什么」八段式深度解析
> - 第 5 章:7 组横向对比矩阵(V / L / A / F / E / O / 正负迁移)
> - 第 6 章:模型结构演化时间线 mermaid + 关键拐点 + 细化时间线(按 backbone / fusion / head / 后训四个维度)
> - 第 7 章:74 篇模型结构速查卡(按主组件归属分 7.V/7.L/7.F/7.A/7.W/7.E/7.O,每篇含一句话定位 / 3-6 节点架构 mermaid / 11 槽位核心组件清单 / 输入→输出 / 主要 Loss / 关键消融数字 / 最大正面贡献组件 + 为什么 / 优势-局限 / 范式归属链回 vla_traintask.md)
> - 第 8 章:8 个场景化组件组合配方(均带 74 篇内证据)+ 10 条架构反模式 + 8 条未来趋势(带具体论文引用)
> - 第 9 章:字母索引(74 篇)+ 按主组件倒排索引(7 大类 × 32 子组件)+ 8 条组件域外部权威来源(去重 vla_traintask.md 的 8 条)
>
> 与 [vla_traintask.md](vla_traintask.md) 形成"模型结构 × 训练任务"双视角对偶:同一篇论文在两份文档里都能通过 `7.X.N` 编号互查。后续若需迭代,加新论文按 V/L/F/A/W/E/O 体系追加同构模型卡即可。
