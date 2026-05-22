
## `IDM`是什么?

IDM (Inverse Dynamics Model) 是合成数据管线中用于从视频反推动作标签的模型 — 给定当前帧和未来帧，预测两帧之间机器人应执行的动作序列。

核心问题：视频生成模型能合成视觉逼真的机器人操作视频，但这些视频没有动作标注，无法直接用于策略训练。IDM 填补这个空缺。

RLDX-1 中的 IDM 规格：
+ 架构：0.1B Diffusion Transformer + SigLIP-2 视觉编码器
+ 训练目标：flow-matching，给定一对输入帧，去噪预测中间动作序列
+ GR-1 版：使用公开预训练 checkpoint（seonghyeonye/IDM_gr1）
+ ALLEX 版：在自有遥操作数据上从头训练，action horizon $H+1=20$，batch 256，60K steps


在合成数据管线中的位置：
```
源视频 → Scene/Task Augmentation → Video Gen (Cosmos-Predict2)
                                        ↓
                                   生成视频 (无动作标签)
                                        ↓
                                   IDM 预测动作序列 a_{t:t+H}
                                        ↓
                               Motion-Consistency Filtering
                               (模拟器回放IDM动作 vs 生成视频对比)
                                        ↓
                                   训练数据 (视频+动作)
```
但 IDM 预测的动作不一定准确，所以论文后面接了 Motion-Consistency Filtering：将 IDM 预测的动作在模拟器中回放，对比回放视频与合成视频的运动一致性，过滤掉不一致的样本。这是整个合成数据管线质量保证的关键环节。
