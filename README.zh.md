# Study 1 — 时间状态在类脑 SNN 中占多大能量份额

**[English](README.md) | [中文](README.zh.md)**

> NSF 提案 *Time as a First-Class Resource* 的 Preliminary Study 1
> 数据采集：2026-11-16
> 数据基础：**SHD 测试集全部 2,264 个样本**

---

## 0. 这份报告回答什么问题

白皮书 Study 1 的目标是：

> "in delay-trained and multi-compartment models, **delay storage and neuron-state traffic, not spikes**, account for a substantial share of energy at matched accuracy. Spike-based estimates therefore misjudge these models."

拆成两个可证伪的判据：

| 判据 | 定义 | 白皮书门槛 |
|---|---|---|
| **(a)** | 时间状态（state memory + delay storage + leakage）占单次推理能量的比例 | ≥ 10% |
| **(b)** | 按 spike 数或 SOP 排序 ≠ 按完整模型排序 | 至少一个比较 |

白皮书原文：

> "Temporal state is a non-trivial share of energy in **at least one** brain-inspired model, and the SOP-only ranking differs from the full-model ranking in **at least one comparison**."

---

## 1. 三个模型

| 模型 | 来源 | 参数量 | 神经元 | epochs | checkpoint 口径 |
|---|---|---|---|---|---|
| **delay 352** | DelRec（`alexmaxad/DelRec`），学习突触延迟 | 306,612 | 724（含输出层 LIF） | 7 | best validation |
| **plain LIF** | PMSN（`xychen-comp/...-PMSN`），$d_{state}=1$ | 187,988 | 704 | 5 | last epoch |
| **multi-compartment** | PMSN，$d_{state}=4$ | 200,660 | 704 | 3 | last epoch |

**为什么是这三个**：对应白皮书要求的 three small SNNs on SHD——plain LIF 基线、学习延迟的网络、多隔室网络。PMSN 的 $d_{state}=1$ 是一阶 IIR，等价于 LIF，所以与 $d_{state}=4$ 共享同一代码框架，架构差异只来自隔室数。

**为什么 PMSN 取最后一轮而不是 best**：MC 在第 1 轮达峰、第 2 轮回落约 1 点。取 best 会把它抬高 1.5–2 点，与 LIF 的差距拉大到 3–4 点，破坏 matched accuracy。

### 三个时间分辨率

同一 1.4 s 覆盖，帧长不同：

| T | 帧长 | delay | LIF | MC | 跨度 |
|---|---|---|---|---|---|
| 50 | 28.0 ms | 86.56% | 88.96% | 91.21% | 4.65 点 |
| **100** | **14.0 ms** | **87.24%** | **86.40%** | **88.30%** | **1.90 点** ✅ |
| **140** | **10.0 ms** | **87.99%** | **87.10%** | **88.96%** | **1.86 点** ✅ |

**T=100 和 T=140 是主结果**（跨度 <2 点）；T=50 保留在表里作交叉验证，但三模型精度差 4.65 点。

---

## 2. 我们在原程序上改了什么，怎么训练的

### 2.0 统一时间轴的概括

**三个模型、三个分辨率、共 9 次训练，全部使用同一个固定时长分箱：**

$$\text{duration} = \frac{1400\ \text{ms}}{T}\ \text{ms/frame}, \quad T \in \{50, 100, 140\}$$

**所以三个模型的第 $t$ 帧始终对应同一段物理时间 $[t \cdot \text{duration},\ (t+1) \cdot \text{duration})$。**

**这是三方可比的前提**：两个开源代码库的默认分箱方式不同（DelRec 用帧数、PMSN 用 `linspace` 等分），如果不改到同一口径，“同一 T”下的帧长不同，SOP、延迟值、state 访问次数都失去可比性。具体改动见 §2.1 与 §2.2。

### 2.1 DelRec：三处补丁

| # | 位置 | 改什么 | 不改会怎样 |
|---|---|---|---|
| 1 | `configs/perf_SHD.py` | 按 `time_step` 重标定全部延迟超参 | 换分辨率后延迟的物理语义漂移 |
| 2 | `src/delrec/datasets/loaders.py` | 用 `fixed_collate_T{T}` 替换 `pad_sequence_collate`；分箱改 `duration=` | **用等分分箱（`frames_number`）精度从 87% 掉到 70.40%** |
| 3 | `src/delrec/utils.py` | 时间维池化 `max` → `mean` | 与官方 checkpoint 不一致 |

**为什么分箱必须固定时长**：DelRec 的延迟是“离散帧数”且跨样本共享。若用等分分箱，同一帧在不同长度的样本上覆盖不同的物理时间，延迟语义被破坏。实测 mismatch 验证：固定时长分箱与 `integrate_events_by_fixed_duration_shd(duration=...)` 逐元素一致。

### 2.1b delay 网络的训练流程改造（原程序 → 我们）

**原程序的行为：**

| 环节 | DelRec 原始 |
|---|---|
| 分箱 | 由 `config.time_step` 传给 `integrate_events_by_fixed_duration_shd(duration=...)`，但下游 `collate_fn=pad_sequence_collate` 保留每个样本的实际帧数 |
| batch 组装 | 变长序列 padding，**同一 batch 内不同样本帧数不同** |
| 时间维池化 | `m, _ = torch.max(output, 0)`（取时间维最大值） |
| 延迟超参 | 硬编码为 `@10 ms` 的值：`max_rec_delay=30`、`max_feedforward_delay=250//time_step`、`sigma_init=10.0`、`init_recdel_offset=10`、`delay_std_init=15` |

**我们改成：**

| 环节 | 改造后 | 为什么 |
|---|---|---|
| 分箱 | **固定 `duration = 1400/T` ms**，超长裁掉、不足隐式补零 | 保“一帧 = 固定物理时长” |
| batch 组装 | **`fixed_collate_T{T}`：裁剪/零填充到恒定的 T 帧** | 延迟按帧计，若 batch 内帧数不齐，同一延迟值的物理时长会变 |
| 时间维池化 | **`m = torch.mean(output, 0)`** | 与官方 checkpoint 的口径一致 |
| 延迟超参 | **全部按物理时间重新推导**（除以 `time_step`），见下表 | 换分辨率时延迟的语义保持 300 ms / 250 ms / 100 ms / 150 ms 不变 |

**注意 `max_feedforward_delay` 的原始写法是 `250//time_step`——它已经做了物理时间换算，但其余四个超参是硬编码的 `@10 ms` 值。如果只改 `time_step` 而不重标定这五个，T=50 时实际延迟会是 300 ms 的两倍以上。**

**超参按物理时间推导**（除 `time_step` 换算成帧数）：

| 参数 | 物理时间 | T=50 | T=100 | T=140 |
|---|---|---|---|---|
| `time_step` | — | 28 | 14 | 10 |
| `max_rec_delay` | 300 ms | 11 | 21 | 30 |
| `max_feedforward_delay` | 250 ms | 9 | 17 | 25 |
| `sigma_init` | 100 ms | 3.5 | 7.0 | 10.0 |
| `init_recdel_offset` | 100 ms | 4 | 7 | 10 |
| `delay_std_init` | 150 ms | 5 | 11 | 15 |

`patch_delrec.py --T {T}` 会自动推导这张表。

### 2.2 PMSN：两处补丁

| # | 改什么 | 不改会怎样 |
|---|---|---|
| 1 | `np.linspace` 等分分箱 → 固定 `duration = 1400/T` ms | 与 DelRec 的“一帧”不对应，两族无法同 T 比较 |
| 2 | `bn=False` → `bn=True` | checkpoint 是 `bn=True` 训的，不匹配则 `load_state_dict` 报错 |

**PMSN 是连续时间常数的 SSM，对分箱方式不敏感**（实测换分箱后精度不变），改分箱纯粹是为了与 delay 网络对齐口径。

### 2.3 训练命令

见 `patches/TRAINING.md`。核心命令：

```bash
# delay（必须 --kernel v2，否则 CuPy shared memory 超限）
python experiments/train.py --dataset shd --model SNN_recurrent_delays \
  --seeds 1111 --epochs 7 --kernel v2 --out /content/work/exp/delay_w352_T100_ts14_7ep

# PMSN
python SHD_PMSN.py --data_path /content/shd --T 100 --d_state 1 --epochs 5 \
  --ckpt_path /content/work/exp/pmsn_lif_T100_fill_5ep --device cuda:0
```

### 2.4 推理时没有 dropout

三个模型训练时都用了 dropout（delay: 前馈 0.4 / 循环 0.2；PMSN: 0.2），**但所有测量都在 `eval()` 下进行，dropout 恒等、成本为零**。统计到的 SOP / spike / flit_hops 全部来自无 dropout 的前向。

---

## 3. Mesh 设计与参考

### 3.1 拓扑

| 档位 | 核数 | 路由 | hop 范围 |
|---|---|---|---|
| 2×2 | 4 | 2D mesh，XY 维序 | 1–2 |
| 4×4 | 16 | 同上 | 1–6 |
| 8×8 | 64 | 同上 | 1–14 |

**hop 数 = 曼哈顿距离**（`net/topology.py` 的最短路径表）。

**没有 2core 档**：1D 直连的 hop 分布（全是 1）与 2D mesh 有长尾的分布性质不同，四档不能在同一拓扑族下横向比较，所以移除。

### 3.2 神经元到核的映射

**顺序映射**（Loihi Nx SDK 与 SpiNNaker 编译器的默认行为）：

$$\text{core}(i) = \left\lfloor \frac{i \cdot n_{cores}}{n_{total}} \right\rfloor, \quad n_{total} = 724$$

**这不是通信优化算法，而是确定性、可复现、不依赖连接图的基线。**

**已知代价**：输出层只有 20 个神经元，全部落在最后一个核。

| mesh | hidden1 占核 | hidden2 占核 | **out 占核** |
|---|---|---|---|
| 2×2 | 0–1 | 1–3 | **3** |
| 4×4 | 0–7 | 7–15 | **15** |
| 8×8 | 0–31 | 31–63 | **62–63** |

**所以 hidden2 → out 的流量汇聚到 1–2 个核，形成结构性热点。**这是“未优化上界”的一部分，见 §7。

### 3.3 通信的计数方式

**一个 spike 发往“所有含下游神经元的核”（multicast 语义），跳数 = 到每个目标核的最短路径长度之和。**

因为 hidden1 与 hidden2 全连接，每个 hidden1 spike 确实需要送达所有含 hidden2 的核，所以 multicast 语义正确。

**但跳数累加等价于“逐目标核的独立单播”，是真实 multicast 树的上界**（见 §7）。

### 3.4 每核容量约束

| 项 | 8×8 下的值 | Loihi 上界 |
|---|---|---|
| 神经元/核 | 11（hidden1/hidden2）、20（out） | 1,024 |
| 突触/核 | ~3,800（hidden2）、7,040（out） | 128 K |

**远低于限值，容量不构成约束。**

---

## 4. 计算方法

### 4.0 为什么自己写解析模型，而不用现成的模拟器

**我们考察了现有的开源 SNN 能耗评估工具，没有找到能同时满足本研究要求的。**要求很具体但不宽：一个 45 nm **数字 CMOS** 的代价模型，覆盖 Study 1 的五类（含 **delay storage** 与 **stored state 的漏电**），能处理**学习延迟**与**多隔室**神经元，并能由两个上游代码库的 PyTorch 模型驱动。

| 工具 | 它建模什么 | 为什么不适用 |
|---|---|---|
| **SpikeSim**（Yale） | 模拟 IMC crossbar（RRAM/SRAM），65 nm | 架构不匹配：crossbar tile 与 ADC/DIFF 流水线，不是数字 MAC 阵列。它的延迟模型（$\alpha$、`passes`）在本工作没有对应物 |
| **NeuroSim** | 模拟/IMC crossbar | 同 SpikeSim |
| **BookSim** | NoC 的延迟与吞吐 | 只建模互连；没有 SNN 能耗，也没有状态与延迟存储 |
| **snnTorch / SpikingJelly** | 训练框架 | 至多给出 MAC/SOP 计数；没有硬件能耗模型 |
| **Lava / Nx SDK** | 面向 Loihi 1 | Loihi 1 的神经元模型是 CUBA LIF，**恰好只有两个内部状态变量**——突触响应电流 $u_i(t)$ 和膜电位 $v_i(t)$（Davies et al. 2018, p.3）。我们的多隔室模型需要 **4 隔室 × 2 个实状态变量 = 8**，在 Loihi 1 上根本无法映射。详见下文 |
| **NeuroBench** | 硬件无关指标 | 明确只提供 SOP 与激活稀疏度，**不做**硬件能耗 |
| **SNNToolBox / N2D2** | 量化与编译 | 能耗模块较粗（只算 SOP） |

**结论**：没有现成工具同时具备（i）45 nm 数字 CMOS 的代价基准、（ii）delay storage 与 state memory 的分项核算、（iii）对学习延迟与多隔室模型的支持、（iv）从这两个训练代码库出发的通路。**所以采用透明的解析模型——每一项都是「计数值 × 文献单价」**，这也正是白皮书 Study 1 Setup 所要求的（"a transparent CMOS energy model with characterized per-operation, per-access, and per-bit-of-storage costs"）。

#### Loihi 1 的结构性限制（为什么它的能耗数字不能直接挪用）

**除了工艺节点，还有三条硬性架构约束使 Loihi 1 无法作为本研究的平台：**

| 约束 | Loihi 1 | 本研究需要 |
|---|---|---|
| **每神经元状态变量数** | **2 个**——$u_i$（突触电流）与 $v_i$（膜电位），固定的 CUBA LIF（Davies et al. 2018, p.3）。它支持树突隔室，但每个隔室携带的是同一套固定 $u/v$ 动力学，没法实例化任意 $N$ 阶的状态空间模型 | **8 个**（4 隔室 PMSN，每隔室实部+虚部）。**装不下** |
| **突触延迟** | 5 元组 $(i, j, \text{wgt}, \text{dly}, \text{tag})$ 里的一个字段——每个**突触**一个延迟值 | **每神经元**一个学习延迟，且带半宽 $\sigma$ 的三角核（DelRec）。语义不同，Loihi 无法表达核的展宽 |
| **时间模型** | 异步、无全局时钟（$t_{NoC}$ 不是设计参数），每核独立遍历自己的神经元隔室 | 时钟驱动、按时间步更新，且有明确的 $t_{inf} = \text{MAC 数} / (n_{cores} F)$ |

**对复用 Loihi 实测值的后果：**

- **23.6 pJ / 突触尖峰操作** 是端到端实测。它把权重 RMW、状态更新、神经元更新、路由打包在一起，**无法拆分**成 Study 1 的五类（synaptic / communication / delay / state / leakage）。用它会把 Study 1 要测的那个区分本身抹掉。
- **81 / 52 pJ / 神经元更新** 同样包含对该神经元状态存储的完整 RMW 循环，无法单独剥离出延迟缓冲与多隔室两项。
- **3.0 / 4.0 pJ / tile hop** **是**可用的——它是唯一一个作用域（router + link、一个 flit）与我们的 `flit_hops` 定义吻合的量，我们把它作为 $E_{hop}$（§4.4），并用 Gorski 的 45 nm 路由器功耗交叉验证。

**所以诚实的定位是**：Loihi 1 提供一个可迁移的单价（$E_{hop}$）和一个定性佐证（大 mesh 下通信主导），但它既装不下 delay 网络也装不下多隔室网络，且其复合能耗数字无法拆成本研究所需的分项核算。

**这是一个真实的局限，而不只是图方便**：周期精确的模拟器还能捕捉缓冲区竞争、仲裁与拥塞，而这些目前都在模型之外（§8.2、§8.3）。这也意味着**绝对**数值不能与已发表的基于工具的结果直接比较。

### 4.1 时间模型

$$t_{inf} = \frac{\text{SOP} + N_{neur} \cdot T \cdot d_{state}}{n_{cores} \cdot F \cdot \text{par}}, \quad F = 2\ \text{GHz},\ \text{par} = 1$$

**物理含义**：总 MAC 数除以（核数 × 频率 × 每周期并行度）。

**`par = 1` 意味着每核每周期完成一个 MAC——这是乐观假设**，见 §7 限制 1。

### 4.2 Study 1 的五个类别

按白皮书 Study 1 的 Procedure（p.12）：

| # | 类别 | 公式 |
|---|---|---|
| 1 | **synaptic operations** | $\text{SOP} \times E_{AC} + N_{neur} T d_{state} \times E_{MAC}$ |
| 2 | **communication** | $fh \times E_{hop}$ |
| 3 | **delay storage** | $N_{delay} \cdot T \cdot 2 \times E_{access}$ |
| 4 | **state memory** | $N_{neur} T d_{state} \cdot 2 \times E_{access}$ |
| 5 | **leakage** | $bits_{state+buffer} \times P_{leak} \times t_{inf}$ |

**不计 $\int P_{idle}dt$、时钟树、数据转换、feature extraction**——那是白皮书 §4.3 的完整公式，Study 1 不要求。

### 4.3 权重漏电单列

**权重也占 SRAM、也漏电，但它与时间机制无关。**把它计入“时间状态占比”会稀释信号（它占 bits 的 99.6%）。

**所以权重漏电单独报告，不并入判据 (a) 的分子。**

### 4.4 单价与出处

| 参数 | 值 | 出处 |
|---|---|---|
| $E_{AC}$ | **0.9 pJ** | Horowitz 2014, Fig 1.1.9（45nm 0.9V，32-bit FP Add） |
| $E_{MAC}$ | **4.6 pJ** | 同上（FP Add 0.9 + FP Mult 3.7） |
| $E_{access}$ | **10 pJ** | 同上（8 KB SRAM read） |
| $P_{leak}$ | **11.9 nW/bit** | Horowitz 2014, p.2（100 mW/MB，45–32nm） |
| $E_{hop}$ | **3.5 pJ** | Davies et al. 2018（Loihi）, Table 2（3.0/4.0 pJ/tile hop） |
| **权重访问** | **0** | Dampfhoffer 2023, L2372（spatially expanded，`remove the weights accesses`） |
| buffer 位宽 | **1 bit/slot** | `delay_layers_pytorch.py:299` |
| $F$ | **2 GHz** | 自定（测量基准） |
| par | **1** | 自定（无文献） |

**$E_{hop}$ 的交叉验证**：由 Gorski 2016 的 45nm 路由器 $P_{dyn}$（$P_{total}$ 16–40 mW 减去主导的 $P_{stat}$，约 5–7.3 mW）除以吞吐 $1.5\times10^9$ flit/s，得 3.3–4.9 pJ/跳，与 Loihi 实测一致。

---

## 5. 计数结果（不含单价）

**以下全部是 SHD 测试集 2,264 个样本的**单样本平均**。**

### 5.1 计数口径的文献依据

**四类计数的定义全部来自 Dampfhoffer et al. 2023 的 LIF+inst 能量模型（式 3.4 与 3.7）：**

**突触操作（式 3.4，L2007）：**

$$E_{IF+inst} = N_{syn} \cdot N_{spikes/syn} \cdot (E_R^{weight} + E_R^{state} + E_W^{state} + E_{AC})$$

**每时间步的神经元更新（式 3.7，L2037–2038）：**

$$E_{LIF+inst} = \underbrace{N_{syn} N_{spikes/syn}(\cdots)}_{\text{事件驱动}} + \underbrace{N_{neur} T (E_R^{state} + E_W^{state} + E_{MAC})}_{\text{时钟驱动}}$$

**据此四个计数的出处：**

| 计数 | 我们的公式 | Dampfhoffer 依据 |
|---|---|---|
| **SOP** | spike × fanout | 式 3.4 的 $N_{syn} N_{spikes/syn}$ 项 |
| **neuron ops** | $N_{neur} \times T \times d_{state}$ | 式 3.7 的 $N_{neur} T$，隔室数 $d_{state}$ 是 PMSN 的扩展 |
| **state 访问** | neuron ops **× 2** | 式 3.7 的 $E_R^{state} + E_W^{state}$；L2032–2034 明说“**reading the state, multiplying it with a constant, and writing back the result**” |
| **buffer 访问** | $N_{delay} \times T \times 2$ | 无直接文献，是按 state 访问的读+写模式类比（读到期 slot + 写当前 spike） |

**为什么只算式 3.7 的第二项、不算第一项**：Dampfhoffer L2039–2044 说明这两项是两条**互斥**的实现路线——第一项对应事件驱动（spike 到达才更新状态），第二项对应时钟驱动（每步更新所有神经元），原文并指出“the update at each timestep is more efficient”。我们的模型是时钟驱动的（`delay_layers_pytorch.py:95` 的 `single_step_forward`），所以只有第二项。

**以下全部是 SHD 测试集 2,264 个样本的**单样本平均**。**

### 5.2 原始计数

| T | 模型 | spikes | SOP | neuron ops | state 访问 | buffer 访问 |
|---|---|---|---|---|---|---|
| **50** | delay | 3,909 | 1,431,230 | 36,200 | 72,400 | 35,200 |
| | LIF | 8,688 | 1,389,789 | 35,200 | 70,400 | 0 |
| | MC | 9,565 | 1,533,348 | 140,800 | 281,600 | 0 |
| **100** | delay | 7,667 | 2,804,570 | 72,400 | 144,800 | 70,400 |
| | LIF | 13,930 | 1,808,584 | 70,400 | 140,800 | 0 |
| | MC | 15,672 | 2,343,106 | 281,600 | 563,200 | 0 |
| **140** | delay | 10,891 | 3,985,830 | 101,360 | 202,720 | 98,560 |
| | LIF | 19,523 | 2,589,276 | 98,560 | 197,120 | 0 |
| | MC | 21,026 | 2,937,174 | 394,240 | 788,480 | 0 |

**neuron ops = $N_{neur} \times T \times d_{state}$，state 访问 = neuron ops × 2（读+写），buffer 访问 = $N_{delay} \times T \times 2$。**

### 5.3 存储位数

| 模型 | weight bits | state bits | buffer bits (T=100) |
|---|---|---|---|
| delay | 9,811,584 | 23,168 | 14,432 |
| LIF | 6,015,616 | 22,528 | 0 |
| MC | 6,421,120 | 90,112 | 0 |

**权重占 bits 总量的 99.6%，是漏电能量的主导项。**

### 5.4 通信跳数

| T | 模型 | 2×2 | 4×4 | 8×8 |
|---|---|---|---|---|
| 50 | delay | 12,468 | 85,422 | 614,151 |
| | LIF | 28,587 | 198,392 | 1,436,650 |
| | MC | 31,381 | 220,566 | 1,591,048 |
| 100 | delay | 24,572 | 169,371 | 1,220,020 |
| | LIF | 45,118 | 308,479 | 2,213,959 |
| | MC | 50,835 | 357,068 | 2,576,434 |
| 140 | delay | 34,980 | 238,466 | 1,706,995 |
| | LIF | 63,203 | 434,092 | 3,125,954 |
| | MC | 68,429 | 474,574 | 3,409,407 |

**delay 的跳数只有 LIF 的 43–56%——更稀疏的 spike 换来了更少的通信。**

### 5.5 乘上单价后的能耗（µJ / 推理）

| T | mesh | 模型 | synaptic | state | delay | leakage | comm | **合计** | weights leak |
|---|---|---|---|---|---|---|---|---|---|
| **50** | 2×2 | delay | 1.455 | 0.724 | 0.352 | 0.071 | 0.044 | **2.646** | 21.417 |
| | | LIF | 1.413 | 0.704 | 0 | 0.048 | 0.100 | **2.265** | 12.751 |
| | | MC | 2.028 | 2.816 | 0 | 0.224 | 0.110 | **5.178** | 15.990 |
| | 4×4 | delay | 1.455 | 0.724 | 0.352 | 0.018 | 0.299 | **2.847** | 5.354 |
| | | LIF | 1.413 | 0.704 | 0 | 0.012 | 0.694 | **2.823** | 3.188 |
| | | MC | 2.028 | 2.816 | 0 | 0.056 | 0.772 | **5.672** | 3.998 |
| | 8×8 | delay | 1.455 | 0.724 | 0.352 | 0.005 | 2.150 | **4.685** | 1.339 |
| | | LIF | 1.413 | 0.704 | 0 | 0.003 | 5.028 | **7.148** | 0.797 |
| | | MC | 2.028 | 2.816 | 0 | 0.014 | 5.569 | **10.426** | 0.999 |
| **100** | 2×2 | delay | 2.857 | 1.448 | 0.704 | 0.161 | 0.086 | **5.256** | 41.989 |
| | | LIF | 1.952 | 1.408 | 0 | 0.063 | 0.158 | **3.580** | 16.814 |
| | | MC | 3.404 | 5.632 | 0 | 0.352 | 0.178 | **9.566** | 25.070 |
| | 4×4 | delay | 2.857 | 1.448 | 0.704 | 0.040 | 0.593 | **5.642** | 10.497 |
| | | LIF | 1.952 | 1.408 | 0 | 0.016 | 1.080 | **4.455** | 4.203 |
| | | MC | 3.404 | 5.632 | 0 | 0.088 | 1.250 | **10.374** | 6.267 |
| | 8×8 | delay | 2.857 | 1.448 | 0.704 | 0.010 | 4.270 | **9.289** | 2.624 |
| | | LIF | 1.952 | 1.408 | 0 | 0.004 | 7.749 | **11.112** | 1.051 |
| | | MC | 3.404 | 5.632 | 0 | 0.022 | 9.018 | **18.076** | 1.567 |
| **140** | 2×2 | delay | 4.053 | 2.027 | 0.986 | 0.244 | 0.122 | **7.432** | 59.651 |
| | | LIF | 2.783 | 1.971 | 0 | 0.090 | 0.221 | **5.066** | 24.051 |
| | | MC | 4.457 | 7.885 | 0 | 0.447 | 0.240 | **13.028** | 31.820 |
| | 4×4 | delay | 4.053 | 2.027 | 0.986 | 0.061 | 0.835 | **7.962** | 14.913 |
| | | LIF | 2.783 | 1.971 | 0 | 0.023 | 1.519 | **6.297** | 6.013 |
| | | MC | 4.457 | 7.885 | 0 | 0.112 | 1.661 | **14.114** | 7.955 |
| | 8×8 | delay | 4.053 | 2.027 | 0.986 | 0.015 | 5.974 | **13.056** | 3.728 |
| | | LIF | 2.783 | 1.971 | 0 | 0.006 | 10.941 | **15.701** | 1.503 |
| | | MC | 4.457 | 7.885 | 0 | 0.028 | 11.933 | **24.303** | 1.989 |

**`weights leak` 单列，不进入“合计”。**

---

## 6. 判据结果

### 6.1 (a) 时间状态占比 ≥ 10%

**9 个配置中 8 个成立。**

| T | mesh | delay | LIF | MC | (a) |
|---|---|---|---|---|---|
| 50 | 2×2 | 43.4% | 33.2% | 58.7% | ✅ |
| 50 | 4×4 | 38.4% | 25.4% | 50.6% | ✅ |
| **50** | **8×8** | 23.1% | **9.9%** | 27.1% | ❌ |
| 100 | 2×2 | 44.0% | 41.1% | 62.6% | ✅ |
| 100 | 4×4 | 38.9% | 32.0% | 55.1% | ✅ |
| 100 | 8×8 | 23.3% | 12.7% | 31.3% | ✅ |
| 140 | 2×2 | 43.8% | 40.7% | 64.0% | ✅ |
| 140 | 4×4 | 38.6% | 31.7% | 56.7% | ✅ |
| 140 | 8×8 | 23.2% | 12.6% | 32.6% | ✅ |

**唯一失败：T=50 的 8×8，LIF 的 9.9% 差 0.1 个点到门槛。**原因：LIF 没有 buffer、$d_{state}=1$，在 64 核下 $t_{inf}$ 只有 11.5 µs，leakage 几乎归零。

#### 关于判据 (a) 的口径

**白皮书原文的门槛是“in *at least one* brain-inspired model”**——即在给定配置下，三个模型里**任意一个**超过 10% 就算成立。

**上面的表用的是更严格的口径**：要求**三个模型全部**超过 10%，所以 T=50 的 8×8 被标成 ❌。

**按白皮书自己的措辞，(a) 是 9/9 全部成立**——就在那个格子里，delay 模型是 **23.1%**，多隔室模型是 **27.1%**。

| 读法 | (a) 成立数 |
|---|---|
| 白皮书措辞：**任意一个**模型 ≥ 10% | **9 / 9** |
| 上表用的更严格读法：**全部**模型 ≥ 10% | 8 / 9 |

**九个配置 × 三个模型的最低值是 9.9%**（LIF，T=50，8×8）；**每个配置内的最大值是 27.1–64.0%**。所以 Study 1 的前提——时间状态在至少一个类脑模型里占非平凡份额——在两种读法下都成立。

**`results/study1_data.json` 是按更严格的 `all` 口径生成的**（脚本里 `a` 字段是 `all(...)`）；但每个模型的 `ts_share` 都存进了 JSON，所以两种读法都能从 JSON 直接重算，不需要重跑测量。

### 6.2 (b) 排序翻转

| T | mesh | spike 序 | SOP 序 | 能量序 | spike≠E | SOP≠E |
|---|---|---|---|---|---|---|
| 50 | 2×2 | DL<LIF<MC | LIF<DL<MC | LIF<DL<MC | **✅** | ❌ |
| 50 | 4×4 | DL<LIF<MC | LIF<DL<MC | LIF<DL<MC | **✅** | ❌ |
| 50 | 8×8 | DL<LIF<MC | LIF<DL<MC | **DL**<LIF<MC | ❌ | **✅** |
| 100 | 2×2 | DL<LIF<MC | LIF<MC<**DL** | LIF<**DL**<MC | **✅** | **✅** |
| 100 | 4×4 | DL<LIF<MC | LIF<MC<**DL** | LIF<**DL**<MC | **✅** | **✅** |
| 100 | 8×8 | DL<LIF<MC | LIF<MC<**DL** | **DL**<LIF<MC | ❌ | **✅** |
| 140 | 2×2 | DL<LIF<MC | LIF<MC<**DL** | LIF<**DL**<MC | **✅** | **✅** |
| 140 | 4×4 | DL<LIF<MC | LIF<MC<**DL** | LIF<**DL**<MC | **✅** | **✅** |
| 140 | 8×8 | DL<LIF<MC | LIF<MC<**DL** | **DL**<LIF<MC | ❌ | **✅** |

**任一翻转：9/9。**

### 6.3 汇总

| 判据 | 成立（严格口径） | 成立（白皮书口径） |
|---|---|---|
| (a) ts ≥ 10% | 8/9 | **9/9** |
| (b) spike ≠ 能量 | 6/9 | 6/9 |
| (b) SOP ≠ 能量 | 7/9 | 7/9 |
| **(b) 任一翻转** | **9/9** | **9/9** |

---

## 7. 分析

### 7.1 两个简化估计从相反方向错判 delay

**T=100, 2×2：**

| 估计方式 | 排序 | 对 delay 的判断 |
|---|---|---|
| **spike 数** | delay(7.7k) < LIF(13.9k) < MC(15.7k) | **最省**（实际第二） |
| **SOP** | LIF(1.81M) < MC(2.34M) < delay(2.80M) | **最贵**（实际第二） |
| **完整模型** | LIF(3.58) < delay(5.26) < MC(9.57) | 第二 |

**为什么相反：**

- **spike 数**看到的是 delay 的稀疏性（3909–10891，是 LIF 的 45–56%），看不到它的 buffer 成本（0.70 µJ，占自身 13%）。
- **SOP** 看到的是 delay 的 recurrent fanout = 372（20 输出 + 352 循环），把 sparse 的优势抹掉，但看不到多隔室网络的 state 成本。

### 7.2 MC 的多隔室成本只在完整模型里可见

| T=100, 2×2 | delay | LIF | **MC** |
|---|---|---|---|
| spike | 7,667 | 13,930 | 15,672 |
| SOP | 2.80M | 1.81M | 2.34M |
| **state** | 1.448 | 1.408 | **5.632（4.0×）** |
| **leakage** | 0.161 | 0.063 | **0.352（5.6×）** |

**$d_{state}=4$ 让 state 访问和 bits 都 ×4，但 SOP 里完全没有这个因子。**

### 7.3 大 mesh 下通信取代 state 成为主导

| T=100, 8×8 | delay | LIF | MC |
|---|---|---|---|
| comm | **4.270** | 7.749 | 9.018 |
| comm 占比 | 46% | 70% | 50% |

**delay 的跳数只有 LIF 的 55%，所以在 64 核下它反而最便宜（9.29 vs 11.11）。**

**此时 SOP 彻底失效：它把 delay 判成最贵（2.80M），实际最便宜。**

### 7.4 时间分辨率的影响

**delay 的精度随帧长变细单调上升（86.56 → 87.24 → 87.99），PMSN 两个变体都是 28 ms 最好。**

**这说明两类模型的时间机制对分辨率的响应方向不同：**

| | 机制 | 偏好 |
|---|---|---|
| delay | 离散帧级延迟 | **细分辨率**（帧越细，延迟可表达的时间粒度越细） |
| PMSN | 连续时间常数 | **28 ms**（更匹配 SHD 音素时长 50–200 ms） |

**这正是 Metrics ③（state cost vs time-step resolution）要量化的内容。**

---

## 8. placement 与功耗计算方法会影响结果

**这是本报告最重要的免责声明：绝对数值依赖下面 5 个建模选择，而这些选择都有替代方案。**

### 8.1 placement 未优化

**顺序映射让输出层全部集中在最后一个核（4×4 的核 15、8×8 的核 62–63）。**

**影响**：hidden2 → out 的流量形成热点，hop 数被系统性抬高。**若换成通信感知的 placement（如把输出层摊开），flit_hops 会下降，comm 项随之下降。**

**8×8 的 comm 占总量 46–70%，所以这一项对结论影响最大。**

### 8.2 multicast 按单播上界计算

**当前跳数 = 逐目标核最短路径之和，等价于“每个目标核发一次单播”。**

**真实的 NoC multicast 用复制树，只在路径分叉处复制。**

**影响**：flit_hops 被高估 2–3 倍，comm 项同比例高估。**但三个模型被同一比例放大，所以排序不变。**

### 8.3 同一时间步的多个 spike 未聚合

**当前每个 spike 独立计数。物理上，同一源核在同一时间步发往同一目标集的多个 spike 可打包进一个 flit 的 payload。**

**影响**：与 8.2 同向，进一步高估 comm。

### 8.4 时间模型 par = 1 无文献依据

**假设 32-bit FP MAC 在 2 GHz 下单周期完成。**

**影响**：$t_{inf}$ 若实际需要 4 周期，会高 4 倍，$P_{leak} \times t_{inf}$ 同比上升。

**但 Study 1 的 leakage 只占 0.2–3.7%（不含权重），所以 (a)(b) 的结论不受影响。权重漏电（41.99 µJ @ T=100, 2×2）会受严重影响，但它是单列的。**

### 8.5 工艺混用

| 参数 | 工艺 |
|---|---|
| $E_{AC}$ / $E_{MAC}$ / $E_{access}$ / $P_{leak}$ | **45 nm** |
| $E_{hop}$ | **14 nm**（Loihi） |

**按节点缩放（14→45 nm 约 2.75×）会把 $E_{hop}$ 变成 ~9.6 pJ/跳，comm 项同比增长。**

### 8.6 所以该怎么读这些数字

**可信的：**
- 三个模型的相对排序（在同一建模假设下）
- 时间状态占比的数量级（10% 量级，不是 1% 也不是 90%）
- “简化估计会错判”这一结论

**不可信的：**
- 绝对 µJ 数值（可差 2–3 倍）
- 任何“比某某芯片省 X 倍”的跨工作对比

---

## 9. 复现方法

### 9.1 依赖

- Python 3.11+，PyTorch，h5py，numpy，matplotlib
- DelRec 源码（`alexmaxad/DelRec`）与 SHD 数据集
- 三个模型的 checkpoint（路径见 §9.4）

### 9.2 从零复现

```bash
# 1. 训练（约 2 小时，见 patches/TRAINING.md 的完整命令）
python patches/patch_delrec.py --root /content/DelRec --T 100
python patches/patch_pmsn.py   --root /content/PMSN   --T 100
# ... 9 次训练

# 2. 测量 + 能量计算（约 12 分钟）
#    注意：尽管文件名带 10samples，默认跑的是全部 2,264 个样本
python scripts/study1_10samples.py

# 3. 作图（约 5 秒）
python scripts/plot_study1.py

# 4. 数据一致性检查（10 样本子集 vs 全部样本，约 2 分钟）
python scripts/check_10_samples.py
```

**关于文件名 `study1_10samples.py` 的说明**：这个名字是历史遗留——脚本最初只是为了“先跑 10 个样本看看”。**它现在的默认行为是跑全部 2,264 个样本**（`N_TEST = -1`，`--n` 参数默认也是 `-1`）。本报告的每一个数字都来自全部 2,264 个样本。要快速冒烟测试用 `--n 10`，要导出 JSON 结果用 `--save <路径>`。另一个脚本 `check_10_samples.py` 是交叉验证工具（对比 10 样本子集与全量，见 §9.3），**它不是本报告数据的来源**。

### 9.3 数据可靠性

**10 样本 vs 2,264 样本的交叉验证：**

| 模型 | T | SOP 比率 | flit_hops 比率 |
|---|---|---|---|
| delay | 50 / 100 / 140 | 1.010 / 0.992 / 0.979 | 1.009 / 0.991 / 0.981 |
| LIF | 50 / 100 / 140 | 1.004 / 1.015 / 1.001 | 1.014 / 1.013 / 1.000 |
| MC | 50 / 100 / 140 | 1.014 / 1.002 / 1.007 | 1.018 / 1.003 / 1.012 |

**全部偏差 <2.1%，说明 2,264 样本的平均已收敛。**

### 9.4 权重位置

```
新SNN\delrec_runs\
├── delay_T50_ts28_7ep\delay_w352_T50_ts28_7ep\
├── delay_w352_T100_ts14_7ep\delay_w352_T100_ts14_7ep\
├── delay_T140_ts10_7ep\delay_w352_T140_ts10_7ep\
├── pmsn_T50_ckpts\pmsn_{lif,mc}_T50_fill_{5,3}ep\...\checkpoint.pth
├── pmsn_new_ckpts\pmsn_{lif,mc}_T100_fill_{5,3}ep\...\checkpoint.pth
└── pmsn_T140_ckpts\pmsn_{lif,mc}_T140_fill_{5,3}ep\...\checkpoint.pth
```

**`scripts/study1_10samples.py` 顶部的 `DELAY_RUNS` 和 `PMSN_RUNS` 是绝对路径，迁移时需改。**

---

## 10. 已知限制汇总

| # | 限制 | 影响 | 详见 |
|---|---|---|---|
| 1 | `par = 1` 无文献依据 | $t_{inf}$ 与权重漏电 | §8.4 |
| 2 | $E_{hop}$ 是 14nm 值混入 45nm | comm 项 | §8.5 |
| 3 | $P_{leak}$ 是 8 MB cache 值（小 SRAM 更高） | leakage 被低估 | `PARAMETERS.md` |
| 4 | flit_hops 是单播上界 | comm 高估 2–3× | §8.2 |
| 5 | 同一时间步 spike 未聚合 | comm 进一步高估 | §8.3 |
| 6 | placement 未优化 | out 层热点 | §8.1 |
| 7 | FP32 与白皮书 §7.1 “low-precision” 不一致 | 叙述层面（INT32 差 <0.5%） | `PARAMETERS.md` |
| 8 | 不计 $\int P_{idle}dt$ | 符合 Study 1 五类，非缺陷 | §4.2 |
| 9 | 权重漏电单列 | 口径选择，非缺陷 | §4.3 |

---

## 11. 图

| 文件 | 回答什么问题 |
|---|---|
| `figs/study1_main_T100_2x2.png` | 白皮书要求的形式：三模型的堆叠能量分解 + spike 数叠加（T=100, 2×2） |
| `figs/study1_grid_resolution_mesh.png` | 分辨率与 mesh 规模如何改变能量的构成（3×2 全网格） |
| `figs/study1_ranking_flip.png` | 三种估计在每个分辨率下的排序对比，标出哪一对翻转 |

---

## 12. 引用

| 用途 | 文献 |
|---|---|
| FP32 单价 | Horowitz, *Computing's Energy Problem (and what we can do about it)*, ISSCC 2014, Fig 1.1.9 |
| 每跳通信 | Davies et al., *Loihi: A Neuromorphic Manycore Processor with On-Chip Learning*, IEEE Micro 2018, Table 2 |
| 架构假设（权重驻留） | Dampfhoffer et al., *Are SNNs Really More Energy Efficient Than ANNs?*, 2023, §3.3 |
| 路由器功耗（交叉验证） | Gorski, *Dissertation*, 2016, Table I.5 / I.7 |
| delay 模型 | DelRec（`alexmaxad/DelRec`） |
| PMSN 模型 | Chen et al., *PMSN: A Parallel Multi-compartment Spiking Neuron*, arXiv:2408.14917 |
| 判据来源 | *Time as a First-Class Resource*, §4.3 与 Preliminary Studies, p.11–12 |
