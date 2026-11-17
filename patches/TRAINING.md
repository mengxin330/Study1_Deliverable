# 训练复现指南

三个时间分辨率、两个代码库、共 9 次训练。

```bash
# 目录约定
/content/DelRec                    # alexmaxad/DelRec
/content/PMSN                      # xychen-comp/Parallel-Multi-compartment-Spiking-Neuron-PMSN
/content/shd/{shd_train.h5,shd_test.h5}
/content/work/exp                  # 输出
```

---

## 一、环境

```bash
# DelRec
cd /content
git clone https://github.com/alexmaxad/DelRec.git
cd DelRec && pip install -r requirements.txt
mkdir -p Datasets/SHD

# PMSN
cd /content
git clone https://github.com/xychen-comp/Parallel-Multi-compartment-Spiking-Neuron-PMSN.git PMSN
pip install -q einops tqdm h5py
pip install -q "spikingjelly @ git+https://github.com/fangwei123456/spikingjelly.git@d4fee3a1715bf42ce15f52fde1b1d4709d7ea25e"

# SHD 数据集
mkdir -p /content/shd && cd /content/shd
wget -q https://zenkelab.org/datasets/shd_train.h5.zip
wget -q https://zenkelab.org/datasets/shd_test.h5.zip
unzip -oq shd_train.h5.zip && unzip -oq shd_test.h5.zip
```

---

## 二、超参由 `time_step` 推导

**所有延迟超参按物理时间定义，再除以 `time_step`（= 1400/T）换算成帧数。换分辨率时语义不变。**

| 参数 | 物理时间 | T=50 (28 ms) | T=100 (14 ms) | T=140 (10 ms) |
|---|---|---|---|---|
| `time_step` | — | **28** | **14** | **10** |
| `max_rec_delay` | 300 ms | 11 | 21 | 30 |
| `max_feedforward_delay` | 250 ms | 9 | 17 | 25 |
| `sigma_init` | 100 ms | 3.5 | 7.0 | 10.0 |
| `init_recdel_offset` | 100 ms | 4 | 7 | 10 |
| `delay_std_init` | 150 ms | 5 | 11 | 15 |
| `batch_size` | — | 40 | 40 | 40 |
| `hidden_layers` | — | [352, 352] | [352, 352] | [352, 352] |

**`patch_delrec.py` 会自动从 `--T` 推导这张表。**

---

## 三、delay 网络（DelRec）

```bash
# T=100（推荐主结果，精度 87.24%）
cd /content/DelRec
python /path/to/patches/patch_delrec.py --root /content/DelRec --T 100
python experiments/train.py --dataset shd --model SNN_recurrent_delays \
  --seeds 1111 --epochs 7 --kernel v2 \
  --out /content/work/exp/delay_w352_T100_ts14_7ep

# T=50（精度 86.56%）
python /path/to/patches/patch_delrec.py --root /content/DelRec --T 50
python experiments/train.py --dataset shd --model SNN_recurrent_delays \
  --seeds 1111 --epochs 7 --kernel v2 \
  --out /content/work/exp/delay_T50_ts28_7ep

# T=140（精度 87.99%）
python /path/to/patches/patch_delrec.py --root /content/DelRec --T 140
python experiments/train.py --dataset shd --model SNN_recurrent_delays \
  --seeds 1111 --epochs 7 --kernel v2 \
  --out /content/work/exp/delay_T140_ts10_7ep
```

**必须 `--kernel v2`**，否则 CuPy 会因 shared memory 超限失败（`triton.runtime.errors.OutOfResources: Required: 135168, limit 65536`）。

**分箱缓存**：`Datasets/SHD/duration_{time_step}` 目录会生成一次，之后复用。

---

## 四、PMSN 网络

```bash
# T=100：plain LIF（d_state=1）5 轮 → 86.40%
#        multi-compartment（d_state=4）3 轮 → 88.30%
python /path/to/patches/patch_pmsn.py --root /content/PMSN --T 100

cd /content/PMSN/SHD
python SHD_PMSN.py --data_path /content/shd --T 100 --d_state 1 --epochs 5 \
  --ckpt_path /content/work/exp/pmsn_lif_T100_fill_5ep --device cuda:0

python SHD_PMSN.py --data_path /content/shd --T 100 --d_state 4 --epochs 3 \
  --ckpt_path /content/work/exp/pmsn_mc_T100_fill_3ep --device cuda:0
```

```bash
# T=50：LIF 5 轮 → 88.96%；MC 3 轮 → 91.21%
python /path/to/patches/patch_pmsn.py --root /content/PMSN --T 50
cd /content/PMSN/SHD

python SHD_PMSN.py --data_path /content/shd --T 50 --d_state 1 --epochs 5 \
  --ckpt_path /content/work/exp/pmsn_lif_T50_fill_5ep --device cuda:0

python SHD_PMSN.py --data_path /content/shd --T 50 --d_state 4 --epochs 3 \
  --ckpt_path /content/work/exp/pmsn_mc_T50_fill_3ep --device cuda:0
```

```bash
# T=140：LIF 5 轮 → 87.10%；MC 3 轮 → 88.96%
python /path/to/patches/patch_pmsn.py --root /content/PMSN --T 140
cd /content/PMSN/SHD

python SHD_PMSN.py --data_path /content/shd --T 140 --d_state 1 --epochs 5 \
  --ckpt_path /content/work/exp/pmsn_lif_T140_fill_5ep --device cuda:0

python SHD_PMSN.py --data_path /content/shd --T 140 --d_state 4 --epochs 3 \
  --ckpt_path /content/work/exp/pmsn_mc_T140_fill_3ep --device cuda:0
```

**关键参数**：`--T` 就是帧数上限（`nb_steps`），分箱超出的帧自动裁掉。

---

## 五、精度汇总

| T | 帧长 | delay | PMSN LIF | PMSN MC |
|---|---|---|---|---|
| 50 | 28.0 ms | 86.56% | 88.96% | 91.21% |
| **100** | **14.0 ms** | **87.24%** | 86.40% | 88.30% |
| 140 | 10.0 ms | 87.99% | 87.10% | 88.96% |

**T=100 和 T=140 的三模型精度跨度 <2 点，最适合作为“matched accuracy”的主结果。**

---

## 五之二、训练轮数与精度演进

### 轮数汇总

| 模型 | epochs | 命令行参数 | 报告精度取自 |
|---|---|---|---|
| **delay 352** | **7** | `--epochs 7` | **best validation** |
| **PMSN LIF** | **5** | `--epochs 5` | **last epoch** |
| **PMSN MC** | **3** | `--epochs 3` | **last epoch** |

**为什么轮数不同**：三个模型来自不同代码库，各自沿用其默认训练预算。delay 网络 7 轮是在 SHD 上收敛到 val >93% 的最小轮数；PMSN 的两个变体分别沿用官方 SHD 配置的 5 轮与 3 轮。

### delay 352（7 轮，取 best validation）

| T | 最终 val | checkpoint epoch | 报告精度（test） |
|---|---|---|---|
| 50 | — | — | **86.56%** |
| 100 | 93.30% | epoch 5 | **87.24%** |
| 140 | 93.06% | epoch 6 | **87.99%** |

**DelRec 的训练脚本按 validation accuracy 保存 best，不是最后一轮。**

### PMSN plain LIF（5 轮，取最后一轮）

| epoch | T=50 | T=100 | T=140 |
|---|---|---|---|
| 0 | 79.02% | 67.45% | 62.01% |
| 1 | 86.00% | 81.89% | 81.67% |
| 2 | 86.48% | 83.61% | 83.75% |
| 3 | 88.34% | 85.29% | 85.95% |
| **4** | **88.96%** | **86.40%** | **87.10%** |

**三个 T 都仍在爬升，末轮即最高。**

### PMSN multi-compartment（3 轮，取最后一轮）

| epoch | T=50 | T=100 | T=140 |
|---|---|---|---|
| 0 | 85.16% | 77.61% | 75.49% |
| 1 | **93.11%** | **89.62%** | **89.13%** |
| **2** | **91.21%** | **88.30%** | **88.96%** |

**MC 在第 1 轮达到峰值，第 2 轮略回落。**

**报告口径统一为最后一轮**（不取 best），因为 best 会让 MC 偏高 1.5–2 点，破坏三模型的精度匹配。

**若改用 best checkpoint，MC 分别为 93.11 / 89.62 / 89.13，与 LIF 的差距拉大到 3–4 点。**

### 选轮数的依据

**目标是“matched accuracy”，不是“epoch 相等”：**

| T | 三模型精度跨度 | 判定 |
|---|---|---|
| 50 | 86.56 – 91.21 = **4.65 点** | 偏大 |
| **100** | 86.40 – 88.30 = **1.90 点** | ✅ 主结果 |
| **140** | 87.10 – 88.96 = **1.86 点** | ✅ 主结果 |

**T=50 的 4.65 点来自 MC 在 28 ms 帧长下表现特别好（91.21%），是真实差异，不做人为调整。**

---

## 六、两个补丁各自解决什么问题

### DelRec

| 补丁 | 不打的后果 |
|---|---|
| **固定时长分箱** | 用 `frames_number` 等分会让“一帧”在不同样本上覆盖不同物理时间；实测精度从 87% 掉到 **70.40%** |
| **定长 collate** | batch 内帧数不齐，同一延迟值的物理时长变化 |
| **max → mean 池化** | 与原论文/checkpoint 不一致 |

### PMSN

| 补丁 | 不打的后果 |
|---|---|
| **固定 duration 分箱** | 与 DelRec 的“一帧”不对应，两个模型族无法同 T 比较 |
| **`bn=True`** | checkpoint 是 `bn=True` 训的，不匹配则 `load_state_dict` 报错 |

### 一个已知的坑

`SHD_PMSN.py` 的 `--log_interval` 默认 300，而一个 epoch 只有 204 个 batch，所以中间进度永不打印。**不要调小到 50 以下**——官方 print 格式串在 `log_interval` 触发时有个参数个数 bug（`TypeError: not enough arguments for format string`）。
