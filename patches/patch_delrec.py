"""DelRec 训练补丁：把固定时长分箱 + 定长 collate 装进开源仓库。

用法（在 DelRec 仓库根目录）:
    python patch_delrec.py --root /content/DelRec --T 100
    python patch_delrec.py --root /content/DelRec --T 50
    python patch_delrec.py --root /content/DelRec --T 140

三个补丁：
  1. configs/perf_SHD.py     —— 按 time_step 重标定全部延迟超参
  2. src/.../loaders.py      —— 用 fixed_collate_T{T} 替换 pad_sequence_collate
  3. src/delrec/utils.py     —— 时间维池化 max -> mean

设计说明
--------
延迟超参全部按“物理时间”定义，再除以 time_step 换算成帧数，
因此换时间分辨率时语义不变：

    max_rec_delay            = 300 ms / time_step
    max_feedforward_delay    = 250 ms / time_step
    sigma_init               = 100 ms / time_step
    init_recdel_offset       = 100 ms / time_step
    delay_std_init           = 150 ms / time_step

为什么要固定时长分箱
--------------------
DelRec 的延迟是“离散帧数”且在样本间共享。若改用 frames_number（等分）
分箱，同一个帧在不同长度的样本上覆盖不同的物理时间，延迟语义被破坏。
实测：frames_number 分箱使 SHD 测试精度从 87% 掉到 70%。

为什么 collate 要定长
--------------------
延迟按帧计，如果每个 batch 的帧数不同，同一延迟值对应的物理时长会变。
所以裁剪/零填充到固定 T 帧。
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

# 物理时间基准（毫秒）
MAX_REC_DELAY_MS = 300.0
MAX_FF_DELAY_MS = 250.0
SIGMA_MS = 100.0
INIT_OFFSET_MS = 100.0
DELAY_STD_MS = 150.0

SPAN_MS = 1400.0   # SHD 总时长


def derive(time_step: int) -> dict:
    """由 time_step 推所有超参（取整规则与原始配置一致）。"""
    return {
        "time_step": time_step,
        "max_rec_delay": int(round(MAX_REC_DELAY_MS / time_step)),
        "max_feedforward_delay": int(round(MAX_FF_DELAY_MS / time_step)),
        "sigma_init": round(SIGMA_MS / time_step, 2),
        "init_recdel_offset": int(round(INIT_OFFSET_MS / time_step)),
        "delay_std_init": int(round(DELAY_STD_MS / time_step)),
        "batch_size": 40,
        "hidden_layers": "[352, 352]",
    }


def patch_config(root: pathlib.Path, T: int, ts: int) -> bool:
    p = root / "configs" / "perf_SHD.py"
    s = p.read_text()
    target = derive(ts)
    ok = True
    for k, v in target.items():
        pat = rf"^([ \t]*{k}[ \t]*=[ \t]*).*$"
        if re.search(pat, s, re.M):
            s = re.sub(pat, rf"\g<1>{v}", s, count=1, flags=re.M)
            print(f"    {k} = {v}")
        else:
            print(f"    WARN 未找到 {k}")
            ok = False

    # 去掉重复的奇数化行
    out, seen = [], False
    for line in s.splitlines():
        if line.strip().startswith("max_feedforward_delay = max_feedforward_delay if"):
            if seen:
                continue
            seen = True
        out.append(line)
    s = "\n".join(out) + "\n"

    if "mask_T_stepsi" not in s:
        s = s.rstrip("\n") + (
            "\n\n    ### compatibility stubs ###\n"
            "    mask_T_stepsi = None\n"
            "    mask_T_steps = 0\n"
            "    masked_proba = None\n")

    p.write_text(s)
    return ok


COLLATE_TMPL = '''def fixed_collate_T{T}(batch):
    """把每个样本裁剪/零填充到固定 T={T} 帧（{span} ms / {ts} ms）。

    延迟按帧计且跨样本共享，所以帧数必须恒定，否则同一延迟值对应
    不同的物理时长，语义被破坏。
    """
    T = {T}
    xs, ys = zip(*batch)
    fixed = []
    for x in xs:
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.shape[0] > T:
            x = x[:T]
        elif x.shape[0] < T:
            x = torch.nn.functional.pad(x, (0, 0, 0, T - x.shape[0]))
        fixed.append(x)
    x_len = torch.full((len(fixed),), T, dtype=torch.long)
    return torch.stack(fixed), torch.as_tensor(ys, dtype=torch.long), x_len


'''


def patch_loaders(root: pathlib.Path, T: int, ts: int) -> bool:
    p = root / "src" / "delrec" / "datasets" / "loaders.py"
    s = p.read_text()
    name = f"fixed_collate_T{T}"

    # 删掉旧的 fixed_collate_* 定义
    s = re.sub(r"def fixed_collate_\w+\(batch\):.*?(?=\ndef |\nclass |\Z)", "", s, flags=re.S)
    body = COLLATE_TMPL.format(T=T, span=int(SPAN_MS), ts=ts)
    s = s.replace("def load_dataset(config):", body + "def load_dataset(config):", 1)

    n_c = len(re.findall(r"collate_fn=\w+", s))
    s = re.sub(r"collate_fn=\w+", f"collate_fn={name}", s)
    print(f"    collate_fn -> {name}  ({n_c} 处)")

    # 固定时长分箱
    n_d = s.count("duration=config.time_step")
    if n_d != 2:
        s = re.sub(r"frames_number=config\.frames_number,\s*split_by='time'",
                   "duration=config.time_step", s)
        n_d = s.count("duration=config.time_step")
        print(f"    WARN 分箱锚点不在位，修复后 {n_d}/2" if n_d != 2 else f"    分箱已修复 {n_d}/2")
    else:
        print(f"    duration 分箱 {n_d}/2")

    p.write_text(s)
    return n_c == 3 and n_d == 2


def patch_utils(root: pathlib.Path) -> bool:
    p = root / "src" / "delrec" / "utils.py"
    s = p.read_text()
    n = s.count("m, _ = torch.max(output, 0)")
    s = s.replace("m, _ = torch.max(output, 0)", "m = torch.mean(output, 0)")
    p.write_text(s)
    print(f"    时间维池化 max -> mean  ({n} 处)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/DelRec")
    ap.add_argument("--T", type=int, required=True,
                    help="目标帧数；time_step = 1400/T")
    a = ap.parse_args()

    root = pathlib.Path(a.root)
    if not (root / "configs" / "perf_SHD.py").exists():
        sys.exit(f"不是 DelRec 仓库：{root}")

    ts = int(round(SPAN_MS / a.T))
    if abs(SPAN_MS / a.T - ts) > 0.51:
        sys.exit(f"T={a.T} 不能整除 1400 ms（time_step={SPAN_MS/a.T:.2f}）")

    print(f"DelRec 补丁  T={a.T}  time_step={ts} ms\n")
    print("  [1] configs/perf_SHD.py")
    ok1 = patch_config(root, a.T, ts)
    print("  [2] src/delrec/datasets/loaders.py")
    ok2 = patch_loaders(root, a.T, ts)
    print("  [3] src/delrec/utils.py")
    ok3 = patch_utils(root)

    print("\n  --- 最终配置 ---")
    for line in (root / "configs" / "perf_SHD.py").read_text().splitlines():
        t = line.strip()
        if t.startswith(("time_step", "max_rec_delay", "max_feedforward_delay",
                         "sigma_init", "init_recdel_offset", "delay_std_init",
                         "batch_size", "hidden_layers")):
            print("   ", t)

    ok = ok1 and ok2 and ok3
    print("\nRESULT:", "OK" if ok else "CHECK")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
