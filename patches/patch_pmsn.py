"""PMSN 训练补丁：把固定时长分箱装进开源仓库。

用法（在 PMSN 仓库根目录）:
    python patch_pmsn.py --root /content/PMSN --T 100
    python patch_pmsn.py --root /content/PMSN --T 50

两个补丁（都在 SHD/SHD_PMSN.py）:
  1. 分箱：np.linspace 等分 -> 固定 duration 分箱
  2. BatchNorm：bn=False -> bn=True（若 checkpoint 是 bn=True）

为什么改分箱
------------
官方用 ``np.linspace(0, max_time, num=nb_steps)`` 做等分分箱，即把每个样本
的事件时间**拉伸**到 T 个 bin。这会破坏与 DelRec 的可比性：DelRec 用固定
时长分箱（duration = 1400/T ms），两者的“一帧”对应相同的物理时间。

PMSN 的神经元内部是连续时间常数的 SSM，对分箱方式不敏感（实测换分箱后
精度不变）；但为了与 delay 网络同口径、同 T 比较，统一改成固定 duration。

等价性验证
----------
固定时长分箱 ``(t_ms // duration).astype(int)`` + ``keep = idx < nb_steps``
与 DelRec 的 ``integrate_events_by_fixed_duration_shd(duration=...)`` 逐元素
一致（实测 mismatch = 0/4），且不需要减去 t[0]。
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

SPAN_MS = 1400.0


def patch(root: pathlib.Path, T: int) -> bool:
    p = root / "SHD" / "SHD_PMSN.py"
    if not p.exists():
        sys.exit(f"找不到 {p}")

    s = p.read_text()
    ok = True

    # --- 1. 分箱：删掉 linspace，改成固定 duration ---
    old1 = "        self.time_bins = np.linspace(0, max_time, num=nb_steps)"
    n1 = s.count(old1)
    if n1 == 1:
        s = s.replace(old1, "        self.time_step_ms = 1400.0 / nb_steps")
        print(f"    [1] linspace -> time_step_ms  ({n1} 处)")
    elif "self.time_step_ms" in s:
        # 已打过补丁，只更新数值
        s = re.sub(r"self\.time_step_ms = [0-9.]+",
                   "self.time_step_ms = 1400.0 / nb_steps", s)
        print("    [1] time_step_ms 已存在，已更新公式")
        n1 = 1
    else:
        print("    [1] FAIL: 找不到 linspace 锚点")
        ok = False
        n1 = 0

    # --- 2. digitize -> 固定时长 ---
    old2 = """            times = np.digitize(self.firing_times[sample_idx], self.time_bins)
            units = self.units_fired[sample_idx]
            batch = [batch_pos for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)"""
    new2 = """            t_ms = 1000.0 * np.asarray(self.firing_times[sample_idx], dtype=np.float64)
            idx = (t_ms // self.time_step_ms).astype(np.int64)
            units = np.asarray(self.units_fired[sample_idx])
            keep = idx < self.nb_steps
            idx, units = idx[keep], units[keep]
            batch = [batch_pos for _ in range(len(idx))]

            coo[0].extend(batch)
            coo[1].extend(idx.tolist())
            coo[2].extend(units.tolist())"""
    n2 = s.count(old2)
    if n2 == 1:
        s = s.replace(old2, new2)
        print(f"    [2] digitize -> 固定时长  ({n2} 处)")
    elif "t_ms // self.time_step_ms" in s:
        print("    [2] 固定时长分箱已存在")
        n2 = 1
    else:
        print("    [2] FAIL: 找不到 digitize 锚点")
        ok = False
        n2 = 0

    # --- 3. BatchNorm ---
    n3 = s.count("bn=False")
    if n3:
        s = s.replace("bn=False", "bn=True")
        print(f"    [3] bn=False -> True  ({n3} 处)")
    else:
        print("    [3] bn 已是 True（或无此项）")

    p.write_text(s)

    # --- 验证 ---
    txt = p.read_text()
    print("\n    --- 验证 ---")
    for pat, want in (("t_ms // self.time_step_ms", True),
                      ("self.time_step_ms = 1400.0 / nb_steps", True)):
        found = pat in txt
        print(f"    {'OK ' if found == want else 'BAD'} {pat}")
        ok = ok and (found == want)

    return ok and n1 == 1 and n2 == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/PMSN")
    ap.add_argument("--T", type=int, required=True)
    a = ap.parse_args()

    root = pathlib.Path(a.root)
    ts = SPAN_MS / a.T
    print(f"PMSN 补丁  T={a.T}  duration={ts:.2f} ms\n")
    ok = patch(root, a.T)
    print("\nRESULT:", "OK" if ok else "CHECK")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
