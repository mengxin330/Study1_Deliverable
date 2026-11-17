"""10 样本快速核对：SOP 与 flit_hops 的量级是否与 2264 样本的平均一致。"""
import os
import sys

import h5py
import numpy as np
import torch

VERIFY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VERIFY)
sys.path.insert(0, r"D:\DelRec")
sys.path.insert(0, r"D:\DelRec\src")

from net.topology import Mesh
from net.placement import Placement, layers_for, fanout_targets

DATA = r"D:\DelRec\Datasets\SHD\extract\shd_test.h5"
BASE = r"C:\Users\yangz\Desktop\deepseek工位\新SNN\delrec_runs"
N_TEST = 10
CONFIGS = [("2core", 2, None), ("2x2", 4, 2), ("4x4", 16, 4), ("8x8", 64, 8)]

DELAY_RUNS = {
    50: rf"{BASE}\delay_T50_ts28_7ep\delay_w352_T50_ts28_7ep",
    100: rf"{BASE}\delay_w352_T100_ts14_7ep\delay_w352_T100_ts14_7ep",
    140: rf"{BASE}\delay_T140_ts10_7ep\delay_w352_T140_ts10_7ep",
}

PMSN_RUNS = {
    ("lif", 50):  rf"{BASE}\pmsn_T50_ckpts\pmsn_lif_T50_fill_5ep\lr0.01_nlr0.001_wd0_drop_0.2_d352_s1_seed1111\checkpoint.pth",
    ("mc", 50):   rf"{BASE}\pmsn_T50_ckpts\pmsn_mc_T50_fill_3ep\lr0.01_nlr0.001_wd0_drop_0.2_d352_s4_seed1111\checkpoint.pth",
    ("lif", 100): rf"{BASE}\pmsn_new_ckpts\pmsn_lif_T100_fill_5ep\lr0.01_nlr0.001_wd0_drop_0.2_d352_s1_seed1111\checkpoint.pth",
    ("mc", 100):  rf"{BASE}\pmsn_new_ckpts\pmsn_mc_T100_fill_3ep\lr0.01_nlr0.001_wd0_drop_0.2_d352_s4_seed1111\checkpoint.pth",
    ("lif", 140): rf"{BASE}\pmsn_T140_ckpts\pmsn_lif_T140_fill_5ep\lr0.01_nlr0.001_wd0_drop_0.2_d352_s1_seed1111\checkpoint.pth",
    ("mc", 140):  rf"{BASE}\pmsn_T140_ckpts\pmsn_mc_T140_fill_3ep\lr0.01_nlr0.001_wd0_drop_0.2_d352_s4_seed1111\checkpoint.pth",
}

# 2264 样本的实测值（单样本平均）
REF_SOP = {
    ("delay", 50): 1431230, ("delay", 100): 2804570, ("delay", 140): 3985830,
    ("lif", 50): 1389789, ("lif", 100): 1808584, ("lif", 140): 2589276,
    ("mc", 50): 1533348, ("mc", 100): 2343106, ("mc", 140): 2937174,
}
REF_FH = {
    ("delay", 50): {7845: 4014, 24572: 12468},  # placeholder, 用 mesh 名
}
REF_FH = {
    ("delay", 50):  [4014, 12468, 85422, 614151],
    ("delay", 100): [7845, 24572, 169371, 1220020],
    ("delay", 140): [11159, 34980, 238466, 1706995],
    ("lif", 50):    [8831, 28587, 198392, 1436650],
    ("lif", 100):   [14171, 45118, 308479, 2213959],
    ("lif", 140):   [19826, 63203, 434092, 3125954],
    ("mc", 50):     [9732, 31381, 220566, 1591048],
    ("mc", 100):    [15977, 50835, 357068, 2576434],
    ("mc", 140):    [21429, 68429, 474574, 3409407],
}


def load_test():
    f = h5py.File(DATA, "r")
    return f["spikes"]["times"], f["spikes"]["units"], np.asarray(f["labels"], dtype=np.int64)


def bin_one(times_s, units, time_step, T, W=700, n_bins=5):
    t = 1000.0 * np.asarray(times_s, dtype=np.float64)
    if t.size == 0:
        return np.zeros((T, W // n_bins), dtype=np.float32)
    t = t - t[0]
    idx = (t // time_step).astype(np.int64)
    keep = idx < T
    idx, u = idx[keep], np.asarray(units, dtype=np.int64)[keep]
    frames = np.zeros((T, W), dtype=np.float32)
    np.add.at(frames, (idx, u), 1.0)
    bl = W // n_bins
    return frames.reshape(T, bl, n_bins).sum(axis=2).astype(np.float32)


def make_hop(ncores, mesh_n=None):
    if mesh_n is None:
        return [[abs(i - j) for j in range(ncores)] for i in range(ncores)]
    tab = Mesh(mesh_n).path_table()
    h = [[0] * ncores for _ in range(ncores)]
    for u in range(ncores):
        for v in range(ncores):
            if u != v:
                p = tab[u][v]
                h[u][v] = len(p) - 1 if p else 0
    return h


def build_tbl(ncores, mesh_n, hidden):
    hop = make_hop(ncores, mesh_n)
    pl = Placement(layers_for(hidden=hidden, out=20), n_cores=ncores)
    tbl = {}
    for src, dss in (("hidden1", ["hidden2"]), ("hidden2", ["out", "hidden2"])):
        for dst in dss:
            d = [0] * ncores
            L = pl.by_name(src)
            for k in range(L.width):
                sc = pl.core_of(L.lo + k)
                tg = fanout_targets(pl, src, [dst], sc)
                d[sc] = sum(hop[sc][x] for x in tg)
            tbl[(src, dst)] = d
    return pl, tbl


def total_hops(rows, tbl, pl):
    tot = 0
    for _t, at_t in rows.items():
        for sl, idxs in at_t.items():
            base = pl.by_name(sl).lo
            per = [a for (s2, _), a in tbl.items() if s2 == sl]
            for li in idxs:
                sc = pl.core_of(base + li)
                for a in per:
                    tot += a[sc]
    return tot


def run_delay(T, times, units, labels, tabs):
    from delay_model import DelayModel
    dm = DelayModel(run_dir=DELAY_RUNS[T], verbose=False)
    dm.model.eval()
    mods = dict(dm.model.named_modules())
    time_step = float(dm.config.time_step)
    fan_h1 = 352
    fan_h2 = 372

    sop, spikes = 0, 0
    fh = [[0] * 4 for _ in range(N_TEST)]
    with torch.no_grad():
        for i in range(N_TEST):
            x = bin_one(times[i], units[i], time_step, T)
            dm.model(torch.from_numpy(x).unsqueeze(1).float())
            rows = {}
            for mi, key, fan, lo in ((3, "hidden1", fan_h1, 0), (8, "hidden2", fan_h2, 352)):
                spk = mods[f"layers.{mi}"].spikes
                if spk is None:
                    continue
                s = np.asarray(spk.cpu())
                if s.ndim == 3:
                    s = s[:, 0, :]
                cnt = int((s > 0.5).sum())
                spikes += cnt
                sop += cnt * fan
                for t in range(T):
                    idx = np.nonzero(s[t] > 0.5)[0].tolist()
                    if idx:
                        rows.setdefault(t, {})[key] = idx
            for j, (cn, _, _) in enumerate(CONFIGS):
                pl, tbl = tabs[cn]
                fh[i][j] = total_hops(rows, tbl, pl)
    n = N_TEST
    return sop / n, spikes / n, [sum(fh[i][j] for i in range(n)) / n for j in range(4)]


def run_pmsn(which, T, times, units, labels, tabs):
    from pmsn_model import SHDNet_ours
    ck = torch.load(PMSN_RUNS[(which, T)], map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = SHDNet_ours(bn=getattr(cfg, "bn", True), d_input=cfg.d_input, d_model=cfg.d_model,
                      d_out=20, d_state=cfg.d_state, dropout=cfg.dropout,
                      lr=cfg.neuron_lr, T=cfg.T)
    net.load_state_dict(ck["best_net"], strict=False)
    net.eval()

    captured = {}
    handles = []
    for name, m in net.named_modules():
        if type(m).__name__ == "PMSN_neuron":
            handles.append(m.register_forward_hook(
                lambda mod, inp, out, n=name: captured.__setitem__(n, out.detach())))
    time_step = 1400.0 / T
    sop, spikes = 0, 0
    fh = [[0] * 4 for _ in range(N_TEST)]
    with torch.no_grad():
        for i in range(N_TEST):
            captured.clear()
            x = bin_one(times[i], units[i], time_step, T)
            xb = torch.from_numpy(x).unsqueeze(0).permute(0, 2, 1)
            net(xb)
            rows = {}
            for k, v in captured.items():
                s = v.reshape(T, 1, -1)
                cnt = int((s > 0.5).sum())
                spikes += cnt
                fan = 352 if "hidden1" in k else 20
                sop += cnt * fan
                key = "hidden1" if "hidden1" in k else "hidden2"
                for t in range(T):
                    idx = torch.nonzero(s[t, 0, :] > 0.5, as_tuple=False).flatten().tolist()
                    if idx:
                        rows.setdefault(t, {})[key] = idx
            for j, (cn, _, _) in enumerate(CONFIGS):
                pl, tbl = tabs[cn]
                fh[i][j] = total_hops(rows, tbl, pl)
    for h in handles:
        h.remove()
    n = N_TEST
    return sop / n, spikes / n, [sum(fh[i][j] for i in range(n)) / n for j in range(4)]


def main():
    times, units, labels = load_test()
    print(f"总样本数 {len(labels)}，本脚本用前 {N_TEST} 个\n")

    tabs_pmsn = {c[0]: build_tbl(c[1], c[2], 352) for c in CONFIGS}
    tabs_delay = {c[0]: build_tbl(c[1], c[2], 352) for c in CONFIGS}

    print(f"{'model':<10}{'T':>5}{'SOP(10)':>12}{'SOP(2264)':>12}{'比率':>8}"
          f"{'fh2c(10)':>11}{'fh2c(2264)':>12}{'比率':>8}")
    print("-" * 82)

    for T in (50, 100, 140):
        s10, spk10, f10 = run_delay(T, times, units, labels, tabs_delay)
        ref = REF_SOP[("delay", T)]
        rf = REF_FH[("delay", T)]
        print(f"{'delay':<10}{T:>5}{s10:>12,.0f}{ref:>12,}{s10/ref:>8.3f}"
              f"{f10[0]:>11,.0f}{rf[0]:>12,}{f10[0]/rf[0]:>8.3f}")

    for which in ("lif", "mc"):
        for T in (50, 100, 140):
            s10, spk10, f10 = run_pmsn(which, T, times, units, labels, tabs_pmsn)
            ref = REF_SOP[(which, T)]
            rf = REF_FH[(which, T)]
            print(f"{which.upper():<10}{T:>5}{s10:>12,.0f}{ref:>12,}{s10/ref:>8.3f}"
                  f"{f10[0]:>11,.0f}{rf[0]:>12,}{f10[0]/rf[0]:>8.3f}")


if __name__ == "__main__":
    main()
