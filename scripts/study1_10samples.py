"""Study 1 正式口径（32-bit FP32），10 个测试样本实测 + 能量计算。

Study 1 的五个类别（白皮书 p.11-12）:
    1. synaptic operations   -> syn + neu
    2. communication         -> comm
    3. delay storage         -> buffer
    4. state memory          -> state
    5. leakage               -> leak（state+buffer 的漏电）

按 Study 1 Hypothesis 的焦点（"delay buffers and state memory"），
权重漏电单列为静态背景，不并入时间状态能耗。

时间模型: t_inf = (SOP + N*T*d_state) / (ncores * F),  par = 1, F = 2 GHz
单价: E_AC=0.9, E_MAC=4.6, E_ACC=10 pJ, P_leak=11.9 nW/bit, E_hop=3.5 pJ
"""
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
N_TEST = -1   # -1 表示用全部测试样本

E_AC, E_MAC, E_ACC, P_LEAK, E_HOP, F = 0.9, 4.6, 10.0, 11.9e-9, 3.5, 2e9
MESHES = [("2x2", 4, 2), ("4x4", 16, 4), ("8x8", 64, 8)]

DELAY_RUNS = {
    50:  rf"{BASE}\delay_T50_ts28_7ep\delay_w352_T50_ts28_7ep",
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
ARCH = {
    "delay": dict(params=306612, neurons=724, d_state=1, n_delay=352),
    "lif":   dict(params=187988, neurons=704, d_state=1, n_delay=0),
    "mc":    dict(params=200660, neurons=704, d_state=4, n_delay=0),
}
ACC = {("delay", 50): 86.56, ("delay", 100): 87.24, ("delay", 140): 87.99,
       ("lif", 50): 88.96, ("lif", 100): 86.40, ("lif", 140): 87.10,
       ("mc", 50): 91.21, ("mc", 100): 88.30, ("mc", 140): 88.96}


def load_test():
    f = h5py.File(DATA, "r")
    return f["spikes"]["times"], f["spikes"]["units"], np.asarray(f["labels"], dtype=np.int64)


def bin_one(times_s, units, ts_ms, T, W=700, nb=5):
    t = 1000.0 * np.asarray(times_s, dtype=np.float64)
    if t.size == 0:
        return np.zeros((T, W // nb), dtype=np.float32)
    t = t - t[0]
    idx = (t // ts_ms).astype(np.int64)
    keep = idx < T
    idx, u = idx[keep], np.asarray(units, dtype=np.int64)[keep]
    fr = np.zeros((T, W), dtype=np.float32)
    np.add.at(fr, (idx, u), 1.0)
    return fr.reshape(T, W // nb, nb).sum(axis=2).astype(np.float32)


def make_hop(nc, mn=None):
    if mn is None:
        return [[abs(i - j) for j in range(nc)] for i in range(nc)]
    tab = Mesh(mn).path_table()
    h = [[0] * nc for _ in range(nc)]
    for u in range(nc):
        for v in range(nc):
            if u != v:
                p = tab[u][v]
                h[u][v] = len(p) - 1 if p else 0
    return h


def build_tabs():
    tabs = {}
    for cn, nc, mn in MESHES:
        hop = make_hop(nc, mn)
        pl = Placement(layers_for(hidden=352, out=20), n_cores=nc)
        tbl = {}
        for src, dss in (("hidden1", ["hidden2"]), ("hidden2", ["out", "hidden2"])):
            for dst in dss:
                d = [0] * nc
                L = pl.by_name(src)
                for k in range(L.width):
                    sc = pl.core_of(L.lo + k)
                    tg = fanout_targets(pl, src, [dst], sc)
                    d[sc] = sum(hop[sc][x] for x in tg)
                tbl[(src, dst)] = d
        tabs[cn] = (pl, tbl)
    return tabs


def hops_of(rows, pl, tbl):
    tot = 0
    for _t, at in rows.items():
        for sl, idxs in at.items():
            base = pl.by_name(sl).lo
            per = [a for (s2, _), a in tbl.items() if s2 == sl]
            for li in idxs:
                sc = pl.core_of(base + li)
                for a in per:
                    tot += a[sc]
    return tot


def measure_delay(T, times, units, tabs):
    from delay_model import DelayModel
    dm = DelayModel(run_dir=DELAY_RUNS[T], verbose=False)
    dm.model.eval()
    mods = dict(dm.model.named_modules())
    ts_ms = float(dm.config.time_step)
    sop, spk = 0, 0
    fh = {c[0]: [] for c in MESHES}
    with torch.no_grad():
        for i in range(N_TEST):
            x = bin_one(times[i], units[i], ts_ms, T)
            dm.model(torch.from_numpy(x).unsqueeze(1).float())
            rows = {}
            for mi, key, fan in ((3, "hidden1", 352), (8, "hidden2", 372)):
                s = mods[f"layers.{mi}"].spikes
                if s is None:
                    continue
                s = np.asarray(s.cpu())
                if s.ndim == 3:
                    s = s[:, 0, :]
                cnt = int((s > 0.5).sum())
                spk += cnt
                sop += cnt * fan
                for t in range(T):
                    idx = np.nonzero(s[t] > 0.5)[0].tolist()
                    if idx:
                        rows.setdefault(t, {})[key] = idx
            for cn, _, _ in MESHES:
                pl, tbl = tabs[cn]
                fh[cn].append(hops_of(rows, pl, tbl))
    return sop / N_TEST, spk / N_TEST, {k: np.mean(v) for k, v in fh.items()}


def measure_pmsn(which, T, times, units, tabs):
    from pmsn_model import SHDNet_ours
    ck = torch.load(PMSN_RUNS[(which, T)], map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = SHDNet_ours(bn=getattr(cfg, "bn", True), d_input=cfg.d_input, d_model=cfg.d_model,
                      d_out=20, d_state=cfg.d_state, dropout=cfg.dropout,
                      lr=cfg.neuron_lr, T=cfg.T)
    net.load_state_dict(ck["best_net"], strict=False)
    net.eval()
    captured, handles = {}, []
    for name, m in net.named_modules():
        if type(m).__name__ == "PMSN_neuron":
            handles.append(m.register_forward_hook(
                lambda mod, inp, out, n=name: captured.__setitem__(n, out.detach())))
    ts_ms = 1400.0 / T
    sop, spk = 0, 0
    fh = {c[0]: [] for c in MESHES}
    with torch.no_grad():
        for i in range(N_TEST):
            captured.clear()
            x = bin_one(times[i], units[i], ts_ms, T)
            net(torch.from_numpy(x).unsqueeze(0).permute(0, 2, 1))
            rows = {}
            for k, v in captured.items():
                s = v.reshape(T, 1, -1)
                cnt = int((s > 0.5).sum())
                spk += cnt
                sop += cnt * (352 if "hidden1" in k else 20)
                key = "hidden1" if "hidden1" in k else "hidden2"
                for t in range(T):
                    idx = torch.nonzero(s[t, 0, :] > 0.5, as_tuple=False).flatten().tolist()
                    if idx:
                        rows.setdefault(t, {})[key] = idx
            for cn, _, _ in MESHES:
                pl, tbl = tabs[cn]
                fh[cn].append(hops_of(rows, pl, tbl))
    for h in handles:
        h.remove()
    return sop / N_TEST, spk / N_TEST, {k: np.mean(v) for k, v in fh.items()}


def energy(model, T, sop, fh, mesh, ncores, slots):
    a = ARCH[model]
    N, ds = a["neurons"], a["d_state"]
    total_mac = sop + N * T * ds
    t_inf = total_mac / (ncores * F)

    syn = sop * E_AC
    neu = N * T * ds * E_MAC
    state = N * T * ds * 2 * E_ACC
    buffer = a["n_delay"] * T * 2 * E_ACC

    # 两种重量口径：见 README §6.4
    #   weights resident —— Dampfhoffer 的假设：权重 spatially expanded，一次突触
    #                       操作只付 E_AC 的算术代价；
    #   weights in SRAM  —— 每次突触操作另读一次 32-bit 权重，代价 E_ACC。
    # 后者是**上界**：真实核按行整读并在多次操作间复用，reuse=1 即最坏情形。
    w_read = sop * E_ACC

    sb_bits = N * ds * 32 + a["n_delay"] * slots
    w_bits = a["params"] * 32
    leak_sb = sb_bits * P_LEAK * t_inf * 1e12
    leak_w = w_bits * P_LEAK * t_inf * 1e12
    comm = fh * E_HOP

    study1 = syn + neu + state + buffer + leak_sb + comm      # Study 1 五项（权重驻留）
    study1_sram = study1 + w_read                             # 权重在 SRAM
    ts = state + buffer + leak_sb                              # 时间状态
    return dict(t_inf=t_inf * 1e6, syn=syn, neu=neu, state=state, buffer=buffer,
                leak_sb=leak_sb, comm=comm, leak_w=leak_w, w_read=w_read,
                study1=study1, study1_sram=study1_sram, ts=ts,
                ts_share=ts / study1, ts_share_sram=ts / study1_sram)


def main():
    times, units, labels = load_test()
    global N_TEST
    if N_TEST < 0:
        N_TEST = len(labels)
    print(f"测试集 {len(labels)} 个，本次用 {N_TEST} 个\n")
    tabs = build_tabs()

    print("实测（10 样本平均）")
    print(f"{'model':<7}{'T':>5}{'spike':>11}{'SOP':>12}", end="")
    for cn, _, _ in MESHES:
        print(f"{'fh-' + cn:>13}", end="")
    print()

    RAW = {}
    for T in (50, 100, 140):
        s, sp, f = measure_delay(T, times, units, tabs)
        RAW[("delay", T)] = (s, sp, f)
        print(f"{'delay':<7}{T:>5}{sp:>11,.0f}{s:>12,}", end="")
        for cn, _, _ in MESHES:
            print(f"{f[cn]:>13,.0f}", end="")
        print()
    for w in ("lif", "mc"):
        for T in (50, 100, 140):
            s, sp, f = measure_pmsn(w, T, times, units, tabs)
            RAW[(w, T)] = (s, sp, f)
            print(f"{w.upper():<7}{T:>5}{sp:>11,.0f}{s:>12,}", end="")
            for cn, _, _ in MESHES:
                print(f"{f[cn]:>13,.0f}", end="")
            print()

    SLOTS = {50: 27, 100: 41, 140: 48}
    VALID = {"delay": 352, "lif": 0, "mc": 0}

    for T in (50, 100, 140):
        print("\n" + "=" * 118)
        print(f"  T = {T}   ({1400/T:.1f} ms/frame)")
        print("=" * 118)
        for cn, nc, _ in MESHES:
            E = {}
            for m in ("delay", "lif", "mc"):
                sop, spk, fhm = RAW[(m, T)]
                E[m] = energy(m, T, sop, fhm[cn], cn, nc, SLOTS[T])
            # 按 spike 数（最粗）与 SOP（中）两种排序
            s_spk = sorted(E, key=lambda m: RAW[(m, T)][1])
            s_sop = sorted(E, key=lambda m: RAW[(m, T)][0])
            s_en = sorted(E, key=lambda m: E[m]["study1"])            # 权重驻留
            s_en_sram = sorted(E, key=lambda m: E[m]["study1_sram"])  # 权重在 SRAM

            print(f"\n--- {cn} ({nc} cores)   t_inf(delay)={E['delay']['t_inf']:.1f} us ---")
            hdr = (f"{'model':<7}{'acc':>7}{'spike':>10}{'syn':>7}{'neu':>7}"
                   f"{'state':>8}{'buf':>7}{'leak_sb':>9}{'comm':>7}"
                   f"{'w_read':>9}{'resident':>10}{'ts%':>6}"
                   f"{'+wread':>9}{'ts%':>6}{'wleak':>9}")
            print(hdr)
            print("-" * len(hdr))
            for m in ("delay", "lif", "mc"):
                e = E[m]
                print(f"{m:<7}{ACC[(m,T)]:>6.2f}%{RAW[(m,T)][1]:>10,.0f}"
                      f"{e['syn']/1e6:>7.3f}{e['neu']/1e6:>7.3f}"
                      f"{e['state']/1e6:>8.3f}{e['buffer']/1e6:>7.3f}"
                      f"{e['leak_sb']/1e6:>9.4f}{e['comm']/1e6:>7.3f}"
                      f"{e['w_read']/1e6:>9.3f}"
                      f"{e['study1']/1e6:>10.3f}{100*e['ts_share']:>6.1f}%"
                      f"{e['study1_sram']/1e6:>9.3f}{100*e['ts_share_sram']:>6.1f}%"
                      f"{e['leak_w']/1e6:>9.3f}")
            print(f"  spike 序 : {' < '.join(m for m in s_spk)}")
            print(f"  SOP   序 : {' < '.join(m for m in s_sop)}")
            print(f"  能量序(resident) : {' < '.join(m for m in s_en)}")
            print(f"  能量序(SRAM)     : {' < '.join(m for m in s_en_sram)}")
            # 判据 (a) 按白皮书原文："a non-trivial share of energy in at least one
            # brain-inspired model" -> any()，不是 all()。
            a_res = any(E[m]["ts_share"] >= 0.10 for m in E)
            a_sram_ = any(E[m]["ts_share_sram"] >= 0.10 for m in E)
            print(f"  (a) ts>=10% (any)  resident:{'Y' if a_res else 'N'}"
                  f"  SRAM:{'Y' if a_sram_ else 'N'}"
                  f"   |  (b) SOP!=E  resident:{'Y' if s_sop != s_en else 'N'}"
                  f"  SRAM:{'Y' if s_sop != s_en_sram else 'N'}")

            OUT[f"T{T}_{cn}"] = {
                "ncores": nc, "t_inf_us": E["delay"]["t_inf"],
                "sop_order": s_sop, "energy_order": s_en, "spike_order": s_spk,
                "energy_order_sram": s_en_sram,
                "a": a_res,                                              # 白皮书口径
                "a_all": all(E[m]["ts_share"] >= 0.10 for m in E),       # 更严格口径，参考
                "a_sram": a_sram_,                                       # 白皮书口径
                "a_all_sram": all(E[m]["ts_share_sram"] >= 0.10 for m in E),
                "b_sop": s_sop != s_en, "b_spike": s_spk != s_en,
                "b_sop_sram": s_sop != s_en_sram, "b_spike_sram": s_spk != s_en_sram,
                "models": {
                    m: {
                        "acc_pct": ACC[(m, T)],
                        "spikes": RAW[(m, T)][1],
                        "sop": RAW[(m, T)][0],
                        "flit_hops": RAW[(m, T)][2][cn],
                        "energy_uJ": {
                            "synaptic": E[m]["syn"] / 1e6,
                            "neuron": E[m]["neu"] / 1e6,
                            "state": E[m]["state"] / 1e6,
                            "delay_storage": E[m]["buffer"] / 1e6,
                            "leakage_sb": E[m]["leak_sb"] / 1e6,
                            "comm": E[m]["comm"] / 1e6,
                            "weight_read": E[m]["w_read"] / 1e6,
                            "total": E[m]["study1"] / 1e6,
                            "total_sram": E[m]["study1_sram"] / 1e6,
                            "weights_leak": E[m]["leak_w"] / 1e6,
                        },
                        "ts_share": E[m]["ts_share"],
                        "ts_share_sram": E[m]["ts_share_sram"],
                    } for m in ("delay", "lif", "mc")
                },
            }

    if SAVE:
        import json
        out = {
            "_meta": {
                "generated_by": "scripts/study1_10samples.py --save",
                "n_test_samples": N_TEST,
                "dataset": "SHD test split",
                "units": "energy in microjoules (uJ) per inference",
                "precision": "FP32 (32-bit floating point)",
                "note": "weights_leak reported separately, excluded from total",
                "weight_read_modes": {
                    "resident": "weights spatially expanded; a synaptic operation "
                                "costs only E_AC (Dampfhoffer 2023, L2372)",
                    "sram": "weights resident in SRAM; every synaptic operation also "
                            "pays one E_access (10 pJ) to read its 32-bit weight. "
                            "This is an UPPER bound: weight reuse would divide it.",
                    "fields": "total / ts_share = resident; "
                              "total_sram / ts_share_sram = weights in SRAM",
                },
                "prices": {k: v for k, v in (
                    ("E_AC_pJ", E_AC), ("E_MAC_pJ", E_MAC), ("E_access_pJ", E_ACC),
                    ("P_leak_W_per_bit", P_LEAK), ("E_hop_pJ", E_HOP),
                    ("F_Hz", F), ("parallelism", 1))},
                "architecture": ARCH,
            },
            "configs": OUT,
            "summary": {
                # (a) 按白皮书口径：at least one model
                "a_holds": f"{sum(v['a'] for v in OUT.values())}/{len(OUT)}",
                "a_holds_sram": f"{sum(v['a_sram'] for v in OUT.values())}/{len(OUT)}",
                # 更严格的 all-模型口径，仅作参考
                "a_all_holds": f"{sum(v['a_all'] for v in OUT.values())}/{len(OUT)}",
                "a_all_holds_sram": f"{sum(v['a_all_sram'] for v in OUT.values())}/{len(OUT)}",
                "b_sop_flip": f"{sum(v['b_sop'] for v in OUT.values())}/{len(OUT)}",
                "b_spike_flip": f"{sum(v['b_spike'] for v in OUT.values())}/{len(OUT)}",
                "b_either_flip": f"{sum(v['b_sop'] or v['b_spike'] for v in OUT.values())}/{len(OUT)}",
                "b_sop_flip_sram": f"{sum(v['b_sop_sram'] for v in OUT.values())}/{len(OUT)}",
                "b_spike_flip_sram": f"{sum(v['b_spike_sram'] for v in OUT.values())}/{len(OUT)}",
                "b_either_flip_sram": f"{sum(v['b_sop_sram'] or v['b_spike_sram'] for v in OUT.values())}/{len(OUT)}",
            },
        }
        with open(SAVE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n已保存 -> {SAVE}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="", help="把结果写成 JSON 到这个路径")
    ap.add_argument("--n", type=int, default=-1, help="样本数，-1 = 全部")
    _a = ap.parse_args()
    SAVE = _a.save
    if _a.n > 0:
        N_TEST = _a.n
    OUT = {}
    main()
