"""Study 1 交付图：堆叠能量分解 + spike 数叠加。

数据：2,264 个 SHD 测试样本实测，32-bit FP32 解析能量模型。
白皮书要求："Stacked energy breakdown for the three models, with spike counts overlaid."
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\yangz\Desktop\deepseek工位\figs"
os.makedirs(OUT, exist_ok=True)
MODELS = ["delay", "LIF", "MC"]
LABEL = {"delay": "Delay-trained\n(DelRec 352)", "LIF": "Plain LIF\n(PMSN d=1)", "MC": "Multi-compartment\n(PMSN d=4)"}

# T -> mesh -> model -> 项 (µJ)
E = {
    50: {
        "2x2": {"delay": dict(syn=1.288, neu=0.167, state=0.724, buf=0.352, leak=0.0713, comm=0.044, tot=2.646),
                "LIF":   dict(syn=1.251, neu=0.162, state=0.704, buf=0.000, leak=0.0478, comm=0.100, tot=2.265),
                "MC":    dict(syn=1.380, neu=0.648, state=2.816, buf=0.000, leak=0.2244, comm=0.110, tot=5.178)},
        "4x4": {"delay": dict(syn=1.288, neu=0.167, state=0.724, buf=0.352, leak=0.0178, comm=0.299, tot=2.847),
                "LIF":   dict(syn=1.251, neu=0.162, state=0.704, buf=0.000, leak=0.0119, comm=0.694, tot=2.823),
                "MC":    dict(syn=1.380, neu=0.648, state=2.816, buf=0.000, leak=0.0561, comm=0.772, tot=5.672)},
        "8x8": {"delay": dict(syn=1.288, neu=0.167, state=0.724, buf=0.352, leak=0.0045, comm=2.150, tot=4.685),
                "LIF":   dict(syn=1.251, neu=0.162, state=0.704, buf=0.000, leak=0.0030, comm=5.028, tot=7.148),
                "MC":    dict(syn=1.380, neu=0.648, state=2.816, buf=0.000, leak=0.0140, comm=5.569, tot=10.426)},
    },
    100: {
        "2x2": {"delay": dict(syn=2.524, neu=0.333, state=1.448, buf=0.704, leak=0.1609, comm=0.086, tot=5.256),
                "LIF":   dict(syn=1.628, neu=0.324, state=1.408, buf=0.000, leak=0.0630, comm=0.158, tot=3.580),
                "MC":    dict(syn=2.109, neu=1.295, state=5.632, buf=0.000, leak=0.3518, comm=0.178, tot=9.566)},
        "4x4": {"delay": dict(syn=2.524, neu=0.333, state=1.448, buf=0.704, leak=0.0402, comm=0.593, tot=5.642),
                "LIF":   dict(syn=1.628, neu=0.324, state=1.408, buf=0.000, leak=0.0157, comm=1.080, tot=4.455),
                "MC":    dict(syn=2.109, neu=1.295, state=5.632, buf=0.000, leak=0.0880, comm=1.250, tot=10.374)},
        "8x8": {"delay": dict(syn=2.524, neu=0.333, state=1.448, buf=0.704, leak=0.0101, comm=4.270, tot=9.289),
                "LIF":   dict(syn=1.628, neu=0.324, state=1.408, buf=0.000, leak=0.0039, comm=7.749, tot=11.112),
                "MC":    dict(syn=2.109, neu=1.295, state=5.632, buf=0.000, leak=0.0220, comm=9.018, tot=18.076)},
    },
    140: {
        "2x2": {"delay": dict(syn=3.587, neu=0.466, state=2.027, buf=0.986, leak=0.2436, comm=0.122, tot=7.432),
                "LIF":   dict(syn=2.330, neu=0.453, state=1.971, buf=0.000, leak=0.0901, comm=0.221, tot=5.066),
                "MC":    dict(syn=2.643, neu=1.814, state=7.885, buf=0.000, leak=0.4465, comm=0.240, tot=13.028)},
        "4x4": {"delay": dict(syn=3.587, neu=0.466, state=2.027, buf=0.986, leak=0.0609, comm=0.835, tot=7.962),
                "LIF":   dict(syn=2.330, neu=0.453, state=1.971, buf=0.000, leak=0.0225, comm=1.519, tot=6.297),
                "MC":    dict(syn=2.643, neu=1.814, state=7.885, buf=0.000, leak=0.1116, comm=1.661, tot=14.114)},
        "8x8": {"delay": dict(syn=3.587, neu=0.466, state=2.027, buf=0.986, leak=0.0152, comm=5.974, tot=13.056),
                "LIF":   dict(syn=2.330, neu=0.453, state=1.971, buf=0.000, leak=0.0056, comm=10.941, tot=15.701),
                "MC":    dict(syn=2.643, neu=1.814, state=7.885, buf=0.000, leak=0.0279, comm=11.933, tot=24.303)},
    },
}

SPIKES = {50: {"delay": 3909, "LIF": 8688, "MC": 9565},
          100: {"delay": 7667, "LIF": 13930, "MC": 15672},
          140: {"delay": 10891, "LIF": 19523, "MC": 21026}}
ACC = {50: {"delay": 86.56, "LIF": 88.96, "MC": 91.21},
       100: {"delay": 87.24, "LIF": 86.40, "MC": 88.30},
       140: {"delay": 87.99, "LIF": 87.10, "MC": 88.96}}
SOP = {50: {"delay": 1431230, "LIF": 1389789, "MC": 1533348},
       100: {"delay": 2804570, "LIF": 1808584, "MC": 2343106},
       140: {"delay": 3985830, "LIF": 2589276, "MC": 2937174}}

PARTS = [("syn", "synaptic ops (AC)", "#4C72B0"),
         ("neu", "neuron update", "#7FA8D6"),
         ("state", "state memory", "#DD8452"),
         ("buf", "delay storage", "#C44E52"),
         ("leak", "leakage (state+buf)", "#E8B04B"),
         ("comm", "communication", "#55A868")]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def panel(ax, T, mesh, title):
    xs = np.arange(3)
    bottoms = np.zeros(3)
    for key, lab, col in PARTS:
        vals = np.array([E[T][mesh][m][key] for m in MODELS])
        ax.bar(xs, vals, 0.62, bottom=bottoms, label=lab, color=col,
               edgecolor="white", linewidth=0.6)
        bottoms += vals
    bottom_max = float(bottoms.max())
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=8)
    ax.set_ylabel("energy / inference (µJ)")
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_ylim(0, bottom_max * 1.32)
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)

    for x, m in zip(xs, MODELS):
        spk = SPIKES[T][m] / 1000
        ax.annotate(f"{E[T][mesh][m]['tot']:.2f} µJ", (x, bottoms[x]),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, fontweight="bold")
        ax.annotate(f"{spk:.1f}k spikes", (x, bottom_max * 1.24),
                    ha="center", fontsize=8, color="#222222", fontweight="bold")
        ax.annotate("●", (x, bottom_max * 1.155), ha="center",
                    fontsize=9, color="#222222")


# ---------------- 图 1：主图（T=100, 2x2） ----------------
fig, ax = plt.subplots(figsize=(7.6, 4.8))
panel(ax, 100, "2x2", "Study 1 — T = 100 (14 ms/frame), 2×2 mesh (4 cores)\n"
                     "stacked energy per inference, with spike counts overlaid")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7.8,
          frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "study1_main_T100_2x2.png"), bbox_inches="tight")
plt.close(fig)

# ---------------- 图 2：3 分辨率 x 2 mesh ----------------
fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=False)
for j, T in enumerate((50, 100, 140)):
    panel(axes[0, j], T, "2x2", f"T = {T} ({1400/T:.0f} ms/frame), 2×2")
    panel(axes[1, j], T, "8x8", f"T = {T} ({1400/T:.0f} ms/frame), 8×8")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.015))
fig.suptitle("Study 1 — energy breakdown across time-step resolutions and mesh sizes",
             fontsize=11, y=0.98)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(os.path.join(OUT, "study1_grid_resolution_mesh.png"), bbox_inches="tight")
plt.close(fig)

# ---------------- 图 3：三种估计的排序对比 ----------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
for ax, T in zip(axes, (50, 100, 140)):
    sp = sorted(MODELS, key=lambda m: SPIKES[T][m])
    so = sorted(MODELS, key=lambda m: SOP[T][m])
    en = sorted(MODELS, key=lambda m: E[T]["2x2"][m]["tot"])
    names = ["spike count", "SOP (× fanout)", "full model energy"]
    rank = [sp, so, en]
    cmap = {"delay": "#C44E52", "LIF": "#4C72B0", "MC": "#DD8452"}
    short = {"delay": "DL", "LIF": "LIF", "MC": "MC"}
    for i, r in enumerate(rank):
        for pos, m in enumerate(r):
            ax.scatter(pos, -i, s=520, color=cmap[m], zorder=3,
                       edgecolor="white", linewidth=1.5)
            ax.annotate(short[m], (pos, -i), ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold", zorder=4)
    ax.set_yticks([0, -1, -2])
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["1st (cheapest)", "2nd", "3rd (most expensive)"], fontsize=8)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-2.5, 0.6)
    flip_sop = "≠" if so != en else "="
    flip_spk = "≠" if sp != en else "="
    ax.set_title(f"T = {T}, 2×2 mesh\nspike{flip_spk}energy   SOP{flip_sop}energy", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
fig.suptitle("Simplified estimates mis-rank the three models", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "study1_ranking_flip.png"), bbox_inches="tight")
plt.close(fig)

print("saved:")
for f in ("study1_main_T100_2x2.png", "study1_grid_resolution_mesh.png",
          "study1_ranking_flip.png"):
    p = os.path.join(OUT, f)
    print(" ", p, f"{os.path.getsize(p)/1024:.0f} KB")
