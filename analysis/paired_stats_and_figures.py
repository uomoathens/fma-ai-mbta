import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "matlab")
FIG = ROOT / "figures"
OUT = ROOT / "analysis"
FIG.mkdir(exist_ok=True)

M = ["Centralised", "Non-Federated MARL", "FMA-AI (FedAvg)"]
C = ["#2a78d6", "#eb6834", "#1baf7a"]
MK = ["s", "D", "o"]
LS = ["-", "--", ":"]


def ci95(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def paired_table(seeds):
    rows = []
    pairs = [(M[2], M[0]), (M[2], M[1]), (M[1], M[0])]
    for split in ["train", "holdout"]:
        d = seeds[seeds.split == split]
        for col, name in [("dE_percent", "dE"), ("OTP_percent", "OTP")]:
            p = d.pivot(index="seed", columns="method", values=col)
            for a, b in pairs:
                diff = p[a] - p[b]
                rows.append({"split": split, "indicator": name, "comparison": f"{a} minus {b}",
                             "mean_difference": diff.mean(), "ci95": ci95(diff)})
    return pd.DataFrame(rows)


def convergence_table(seeds):
    d = seeds[seeds.split == "train"]
    return d.groupby("method").R_conv.agg(["median", "min", "max", "count"]).reset_index()


def figure_tradeoff(summary, sweep):
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    tr = summary[summary.split == "train"]
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for m, c, mk, ls in zip(M, C, MK, LS):
        d = sweep[sweep.method == m].sort_values("OTP_tolerance_pp")
        ax.plot(d.OTP_mean, d.dE_mean, ls=ls, color=c, lw=1.6, marker=mk, ms=5, mfc="white", mew=1.4, label=f"{m}, tolerance sweep")
    for m, c, mk in zip(M, C, MK):
        r = tr[tr.method == m].iloc[0]
        ax.errorbar(r.OTP_mean, r.dE_mean, xerr=r.OTP_ci95, yerr=r.dE_ci95, fmt=mk, color=c, ms=8, mec="white", mew=1.2,
                    elinewidth=1.4, capsize=3, label=f"{m}, main run (20 seeds)", zorder=5)
    ax.plot(tr.OTP_base_mean.iloc[0], 0, "kx", ms=9, mew=2, label="Timetable baseline", zorder=6)
    d = sweep[sweep.method == M[0]].sort_values("OTP_tolerance_pp")
    for _, r in d.iterrows():
        ax.annotate(f"{int(r.OTP_tolerance_pp)} pp", (r.OTP_mean, r.dE_mean), textcoords="offset points", xytext=(7, 6), fontsize=8, color="#444")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("On-time performance, OTP (%)")
    ax.set_ylabel("Reduction in energy and CO$_2$ (%)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_tradeoff.png", dpi=300)


def figure_convergence(conv):
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    ax.grid(True, color="#e6e6e6", lw=0.6)
    for m, c, ls in zip(M, C, LS):
        ax.semilogy(conv["round"], conv[m].rolling(20, min_periods=1).mean(), ls=ls, color=c, lw=1.8, label=m)
    ax.axhline(2e-3, color="#444", lw=1, ls=(0, (4, 3)))
    ax.text(conv["round"].max(), 2.15e-3, r"$\delta_\theta = 2\times10^{-3}$", ha="right", va="bottom", fontsize=8.5, color="#333")
    ax.set_xlabel("Communication round $k$")
    ax.set_ylabel(r"$\|\theta^{(k)}-\theta^{(k-1)}\|_2$, 20-round moving mean")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_convergence.png", dpi=300)


def main():
    seeds = pd.read_csv(RES / "results_seeds.csv")
    summary = pd.read_csv(RES / "results_summary.csv")
    sweep = pd.read_csv(RES / "tolerance_sweep.csv")
    conv = pd.read_csv(RES / "convergence.csv")
    paired_table(seeds).round(4).to_csv(OUT / "paired_differences.csv", index=False)
    convergence_table(seeds).to_csv(OUT / "convergence_rounds.csv", index=False)
    figure_tradeoff(summary, sweep)
    figure_convergence(conv)
    print(paired_table(seeds).round(3).to_string(index=False))
    print(convergence_table(seeds).to_string(index=False))


if __name__ == "__main__":
    main()
