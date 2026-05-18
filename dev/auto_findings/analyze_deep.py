"""
analyze_deep.py — deeper analysis layer beyond progress.png.

Produces four plots that go beyond "improvement over time":

  1. knob_attribution.png — waterfall of how each KEEP commit contributed
  2. throughput_vs_capability.png — scatter val_bpb vs total_tokens_M,
     fits a Chinchilla-style scaling curve, identifies throughput vs HP regime
  3. diminishing_returns.png — log-scale gain per iteration, asymptote estimate
  4. cost_per_finding.png — $ spent vs information gained

Hardcoded session data tables below — the headline TSVs don't carry tokens_M
or num_steps, so we cite from the run logs preserved in the findings docs.

Run: python dev/auto_findings/analyze_deep.py
Output: dev/auto_findings/plots/deep_*.png
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =============================================================================
# Curated data from sessions 1+2 (val_bpb + tokens_M + steps; pulled from
# run logs preserved in findings docs).
# =============================================================================

# Format: (label, val_bpb, tokens_M, steps, num_params_M, status, session, is_keep)
EXPERIMENTS = [
    # session 1
    ("S1 baseline (MoE)",          1.120724, 122.2, 466, 499.2, "keep",    1, True),
    ("S1 anchor (dense+K-HPs)",    1.084925, 205.5, 392,  50.3, "keep",    1, True),
    ("S1 anchor v2 (+SSSL)",       1.081073, 208.7, 398,  50.3, "keep",    1, True),
    ("S1 iter1 (MoE+MATRIX=0.04)", 1.109074, 125.0, 477,  49.3, "keep",    1, True),
    ("S1 iter2 (+EMBED=0.6)",      1.105286, 124.8, 476,  49.3, "keep",    1, True),
    ("S1 iter3 (NUM_EXPERTS=8)",   1.142020,  92.8, 354,  49.3, "discard", 1, False),
    ("S1 iter4 (dense @iter2 HPs)",1.062571, 190.1, 725,  50.3, "keep",    1, True),
    # session 2
    ("S2 baseline (MoE)",          1.127507, 119.0, 454,  49.3, "keep",    2, True),
    ("S2 iter1 (replica)",         1.062026, 191.1, 729,  50.3, "keep",    2, True),
    ("S2 iter2 (MATRIX=0.06)",     1.063568, 191.4, 730,  50.3, "discard", 2, False),
    ("S2 iter3 (BATCH=2^19)",      1.099455, 194.0, 370,  50.3, "discard", 2, False),
    ("S2 iter4 (WARMUP=0)",        1.087347, 190.3, 726,  50.3, "discard", 2, False),
    ("S2 iter5 (+WD=0.2)",         1.061015, 191.4, 730,  50.3, "keep",    2, True),
    ("S2 iter6 (+EMBED=0.8)",      1.060142, 192.2, 733,  50.3, "keep",    2, True),
    ("S2 iter7 (+β1=0.8)",         1.058859, 192.4, 734,  50.3, "keep",    2, True),
    ("S2 iter8 (+UNEMBED=0.004)",  1.058836, 192.7, 735,  50.3, "keep",    2, True),
    ("S2 iter9 (+SSSL)",           1.057578, 194.5, 742,  50.3, "keep",    2, True),
]

# Knob attribution — sequence of cumulative wins on the best track.
# Each row: (knob_change_description, val_bpb_after, delta_from_prev).
ATTRIBUTION = [
    ("0. baseline (MoE-on, conservative LRs)",     1.120724,  0.000),
    ("1. switch to dense (NUM_EXPERTS=1)",         1.084925, -0.036),  # using anchor as proxy
    ("2. MATRIX_LR 0.02 → 0.04 + EMBED_LR → 0.6",  1.062571, -0.022),  # iter4 of session 1
    ("3. WEIGHT_DECAY 0.0 → 0.2",                  1.061015, -0.002),
    ("4. EMBEDDING_LR 0.6 → 0.8",                  1.060142, -0.001),
    ("5. ADAM_BETAS β1 0.9 → 0.8",                 1.058859, -0.001),
    ("6. UNEMBEDDING_LR 0.008 → 0.004",            1.058836, -0.00002),
    ("7. WINDOW_PATTERN L → SSSL",                 1.057578, -0.001),
]

KARPATHY_PUBLISHED = 0.998   # d=8 dense baseline from Karpathy's autoresearch readme
KARPATHY_TOKENS_M = 500       # he runs 500M tokens in 5min on H100 + FA3

OUT_DIR = Path("dev/auto_findings/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Plot 1 — Knob attribution waterfall
# =============================================================================
def plot_knob_attribution():
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [r[0] for r in ATTRIBUTION]
    vals = [r[1] for r in ATTRIBUTION]
    deltas = [r[2] for r in ATTRIBUTION]

    xs = list(range(len(labels)))
    # bars showing each step's improvement (or starting baseline as a single tall bar)
    ax.bar(xs[0], vals[0] - KARPATHY_PUBLISHED, bottom=KARPATHY_PUBLISHED,
           color="#1f77b4", alpha=0.4, label="baseline value above Karpathy floor")

    for i in range(1, len(labels)):
        ax.bar(xs[i], -deltas[i], bottom=vals[i], color="#2ca02c", alpha=0.85,
               edgecolor="white", linewidth=0.5)
        # annotate delta
        ax.text(xs[i], vals[i] + (-deltas[i])/2, f"{deltas[i]:+.4f}",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # connecting line for cumulative best
    ax.plot(xs, vals, color="#1f77b4", linewidth=2, marker="o", zorder=5, label="cumulative best")

    # Karpathy reference
    ax.axhline(KARPATHY_PUBLISHED, color="#aaa", linestyle="--", linewidth=1.5,
               label=f"Karpathy published ({KARPATHY_PUBLISHED:.3f})")
    ax.fill_between(xs, KARPATHY_PUBLISHED, KARPATHY_PUBLISHED - 0.05,
                    color="#aaa", alpha=0.1)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("val_bpb (lower = better)", fontsize=11)
    ax.set_title("knob attribution — what gave the 0.063 total improvement", fontsize=12, pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0.99, 1.14)

    # summary
    total_gain = vals[0] - vals[-1]
    summary = (
        f"baseline:    {vals[0]:.4f}\n"
        f"final best:  {vals[-1]:.4f}\n"
        f"total gain:  {total_gain:.4f} ({100*total_gain/vals[0]:.1f}%)\n"
        f"remaining to Karpathy: {vals[-1] - KARPATHY_PUBLISHED:.4f}"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            family="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#999"))

    plt.tight_layout()
    out = OUT_DIR / "deep_knob_attribution.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# =============================================================================
# Plot 2 — Throughput vs capability (val_bpb vs tokens_M, with Chinchilla fit)
# =============================================================================
def plot_throughput_capability():
    fig, ax = plt.subplots(figsize=(11, 6))

    # Filter to ~50M active param experiments (apples-to-apples per Chinchilla).
    # Tuple indices: 0=label, 1=val_bpb, 2=tokens_M, 3=steps, 4=num_params_M, 5=status, 6=session, 7=is_keep.
    pts = [r for r in EXPERIMENTS if r[4] in (50.3, 49.3)]
    keeps = [r for r in pts if r[7]]
    discards = [r for r in pts if not r[7]]

    ax.scatter([r[2] for r in keeps], [r[1] for r in keeps],
               c="#2ca02c", s=90, alpha=0.85, edgecolors="white", linewidths=0.7,
               label=f"keep (n={len(keeps)})", zorder=3)
    ax.scatter([r[2] for r in discards], [r[1] for r in discards],
               c="#d62728", marker="x", s=110, linewidths=2.5,
               label=f"discard (n={len(discards)})", zorder=3)

    # Annotate the four most informative points
    annotate_labels = {
        "S1 iter4 (dense @iter2 HPs)": (10, 12),
        "S2 iter9 (+SSSL)":            (10, -20),
        "S1 baseline (MoE)":           (-10, 15),
        "S1 iter3 (NUM_EXPERTS=8)":    (10, 12),
    }
    for r in pts:
        if r[0] in annotate_labels:
            offset = annotate_labels[r[0]]
            ax.annotate(r[0].replace("S1 ", "").replace("S2 ", ""),
                        xy=(r[2], r[1]), xytext=offset, textcoords="offset points",
                        fontsize=8, color="#333")

    # Chinchilla-style fit: val_bpb = a + b * tokens_M^(-c)
    # Use only KEEPS at 50.3M params for the fit.
    tokens = np.array([r[2] for r in keeps])
    vals = np.array([r[1] for r in keeps])
    # Simple log-linear fit: log(val_bpb - asymptote) = log(b) - c*log(tokens)
    # Try a few asymptotes; use Karpathy's 0.998 as a reasonable lower bound.
    asym = 0.95   # below Karpathy slightly to allow extrapolation
    log_t = np.log(tokens)
    log_v = np.log(vals - asym)
    c_fit, log_b_fit = np.polyfit(log_t, log_v, 1)
    c_fit = -c_fit
    b_fit = np.exp(log_b_fit)
    print(f"[fit] val_bpb ≈ {asym:.3f} + {b_fit:.4f} * tokens_M^(-{c_fit:.3f})")

    # Plot the fit
    t_range = np.linspace(80, 600, 200)
    fit_curve = asym + b_fit * t_range**(-c_fit)
    ax.plot(t_range, fit_curve, color="#1f77b4", linewidth=2, alpha=0.7,
            label=f"fit: val_bpb ≈ {asym:.3f} + {b_fit:.3f}·tokens^(-{c_fit:.2f})")

    # Karpathy reference point
    ax.scatter([KARPATHY_TOKENS_M], [KARPATHY_PUBLISHED], marker="*", s=400,
               c="#ff7f0e", edgecolors="black", linewidths=1.5, zorder=4,
               label=f"Karpathy published ({KARPATHY_TOKENS_M}M tokens, val_bpb={KARPATHY_PUBLISHED:.3f}) — FA3")
    ax.axhline(KARPATHY_PUBLISHED, color="#aaa", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("total tokens trained (M) in 5-min budget", fontsize=11)
    ax.set_ylabel("val_bpb (lower = better)", fontsize=11)
    ax.set_title("throughput vs val_bpb — Chinchilla scaling explains the FA2↔FA3 gap", fontsize=12, pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)

    # Prediction: if we got Karpathy's throughput, what val_bpb?
    predicted_at_500 = asym + b_fit * KARPATHY_TOKENS_M**(-c_fit)
    summary = (
        f"Chinchilla-style fit: val_bpb ≈ {asym} + {b_fit:.3f}·T^(-{c_fit:.2f})\n"
        f"Our best (195M tokens, FA2): {min(r[1] for r in keeps):.4f}\n"
        f"Karpathy (500M tokens, FA3):  {KARPATHY_PUBLISHED:.4f}\n"
        f"Fit prediction at 500M:      {predicted_at_500:.4f}\n"
        f"→ if FA3 doubles throughput, predicted val_bpb: {asym + b_fit * (2*195)**(-c_fit):.4f}\n"
        f"  → would close {100*(1.0576 - (asym + b_fit*(2*195)**(-c_fit)))/(1.0576 - 0.998):.0f}% of gap to Karpathy"
    )
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=9,
            family="monospace", verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#999"))

    plt.tight_layout()
    out = OUT_DIR / "deep_throughput_vs_capability.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# =============================================================================
# Plot 3 — Diminishing returns
# =============================================================================
def plot_diminishing_returns():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    iters = list(range(1, len(ATTRIBUTION)))
    gains = [-ATTRIBUTION[i][2] for i in iters]
    ax.bar(iters, gains, color="#2ca02c", alpha=0.85, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("knob iteration (in attribution order)", fontsize=11)
    ax.set_ylabel("val_bpb gain (log scale)", fontsize=11)
    ax.set_title("diminishing returns per knob", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    ax.axhline(0.0001, color="#aaa", linestyle="--", label="noise floor (~0.0001)")
    ax.axhline(0.001, color="#888", linestyle=":", label="signal threshold (~0.001)")
    ax.legend()
    for i, g in zip(iters, gains):
        ax.text(i, g, f"{g:.4f}", ha="center", va="bottom", fontsize=8)

    # Right panel: cumulative gain
    ax = axes[1]
    cum = np.cumsum(gains)
    ax.plot([0] + list(iters), [0] + list(cum), "o-", color="#1f77b4", linewidth=2)
    ax.set_xlabel("knob iteration", fontsize=11)
    ax.set_ylabel("cumulative val_bpb gain", fontsize=11)
    ax.set_title("cumulative improvement — flattening", fontsize=12)
    ax.grid(alpha=0.3)
    # Predict where curve plateaus
    if len(gains) >= 4:
        # Fit exponential decay: gains[i] ≈ A * r^i
        log_gains = np.log(np.array(gains[1:]))   # skip first big jump
        i_arr = np.array(iters[1:])
        slope, intercept = np.polyfit(i_arr, log_gains, 1)
        decay = np.exp(slope)
        amp = np.exp(intercept)
        # Predict total remaining gain (geometric series sum)
        next_i = iters[-1] + 1
        remaining = amp * decay**next_i / (1 - decay) if decay < 1 else float("inf")
        predicted_floor = cum[-1] + remaining
        ax.axhline(predicted_floor, color="#d62728", linestyle="--",
                   label=f"predicted asymptote: +{predicted_floor:.4f} cumulative")
        ax.text(0.5, predicted_floor, f"  HP-only ceiling ≈ +{predicted_floor:.4f}",
                color="#d62728", fontsize=9, va="bottom")
    ax.legend()

    plt.tight_layout()
    out = OUT_DIR / "deep_diminishing_returns.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# =============================================================================
# Plot 4 — Cost per finding
# =============================================================================
def plot_cost_per_finding():
    fig, ax = plt.subplots(figsize=(11, 5))
    # Cumulative cost and cumulative best val_bpb after each experiment
    cost_per_exp = 4.51 / 7  # session 1 cost / experiments
    cost_per_exp_s2 = 5.51 / 10

    cum_cost = []
    cum_best = []
    running_best = float("inf")
    running_cost = 0.0
    for r in EXPERIMENTS:
        c = cost_per_exp if r[6] == 1 else cost_per_exp_s2
        running_cost += c
        if r[1] > 0:
            running_best = min(running_best, r[1])
        cum_cost.append(running_cost)
        cum_best.append(running_best)

    ax.plot(cum_cost, cum_best, "o-", color="#1f77b4", linewidth=2, markersize=8,
            label="best val_bpb so far")
    ax.set_xlabel("cumulative $ spent on compute", fontsize=11)
    ax.set_ylabel("best val_bpb", fontsize=11)
    ax.set_title("cost efficiency — $ per unit val_bpb improvement", fontsize=12, pad=12)
    ax.grid(alpha=0.3)
    ax.axhline(KARPATHY_PUBLISHED, color="#aaa", linestyle="--",
               label=f"Karpathy ({KARPATHY_PUBLISHED:.3f})")

    # Add markers for session boundaries
    s1_end = sum(1 for r in EXPERIMENTS if r[6] == 1) * cost_per_exp
    ax.axvline(s1_end, color="#888", linestyle=":", alpha=0.5)
    ax.text(s1_end, 1.13, "  ← session 1 end", fontsize=9, color="#666", va="top")

    # Estimate $/val_bpb
    total_cost = cum_cost[-1]
    total_gain = EXPERIMENTS[0][1] - min(r[1] for r in EXPERIMENTS if r[1] > 0)
    summary = (
        f"total compute: ~${total_cost:.2f}\n"
        f"total gain:    {total_gain:.4f} val_bpb\n"
        f"$/0.001 val_bpb: ${total_cost/(total_gain*1000):.2f}"
    )
    ax.text(0.55, 0.95, summary, transform=ax.transAxes, fontsize=9,
            family="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#999"))
    ax.legend(loc="lower left")

    plt.tight_layout()
    out = OUT_DIR / "deep_cost_per_finding.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    plot_knob_attribution()
    plot_throughput_capability()
    plot_diminishing_returns()
    plot_cost_per_finding()
    print("\ndone. plots in dev/auto_findings/plots/deep_*.png")
