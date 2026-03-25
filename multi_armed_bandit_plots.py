"""
Multi-Armed Bandit: Extended Visualization Suite
=================================================
Produces 5 additional chart types beyond the basic cumulative-average curve:
  1. Box Plot      – reward distribution per arm (explore phase samples)
  2. Violin Plot   – smoother KDE view of the same distributions
  3. Histogram     – overlapping per-arm reward histograms
  4. Bar Chart     – empirical vs. true means with error bars
  5. Regret Curve  – cumulative regret vs. optimal policy over time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── Environment (must match main simulation) ──────────────────────────────────
TOTAL_STEPS    = 10_000
EXPLORE_STEPS  = 1_500
EXPLOIT_STEPS  = TOTAL_STEPS - EXPLORE_STEPS

ARM_NAMES      = ["A", "B", "C"]
TRUE_MEANS     = np.array([0.8, 0.7, 0.5])
STD_DEV        = 1.0
N_ARMS         = len(ARM_NAMES)
STEPS_PER_ARM  = EXPLORE_STEPS // N_ARMS   # 500 each

COLORS         = ["#2196F3", "#FF9800", "#9C27B0"]   # Blue, Orange, Purple
BEST_ARM_IDX   = 0   # Arm A is optimal


# ─────────────────────────────────────────────────────────────────────────────
# Data generation (re-run exactly as in main script)
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_data():
    """Simulate the full A/B test and return raw sample arrays."""
    explore_samples = []
    for arm in range(N_ARMS):
        samples = rng.normal(loc=TRUE_MEANS[arm], scale=STD_DEV, size=STEPS_PER_ARM)
        explore_samples.append(samples)

    # Empirical means → pick best arm
    arm_means  = np.array([s.mean() for s in explore_samples])
    best_arm   = int(np.argmax(arm_means))

    # Exploration rewards in chronological order (A then B then C)
    explore_rewards = np.concatenate(explore_samples)

    # Exploitation rewards
    exploit_rewards = rng.normal(loc=TRUE_MEANS[best_arm], scale=STD_DEV, size=EXPLOIT_STEPS)

    all_rewards = np.concatenate([explore_rewards, exploit_rewards])

    return explore_samples, arm_means, best_arm, all_rewards


explore_samples, arm_means, best_arm, all_rewards = generate_all_data()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: cumulative regret
# ─────────────────────────────────────────────────────────────────────────────

def compute_cumulative_regret(all_rewards: np.ndarray) -> np.ndarray:
    """
    Regret = (optimal expected return per step) – (actual reward).
    Cumulative regret is the running sum of per-step regret.
    """
    optimal = TRUE_MEANS[BEST_ARM_IDX]
    per_step_regret = optimal - all_rewards
    return np.cumsum(per_step_regret)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 – Box Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplot(ax):
    bp = ax.boxplot(
        explore_samples,
        patch_artist=True,
        notch=True,          # notched = 95% CI around median
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set(color="white", linewidth=2)

    # Overlay true means as dashed horizontal segments
    for i, mean in enumerate(TRUE_MEANS):
        ax.axhline(mean, xmin=(i * 1/3) + 0.05, xmax=((i + 1) * 1/3) - 0.05,
                   color=COLORS[i], linewidth=2, linestyle="--", alpha=0.9)

    # ── Annotations ──────────────────────────────────────────────────────────
    # Label IQR range for Arm A (box spans Q1 to Q3 ≈ ±0.674σ)
    q1_a, q3_a = np.percentile(explore_samples[0], [25, 75])
    ax.annotate(
        f"IQR≈{q3_a - q1_a:.2f}\n(~0.67σ×2)",
        xy=(1, q1_a), xytext=(1.55, q1_a - 0.4),
        fontsize=7.5, color=COLORS[0],
        arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=0.8),
    )
    # Point out whisker extent (≈ 1.5 × IQR rule)
    ax.text(3.38, 2.9, "Whisker\n=1.5×IQR", fontsize=7, color="#555", va="top", ha="left")
    ax.annotate("", xy=(3, max(explore_samples[2])),
                xytext=(3.35, 2.9),
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
    # Mark the best arm
    ax.text(1, TRUE_MEANS[0] + 0.12, "★ Best arm", ha="center",
            fontsize=8, color=COLORS[0], fontweight="bold")
    # Note: notch = 95% CI of median
    ax.text(0.62, -2.6, "Notch = 95% CI of median", fontsize=7, color="#777")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"Arm {n}" for n in ARM_NAMES])
    ax.set_ylabel("Reward")
    ax.set_title("① Box Plot — Reward Distribution per Arm\n(dashed = true mean)")
    ax.grid(axis="y", alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 – Violin Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_violin(ax):
    vp = ax.violinplot(
        explore_samples,
        positions=[1, 2, 3],
        showmeans=True,
        showmedians=True,
        widths=0.6,
    )
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(COLORS[i])
        body.set_alpha(0.6)
    vp["cmeans"].set_color("black")    # black tick = mean
    vp["cmedians"].set_color("white")  # white tick = median

    # ── Annotations ──────────────────────────────────────────────────────────
    # Explain mean / median markers
    ax.text(3.38, 1.05, "─ Mean",   fontsize=7.5, color="black", va="center")
    ax.text(3.38, 0.55, "─ Median", fontsize=7.5, color="white",
            va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="#555", ec="none", alpha=0.7))
    # Annotate width = probability density
    ax.annotate(
        "Width ∝\nprobability\ndensity",
        xy=(1.35, 0.8), xytext=(1.7, -1.5),
        fontsize=7.5, color="#333", ha="center",
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
    )
    # Note the heavy σ=1 overlap
    ax.text(1.5, -3.5,
            "All arms share σ=1 → large visual overlap\ndespite different true means",
            fontsize=7.5, color="#555", ha="center",
            style="italic")
    for i, m in enumerate(arm_means):
        ax.text(i + 1, m + 0.18, f"μ̂={m:.2f}", ha="center",
                fontsize=7.5, color=COLORS[i], fontweight="bold")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"Arm {n}" for n in ARM_NAMES])
    ax.set_ylabel("Reward")
    ax.set_title("② Violin Plot — KDE of Reward Distributions")
    ax.grid(axis="y", alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 – Overlapping Histograms
# ─────────────────────────────────────────────────────────────────────────────

def plot_histogram(ax):
    bins = np.linspace(-3, 4, 50)
    for i, (samples, name) in enumerate(zip(explore_samples, ARM_NAMES)):
        ax.hist(
            samples, bins=bins, alpha=0.5, color=COLORS[i],
            label=f"Arm {name} (μ={TRUE_MEANS[i]})", density=True,
        )
        ax.axvline(samples.mean(), color=COLORS[i], linestyle="--",
                   linewidth=1.5, alpha=0.9)

    # ── Annotations ──────────────────────────────────────────────────────────
    # Label each empirical mean line at the top
    for i, (samples, name) in enumerate(zip(explore_samples, ARM_NAMES)):
        ax.text(samples.mean() + 0.08, 0.46 - i * 0.04,
                f"Arm {name}\nμ̂={samples.mean():.3f}",
                fontsize=7, color=COLORS[i], va="top")
    # Note the ±1σ range that covers 68% of rewards
    ax.annotate(
        "±1σ covers\n~68% of rewards",
        xy=(TRUE_MEANS[0] + STD_DEV, 0.22),
        xytext=(2.8, 0.38),
        fontsize=7.5, color="#333",
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
    )
    # Bracket showing ±1σ for Arm A
    ax.annotate("",
        xy=(TRUE_MEANS[0] - STD_DEV, 0.20),
        xytext=(TRUE_MEANS[0] + STD_DEV, 0.20),
        arrowprops=dict(arrowstyle="<->", color="#888", lw=1.2))
    ax.text(TRUE_MEANS[0], 0.21, "2σ", ha="center", fontsize=7.5, color="#555")
    # Highlight challenge: distributions overlap heavily
    ax.text(-2.8, 0.50,
            "Heavy overlap → hard to\ndistinguish arms from\na single sample",
            fontsize=7.5, color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8))

    ax.set_xlabel("Reward Value")
    ax.set_ylabel("Density")
    ax.set_title("③ Histogram — Overlapping Reward Distributions\n(dashed = empirical mean)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 – Bar Chart: Empirical vs. True Means
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_comparison(ax):
    x = np.arange(N_ARMS)
    width = 0.35

    # Standard errors for empirical means (SE = σ / √n)
    std_errors = np.array([s.std() / np.sqrt(len(s)) for s in explore_samples])

    bars_true = ax.bar(x - width/2, TRUE_MEANS, width,
                       label="True Mean", color="#607D8B", alpha=0.8)
    bars_emp  = ax.bar(x + width/2, arm_means, width, yerr=std_errors,
                       label="Empirical Mean (±SE)", color=COLORS, alpha=0.85,
                       capsize=6, error_kw={"elinewidth": 2})

    # Value labels on top of each bar
    for bar in bars_true:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_emp:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    # ── Annotations ──────────────────────────────────────────────────────────
    # Highlight best arm
    ax.annotate(
        f"★ Best arm selected\n(μ̂={arm_means[best_arm]:.3f} vs true {TRUE_MEANS[best_arm]})",
        xy=(best_arm + width/2, arm_means[best_arm] + std_errors[best_arm] + 0.04),
        xytext=(best_arm + 0.65, arm_means[best_arm] + 0.25),
        fontsize=7.5, color="red", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=0.9),
    )
    # Annotate SE formula for Arm A
    ax.text(-0.35, 0.04,
            f"SE = σ/√n\n= {explore_samples[0].std():.2f}/√{STEPS_PER_ARM}\n≈ {std_errors[0]:.3f}",
            fontsize=7, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))
    # Gap annotation between true and empirical for Arm A
    gap = abs(TRUE_MEANS[0] - arm_means[0])
    ax.annotate("",
        xy=(0 - width/2, TRUE_MEANS[0]),
        xytext=(0 + width/2, arm_means[0]),
        arrowprops=dict(arrowstyle="<->", color="#E91E63", lw=1.2))
    ax.text(0.02, (TRUE_MEANS[0] + arm_means[0]) / 2,
            f"Δ={gap:.3f}", fontsize=7, color="#E91E63", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Arm {n}" for n in ARM_NAMES])
    ax.set_ylabel("Mean Return")
    ax.set_ylim(0, 1.25)
    ax.set_title("④ Bar Chart — True vs. Empirical Mean\n(error bars = ±standard error)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 – Cumulative Regret Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_regret(ax):
    steps = np.arange(1, TOTAL_STEPS + 1)
    cum_regret = compute_cumulative_regret(all_rewards)

    ax.plot(steps, cum_regret, color="#F44336", linewidth=1.8,
            label="Cumulative Regret (A/B Test)")

    # Linear regret baseline for random policy (always pulls random arm)
    random_baseline = (TRUE_MEANS[BEST_ARM_IDX] - TRUE_MEANS.mean()) * steps
    ax.plot(steps, random_baseline, color="#9E9E9E", linewidth=1.2,
            linestyle="--", label="Random Policy Baseline")

    ax.axvline(EXPLORE_STEPS, color="#FF5722", linewidth=1.4, linestyle=":",
               label=f"Explore→Exploit (step {EXPLORE_STEPS})")

    # ── Annotations ──────────────────────────────────────────────────────────
    # 1. Steep slope during exploration
    ax.annotate(
        "Steep slope:\nExploration forces\nsub-optimal arms",
        xy=(750, cum_regret[749]),
        xytext=(1800, cum_regret[749] - 80),
        fontsize=8, color="#333",
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd", alpha=0.9),
    )
    # 2. Exploitation phase — slope flattens
    exploit_mid = EXPLORE_STEPS + (EXPLOIT_STEPS // 2)
    ax.annotate(
        "Slope flattens:\nOnly Arm A pulled\n(regret≈0 per step)",
        xy=(exploit_mid, cum_regret[exploit_mid - 1]),
        xytext=(exploit_mid + 800, cum_regret[exploit_mid - 1] - 60),
        fontsize=8, color="#1565C0",
        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#90CAF9", alpha=0.9),
    )
    # 3. Final cumulative regret value
    final_regret = cum_regret[-1]
    ax.text(9800, final_regret + 15,
            f"Final regret\n= {final_regret:.0f}",
            fontsize=8, color="#F44336", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#F44336", alpha=0.9))
    # 4. Show gap vs random policy at end
    final_random = random_baseline[-1]
    ax.annotate("",
        xy=(9950, final_regret),
        xytext=(9950, final_random),
        arrowprops=dict(arrowstyle="<->", color="#4CAF50", lw=1.5))
    ax.text(9400, (final_regret + final_random) / 2,
            f"Saved\n{final_random - final_regret:.0f}\nvs random",
            fontsize=7.5, color="#2E7D32", ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#E8F5E9", ec="#A5D6A7", alpha=0.9))
    # 5. Label transition point
    ax.text(EXPLORE_STEPS + 100, 20,
            f"← Step {EXPLORE_STEPS:,}\nSwitch to Arm A",
            fontsize=8, color="#FF5722", va="bottom")

    ax.set_xlabel("Dollars Spent / Steps")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("⑤ Regret Curve — Cumulative Regret vs. Optimal\n"
                 "(lower is better; slope flattens after exploitation starts)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Compose 2×3 figure
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "Multi-Armed Bandit A/B Test — Extended Visualization Suite\n"
    f"(Arms A/B/C, true means {TRUE_MEANS}, σ={STD_DEV}, "
    f"explore {EXPLORE_STEPS} | exploit {EXPLOIT_STEPS} steps)",
    fontsize=14, fontweight="bold", y=1.01,
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1:])   # regret curve spans two columns

plot_boxplot(ax1)
plot_violin(ax2)
plot_histogram(ax3)
plot_bar_comparison(ax4)
plot_regret(ax5)

plt.savefig(
    "C:\\Users\\user\\Documents\\DIC3\\ab_test_extended_plots.png",
    dpi=150, bbox_inches="tight",
)
plt.show()
print("Chart saved → ab_test_extended_plots.png")
