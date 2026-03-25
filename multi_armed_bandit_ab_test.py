"""
Multi-Armed Bandit: Traditional A/B Testing Simulation
=======================================================
Strategy:
  - Exploration phase : First 1500 steps, evenly split across 3 arms (500 each)
  - Exploitation phase: Remaining 8500 steps, all-in on the best-performing arm

Distribution choice: Gaussian (Normal) with std=1.0
  Rationale: Gaussian rewards are more general and realistic for continuous
  return scenarios (e.g., revenue, click value). Bernoulli is appropriate
  for binary outcomes (click / no-click); a comment block at the bottom
  shows how to swap to Bernoulli if desired.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── Environment Configuration ─────────────────────────────────────────────────
TOTAL_STEPS       = 10_000          # Total budget (dollars / steps)
EXPLORE_STEPS     = 1_500           # Exploration phase budget
EXPLOIT_STEPS     = TOTAL_STEPS - EXPLORE_STEPS   # 8500

ARM_NAMES         = ["A", "B", "C"]
TRUE_MEANS        = np.array([0.8, 0.7, 0.5])    # Intrinsic expected returns
STD_DEV           = 1.0                            # Gaussian noise std

N_ARMS            = len(ARM_NAMES)
STEPS_PER_ARM     = EXPLORE_STEPS // N_ARMS        # 500 per arm


# ─────────────────────────────────────────────────────────────────────────────
# Reward samplers
# ─────────────────────────────────────────────────────────────────────────────

def sample_gaussian(arm_idx: int, n: int) -> np.ndarray:
    """Sample n Gaussian rewards from arm `arm_idx`."""
    return rng.normal(loc=TRUE_MEANS[arm_idx], scale=STD_DEV, size=n)


# Bernoulli alternative (swap in by replacing sample_gaussian calls):
# def sample_bernoulli(arm_idx: int, n: int) -> np.ndarray:
#     """Sample n Bernoulli rewards; TRUE_MEANS[arm_idx] used as success prob."""
#     return rng.binomial(n=1, p=TRUE_MEANS[arm_idx], size=n).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – Exploration
# ─────────────────────────────────────────────────────────────────────────────

def run_exploration() -> tuple[np.ndarray, np.ndarray, int]:
    """
    Uniformly pull each arm STEPS_PER_ARM times.

    Returns
    -------
    explore_rewards : ndarray shape (EXPLORE_STEPS,)  – per-step rewards
    arm_means       : ndarray shape (N_ARMS,)          – empirical means
    best_arm        : int                              – index of best arm
    """
    explore_rewards = np.empty(EXPLORE_STEPS)

    for i, arm in enumerate(range(N_ARMS)):
        start = i * STEPS_PER_ARM
        end   = start + STEPS_PER_ARM
        explore_rewards[start:end] = sample_gaussian(arm, STEPS_PER_ARM)

    # Empirical means per arm
    arm_means = np.array([
        explore_rewards[i * STEPS_PER_ARM:(i + 1) * STEPS_PER_ARM].mean()
        for i in range(N_ARMS)
    ])

    best_arm = int(np.argmax(arm_means))
    return explore_rewards, arm_means, best_arm


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – Exploitation
# ─────────────────────────────────────────────────────────────────────────────

def run_exploitation(best_arm: int) -> np.ndarray:
    """
    Pull the best arm for all remaining EXPLOIT_STEPS.

    Returns
    -------
    exploit_rewards : ndarray shape (EXPLOIT_STEPS,)
    """
    return sample_gaussian(best_arm, EXPLOIT_STEPS)


# ─────────────────────────────────────────────────────────────────────────────
# Cumulative average helper
# ─────────────────────────────────────────────────────────────────────────────

def cumulative_average(rewards: np.ndarray) -> np.ndarray:
    """Vectorised cumulative mean of a 1-D reward array."""
    return np.cumsum(rewards) / np.arange(1, len(rewards) + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    cum_avg      : np.ndarray,
    arm_means    : np.ndarray,
    best_arm     : int,
) -> None:
    """
    Plot cumulative average return over all steps.
    A vertical dashed line marks the exploration → exploitation transition.
    """
    steps = np.arange(1, TOTAL_STEPS + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # ── Main curve ────────────────────────────────────────────────────────────
    ax.plot(steps, cum_avg, color="#2196F3", linewidth=1.8,
            label="Cumulative Avg Return (A/B Test)")

    # ── True optimal reference line ───────────────────────────────────────────
    ax.axhline(TRUE_MEANS[0], color="#4CAF50", linewidth=1.2,
               linestyle="--", label=f"True best mean (Arm A = {TRUE_MEANS[0]})")

    # ── Transition marker ─────────────────────────────────────────────────────
    ax.axvline(EXPLORE_STEPS, color="#FF5722", linewidth=1.6,
               linestyle=":", label=f"Explore→Exploit transition (step {EXPLORE_STEPS})")

    # Annotation for transition point
    ax.annotate(
        f"Switch to Arm {ARM_NAMES[best_arm]}\n"
        f"(empirical means: {', '.join(f'{ARM_NAMES[i]}={arm_means[i]:.3f}' for i in range(N_ARMS))})",
        xy=(EXPLORE_STEPS, cum_avg[EXPLORE_STEPS - 1]),
        xytext=(EXPLORE_STEPS + 300, cum_avg[EXPLORE_STEPS - 1] - 0.15),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#555"),
        color="#333",
    )

    # ── Labels & style ────────────────────────────────────────────────────────
    ax.set_xlabel("Dollars Spent / Steps", fontsize=12)
    ax.set_ylabel("Cumulative Average Return", fontsize=12)
    ax.set_title(
        "Multi-Armed Bandit – Traditional A/B Testing\n"
        f"(Gaussian rewards, σ={STD_DEV}; Explore {EXPLORE_STEPS} | Exploit {EXPLOIT_STEPS})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ab_test_result.png", dpi=150)
    plt.show()
    print("Chart saved → ab_test_result.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Exploration ───────────────────────────────────────────────────────────
    explore_rewards, arm_means, best_arm = run_exploration()

    print("=" * 50)
    print("EXPLORATION PHASE RESULTS")
    print("=" * 50)
    for i, name in enumerate(ARM_NAMES):
        marker = " ← best" if i == best_arm else ""
        print(f"  Arm {name}: empirical mean = {arm_means[i]:.4f}{marker}")
    print(f"\nSelected arm for exploitation: {ARM_NAMES[best_arm]}")

    # ── Exploitation ──────────────────────────────────────────────────────────
    exploit_rewards = run_exploitation(best_arm)

    # ── Combine & compute cumulative average ──────────────────────────────────
    all_rewards = np.concatenate([explore_rewards, exploit_rewards])
    cum_avg     = cumulative_average(all_rewards)

    print("\nSUMMARY")
    print("=" * 50)
    print(f"  Total steps          : {TOTAL_STEPS:,}")
    print(f"  Exploration steps    : {EXPLORE_STEPS:,}")
    print(f"  Exploitation steps   : {EXPLOIT_STEPS:,}")
    print(f"  Final cumulative avg : {cum_avg[-1]:.4f}")
    print(f"  True optimal mean    : {TRUE_MEANS[best_arm]:.4f}")
    print("=" * 50)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(cum_avg, arm_means, best_arm)


if __name__ == "__main__":
    main()
