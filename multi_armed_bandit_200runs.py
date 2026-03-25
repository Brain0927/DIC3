"""
Explore-then-Exploit Multi-Armed Bandit Simulation (200 Runs)
=============================================================
This script generates the 1x2 dual subplot requested in the prompt.
1. Left: Cumulative Average Return over 10,000 steps with shaded +/-1 std dev.
2. Right: Grouped bar chart comparing true means vs estimated means and showing selection probability.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 環境與演算法設定 ────────────────────────────────────────────────────────
N_RUNS        = 200
TOTAL_STEPS   = 10_000
EXPLORE_STEPS = 2_000
EXPLOIT_STEPS = TOTAL_STEPS - EXPLORE_STEPS

TRUE_MEANS    = np.array([0.8, 0.7, 0.5])
STD_DEV       = 1.0
N_ARMS        = len(TRUE_MEANS)
ARM_NAMES     = ["Bandit A", "Bandit B", "Bandit C"]

SEED          = 42
rng           = np.random.default_rng(SEED)

# ── 執行 200 次蒙地卡羅模擬 (向量化加速) ────────────────────────────────────

# 1. 探索期：完全隨機選擇機器 (Shape: 200 runs x 2000 steps)
explore_actions = rng.integers(0, N_ARMS, size=(N_RUNS, EXPLORE_STEPS))
explore_rewards = rng.normal(loc=TRUE_MEANS[explore_actions], scale=STD_DEV)

# 計算各機器在探索期內的實測平均值 (Estimated Mean)
estimated_means = np.zeros((N_RUNS, N_ARMS))
for arm in range(N_ARMS):
    mask = (explore_actions == arm)
    sums = (explore_rewards * mask).sum(axis=1)    # 每個 run 對應機器的總和
    counts = mask.sum(axis=1)                      # 每個 run 抽到該機器的次數
    # 避免除以零 (若某個 run 剛好 2000 次都沒抽到某機器)
    estimated_means[:, arm] = np.divide(sums, counts, out=np.zeros_like(sums), where=counts!=0)

# 找出每回合實測最佳的機器索引
best_bandits = np.argmax(estimated_means, axis=1)  # Shape: (200,)

# 2. 利用期：針對最好的機器持續投入 (Shape: 200 runs x 8000 steps)
# TRUE_MEANS[best_bandits] 會取出每個 run 的最佳機器的真實平均
exploit_rewards = rng.normal(loc=TRUE_MEANS[best_bandits][:, None], scale=STD_DEV, size=(N_RUNS, EXPLOIT_STEPS))

# 3. 合併結果並計算統計數據
all_rewards = np.concatenate([explore_rewards, exploit_rewards], axis=1)
cumulative_sums = np.cumsum(all_rewards, axis=1)
steps_arr = np.arange(1, TOTAL_STEPS + 1)
cumulative_avg = cumulative_sums / steps_arr

# 形狀皆為 (10000,)
mean_cum_avg = cumulative_avg.mean(axis=0)
std_cum_avg  = cumulative_avg.std(axis=0)

# 機器選擇率 & 平均預估值
selection_probs = np.zeros(N_ARMS)
unique, counts = np.unique(best_bandits, return_counts=True)
for u, c in zip(unique, counts):
    selection_probs[u] = c / N_RUNS

mean_estimated_means = estimated_means.mean(axis=0)


# ── 繪圖 ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Explore-then-Exploit | Explore: 2,000(random) $\\rightarrow$ Exploit : 8,000 (best bandit)\n"
    f"Bandits: A=0.8, B=0.7, C=0.5 | {N_RUNS} runs",
    fontsize=14, fontweight="bold"
)

# ---------------------------------------------------------
# 左圖：累積平均回報曲線
# ---------------------------------------------------------
ax1.plot(steps_arr, mean_cum_avg, color="#1f77b4", linewidth=2.5, label="Avg Return (all runs)")
ax1.fill_between(steps_arr, 
                 mean_cum_avg - std_cum_avg, 
                 mean_cum_avg + std_cum_avg, 
                 color="#1f77b4", alpha=0.2, label="±1 Std Dev")

# 垂直虛線 (探索/利用分界)
ax1.axvline(EXPLORE_STEPS, color="orange", linestyle="--", linewidth=1.5, 
            label=f"Explore $\\rightarrow$ Exploit boundary (${EXPLORE_STEPS:,})")

# 水平點線 (最佳真實期望值)
ax1.axhline(TRUE_MEANS[0], color="red", linestyle=":", linewidth=1.5, 
            label=f"Optimal mean (Bandit A = {TRUE_MEANS[0]})")

# 背景填色
ax1.axvspan(0, EXPLORE_STEPS, facecolor="orange", alpha=0.08)
ax1.axvspan(EXPLORE_STEPS, TOTAL_STEPS, facecolor="green", alpha=0.08)

# 文字標記
ax1.text(EXPLORE_STEPS / 2, 0.45, "EXPLORE\n(random)", color="darkorange", 
         ha="center", va="center", fontweight="bold", fontsize=11)
ax1.text(EXPLORE_STEPS + (EXPLOIT_STEPS / 2), 0.45, "EXPLOIT\n(best bandit)", color="darkgreen", 
         ha="center", va="center", fontweight="bold", fontsize=11)

ax1.set_xlim(0, TOTAL_STEPS)
ax1.set_ylim(0.35, 0.95)
ax1.set_xlabel("Dollars Spent (Total Budget)", fontsize=11)
ax1.set_ylabel("Average Return per Dollar", fontsize=11)
ax1.set_title("Cumulative Average Return vs. Dollars Spent", fontsize=12)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)


# ---------------------------------------------------------
# 右圖：真實預估均值與選擇機率
# ---------------------------------------------------------
x = np.arange(N_ARMS)
width = 0.35

# 顏色設定 (淺色與深色)
colors_light = ["#c6cbf8", "#f8c1b6", "#a1edd1"]  # A, B, C (亮色)
colors_dark  = ["#5a6bf7", "#f06346", "#2fcf9e"]  # A, B, C (深色)

bars_true = ax2.bar(x - width/2, TRUE_MEANS, width, label="True Mean", color=colors_light)
bars_est  = ax2.bar(x + width/2, mean_estimated_means, width, label="Estimated Mean\n(from random explore)", color=colors_dark)

# 文字標記 (數值與勝出率)
for i in range(N_ARMS):
    # 預估平均值放上方
    ax2.text(x[i] + width/2, mean_estimated_means[i] + 0.02, 
             f"{mean_estimated_means[i]:.3f}\n(best {selection_probs[i]*100:.0f}%)", 
             ha="center", va="bottom", fontsize=10)

ax2.set_xticks(x)
ax2.set_xticklabels(ARM_NAMES, fontsize=11)
ax2.set_ylabel("Mean Return", fontsize=11)
ax2.set_ylim(0, 1.2)
ax2.set_title("True vs. Estimated Mean\n(% = how often chosen as best bandit)", fontsize=12)

# 自訂 Legend 合併標籤
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#d6dcfb", edgecolor='none', label='True Mean'),
    Patch(facecolor="#5a6bf7", edgecolor='none', label='Estimated Mean\n(from random explore)')
]
ax2.legend(handles=legend_elements, loc="upper right", fontsize=10)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.85)  # 讓總標題有空間
plt.savefig("C:\\Users\\user\\Documents\\DIC3\\ab_test_200runs_simulation.png", dpi=150)
plt.show()

print("已產生存檔：ab_test_200runs_simulation.png")
