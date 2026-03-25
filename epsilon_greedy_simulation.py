"""
Epsilon-Greedy Bandit Simulation Script generated from EpsilonGreedyPrompt.md.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker

# ── 環境參數與設定 ─────────────────────────────────────────────────────────
N_RUNS = 200
TOTAL_STEPS = 10000
TRUE_MEANS = np.array([0.8, 0.7, 0.5])
STD_DEV = 1.0
EPSILONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_ARMS = len(TRUE_MEANS)
SEED = 42

rng = np.random.default_rng(SEED)

# 保存原本的 Prompt 字串
PROMPT_ARCHIVE = """# Epsilon-Greedy Bandit Simulation Prompt

Please act as an expert in Reinforcement Learning and Data Visualization. Write a complete, standalone Python script (using `numpy`, `matplotlib`, and `matplotlib.cm` for colors) that simulates the "Epsilon-Greedy" Multi-Armed Bandit algorithm and visualizes the results of tuning the epsilon hyperparameter.

### Critical Requirements for the Script Outputs:
1. **Save the Figure**: The script must save the generated 1x2 subplot as an image file named `epsilon_greedy_simulation.png`.
2. **Save the Prompt**: The script must contain a string variable holding THIS EXACT PROMPT, and write it out to a markdown file named `EpsilonGreedyPrompt_Archive.md` alongside the image.

### 【Environment & Algorithm Settings】
1. **3 Bandits (Arms)**:
   - Bandit A: True Mean = 0.8
   - Bandit B: True Mean = 0.7
   - Bandit C: True Mean = 0.5
   - Reward distribution: Normal (Gaussian) with Standard Deviation = 1.0.
2. **Budget & Strategy**: Total budget = 10,000 steps.
   - **Epsilon values to test (`eps_sweep`)**: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`
   - **Epsilon-Greedy Logic**: At each step, with probability `e`, explore by picking a random arm uniformly. With probability `1 - e`, exploit by picking the arm with the highest current empirical mean.
   - Initial values for empirical means should be 0, and the initial step counts should be 0.
3. **Runs**: To ensure statistical stability, run this entire 10,000-step process for **200 independent runs** per epsilon value. Please heavily optimize with vectorized numpy operations.

### 【Visualization Requirements】
Create a matplotlib figure with 1 row and 2 columns (approx. size 15x6).
- **Main Title**: "Epsilon-Greedy Bandit Simulation | ε sweep: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]"
- **Subtitle**: "Bandits: A=0.8, B=0.7, C=0.5 | Budget: $10,000 | 200 runs"

#### Left Subplot: Cumulative Average Return vs. Dollars Spent
1. X-axis: 0 to 10,000 steps. Y-axis: Average Return per Dollar.
2. For each epsilon value, plot the **Mean** of the cumulative average returns across the 200 runs as a solid line.
3. Plot **±1 Standard Deviation** of these cumulative averages as a very light, semi-transparent shaded area (using `fill_between`).
4. **Colors**: Use a colormap (e.g., `coolwarm`). Make `ε=0.0` dark blue and map smoothly up to `ε=0.5` as dark red.
5. Add a horizontal dotted gray line at Y=0.8 labeled "Optimal mean = 0.8".
6. **Legend**: Show each epsilon value along with its final average total reward formatted without decimals (e.g., "ε = 0.1 (total ≈ 7849)"). Place it in the lower-left.

#### Right Subplot: Total Reward vs. Epsilon Value
1. X-axis: Epsilon ($\epsilon$) showing clearly values from 0.0 to 0.5. Y-axis: Average Total Reward (limit Y axis bounds from roughly 7100 to 8400 for better scale).
2. Plot a Bar Chart showing the final *average total reward* for each epsilon value.
3. **Colors**: Ensure the bar color strictly matches the line color used for that epsilon in the Left Subplot.
4. Add text annotations centered above each bar showing the exact total reward formatted with commas (e.g., "7,849"). Make the text bold.
5. Add a horizontal dotted gray line at Y=8000 labeled "Theoretical max (8,000)" in a legend box in the top-right.

Please provide the complete, runnable Python code that handles the simulation, generates these exact plots, saves the `.png`, and writes this prompt string to `.md`."""

# 將 Prompt 寫入檔案
with open("C:\\Users\\user\\Documents\\DIC3\\EpsilonGreedyPrompt_Archive.md", "w", encoding="utf-8") as f:
    f.write(PROMPT_ARCHIVE)
print("Saved prompt archive -> EpsilonGreedyPrompt_Archive.md")

# ── 執行向量化模擬 ─────────────────────────────────────────────────────────

results = {}

for eps in EPSILONS:
    # 紀錄各 run 對每台機器的估計期望值與選擇次數
    Q = np.zeros((N_RUNS, N_ARMS))
    N_counts = np.zeros((N_RUNS, N_ARMS))
    rewards = np.zeros((N_RUNS, TOTAL_STEPS))
    
    for step in range(TOTAL_STEPS):
        # 決定是 exploration 還是 exploitation (200個 run 同時判定)
        explore_mask = rng.random(N_RUNS) < eps
        
        # Exploration: 均勻隨機選取
        explore_actions = rng.integers(0, N_ARMS, size=N_RUNS)
        
        # Exploitation: 取 Q 值最大者。加上極微小雜訊以隨機打破平手局面
        noise = rng.normal(0, 1e-8, size=(N_RUNS, N_ARMS))
        exploit_actions = np.argmax(Q + noise, axis=1)
        
        # 最終行動
        actions = np.where(explore_mask, explore_actions, exploit_actions)
        
        # 取得各自機器的回報
        step_rewards = rng.normal(loc=TRUE_MEANS[actions], scale=STD_DEV)
        rewards[:, step] = step_rewards
        
        # 更新計數與 Q 值 (Incremental update rule)
        run_indices = np.arange(N_RUNS)
        N_counts[run_indices, actions] += 1
        Q[run_indices, actions] += (step_rewards - Q[run_indices, actions]) / N_counts[run_indices, actions]

    # 計算統計數據
    cum_rewards = np.cumsum(rewards, axis=1)
    steps_arr = np.arange(1, TOTAL_STEPS + 1)
    cum_avg = cum_rewards / steps_arr
    
    mean_cum_avg = cum_avg.mean(axis=0)
    std_cum_avg  = cum_avg.std(axis=0)
    
    mean_total = rewards.sum(axis=1).mean()
    
    results[eps] = {
        'mean_cum_avg': mean_cum_avg,
        'std_cum_avg': std_cum_avg,
        'mean_total': mean_total
    }

print("Simulation finished. Plotting...")

# ── 繪圖 ──────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

fig.suptitle(
    "Epsilon-Greedy Bandit Simulation | ε sweep: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]\n"
    "Bandits: A=0.8, B=0.7, C=0.5 | Budget: $10,000 | 200 runs",
    fontsize=14, fontweight="bold"
)

# 使用 coolwarm 漸層色彩 (eps=0 深藍到 eps=0.5 深紅)
cmap = cm.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0, 1, len(EPSILONS))]

steps_arr = np.arange(1, TOTAL_STEPS + 1)

# --- 左側圖：Cumulative Average Return ---
for i, eps in enumerate(EPSILONS):
    res = results[eps]
    color = colors[i]
    label = f"ε = {eps:.1f} (total ≈ {res['mean_total']:.0f})"
    
    # 畫出平均線
    ax1.plot(steps_arr, res['mean_cum_avg'], color=color, linewidth=2, label=label)
    # 畫出極淺色變異範圍 shaded area (\u00B11 Std Dev)
    ax1.fill_between(steps_arr, 
                     res['mean_cum_avg'] - res['std_cum_avg'], 
                     res['mean_cum_avg'] + res['std_cum_avg'], 
                     color=color, alpha=0.08)

ax1.axhline(TRUE_MEANS[0], color="gray", linestyle=":", label="Optimal mean = 0.8")
ax1.set_xlim(0, TOTAL_STEPS)
ax1.set_ylim(0.3, 0.95)
ax1.set_xlabel("Dollars Spent", fontsize=11)
ax1.set_ylabel("Average Return per Dollar", fontsize=11)
ax1.set_title("Cumulative Average Return vs. Dollars Spent", fontsize=12)

# Legend 順序重排，確保 Optimal Mean 在上面
handles, labels = ax1.get_legend_handles_labels()
handles = [handles[-1]] + handles[:-1]
labels = [labels[-1]] + labels[:-1]
ax1.legend(handles, labels, loc="lower left", fontsize=10, 
           framealpha=0.9, edgecolor="#ccc")
ax1.grid(True, alpha=0.3)


# --- 右側圖：Total Reward vs Epsilon ---
x_bars = np.arange(len(EPSILONS))
bar_width = 0.55

for i, eps in enumerate(EPSILONS):
    res = results[eps]
    color = colors[i]
    total_val = res['mean_total']
    
    # 長條圖
    bar = ax2.bar(x_bars[i], total_val, width=bar_width, color=color, alpha=0.85)
    
    # 長條圖上方文字
    ax2.text(x_bars[i], total_val + 20, 
             f"{total_val:,.0f}", 
             ha="center", va="bottom", fontweight="bold", fontsize=11, color="#333")

theoretical_max = TOTAL_STEPS * TRUE_MEANS[0]
ax2.axhline(theoretical_max, color="gray", linestyle=":", label=f"Theoretical max ({theoretical_max:,.0f})")

ax2.set_xticks(x_bars)
ax2.set_xticklabels([f"{eps:.1f}" for eps in EPSILONS], fontsize=11)
ax2.set_xlabel("Epsilon (ε)", fontsize=11)
ax2.set_ylabel("Average Total Reward", fontsize=11)
ax2.set_ylim(7100, 8400)
ax2.set_title("Total Reward vs. Epsilon Value", fontsize=12)
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("C:\\Users\\user\\Documents\\DIC3\\epsilon_greedy_simulation.png", dpi=150)
plt.show()

print("Saved figure -> epsilon_greedy_simulation.png")
