import numpy as np
import matplotlib.pyplot as plt

# ── 環境設定 ─────────────────────────────────────────────────────────────────
TOTAL_STEPS = 10_000
STEPS_PER_BANDIT = TOTAL_STEPS // 3  # 每台老虎機分配到的步數 (3333)

TRUE_MEANS = [0.8, 0.7, 0.5]
BANDIT_NAMES = ["Bandit A", "Bandit B", "Bandit C"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 對應圖片中的藍色、橘色、綠色
STD_DEV = 1.0

np.random.seed(42)

# ── 繪圖 ─────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))

for i in range(3):
    # 為每台機器生成獨立的報酬樣本
    rewards = np.random.normal(loc=TRUE_MEANS[i], scale=STD_DEV, size=STEPS_PER_BANDIT)
    
    # 計算累積平均回報
    cum_avg = np.cumsum(rewards) / np.arange(1, STEPS_PER_BANDIT + 1)
    
    # 繪製曲線
    plt.plot(cum_avg, label=BANDIT_NAMES[i], color=COLORS[i])

# ── 圖表樣式設定（參考使用者提供的截圖） ─────────────────────────────────────
plt.title("A/B Test Simulation: Average Return vs Dollars Spent", fontsize=14)
plt.xlabel("Dollars Spent per Bandit", fontsize=12)
plt.ylabel("Average Return", fontsize=12)
plt.legend(loc="lower right", framealpha=1.0)
plt.grid(True)

plt.tight_layout()
plt.savefig("C:\\Users\\user\\Documents\\DIC3\\reference_plot.png", dpi=150)
plt.show()

print("已產生存檔：reference_plot.png")
