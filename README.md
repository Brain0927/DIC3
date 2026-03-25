# 多臂賭博機演算法實驗紀錄 (Multi-Armed Bandit)
**A/B Testing vs. $\epsilon$-Greedy 策略**

本專案記錄了一次完整的強化學習（Reinforcement Learning）多臂賭博機問題的探索過程。我們透過 Python 模擬了不同策略在面對未知機率機器時的表現，並利用 Matplotlib 產生了詳細的視覺化分析圖表，最終產出了可用於讓 LLM 自動生成這些程式碼的提示詞 (Prompts)。

---

## 💡 核心環境設定
- **3 台老虎機 (Bandits)**：
  - Bandit A：真實期望值 (True Mean) = 0.8
  - Bandit B：真實期望值 (True Mean) = 0.7
  - Bandit C：真實期望值 (True Mean) = 0.5
- **回報分佈**：常態分佈 (Gaussian Distribution)，標準差 $\sigma$ = 1.0 (這導致極大的隨機雜訊，單次拉動難以辨別好壞)。
- **總預算**：10,000 步 (Dollars / Steps)。

---

## 📈 階段一：傳統 A/B 測試 (Explore-then-Exploit)

第一階段我們模擬了傳統業界最常使用的 A/B 測試：**先純探索，後純利用**。
我們將前 1,500 ~ 2,000 步作為「純探索期」，平均測試三台機器；剩下的步數全數投入探索期中表現最好的機器。

### 相關檔案：
- [`multi_armed_bandit_ab_test.py`](multi_armed_bandit_ab_test.py) - 單次執行的基礎 A/B 測試模擬。
- [`multi_armed_bandit_plots.py`](multi_armed_bandit_plots.py) - 進階視覺化套件（包含箱型圖、小提琴圖、分佈直方圖、長條圖與預估值誤差分析，以及累積後悔曲線）。
- [`multi_armed_bandit_200runs.py`](multi_armed_bandit_200runs.py) - 為了處理隨機性帶來的誤差，執行 200 次獨立實驗的進階版本。
- [`ABtestPrompt.md`](ABtestPrompt.md) / [`ABtestSimulationPrompt.md`](ABtestSimulationPrompt.md) - 能讓 AI 直接生成上述 200 runs 程式碼的最佳化提示詞。

### 📌 關鍵學習洞察 (Insights)：
1. **探索的機會成本 (Opportunity Cost)**：在純探索期，為了蒐集資料，我們被迫去拉低期望值的機器 (B 和 C)，這在折線圖上會形成一個明顯的「累積回報凹陷」。
2. **切換死板**：硬性規定何時切換，缺乏彈性。若探索期設定太短，容易選錯機器；設定太長，則浪費太多預算在劣質機器上。
3. **穩定性**：透過 200 次實驗的陰影圖 (±1 標準差)，我們能看到即使整體平均能收斂，極大標準差 ($\sigma=1.0$) 依然會讓單次實驗充滿風險。

---

## 🚀 階段二：$\epsilon$-Greedy (貪婪演算法)

第二階段我們引入了更聰明的演算法：**Epsilon-Greedy**。
這個策略不再把「探索」和「利用」分成兩個截然不同的時期，而是將兩者混合：
- 每一回合，有 $\epsilon$ 的機率（例如 10%）隨機亂選（探索）。
- 有 $1 - \epsilon$ 的機率（例如 90%）挑選當前估計值最高的機器（利用）。

### 相關檔案：
- [`epsilon_greedy_simulation.py`](epsilon_greedy_simulation.py) - 掃描多個 $\epsilon$ 值 (0.0 到 0.5) 並執行 200 次實驗的模擬腳本。
- [`EpsilonGreedyPrompt.md`](EpsilonGreedyPrompt.md) - 能讓 AI 直接生成 $\epsilon$-Greedy 程式碼的最佳化提示詞。

### 📌 關鍵學習洞察 (Insights)：
1. **探索與利用的完美平衡 (Sweet Spot)**：
   - 當 $\epsilon = 0.0$ (完全不探索)：容易陷入局部最佳解（盲目自信），一旦開局被雜訊誤導，便永不翻身。
   - 當 $\epsilon = 0.5$ (過度探索)：浪費太多預算在已經確認較差的機器上，最終報酬低下。
   - **當 $\epsilon = 0.1$**：在我們的模擬中取得最高平均總報酬 (7,849)，幾乎逼近理論上限 (8,000)。
2. **賺錢效率大勝 A/B 測試**：傳統 2000 步探索的 A/B 測試最後總回報約在 7600~7700 上下，而 $\epsilon=0.1$ 透過動態分配探索頻率，將收益提升至 7840 以上。這證明了「邊做邊學」在不確定環境下的巨大優勢。

---

## 📂 如何使用本專案的 Prompt

如果您想要用 AI 重新產生這些強大的視覺化與模擬程式，只需打開 `.md` 結尾的 Prompt 檔案並複製內容給大型語言模型 (如 ChatGPT, Claude, Gemini)：
- `ABtestPrompt.md` -> 產出 Explore-then-Exploit 200 次實驗雙拼圖
- `EpsilonGreedyPrompt.md` -> 產出 $\epsilon$-Greedy 參數掃描與對比圖
- `UCBPrompt.md` -> 產出 UCB (Upper Confidence Bound) 的參數掃描與對應長條圖
- `ThompsonSamplingPrompt.md` -> 產出「業界最強大」的湯普森採樣，並直接跨演算法 PK (大戰) 貪婪演算法與 UCB

圖表本身會自動儲存為 `.png`，並確保色彩映射、標籤與文字標註符合專業數據科學簡報水準。
