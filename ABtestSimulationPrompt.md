# Multi-Armed Bandit: Explore-then-Exploit Simulation Prompt

請扮演強化學習與資料視覺化專家的角色，使用 Python (Numpy + Matplotlib) 撰寫一段「Explore-then-Exploit」多臂賭博機策略的模擬與視覺化程式碼。
請務必精準重現我描述的雙拼圖表 (1 Row, 2 Columns) 樣式。

## 【環境與演算法設定】
1. 3 台老虎機 (Bandits)：
   - 機器 A 真實期望值為 0.8
   - 機器 B 真實期望值為 0.7
   - 機器 C 真實期望值為 0.5
   - 每次回報服從常態分佈 (Normal Distribution)，標準差 (Std Dev) 設為 1.0。
2. 預算與策略：總預算為 10,000 步。
   - 探索期 (Explore)：前 2,000 步，完全「隨機 (random)」選擇 A、B、C 其中一台。
   - 利用期 (Exploit)：剩餘的 8,000 步，固定選擇在「探索期」中實測平均回報最高的那一台機器 (best bandit)。
3. 重複實驗：為了統計穩定性，請將上述整個完整過程跑 200 次獨立實驗 (200 runs)。

## 【圖表整體要求】
- 使用 matplotlib 建立 1x2 的子圖 (subplots)，整體尺寸約 14x6。
- 總標題設定為：
  "Explore-then-Exploit | Explore: 2,000(random) -> Exploit : 8,000 (best bandit)"
- 副標題設定為："Bandits: A=0.8, B=0.7, C=0.5 | 200 runs"

## 【左側圖表：累積平均回報曲線 (Cumulative Average Return vs. Dollars Spent)】
1. X 軸為「Dollars Spent (Total Budget)」，從 0 到 10000。
2. Y 軸為「Average Return per Dollar」。
3. 將 200 次實驗的每一步「累積平均回報」計算出：
   - 平均值 (Mean)，畫出深藍色實線 (Avg Return all runs)。
   - 加減 1 倍標準差 (±1 Std Dev)，畫出淺藍色半透明陰影範圍。
4. 畫一條 X=2000 的垂直橘色虛線，標籤為 "Explore -> Exploit boundary ($2,000)"。
5. 畫一條 Y=0.8 的水平紅色點線，標籤為 "Optimal mean (Bandit A = 0.8)"。
6. 使用 axvspan 背景填色：
   - 0 到 2000 步區間：背景填極淺的黃橘色，並在區間內寫上橘色文字 "EXPLORE(random)"。
   - 2000 到 10000 步區間：背景填極淺的綠色，並在區間內寫上綠色文字 "EXPLOIT(best bandit)"。

## 【右側圖表：真實與預估均值及選擇率 (True vs. Estimated Mean)】
1. X 軸項目為 "Bandit A", "Bandit B", "Bandit C"。
2. Y 軸從 0 到 1.2。
3. 為每個 Bandit 劃出兩根並排的長條圖 (Grouped Bar Chart)：
   - 左側長條：真實平均值 (True Mean)，顏色分別為淺藍、淺紅、淺綠 (透明度較高)。
   - 右側長條：200 次實驗出來的「探索期預估平均值 (Estimated Mean)」，顏色為深藍、深紅、深綠 (不透明)。
4. 在每一組長條圖的上方，加上文字標註，包含：
   - 預估平均數值 (小數點後三位)
   - 換行標註該機器在這 200 次實驗中，最終被判定為最佳機器而選用的機率 (例如 "(best 100%)"、"(best 0%)")。
5. 請加上合適的 Legend 說明長條代表意義。

請確保程式碼具備良好的向量化 (Vectorized) 寫法以加速 200 次實驗的運算，並確保圖表版面清爽、有格線 (grid)，符合專業報告水準。
