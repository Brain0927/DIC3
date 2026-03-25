# UCB (Upper Confidence Bound) Bandit Simulation Prompt

Please act as an expert in Reinforcement Learning and Data Visualization. Write a complete, standalone Python script (using `numpy`, `matplotlib`, and `matplotlib.cm` for colors) that simulates the "UCB (Upper Confidence Bound) / UCB1" Multi-Armed Bandit algorithm and visualizes the results of tuning the exploration parameter $c$.

### Critical Requirements for the Script Outputs:
1. **Save the Figure**: The script must save the generated 1x2 subplot as an image file named `ucb_simulation.png`.
2. **Save the Prompt**: The script must contain a string variable holding THIS EXACT PROMPT, and write it out to a markdown file named `UCBPrompt_Archive.md` alongside the image.

### 【Environment & Algorithm Settings】
1. **3 Bandits (Arms)**:
   - Bandit A: True Mean = 0.8
   - Bandit B: True Mean = 0.7
   - Bandit C: True Mean = 0.5
   - Reward distribution: Normal (Gaussian) with Standard Deviation = 1.0.
2. **Budget & Strategy**: Total budget = 10,000 steps.
   - **c values to test (`c_sweep`)**: `[0.1, 0.5, 1.0, 2.0, 5.0]`
   - **UCB Logic**: At step $t$, for each arm $a$, calculate $UCB_a = \hat{\mu}_a + c \cdot \sqrt{\frac{\ln(t)}{N_a}}$. Exploit by picking the arm with the highest $UCB_a$. If any arm has $N_a=0$, it must be pulled first (play each arm once at the beginning).
3. **Runs**: To ensure statistical stability, run this entire 10,000-step process for **200 independent runs** per $c$ value. Please heavily optimize with vectorized numpy operations.

### 【Visualization Requirements】
Create a matplotlib figure with 1 row and 2 columns (approx. size 15x6).
- **Main Title**: "UCB (Upper Confidence Bound) Simulation | c parameter sweep"
- **Subtitle**: "Bandits: A=0.8, B=0.7, C=0.5 | Budget: $10,000 | 200 runs"

#### Left Subplot: Cumulative Average Return vs. Dollars Spent
1. X-axis: 0 to 10,000 steps. Y-axis: Average Return per Dollar.
2. For each $c$ value, plot the **Mean** of the cumulative average returns across the 200 runs as a solid line.
3. Plot **±1 Standard Deviation** of these cumulative averages as a very light, semi-transparent shaded area (using `fill_between`).
4. **Colors**: Use a colormap (e.g., `viridis` or `YlOrRd`). Ensure the line colors map distinctly to the 5 `c` values.
5. Add a horizontal dotted gray line at Y=0.8 labeled "Optimal mean = 0.8".
6. **Legend**: Show each $c$ value along with its final average total reward formatted without decimals (e.g., "c = 1.0 (total ≈ 7910)"). Place it in the lower-right.

#### Right Subplot: Total Reward vs. c Value
1. X-axis: Exploration strictness ($c$). Y-axis: Average Total Reward (limit Y bounds appropriately from roughly 7200 to 8200).
2. Plot a Bar Chart showing the final *average total reward* for each $c$ value.
3. **Colors**: Ensure the bar color strictly matches the line color used for that $c$ value in the Left Subplot.
4. Add text annotations centered above each bar showing the exact total reward formatted with commas (e.g., "7,910"). Make the text bold.
5. Add a horizontal dotted gray line at Y=8000 labeled "Theoretical max (8,000)".

Please provide the complete, runnable Python code that handles the simulation, generates these exact plots, saves the `.png`, and writes this prompt string to `.md`.
