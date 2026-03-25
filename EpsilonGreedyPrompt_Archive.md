# Epsilon-Greedy Bandit Simulation Prompt

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

Please provide the complete, runnable Python code that handles the simulation, generates these exact plots, saves the `.png`, and writes this prompt string to `.md`.