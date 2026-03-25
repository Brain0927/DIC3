# Multi-Armed Bandit: Explore-then-Exploit Simulation Prompt

Please act as an expert in Reinforcement Learning and Data Visualization. Write a complete, standalone Python script (using numpy and matplotlib) that simulates an "Explore-then-Exploit" Multi-Armed Bandit strategy and visualizes the results.

### Critical Requirements for the Script Outputs:
1. **Save the Figure**: The script must save the generated 1x2 subplot as an image file named `ab_test_200runs_simulation.png`.
2. **Save the Prompt**: The script must also contain a string variable holding THIS EXACT PROMPT, and write it out to a markdown file named `ABtestPrompt_Archive.md` alongside the image.

### 【Environment & Algorithm Settings】
1. **3 Bandits (Arms)**:
   - Bandit A: True Mean = 0.8
   - Bandit B: True Mean = 0.7
   - Bandit C: True Mean = 0.5
   - Reward distribution: Normal (Gaussian) with Standard Deviation = 1.0.
2. **Budget & Strategy**: Total budget = 10,000 steps.
   - **Explore Phase**: First 2,000 steps. Choose 1 of the 3 bandits completely at random (uniform probability) at each step.
   - **Exploit Phase**: Remaining 8,000 steps. Purely exploit the bandit that had the highest estimated empirical mean during the Explore phase.
3. **Runs**: To ensure statistical stability, run this entire 10,000-step process for **200 independent runs**. Please use vectorized numpy operations for speed.

### 【Visualization Requirements】
Create a matplotlib figure with 1 row and 2 columns (approx. size 14x6).
- **Main Title**: "Explore-then-Exploit | Explore: 2,000(random) -> Exploit : 8,000 (best bandit)"
- **Subtitle**: "Bandits: A=0.8, B=0.7, C=0.5 | 200 runs"

#### Left Subplot: Cumulative Average Return vs. Dollars Spent
1. X-axis: 0 to 10,000 steps. Y-axis: Average Return per Dollar.
2. For the 200 runs, calculate the cumulative average return at each step. 
3. Plot the **Mean** of these cumulative averages across all runs as a solid blue line.
4. Plot **±1 Standard Deviation** of these cumulative averages as a light blue, semi-transparent shaded area (using `fill_between`).
5. Add a vertical dashed orange line at X=2000 labeled "Explore -> Exploit boundary ($2,000)".
6. Add a horizontal dotted red line at Y=0.8 labeled "Optimal mean (Bandit A = 0.8)".
7. Use `axvspan` to shade the background:
   - X=0 to 2000: light orange background, with text "EXPLORE (random)" inside.
   - X=2000 to 10000: light green background, with text "EXPLOIT (best bandit)" inside.

#### Right Subplot: True vs. Estimated Mean & Selection Probability
1. Grouped Bar Chart for Bandit A, Bandit B, and Bandit C.
2. For each Bandit, plot two bars side-by-side:
   - **Left Bar**: True Mean (light opacity colors: light blue, light red, light green).
   - **Right Bar**: Estimated Mean calculated *only from the random explore phase*, averaged across the 200 runs (solid colors: dark blue, dark red, dark green).
3. Above each group of bars, add data labels showing:
   - The numerical Estimated Mean (rounded to 3 decimals).
   - The percentage of times (out of 200 runs) that this specific bandit was selected as the "best bandit" moving into the Exploit phase (e.g., "(best 100%)").
4. Include a legend to distinguish between True Mean and Estimated Mean.

Please provide the complete, runnable Python code that handles the simulation, generates these exact plots, saves the `.png`, and writes this prompt string to `.md`.
