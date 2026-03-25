# Thompson Sampling Bandit Simulation Prompt

Please act as an expert in Reinforcement Learning and Data Visualization. Write a complete, standalone Python script (using `numpy`, `matplotlib`, and `scipy.stats`) that simulates the "Thompson Sampling (Gaussian)" Multi-Armed Bandit algorithm and visualizes the results against Epsilon-Greedy, UCB, and A/B Testing.

### Critical Requirements for the Script Outputs:
1. **Save the Figure**: The script must save the generated 1x3 subplot as an image file named `thompson_sampling_comparison.png`.
2. **Save the Prompt**: The script must contain a string variable holding THIS EXACT PROMPT, and write it out to a markdown file named `ThompsonSamplingPrompt_Archive.md` alongside the image.

### 【Environment Settings】
1. **3 Bandits (Arms)**: Bandit A: True Mean = 0.8, Bandit B: True Mean = 0.7, Bandit C: True Mean = 0.5
2. **Reward distribution**: Normal (Gaussian) with Standard Deviation ($\sigma$) = 1.0.
3. **Budget & Runs**: Total budget = 10,000 steps. Conduct exactly **200 independent runs** for stability, perfectly vectorized via numpy arrays.

### 【Algorithms to Simulate & Compare】
1. **Epsilon-Greedy** ($\epsilon=0.1$): 10% explore, 90% exploit highest empirical mean.
2. **UCB** ($c=1.0$): Pulled based on $\hat{\mu}_a + c \sqrt{\ln(t)/N_a}$.
3. **Thompson Sampling (Gaussian/Normal Conjugate Prior)**:
   - For a known reward variance $\sigma^2=1$, assume prior mean $\mu_0=0$, prior variance $\tau_0^2 = 100$.
   - The posterior parameters after $n$ pulls with sum of rewards $S$ are:
     $\tau_n^2 = \frac{1}{\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}}$ and $\mu_n = \tau_n^2 \left(\frac{\mu_0}{\tau_0^2} + \frac{S}{\sigma^2}\right)$
   - At each step, sample $\theta_a \sim \text{Normal}(\mu_n, \tau_n)$ for each arm, and pull the arm with max $\theta_a$.

### 【Visualization Requirements】
Create a matplotlib figure with 1 row, 3 columns (approx. size 18x6).
- **Main Title**: "MAB Algorithm Battle: Thompson Sampling vs UCB vs ε-Greedy"
- **Subtitle**: "Gaussian Bandits (A=0.8, B=0.7, C=0.5) | std=1.0 | 200 runs"

#### Subplot 1: Cumulative Regret vs Steps
- X-axis: 0 to 10,000 steps. Y-axis: Cumulative Regret.
- Plot the Mean Cumulative Regret over 200 runs for:
  - $\epsilon=0.1$-Greedy (Blue)
  - UCB $c=1.0$ (Orange)
  - Thompson Sampling (Green)
- Add shaded ±1 standard deviation bounds around only the Thompson Sampling curve to keep the plot clean.
- Ensure Thompson Sampling (typically logarithmic regret) visually outperforms the linear slope of $\epsilon$-greedy. 

#### Subplot 2: Total Average Reward (Bar Chart)
- Plot a bar chart comparing the three algorithms for final total reward collected after 10,000 steps.
- Add Theoretical Max (8,000) as a dotted red line.
- Annotate exact values strongly formatted with commas on top of each bar.

#### Subplot 3: Arm Selection Breakdown (Stacked Bar Chart)
- For the three algorithms + "Oracle" (which always picks A), plot the final 10,000 step allocation percentage.
- Stacked Bar Chart: X-axis = Algorithm Name, Y-axis = Percentage (0 to 100%).
- Show the percentage of times Bandit A (Best), B, and C were selected across all 200 runs combined.
- E.g., Thompson Sampling might be (99% A, 0.8% B, 0.2% C). Color code Arm A (Blue), Arm B (Orange), Arm C (Green).

Please provide the complete, runnable Python code that handles the simulations, generates these distinct plots, saves the `.png`, and writes this prompt string to `.md`.
