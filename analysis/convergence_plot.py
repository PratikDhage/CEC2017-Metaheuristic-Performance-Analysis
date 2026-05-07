import numpy as np
import matplotlib.pyplot as plt
import os

# Load history (Shape: [TotalConfigs, Runs, Iterations])
history_path = "results/all_histories.npy"
histories = np.load(history_path, allow_pickle=True)

# Define your algorithm names in the order they were run
algos = ["PSO", "ABC", "GA", "DE", "WOA"]
n_algos = len(algos)

plt.figure(figsize=(10, 6))

for i, name in enumerate(algos):
    # Take the first function's history (Index 0)
    # Average across all 30 runs
    avg_convergence = np.mean(histories[i], axis=0)
    
    plt.plot(avg_convergence, label=name, linewidth=2)

plt.yscale('log')
plt.title("Convergence Curve: F1 (D=10)")
plt.xlabel("Iterations")
plt.ylabel("Best Fitness (Log Scale)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/plots/convergence_F1.png")
print("[✓] Convergence plot saved to results/plots/convergence_F1.png")