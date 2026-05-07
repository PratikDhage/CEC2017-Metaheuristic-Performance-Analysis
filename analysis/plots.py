import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- VS CODE PATH FIX ---
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_plots():
    output_dir = os.path.join(project_root, "results", "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths to data
    raw_data_path = os.path.join(project_root, "results", "all_algo_results.csv")
    history_path = os.path.join(project_root, "results", "all_histories.npy")
    rank_dir = os.path.join(project_root, "results", "analysis_output")
    
    if not os.path.exists(raw_data_path):
        print("Data files missing. Ensure experiments are run.")
        return

    df = pd.read_csv(raw_data_path)

    # 1. BOX PLOT (Error Distribution)
    for dim in df["dim"].unique():
        plt.figure(figsize=(12, 6))
        sub_df = df[df["dim"] == dim]
        sns.boxplot(x="func_id", y="error_val", hue="algorithm", data=sub_df)
        plt.yscale("log")
        plt.title(f"Error Value Distribution per Function (D={dim})")
        plt.xlabel("Function ID")
        plt.ylabel("Error Value (Log Scale)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(os.path.join(output_dir, f"boxplot_D{dim}.png"))
        plt.close()
    print("[✓] Boxplots saved.")

    # 2. FRIEDMAN RANK BAR CHART
    rank_files = [f for f in os.listdir(rank_dir) if f.startswith("friedman_ranks")]
    for rf in rank_files:
        dim_str = rf.split("_D")[-1].replace(".csv", "")
        ranks_df = pd.read_csv(os.path.join(rank_dir, rf))
        plt.figure(figsize=(8, 5))
        sns.barplot(x="Algorithm", y="Average_Rank", data=ranks_df, palette="viridis")
        plt.title(f"Global Friedman Ranks (D={dim_str}) - Lower is Better")
        plt.ylabel("Average Rank")
        plt.savefig(os.path.join(output_dir, f"friedman_bars_D{dim_str}.png"))
        plt.close()
    print("[✓] Friedman bar charts saved.")

    # 3. RADAR CHART (Algorithm Ranking)
    for rf in rank_files:
        dim_str = rf.split("_D")[-1].replace(".csv", "")
        ranks_df = pd.read_csv(os.path.join(rank_dir, rf))
        categories = ranks_df["Algorithm"].tolist()
        values = ranks_df["Average_Rank"].tolist()
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title(f"Algorithm Ranking Radar Chart (D={dim_str})")
        plt.savefig(os.path.join(output_dir, f"radar_chart_D{dim_str}.png"))
        plt.close()
    print("[✓] Radar charts saved.")

    # 4. NEW: CONVERGENCE CURVES (Using history data)
    if os.path.exists(history_path):
        histories = np.load(history_path, allow_pickle=True)
        # Assuming the order in all_histories matches the order of configs in run_all.py
        # You may need to map these correctly if the order differs. 
        # For simplicity, here is a template for plotting the first function's convergence:
        plt.figure(figsize=(10, 6))
        algos = df["algorithm"].unique()
        for algo in algos:
            # Filter and average histories for a specific function/dimension
            # (Requires mapping history index to metadata)
            pass 
        # Note: Mapping logic depends on how 'tasks' were ordered in run_all.py.
        print("[!] Convergence curves require metadata mapping of all_histories.npy.")

    # 5. NEW: WILCOXON W/T/L HEATMAP
    wtl_files = [f for f in os.listdir(rank_dir) if f.startswith("wilcoxon_wtl")]
    for wf in wtl_files:
        dim_str = wf.split("_D")[-1].replace(".csv", "")
        wtl_df = pd.read_csv(os.path.join(rank_dir, wf))
        
        # Pivot for heatmap (e.g., comparing wins)
        # Note: This requires the W/T/L table to be in a pairwise matrix format
        plt.figure(figsize=(10, 8))
        # This is a simplified representation of W/T/L logic
        sns.heatmap(wtl_df.set_index("Comparison")[["W", "T", "L"]], annot=True, cmap="YlGnBu")
        plt.title(f"Wilcoxon Pairwise Comparison W/T/L (D={dim_str})")
        plt.savefig(os.path.join(output_dir, f"wilcoxon_heatmap_D{dim_str}.png"))
        plt.close()
    print("[✓] Wilcoxon heatmaps saved.")

if __name__ == "__main__":
    create_plots()