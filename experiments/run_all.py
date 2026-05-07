import os
import time
import warnings
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor

# Path setup for local VS Code environment
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from algorithms import ALGORITHMS
import opfunu

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# FAST CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "dimensions"    : [10], 
    "functions"     : [1, 3, 4, 10, 11, 15, 20, 21, 25, 29],
    "n_runs"        : 30,
    "pop_size"      : 20,
    "fes_factor"    : 200, 
    "master_csv"    : os.path.join(project_root, "results", "all_algo_results.csv"),
    "master_history": os.path.join(project_root, "results", "all_histories.npy"),
    "algorithms"    : ["PSO", "ABC", "GA", "DE", "WOA"], 
}

def get_function(func_id: int, dim: int):
    try:
        classes = opfunu.get_functions_based_classname("2017")
        for func_class in classes:
            if func_class.__name__ == f"F{func_id}2017":
                return func_class(ndim=dim)
    except: pass
    return None

def run_config_worker(params):
    algo_name, func_id, dim = params
    f_instance = get_function(func_id, dim)
    if f_instance is None: return [], []

    results = []
    histories = []
    # Loop for independent runs
    for run_id in range(1, CONFIG["n_runs"] + 1):
        seed = run_id * 1000 + func_id * 10 + dim
        # Budget is drastically reduced here
        max_fes = CONFIG["fes_factor"] * dim
        
        AlgoClass = ALGORITHMS[algo_name]
        kwargs = {
            "func": f_instance.evaluate,
            "dim": dim,
            "bounds": (-100.0, 100.0),
            "max_fes": max_fes,
            "seed": seed
        }
        if algo_name == "ABC":
            kwargs["colony_size"] = CONFIG["pop_size"]
        elif algo_name == "Tabu":
            kwargs["n_neighbors"] = CONFIG["pop_size"]
        else:
            kwargs["pop_size"] = CONFIG["pop_size"]
            
        algo = AlgoClass(**kwargs)
        
        best_fit, _, history = algo.optimize()
        
        results.append({
            "algorithm": algo_name,
            "func_id": func_id,
            "dim": dim,
            "run_id": run_id,
            "error_val": best_fit - f_instance.f_global
        })
        histories.append(history)

    return results, histories

def run_experiments():
    os.makedirs(os.path.dirname(CONFIG["master_csv"]), exist_ok=True)
    
    tasks = list(product(CONFIG["algorithms"], CONFIG["functions"], CONFIG["dimensions"]))
    all_results_list = []
    all_histories_list = []

    print(f"Starting Fast-Track: {len(tasks)} configs | {CONFIG['fes_factor']} FEs factor")

    # Utilize all CPU cores for maximum speed
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for res, hist in tqdm(executor.map(run_config_worker, tasks), total=len(tasks), desc="Progress"):
            all_results_list.extend(res)
            all_histories_list.append(hist)

    # Save to one master CSV
    df = pd.DataFrame(all_results_list)
    df.to_csv(CONFIG["master_csv"], index=False)
    
    # Save the master history array
    np.save(CONFIG["master_history"], np.array(all_histories_list, dtype=object))
    
    print(f"\n[v] Finished in record time! Data saved to: {CONFIG['master_csv']}")

if __name__ == "__main__":
    run_experiments()