# Optimization Algorithm Benchmark Framework

This repository contains a comprehensive experimental pipeline for evaluating and comparing various metaheuristic optimization algorithms against the CEC 2017 benchmark suite using the `opfunu` library. It includes the algorithm implementations, execution pipeline, statistical analysis tools, and visualization scripts necessary for producing academic-grade research results.

## Project Structure

```
.
├── algorithms/                 # Metaheuristic algorithm implementations
│   ├── abc.py                  # Artificial Bee Colony
│   ├── de.py                   # Differential Evolution
│   ├── ga.py                   # Genetic Algorithm
│   ├── pso.py                  # Particle Swarm Optimization
│   └── woa.py                  # Whale Optimization Algorithm
├── analysis/                   # Statistical tests and plotting scripts
│   ├── convergence_plot.py     # Convergence curve plotting
│   ├── plots.py                # Additional visualization utilities
│   └── statistical_tests.py    # Friedman and Wilcoxon statistical tests
├── benchmarks/                 # Benchmark objective functions setup
│   └── cec2017_loader.py       # Opfunu CEC 2017 loader
├── experiments/                # Experiment runners
│   └── run_all.py              # Multiprocessing execution pipeline
├── report/                     # LaTeX source for the research paper/report
│   └── report.tex
└── results/                    # Generated data and outputs
    ├── all_algo_results.csv    # Master CSV with all runs (generated)
    ├── all_histories.npy       # Convergence history arrays (generated)
    └── analysis_output/        # Output from statistical_tests.py (generated)
```

## Features

- **Algorithms**: Implements classical and modern metaheuristics (PSO, ABC, GA, DE, WOA).
- **Benchmarks**: Easy integration with CEC 2017 benchmark functions using the `opfunu` package.
- **Fast Multiprocessing Pipeline**: `experiments/run_all.py` uses python's `ProcessPoolExecutor` to run independent algorithm runs concurrently across all CPU cores for rapid experimentation.
- **Robust Statistical Analysis**: Automatically calculates Mean ± Standard Deviation, Friedman average rankings, and pairwise Wilcoxon Signed-Rank tests for rigorous algorithm comparison.
- **LaTeX Ready Outputs**: Extracts metrics and formatted data tables suited directly for academic papers.

## Requirements

Ensure you have Python 3.8+ installed. You can install the required dependencies using `pip`:

```bash
pip install numpy pandas scipy matplotlib tqdm opfunu
```

## Usage Instructions

### 1. Run the Experiments
The main experiment script runs all defined algorithms across specified CEC2017 functions and dimensions. It handles seed generation and saves the raw data.

```bash
python experiments/run_all.py
```
*Note: This will generate `results/all_algo_results.csv` and `results/all_histories.npy`.*

**Configuration:** 
You can edit the `CONFIG` dictionary at the top of `experiments/run_all.py` to change the functions tested, number of independent runs, population size, dimensions, and the function evaluations (FEs) budget.

### 2. Statistical Analysis
Once the experiments are completed, run the statistical module to calculate the Mean±Std, Friedman Ranks, and Wilcoxon Wins/Ties/Losses tables:

```bash
python analysis/statistical_tests.py
```
*Note: The generated CSV tables will be saved in `results/analysis_output/`.*

### 3. Generate Visualizations
To generate convergence plots comparing the performance of the algorithms across the iterations:

```bash
python analysis/convergence_plot.py
python analysis/plots.py
```

## Contributing
When adding a new algorithm, place the implementation in the `algorithms/` directory and expose it in `algorithms/__init__.py`. Ensure your class has an `optimize()` method returning `(best_fitness, best_solution, convergence_history)` to stay compatible with the `run_all.py` pipeline.

## License
MIT License
