# Neuro Courier Project

A Traveling Salesman Problem (TSP) solver project implementing multiple algorithms to find optimal or near-optimal routes for courier delivery.

## Project Structure

ALGO_TERM_PROJECT/
│
├── .venv/                              # Python Virtual Environment
│
├── data/                               # Dataset files
│   ├── exact_solutions.json
│   ├── large_instances.json
│   ├── medium_instances.json
│   └── small_instances.json
│
├── results/                            # Experiment results
│   ├── logs/
│   ├── plots/
│   └── Small_Instance_exact_solution.json
│
├── src/                                # Source code package
│   ├── solvers/                        # TSP Algorithms
│   │   ├── __init__.py
│   │   ├── aco_solver.py               # Ant Colony Optimization
│   │   ├── exact_solver.py             # Brute Force / Naive Exact
│   │   ├── optimized_exact_solver.py   # Improved Exact Solver
│   │   └── sa_solver.py                # Simulated Annealing
│   │
│   ├── __init__.py                     # Makes 'src' a package
│   ├── generator.py                    # Map generator
│   ├── models.py                       # Data models (City, Map)
│   └── utils.py                        # Helper functions
│
├── tests/                              # Unit tests
│   ├── __init__.py
│   ├── test_generator.py           
│   ├── test_models.py              
│   ├── test_solvers.py             
│   └── test_utils.py               
│
├── .gitignore                          # Git ignore rules
├── compute_exact_solutions.py          # Script to pre-compute ground truths
├── main.py                             # Main application entry point
├── README.md                           # Project documentation
├── requirements.txt                    # Dependencies
├── scalability_analysis.py             # Script for performance/size analysis
└── tune_parameters.py                  # Script for hyperparameter tuning

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Generate TSP Instances

Generate small, medium, and large TSP instances:

```bash
python main.py --generate
```

This creates three JSON files in the `data/` directory:
- `small_instances.json` (20 cities)
- `medium_instances.json` (50 cities)
- `large_instances.json` (100 cities)

### Step 2: Compute Exact Solutions (Recommended)

Compute exact optimal solutions for approximation ratio calculations:

```bash
python compute_exact_solutions.py
```

This script:
- Calculates optimal solutions using brute force for all instances (where feasible)
- Saves results to `data/exact_solutions.json`
- Only computes for instances with ≤21 cities (larger instances marked as "too_large")

**Note:** This step is optional but recommended. If skipped, approximation ratios will show as "N/A" in results.

### Step 3: Run Experiments

Run all solvers on all instances with multiple runs:

```bash
python main.py --run
```

The script will:
- Load exact solutions from `data/exact_solutions.json` if available
- Calculate approximation ratios for all instances with exact solutions
- Generate plots and CSV/JSON results

### Custom Number of Runs

Run experiments with a custom number of runs per algorithm:

```bash
python main.py --run --runs 10
```

### Generate and Run (Default)

Generate instances and run experiments in one command:

```bash
python main.py
```

## Algorithms

### 1. Exact Solver (Brute Force)
- **Method**: Tries all possible permutations
- **Complexity**: O(n!)
- **Use Case**: Only suitable for small instances (≤21 cities)
- **Guarantee**: Finds optimal solution

### 2. ACO (Ant Colony Optimization)
- **Method**: Metaheuristic inspired by ant behavior
- **Key Components**:
  - Pheromone trails on edges
  - Heuristic information (inverse distance)
  - Probabilistic transition rule
  - Pheromone evaporation and reinforcement
  - Elitist strategy (bonus pheromone for best tour)

#### ACO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_ants` | 30 | Number of ants per iteration |
| `alpha` | 1.0 | Pheromone importance (higher = more influence) |
| `beta` | 3.0 | Heuristic importance (higher = prefer shorter edges) |
| `evaporation_rate` | 0.5 | Rate at which pheromone evaporates (0-1) |
| `q` | 100.0 | Pheromone deposit constant |
| `max_iterations` | 100 | Maximum number of iterations |
| `elitist_weight` | 2.0 | Elitist strategy weight |

**Parameter Tuning Tips**:
- Higher `alpha`: More exploitation (follow existing trails)
- Higher `beta`: More exploration (prefer shorter edges)
- Higher `evaporation_rate`: Faster convergence but may lose good paths
- More `num_ants`: Better exploration but slower

### 3. SA (Simulated Annealing)
- **Method**: Metaheuristic that accepts worse solutions probabilistically
- **Key Components**:
  - Temperature schedule (geometric cooling)
  - Neighborhood moves (swap, reverse segment, 2-opt)
  - Metropolis acceptance criterion

#### SA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_temperature` | 5000.0 | Starting temperature |
| `cooling_rate` | 0.9995 | Temperature reduction factor per iteration |
| `min_temperature` | 0.001 | Minimum temperature (stopping criterion) |
| `max_iterations` | 50000 | Maximum number of iterations |

**Parameter Tuning Tips**:
- Higher `initial_temperature`: More exploration early on
- Lower `cooling_rate`: Slower cooling, more exploration
- Higher `max_iterations`: More time to find better solutions

## Output

### Plots (`results/plots/`)
- **Route Visualizations**: `exact_small.png`, `aco_*.png`, `sa_*.png`
  - Shows the tour path connecting all cities
  - Cities numbered in order of visit
  
- **Convergence Plots**: `*_convergence.png`
  - Shows how solution quality improves over iterations
  - Multiple runs shown with average line

### Logs (`results/logs/`)
- **CSV Files**: `results_TIMESTAMP.csv`
  - Excel/Google Sheets compatible
  - Contains: solver, instance, statistics, approximation ratio
  
- **JSON Files**: `results_TIMESTAMP.json`
  - Detailed results in JSON format
  - Includes all runs, statistics, and metadata

## Reproducibility

All random operations use seeds for reproducibility:
- Instance generation: `seed=42`
- Solver runs: `seed=42 + run_number`

To reproduce exact results:
1. Use the same seed values
2. Use the same Python version (3.7+)
3. Use the same dependencies (see `requirements.txt`)

## Experimental Features

### Approximation Ratio
For small instances, the approximation ratio is calculated:
```
Approximation Ratio = Heuristic Solution / Optimal Solution
```
- Ratio = 1.0: Optimal solution found
- Ratio > 1.0: How much worse than optimal
- Only calculated for small instances (where optimal is known)

### Multiple Runs
Each algorithm runs multiple times (default: 5) to:
- Account for randomness
- Calculate statistics (mean, std, min, max)
- Show solution quality variability

### Convergence Analysis
Convergence plots show:
- How each run improves over iterations
- Average convergence behavior
- Premature convergence detection

## Advanced Usage

### Parameter Tuning
Use the parameter tuning script to find optimal parameters:

```bash
python tune_parameters.py
```

### Scalability Analysis
Analyze how algorithms scale with problem size:

```bash
python scalability_analysis.py
```

### Unit Tests
Run unit tests to verify correctness:

```bash
python -m pytest tests/
```

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- pulp

## Project Context

This project is part of an algorithm engineering course focusing on:
- NP-hard problem solving
- Metaheuristic algorithms
- Approximation algorithms
- Experimental analysis

## License

This project is for educational purposes.

## Citation

If you use this code in your research or project, please cite appropriately.
"""

Bersun Ustuner
Yusuf Efe Erer
Zeynep Rana Secen

"""