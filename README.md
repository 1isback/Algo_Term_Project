# Neuro Courier Project

A Traveling Salesman Problem (TSP) solver project implementing multiple algorithms to find optimal or near-optimal routes for courier delivery.

## Project Structure

```
neuro_courier_project/
│
├── data/                       # Generated map data
│   ├── small_instances.json    # 10 cities
│   ├── medium_instances.json   # 50 cities
│   └── large_instances.json    # 100 cities
│
├── results/                    # Experiment results
│   ├── plots/                  # Route visualizations (.png)
│   └── logs/                   # Result tables (distance, time, etc.)
│
├── src/                        # Source code
│   ├── models.py               # City and Map classes
│   ├── generator.py            # Random city generator
│   ├── utils.py                # Distance calculation and plotting
│   │
│   └── solvers/                # TSP solving algorithms
│       ├── exact_solver.py     # Brute Force (exact solution)
│       ├── aco_solver.py       # Ant Colony Optimization
│       └── sa_solver.py        # Simulated Annealing
│
├── main.py                     # Main entry point
├── requirements.txt            # Required libraries
└── README.md                   # This file
```

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Generate TSP Instances

Generate small, medium, and large TSP instances:

```bash
python main.py --generate
```

### Run Experiments

Run all solvers on all instances:

```bash
python main.py --run
```

### Generate and Run (Default)

Generate instances and run experiments:

```bash
python main.py
```

## Algorithms

1. **Exact Solver (Brute Force)**: Tries all possible permutations. Only suitable for small instances (< 12 cities).

2. **ACO (Ant Colony Optimization)**: Metaheuristic inspired by ant behavior. Uses pheromone trails to find good solutions.

3. **SA (Simulated Annealing)**: Metaheuristic that accepts worse solutions probabilistically to escape local optima.

## Output

- **Plots**: Visualizations of routes saved in `results/plots/`
- **Logs**: Detailed results (distance, time) saved in `results/logs/`

## Requirements

- Python 3.7+
- numpy
- matplotlib

## License

This project is for educational purposes.

