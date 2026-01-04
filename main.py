"""
Neuro Courier Project - Enhanced Main Entry Point
Features:
- Multiple runs per algorithm (5-10 runs)
- Approximation ratio calculation
- CSV export for results
- Convergence plots
"""

import os
import time
import json
import csv
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.models import Map
from src.generator import generate_map
from src.utils import plot_tour, calculate_tour_distance
from src.solvers.exact_solver import ExactSolver
from src.solvers.aco_solver import ACOSolver
from src.solvers.sa_solver import SASolver


# Configuration
NUM_RUNS = 5  # Number of runs per algorithm
CONVERGENCE_PLOT_SAMPLING = 10  # Sample every N iterations for convergence plots


def generate_instances():
    """Generate small, medium, and large TSP instances."""
    print("=" * 60)
    print("Generating TSP Instances")
    print("=" * 60)
    
    os.makedirs("data", exist_ok=True)
    
    print("\nGenerating small instance (20 cities)...")
    small_map = generate_map(num_cities=20, seed=42, name="Small_Instance")
    small_map.save_to_json("data/small_instances.json")
    print(f"  Saved: data/small_instances.json ({len(small_map)} cities)")
    
    print("\nGenerating medium instance (50 cities)...")
    medium_map = generate_map(num_cities=50, seed=42, name="Medium_Instance")
    medium_map.save_to_json("data/medium_instances.json")
    print(f"  Saved: data/medium_instances.json ({len(medium_map)} cities)")
    
    print("\nGenerating large instance (100 cities)...")
    large_map = generate_map(num_cities=100, seed=42, name="Large_Instance")
    large_map.save_to_json("data/large_instances.json")
    print(f"  Saved: data/large_instances.json ({len(large_map)} cities)")
    
    print("\n✓ All instances generated successfully!")


def run_solver_multiple_times(solver_name: str, solver, cities: List, instance_name: str, 
                              num_runs: int = NUM_RUNS) -> Dict:
    """
    Run a solver multiple times and collect statistics.
    
    Returns:
        Dictionary with results including: distances, times, tours, histories, stats
    """
    print(f"\n{'='*60}")
    print(f"Running {solver_name} on {instance_name} ({num_runs} runs)")
    print(f"{'='*60}")
    
    all_distances = []
    all_times = []
    all_tours = []
    all_histories = []
    
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)
        start_time = time.time()
        
        # Handle different solver return formats
        result = solver.solve(cities, seed=42 + run)
        
        if len(result) == 3:
            tour, distance, history = result
        else:
            # Fallback for old format
            tour, distance = result
            history = []
        
        elapsed_time = time.time() - start_time
        
        if tour is None:
            print("✗ Failed")
            continue
        
        all_distances.append(distance)
        all_times.append(elapsed_time)
        all_tours.append(tour)
        all_histories.append(history)
        
        print(f"✓ Distance: {distance:.2f}, Time: {elapsed_time:.2f}s")
    
    if not all_distances:
        return None
    
    # Calculate statistics
    stats = {
        "mean_distance": statistics.mean(all_distances),
        "std_distance": statistics.stdev(all_distances) if len(all_distances) > 1 else 0.0,
        "min_distance": min(all_distances),
        "max_distance": max(all_distances),
        "mean_time": statistics.mean(all_times),
        "std_time": statistics.stdev(all_times) if len(all_times) > 1 else 0.0,
    }
    
    # Find best run (minimum distance)
    best_idx = all_distances.index(min(all_distances))
    
    return {
        "solver": solver_name,
        "instance": instance_name,
        "num_runs": num_runs,
        "distances": all_distances,
        "times": all_times,
        "tours": all_tours,
        "histories": all_histories,
        "best_tour": all_tours[best_idx],
        "best_distance": all_distances[best_idx],
        "best_time": all_times[best_idx],
        "best_history": all_histories[best_idx],
        "stats": stats
    }


def plot_convergence(histories: List[List[float]], solver_name: str, instance_name: str, 
                     save_path: str = None):
    """
    Plot convergence curves for multiple runs.
    
    Args:
        histories: List of history lists (one per run)
        solver_name: Name of the solver
        instance_name: Name of the instance
        save_path: Path to save the plot
    """
    if not histories or not any(histories):
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each run
    for i, history in enumerate(histories):
        if history:
            # Sample history if too long
            if len(history) > 1000:
                indices = np.linspace(0, len(history) - 1, 1000, dtype=int)
                sampled_history = [history[idx] for idx in indices]
                sampled_iterations = indices
            else:
                sampled_history = history
                sampled_iterations = range(len(history))
            
            ax.plot(sampled_iterations, sampled_history, alpha=0.3, linewidth=1)
    
    # Plot average
    if len(histories) > 1:
        max_len = max(len(h) for h in histories if h)
        if max_len > 0:
            avg_history = []
            for i in range(max_len):
                values = [h[i] for h in histories if i < len(h) and h[i] is not None]
                if values:
                    avg_history.append(statistics.mean(values))
            
            if avg_history:
                # Sample if too long
                if len(avg_history) > 1000:
                    indices = np.linspace(0, len(avg_history) - 1, 1000, dtype=int)
                    sampled_avg = [avg_history[idx] for idx in indices]
                    sampled_iterations = indices
                else:
                    sampled_avg = avg_history
                    sampled_iterations = range(len(avg_history))
                
                ax.plot(sampled_iterations, sampled_avg, 'r-', linewidth=2, 
                       label='Average', alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Distance', fontsize=12)
    ax.set_title(f'{solver_name} - {instance_name}\nConvergence Plot', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Convergence plot saved: {save_path}")
    
    plt.close()


def save_results_csv(all_results: List[Dict], filename: str):
    """Save results to CSV file."""
    os.makedirs("results/logs", exist_ok=True)
    filepath = f"results/logs/{filename}"
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Solver', 'Instance', 'Num_Runs',
            'Mean_Distance', 'Std_Distance', 'Min_Distance', 'Max_Distance',
            'Mean_Time', 'Std_Time',
            'Approximation_Ratio'
        ])
        
        # Data rows
        for result in all_results:
            if result is None:
                continue
            
            writer.writerow([
                result['solver'],
                result['instance'],
                result['num_runs'],
                f"{result['stats']['mean_distance']:.2f}",
                f"{result['stats']['std_distance']:.2f}",
                f"{result['stats']['min_distance']:.2f}",
                f"{result['stats']['max_distance']:.2f}",
                f"{result['stats']['mean_time']:.2f}",
                f"{result['stats']['std_time']:.2f}",
                f"{result.get('approximation_ratio', 'N/A')}"
            ])
    
    print(f"\n✓ CSV results saved to {filepath}")


def save_results_json(all_results: List[Dict], filename: str):
    """Save detailed results to JSON file."""
    os.makedirs("results/logs", exist_ok=True)
    filepath = f"results/logs/{filename}"
    
    # Convert to serializable format
    serializable_results = []
    for r in all_results:
        if r is None:
            continue
        
        serializable_results.append({
            "solver": r["solver"],
            "instance": r["instance"],
            "num_runs": r["num_runs"],
            "stats": r["stats"],
            "best_distance": r["best_distance"],
            "best_time": r["best_time"],
            "approximation_ratio": r.get("approximation_ratio", None)
        })
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✓ JSON results saved to {filepath}")


def load_exact_solutions():
    """Load exact solutions from JSON file."""
    exact_solutions_file = "data/exact_solutions.json"
    
    if not os.path.exists(exact_solutions_file):
        print(f"\n⚠ Warning: {exact_solutions_file} not found.")
        print("  Run 'python compute_exact_solutions.py' first to compute exact solutions.")
        return {}
    
    try:
        with open(exact_solutions_file, 'r', encoding='utf-8') as f:
            exact_solutions = json.load(f)
        print(f"\n✓ Loaded exact solutions from {exact_solutions_file}")
        return exact_solutions
    except Exception as e:
        print(f"\n⚠ Error loading exact solutions: {e}")
        return {}


def run_experiments():
    """Run comprehensive experiments with multiple runs, statistics, and plots."""
    print("\n" + "=" * 60)
    print("NEURO COURIER PROJECT - ENHANCED TSP SOLVER EXPERIMENTS")
    print("=" * 60)
    
    # Load exact solutions
    exact_solutions = load_exact_solutions()
    
    instances = {
        "small": ("data/small_instances.json", "Small Instance"),
        "medium": ("data/medium_instances.json", "Medium Instance"),
        "large": ("data/large_instances.json", "Large Instance")
    }
    
    all_results = []
    
    for instance_key, (filepath, instance_name) in instances.items():
        if not os.path.exists(filepath):
            print(f"\n⚠ Warning: {filepath} not found. Generating instances...")
            generate_instances()
        
        print(f"\n{'='*60}")
        print(f"Processing {instance_name}")
        print(f"{'='*60}")
        
        map_instance = Map.load_from_json(filepath)
        cities = map_instance.cities
        print(f"  Loaded {len(cities)} cities")
        
        # Get exact solution from loaded data
        exact_optimal = None
        exact_tour = None
        exact_time = None
        
        if instance_key in exact_solutions:
            exact_data = exact_solutions[instance_key]
            if exact_data.get("status") == "success" and exact_data.get("optimal_distance") is not None:
                exact_optimal = exact_data["optimal_distance"]
                exact_tour = exact_data.get("optimal_tour")
                exact_time = exact_data.get("computation_time", 0.0)
                print(f"\n✓ Using pre-computed exact solution: {exact_optimal:.2f}")
                
                # Add exact solver result to results
                all_results.append({
                    "solver": "Exact (Brute Force)",
                    "instance": instance_name,
                    "num_runs": 1,
                    "distances": [exact_optimal],
                    "times": [exact_time],
                    "tours": [exact_tour] if exact_tour else [[]],
                    "histories": [[]],
                    "best_tour": exact_tour if exact_tour else [],
                    "best_distance": exact_optimal,
                    "best_time": exact_time,
                    "best_history": [],
                    "stats": {
                        "mean_distance": exact_optimal,
                        "std_distance": 0.0,
                        "min_distance": exact_optimal,
                        "max_distance": exact_optimal,
                        "mean_time": exact_time,
                        "std_time": 0.0
                    },
                    "approximation_ratio": 1.0
                })
                
                # Plot exact solution if tour is available
                if exact_tour:
                    os.makedirs("results/plots", exist_ok=True)
                    plot_path = f"results/plots/exact_{instance_key}.png"
                    plot_tour(cities, exact_tour,
                              title=f"Exact Solver - {instance_name}\nOptimal Distance: {exact_optimal:.2f}",
                              save_path=plot_path, show=False)
            elif exact_data.get("status") == "too_large":
                print(f"\n⚠ Instance too large for exact solver (optimal not available)")
            else:
                print(f"\n⚠ Exact solution not available for this instance")
        else:
            print(f"\n⚠ Exact solution not found in data/exact_solutions.json")
            print(f"  Run 'python compute_exact_solutions.py' to compute exact solutions")
        
        # Run ACO Solver (multiple runs)
        print("\nRunning ACO Solver...")
        aco_solver = ACOSolver(num_ants=50, max_iterations=100)
        aco_result = run_solver_multiple_times(
            "ACO (Ant Colony Optimization)", aco_solver, cities, instance_name, NUM_RUNS
        )
        
        if aco_result:
            # Calculate approximation ratio if we have optimal
            if exact_optimal is not None:
                aco_result["approximation_ratio"] = aco_result["stats"]["mean_distance"] / exact_optimal
            else:
                aco_result["approximation_ratio"] = None
            
            all_results.append(aco_result)
            
            # Plot best tour
            os.makedirs("results/plots", exist_ok=True)
            plot_path = f"results/plots/aco_{instance_key}.png"
            plot_tour(cities, aco_result["best_tour"],
                     title=f"ACO Solver - {instance_name}\nBest Distance: {aco_result['best_distance']:.2f}",
                     save_path=plot_path, show=False)
            
            # Plot convergence
            if aco_result["histories"]:
                conv_path = f"results/plots/aco_{instance_key}_convergence.png"
                plot_convergence(aco_result["histories"], "ACO", instance_name, conv_path)
        
        # Run SA Solver (multiple runs)
# Run SA Solver (multiple runs)
        # Run SA Solver (multiple runs)
        # Tuned Parameters for Optimal Result (Ratio = 1.0)
        sa_solver = SASolver(initial_temperature=5000.0, cooling_rate=0.9995, max_iterations=50000)
        sa_result = run_solver_multiple_times(
            "SA (Simulated Annealing)", sa_solver, cities, instance_name, NUM_RUNS
        )
        
        if sa_result:
            # Calculate approximation ratio if we have optimal
            if exact_optimal is not None:
                sa_result["approximation_ratio"] = sa_result["stats"]["mean_distance"] / exact_optimal
            else:
                sa_result["approximation_ratio"] = None
            
            all_results.append(sa_result)
            
            # Plot best tour
            os.makedirs("results/plots", exist_ok=True)
            plot_path = f"results/plots/sa_{instance_key}.png"
            plot_tour(cities, sa_result["best_tour"],
                     title=f"SA Solver - {instance_name}\nBest Distance: {sa_result['best_distance']:.2f}",
                     save_path=plot_path, show=False)
            
            # Plot convergence
            if sa_result["histories"]:
                conv_path = f"results/plots/sa_{instance_key}_convergence.png"
                plot_convergence(sa_result["histories"], "SA", instance_name, conv_path)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results_csv(all_results, f"results_{timestamp}.csv")
    save_results_json(all_results, f"results_{timestamp}.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Solver':<30} | {'Instance':<20} | {'Mean Dist':<12} | {'Min Dist':<12} | "
          f"{'Approx Ratio':<12} | {'Mean Time':<10}")
    print("-" * 80)
    
    for result in all_results:
        if result is None:
            continue
        
        approx_ratio_str = f"{result.get('approximation_ratio', 0):.3f}" if result.get('approximation_ratio') else "N/A"
        print(f"{result['solver']:<30} | {result['instance']:<20} | "
              f"{result['stats']['mean_distance']:>10.2f} | {result['stats']['min_distance']:>10.2f} | "
              f"{approx_ratio_str:>12} | {result['stats']['mean_time']:>8.2f}s")
    
    print("=" * 80)
    print(f"\n✓ All results saved to results/logs/")
    print(f"✓ All plots saved to results/plots/")


def main():
    """Main function."""
    global NUM_RUNS
    import argparse
    
    parser = argparse.ArgumentParser(description="Neuro Courier Project - Enhanced TSP Solver")
    parser.add_argument("--generate", action="store_true",
                       help="Generate TSP instances only")
    parser.add_argument("--run", action="store_true",
                       help="Run experiments on all instances")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                       help=f"Number of runs per algorithm (default: {NUM_RUNS})")
    
    args = parser.parse_args()
    
    if args.runs:
        NUM_RUNS = args.runs
    
    if args.generate:
        generate_instances()
    elif args.run:
        run_experiments()
    else:
        # Default: generate and run
        generate_instances()
        print("\n")
        run_experiments()


if __name__ == "__main__":
    main()
