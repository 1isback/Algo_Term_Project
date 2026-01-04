"""
Fast Parameter Tuning Script for ACO and SA Solvers
Optimized with progress tracking and time limits
"""

import os
import time
import json
import itertools
from datetime import datetime
from typing import Dict, List
import sys

from src.models import Map
from src.generator import generate_map
from src.solvers.aco_solver import ACOSolver
from src.solvers.sa_solver import SASolver
from src.utils import calculate_tour_distance


def print_progress_bar(iteration, total, prefix='', suffix='', length=40):
    """Print a progress bar with percentage."""
    percent = 100 * (iteration / float(total))
    filled = int(length * iteration // total)
    bar = '█' * filled + '░' * (length - filled)
    sys.stdout.write(f'\r{prefix} [{bar}] {percent:.1f}% {suffix}')
    sys.stdout.flush()


def test_aco_parameters(cities, param_combinations: List[Dict], num_runs: int = 2, max_time: int = 600):
    """Test different ACO parameter combinations with time limit."""
    print("\n" + "=" * 70)
    print("ACO PARAMETER TUNING (Optimized Grid Search)")
    print("=" * 70)
    
    results = []
    total = len(param_combinations)
    start_time = time.time()
    
    for i, params in enumerate(param_combinations):
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > max_time:
            print(f"\n⚠ Time limit reached ({max_time}s). Tested {i}/{total} combinations.")
            break
        
        # Progress bar
        print_progress_bar(i, total, 
                          prefix=f'ACO Testing',
                          suffix=f'({i}/{total}) Elapsed: {elapsed:.1f}s')
        
        distances = []
        times = []
        
        # Run tests for this parameter set
        for run in range(num_runs):
            try:
                solver = ACOSolver(**params)
                run_start = time.time()
                tour, distance, _ = solver.solve(cities, seed=42 + run)
                run_time = time.time() - run_start
                
                if tour is not None:
                    distances.append(distance)
                    times.append(run_time)
            except Exception as e:
                continue
        
        if distances:
            avg_distance = sum(distances) / len(distances)
            avg_time = sum(times) / len(times)
            min_distance = min(distances)
            
            result = {
                "params": params,
                "avg_distance": avg_distance,
                "min_distance": min_distance,
                "avg_time": avg_time,
                "num_successful_runs": len(distances)
            }
            results.append(result)
    
    print_progress_bar(min(i+1, total), total, 
                      prefix=f'ACO Testing',
                      suffix=f'({min(i+1, total)}/{total}) DONE')
    print()  # New line after progress bar
    
    return results


def test_sa_parameters(cities, param_combinations: List[Dict], num_runs: int = 2, max_time: int = 600):
    """Test different SA parameter combinations with time limit."""
    print("\n" + "=" * 70)
    print("SA PARAMETER TUNING (Optimized Grid Search)")
    print("=" * 70)
    
    results = []
    total = len(param_combinations)
    start_time = time.time()
    
    for i, params in enumerate(param_combinations):
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > max_time:
            print(f"\n⚠ Time limit reached ({max_time}s). Tested {i}/{total} combinations.")
            break
        
        # Progress bar
        print_progress_bar(i, total, 
                          prefix=f'SA Testing',
                          suffix=f'({i}/{total}) Elapsed: {elapsed:.1f}s')
        
        distances = []
        times = []
        
        # Run tests for this parameter set
        for run in range(num_runs):
            try:
                solver = SASolver(**params)
                run_start = time.time()
                tour, distance, _ = solver.solve(cities, seed=42 + run)
                run_time = time.time() - run_start
                
                if tour is not None:
                    distances.append(distance)
                    times.append(run_time)
            except Exception as e:
                continue
        
        if distances:
            avg_distance = sum(distances) / len(distances)
            avg_time = sum(times) / len(times)
            min_distance = min(distances)
            
            result = {
                "params": params,
                "avg_distance": avg_distance,
                "min_distance": min_distance,
                "avg_time": avg_time,
                "num_successful_runs": len(distances)
            }
            results.append(result)
    
    print_progress_bar(min(i+1, total), total, 
                      prefix=f'SA Testing',
                      suffix=f'({min(i+1, total)}/{total}) DONE')
    print()  # New line after progress bar
    
    return results


def generate_aco_grid_search_combinations():
    """
    Generate reduced ACO parameter combinations for faster tuning.
    Focuses on most impactful parameters.
    """
    # REDUCED parameter ranges - focus on most important
    param_grid = {
        "num_ants": [30, 50, 70],           # 3 values (was 4)
        "alpha": [0.5, 1.0, 2.0],           # 3 values (was 4)
        "beta": [2.0, 3.0, 4.0],            # 3 values (was 4)
        "evaporation_rate": [0.3, 0.5],     # 2 values (was 3)
        "max_iterations": [50, 100],        # 2 values (was 3)
        "elitist_weight": [1.0, 2.0]        # 2 values (was 3)
    }
    
    # Total: 3 × 3 × 3 × 2 × 2 × 2 = 216 combinations (was 1728)
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def generate_sa_grid_search_combinations():
    """
    Generate all SA parameter combinations using grid search.
    SA is fast so we keep all combinations.
    """
    # Full parameter ranges for comprehensive search
    param_grid = {
        "initial_temperature": [500.0, 1000.0, 2000.0, 3000.0],
        "cooling_rate": [0.99, 0.995, 0.998],
        "min_temperature": [0.1, 0.5, 1.0],
        "max_iterations": [5000, 10000, 15000]
    }
    
    # Total: 4 × 3 × 3 × 3 = 108 combinations
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def main():
    """Main tuning function with time management."""
    print("=" * 70)
    print("FAST PARAMETER TUNING FOR TSP SOLVERS")
    print("=" * 70)
    
    # Time allocation
    MAX_TOTAL_TIME = 20 * 60  # 20 minutes in seconds
    ACO_TIME_BUDGET = 12 * 60  # 12 minutes for ACO
    SA_TIME_BUDGET = 7 * 60    # 7 minutes for SA (fast anyway)
    
    # Generate or load test instance
    test_file = "data/small_instances.json"
    if not os.path.exists(test_file):
        print("\nGenerating test instance...")
        os.makedirs("data", exist_ok=True)
        test_map = generate_map(num_cities=20, seed=42, name="Tuning_Test")
        test_map.save_to_json(test_file)
    
    map_instance = Map.load_from_json(test_file)
    cities = map_instance.cities
    print(f"\nUsing test instance: {len(cities)} cities")
    
    # Generate ACO parameter combinations
    print("\nGenerating ACO parameter combinations (OPTIMIZED)...")
    aco_combinations = generate_aco_grid_search_combinations()
    print(f"Total ACO combinations: {len(aco_combinations)}")
    print("  Reduced parameter ranges:")
    print("    num_ants: [30, 50, 70]")
    print("    alpha: [0.5, 1.0, 2.0]")
    print("    beta: [2.0, 3.0, 4.0]")
    print("    evaporation_rate: [0.3, 0.5]")
    print("    max_iterations: [50, 100]")
    print("    elitist_weight: [1.0, 2.0]")
    print(f"  Total: 3×3×3×2×2×2 = {len(aco_combinations)} (was 1728)")
    
    # Generate SA parameter combinations
    print("\nGenerating SA parameter combinations (FULL SEARCH)...")
    sa_combinations = generate_sa_grid_search_combinations()
    print(f"Total SA combinations: {len(sa_combinations)}")
    print("  Full parameter ranges:")
    print("    initial_temperature: [500, 1000, 2000, 3000]")
    print("    cooling_rate: [0.99, 0.995, 0.998]")
    print("    min_temperature: [0.1, 0.5, 1.0]")
    print("    max_iterations: [5000, 10000, 15000]")
    print(f"  Total: 4×3×3×3 = {len(sa_combinations)}")
    
    print("\n" + "=" * 70)
    print(f"Time budget: {MAX_TOTAL_TIME/60:.0f} minutes total")
    print(f"  ACO: {ACO_TIME_BUDGET/60:.0f} minutes")
    print(f"  SA: {SA_TIME_BUDGET/60:.0f} minutes")
    print("=" * 70)
    
    overall_start = time.time()
    
    # Run ACO tuning
    aco_results = test_aco_parameters(cities, aco_combinations, 
                                      num_runs=2, max_time=ACO_TIME_BUDGET)
    
    # Run SA tuning (with 3 runs since it's fast)
    sa_results = test_sa_parameters(cities, sa_combinations, 
                                    num_runs=3, max_time=SA_TIME_BUDGET)
    
    total_time = time.time() - overall_start
    
    # Find best parameters
    if aco_results:
        best_aco = min(aco_results, key=lambda x: x["avg_distance"])
        print("\n" + "=" * 70)
        print("BEST ACO PARAMETERS")
        print("=" * 70)
        print(f"Parameters:")
        for key, value in best_aco['params'].items():
            print(f"  {key}: {value}")
        print(f"\nResults:")
        print(f"  Average Distance: {best_aco['avg_distance']:.2f}")
        print(f"  Minimum Distance: {best_aco['min_distance']:.2f}")
        print(f"  Average Time: {best_aco['avg_time']:.2f}s")
        print(f"  Successful Runs: {best_aco['num_successful_runs']}/2")
        
        # Show top 5
        sorted_aco = sorted(aco_results, key=lambda x: x["avg_distance"])[:5]
        print(f"\nTop 5 ACO Parameter Sets:")
        for i, result in enumerate(sorted_aco, 1):
            print(f"  {i}. Dist: {result['avg_distance']:.2f} | {result['params']}")
    
    if sa_results:
        best_sa = min(sa_results, key=lambda x: x["avg_distance"])
        print("\n" + "=" * 70)
        print("BEST SA PARAMETERS")
        print("=" * 70)
        print(f"Parameters:")
        for key, value in best_sa['params'].items():
            print(f"  {key}: {value}")
        print(f"\nResults:")
        print(f"  Average Distance: {best_sa['avg_distance']:.2f}")
        print(f"  Minimum Distance: {best_sa['min_distance']:.2f}")
        print(f"  Average Time: {best_sa['avg_time']:.2f}s")
        print(f"  Successful Runs: {best_sa['num_successful_runs']}/3")
        
        # Show top 5
        sorted_sa = sorted(sa_results, key=lambda x: x["avg_distance"])[:5]
        print(f"\nTop 5 SA Parameter Sets:")
        for i, result in enumerate(sorted_sa, 1):
            print(f"  {i}. Dist: {result['avg_distance']:.2f} | {result['params']}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"ACO combinations tested: {len(aco_results)}/{len(aco_combinations)}")
    print(f"SA combinations tested: {len(sa_results)}/{len(sa_combinations)}")
    print("=" * 70)
    
    # Save results
    os.makedirs("results/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {
        "tuning_time_seconds": total_time,
        "aco_combinations_tested": len(aco_results),
        "sa_combinations_tested": len(sa_results),
        "aco_results": aco_results,
        "sa_results": sa_results,
        "best_aco": best_aco if aco_results else None,
        "best_sa": best_sa if sa_results else None
    }
    
    filepath = f"results/logs/fast_tuning_{timestamp}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {filepath}")


if __name__ == "__main__":
    main()
    