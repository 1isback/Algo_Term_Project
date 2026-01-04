"""
Parameter Tuning Script for ACO and SA Solvers
Uses Grid Search to test all parameter combinations and find optimal settings.
"""

import os
import time
import json
import itertools
from datetime import datetime
from typing import Dict, List

from src.models import Map
from src.generator import generate_map
from src.solvers.aco_solver import ACOSolver
from src.solvers.sa_solver import SASolver
from src.utils import calculate_tour_distance


def test_aco_parameters(cities, param_combinations: List[Dict], num_runs: int = 3):
    """Test different ACO parameter combinations."""
    print("\n" + "=" * 60)
    print("ACO PARAMETER TUNING (Grid Search)")
    print("=" * 60)
    
    results = []
    total = len(param_combinations)
    
    for i, params in enumerate(param_combinations):
        progress = (i + 1) / total * 100
        print(f"\n[{i+1}/{total}] ({progress:.1f}%) Testing: {params}")
        
        distances = []
        times = []
        
        for run in range(num_runs):
            solver = ACOSolver(**params)
            start_time = time.time()
            tour, distance, _ = solver.solve(cities, seed=42 + run)
            elapsed_time = time.time() - start_time
            
            if tour is not None:
                distances.append(distance)
                times.append(elapsed_time)
        
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
            
            print(f"  ✓ Avg Distance: {avg_distance:.2f}, Min: {min_distance:.2f}, Time: {avg_time:.2f}s")
        else:
            print(f"  ✗ All runs failed")
    
    return results


def test_sa_parameters(cities, param_combinations: List[Dict], num_runs: int = 3):
    """Test different SA parameter combinations."""
    print("\n" + "=" * 60)
    print("SA PARAMETER TUNING (Grid Search)")
    print("=" * 60)
    
    results = []
    total = len(param_combinations)
    
    for i, params in enumerate(param_combinations):
        progress = (i + 1) / total * 100
        print(f"\n[{i+1}/{total}] ({progress:.1f}%) Testing: {params}")
        
        distances = []
        times = []
        
        for run in range(num_runs):
            solver = SASolver(**params)
            start_time = time.time()
            tour, distance, _ = solver.solve(cities, seed=42 + run)
            elapsed_time = time.time() - start_time
            
            if tour is not None:
                distances.append(distance)
                times.append(elapsed_time)
        
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
            
            print(f"  ✓ Avg Distance: {avg_distance:.2f}, Min: {min_distance:.2f}, Time: {avg_time:.2f}s")
        else:
            print(f"  ✗ All runs failed")
    
    return results


def generate_aco_grid_search_combinations():
    """
    Generate all ACO parameter combinations using grid search.
    Returns list of all parameter dictionaries.
    """
    # Expanded parameter ranges for comprehensive search
    param_grid = {
        "num_ants": [30, 50, 70, 100],
        "alpha": [0.5, 1.0, 1.5, 2.0],
        "beta": [1.0, 2.0, 3.0, 4.0],
        "evaporation_rate": [0.3, 0.5, 0.7],
        "max_iterations": [50, 100, 150],
        "elitist_weight": [1.0, 2.0, 3.0]
    }
    
    # Generate all combinations using itertools.product
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
    Returns list of all parameter dictionaries.
    """
    # Expanded parameter ranges for comprehensive search
    param_grid = {
        "initial_temperature": [500.0, 1000.0, 2000.0, 3000.0],
        "cooling_rate": [0.99, 0.995, 0.998],
        "min_temperature": [0.1, 0.5, 1.0],
        "max_iterations": [5000, 10000, 15000]
    }
    
    # Generate all combinations using itertools.product
    keys = param_grid.keys()
    values = param_grid.values()
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def main():
    """Main tuning function."""
    print("=" * 60)
    print("PARAMETER TUNING FOR TSP SOLVERS (GRID SEARCH)")
    print("=" * 60)
    
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
    
    # Generate ACO parameter combinations using grid search
    print("\nGenerating ACO parameter combinations (Grid Search)...")
    aco_combinations = generate_aco_grid_search_combinations()
    print(f"Total ACO combinations: {len(aco_combinations)}")
    print("  Parameter ranges:")
    print("    num_ants: [30, 50, 70, 100]")
    print("    alpha: [0.5, 1.0, 1.5, 2.0]")
    print("    beta: [1.0, 2.0, 3.0, 4.0]")
    print("    evaporation_rate: [0.3, 0.5, 0.7]")
    print("    max_iterations: [50, 100, 150]")
    print("    elitist_weight: [1.0, 2.0, 3.0]")
    print(f"  Total: 4 × 4 × 4 × 3 × 3 × 3 = {len(aco_combinations)} combinations")
    
    # Generate SA parameter combinations using grid search
    print("\nGenerating SA parameter combinations (Grid Search)...")
    sa_combinations = generate_sa_grid_search_combinations()
    print(f"Total SA combinations: {len(sa_combinations)}")
    print("  Parameter ranges:")
    print("    initial_temperature: [500, 1000, 2000, 3000]")
    print("    cooling_rate: [0.99, 0.995, 0.998]")
    print("    min_temperature: [0.1, 0.5, 1.0]")
    print("    max_iterations: [5000, 10000, 15000]")
    print(f"  Total: 4 × 3 × 3 × 3 = {len(sa_combinations)} combinations")
    
    print("\n" + "=" * 60)
    print("WARNING: This will test many combinations and may take a long time!")
    print(f"Estimated time: ~{len(aco_combinations) * 3 * 0.5 + len(sa_combinations) * 3 * 0.01:.1f} seconds")
    print("=" * 60)
    
    # Run tuning
    aco_results = test_aco_parameters(cities, aco_combinations, num_runs=3)
    sa_results = test_sa_parameters(cities, sa_combinations, num_runs=3)
    
    # Find best parameters
    if aco_results:
        best_aco = min(aco_results, key=lambda x: x["avg_distance"])
        print("\n" + "=" * 60)
        print("BEST ACO PARAMETERS (Grid Search)")
        print("=" * 60)
        print(f"Parameters:")
        for key, value in best_aco['params'].items():
            print(f"  {key}: {value}")
        print(f"\nResults:")
        print(f"  Average Distance: {best_aco['avg_distance']:.2f}")
        print(f"  Minimum Distance: {best_aco['min_distance']:.2f}")
        print(f"  Average Time: {best_aco['avg_time']:.2f}s")
        print(f"  Successful Runs: {best_aco['num_successful_runs']}/3")
        
        # Show top 5
        sorted_aco = sorted(aco_results, key=lambda x: x["avg_distance"])[:5]
        print(f"\nTop 5 ACO Parameter Sets:")
        for i, result in enumerate(sorted_aco, 1):
            print(f"  {i}. Distance: {result['avg_distance']:.2f}, Params: {result['params']}")
    
    if sa_results:
        best_sa = min(sa_results, key=lambda x: x["avg_distance"])
        print("\n" + "=" * 60)
        print("BEST SA PARAMETERS (Grid Search)")
        print("=" * 60)
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
            print(f"  {i}. Distance: {result['avg_distance']:.2f}, Params: {result['params']}")
    
    # Save results
    os.makedirs("results/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {
        "aco_results": aco_results,
        "sa_results": sa_results,
        "best_aco": best_aco if aco_results else None,
        "best_sa": best_sa if sa_results else None
    }
    
    filepath = f"results/logs/parameter_tuning_{timestamp}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {filepath}")


if __name__ == "__main__":
    main()

