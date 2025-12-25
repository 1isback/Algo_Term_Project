"""
Parameter Tuning Script for ACO and SA Solvers
Tests different parameter combinations to find optimal settings.
"""

import os
import time
import json
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
    print("ACO PARAMETER TUNING")
    print("=" * 60)
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] Testing: {params}")
        
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
    print("SA PARAMETER TUNING")
    print("=" * 60)
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] Testing: {params}")
        
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


def main():
    """Main tuning function."""
    print("=" * 60)
    print("PARAMETER TUNING FOR TSP SOLVERS")
    print("=" * 60)
    
    # Generate or load test instance
    test_file = "data/small_instances.json"
    if not os.path.exists(test_file):
        print("\nGenerating test instance...")
        os.makedirs("data", exist_ok=True)
        test_map = generate_map(num_cities=10, seed=42, name="Tuning_Test")
        test_map.save_to_json(test_file)
    
    map_instance = Map.load_from_json(test_file)
    cities = map_instance.cities
    print(f"\nUsing test instance: {len(cities)} cities")
    
    # ACO Parameter combinations
    aco_combinations = [
        {"num_ants": 30, "alpha": 1.0, "beta": 2.0, "evaporation_rate": 0.5, "max_iterations": 50},
        {"num_ants": 50, "alpha": 1.0, "beta": 2.0, "evaporation_rate": 0.5, "max_iterations": 50},
        {"num_ants": 50, "alpha": 1.0, "beta": 2.0, "evaporation_rate": 0.3, "max_iterations": 50},
        {"num_ants": 50, "alpha": 1.0, "beta": 2.0, "evaporation_rate": 0.7, "max_iterations": 50},
        {"num_ants": 50, "alpha": 1.5, "beta": 2.0, "evaporation_rate": 0.5, "max_iterations": 50},
        {"num_ants": 50, "alpha": 1.0, "beta": 3.0, "evaporation_rate": 0.5, "max_iterations": 50},
        {"num_ants": 50, "alpha": 1.0, "beta": 2.0, "evaporation_rate": 0.5, "max_iterations": 100},
    ]
    
    # SA Parameter combinations
    sa_combinations = [
        {"initial_temperature": 500.0, "cooling_rate": 0.99, "max_iterations": 5000},
        {"initial_temperature": 1000.0, "cooling_rate": 0.995, "max_iterations": 5000},
        {"initial_temperature": 2000.0, "cooling_rate": 0.995, "max_iterations": 5000},
        {"initial_temperature": 1000.0, "cooling_rate": 0.99, "max_iterations": 5000},
        {"initial_temperature": 1000.0, "cooling_rate": 0.998, "max_iterations": 5000},
        {"initial_temperature": 1000.0, "cooling_rate": 0.995, "max_iterations": 10000},
    ]
    
    # Run tuning
    aco_results = test_aco_parameters(cities, aco_combinations, num_runs=3)
    sa_results = test_sa_parameters(cities, sa_combinations, num_runs=3)
    
    # Find best parameters
    if aco_results:
        best_aco = min(aco_results, key=lambda x: x["avg_distance"])
        print("\n" + "=" * 60)
        print("BEST ACO PARAMETERS")
        print("=" * 60)
        print(f"Parameters: {best_aco['params']}")
        print(f"Average Distance: {best_aco['avg_distance']:.2f}")
        print(f"Minimum Distance: {best_aco['min_distance']:.2f}")
        print(f"Average Time: {best_aco['avg_time']:.2f}s")
    
    if sa_results:
        best_sa = min(sa_results, key=lambda x: x["avg_distance"])
        print("\n" + "=" * 60)
        print("BEST SA PARAMETERS")
        print("=" * 60)
        print(f"Parameters: {best_sa['params']}")
        print(f"Average Distance: {best_sa['avg_distance']:.2f}")
        print(f"Minimum Distance: {best_sa['min_distance']:.2f}")
        print(f"Average Time: {best_sa['avg_time']:.2f}s")
    
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

