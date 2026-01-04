"""
Parameter Tuning Script for ACO and SA Solvers
Tests parameters across multiple instance sizes to find robust settings.
"""

import os
import time
import json
import itertools
from datetime import datetime
from typing import Dict, List, Tuple

from src.models import Map
from src.generator import generate_map
from src.solvers.aco_solver import ACOSolver
from src.solvers.sa_solver import SASolver
from src.utils import calculate_tour_distance


def load_existing_instances():
    """Load existing instances from data folder."""
    instance_files = [
        "data/small_instances.json",
        "data/medium_instances.json",
        "data/large_instances.json"
    ]
    
    instances = []
    for filepath in instance_files:
        if os.path.exists(filepath):
            print(f"  Loading {os.path.basename(filepath)}...")
            map_instance = Map.load_from_json(filepath)
            
            # Extract instance name from filename
            filename = os.path.basename(filepath)
            name = filename.replace('_instances.json', '').capitalize()
            
            instances.append({
                "name": name,
                "size": len(map_instance.cities),
                "cities": map_instance.cities,
                "file": filepath
            })
        else:
            print(f"  Warning: {filepath} not found, skipping...")
    
    if not instances:
        raise FileNotFoundError("No instance files found in data/ folder!")
    
    return instances


def test_aco_on_instances(instances: List[Dict], param_combinations: List[Dict], num_runs: int = 3):
    """Test ACO parameters across multiple instances."""
    print("\n" + "=" * 70)
    print("ACO PARAMETER TUNING - MULTI-INSTANCE GRID SEARCH")
    print("=" * 70)
    
    results = []
    total = len(param_combinations)
    
    for i, params in enumerate(param_combinations):
        progress = (i + 1) / total * 100
        print(f"\n[{i+1}/{total}] ({progress:.1f}%) Testing: {params}")
        
        instance_results = {}
        
        for instance in instances:
            distances = []
            times = []
            
            for run in range(num_runs):
                solver = ACOSolver(**params)
                start_time = time.time()
                tour, distance, _ = solver.solve(instance['cities'], seed=42 + run)
                elapsed_time = time.time() - start_time
                
                if tour is not None:
                    distances.append(distance)
                    times.append(elapsed_time)
            
            if distances:
                instance_results[instance['name']] = {
                    "avg_distance": sum(distances) / len(distances),
                    "min_distance": min(distances),
                    "avg_time": sum(times) / len(times),
                    "successful_runs": len(distances)
                }
                print(f"  {instance['name']:12} -> Avg: {instance_results[instance['name']]['avg_distance']:7.2f}, "
                      f"Min: {instance_results[instance['name']]['min_distance']:7.2f}, "
                      f"Time: {instance_results[instance['name']]['avg_time']:5.2f}s")
            else:
                print(f"  {instance['name']:12} -> FAILED")
        
        if instance_results:
            # Calculate normalized scores for fair comparison across instance sizes
            # Lower score is better
            avg_distances = [r['avg_distance'] for r in instance_results.values()]
            avg_times = [r['avg_time'] for r in instance_results.values()]
            
            result = {
                "params": params,
                "instance_results": instance_results,
                "overall_avg_distance": sum(avg_distances) / len(avg_distances),
                "overall_avg_time": sum(avg_times) / len(avg_times),
                "total_successful": sum(r['successful_runs'] for r in instance_results.values()),
                "total_possible": len(instances) * num_runs
            }
            results.append(result)
        else:
            print(f"  ✗ All instances failed")
    
    return results


def test_sa_on_instances(instances: List[Dict], param_combinations: List[Dict], num_runs: int = 3):
    """Test SA parameters across multiple instances."""
    print("\n" + "=" * 70)
    print("SA PARAMETER TUNING - MULTI-INSTANCE GRID SEARCH")
    print("=" * 70)
    
    results = []
    total = len(param_combinations)
    
    for i, params in enumerate(param_combinations):
        progress = (i + 1) / total * 100
        print(f"\n[{i+1}/{total}] ({progress:.1f}%) Testing: {params}")
        
        instance_results = {}
        
        for instance in instances:
            distances = []
            times = []
            
            for run in range(num_runs):
                solver = SASolver(**params)
                start_time = time.time()
                tour, distance, _ = solver.solve(instance['cities'], seed=42 + run)
                elapsed_time = time.time() - start_time
                
                if tour is not None:
                    distances.append(distance)
                    times.append(elapsed_time)
            
            if distances:
                instance_results[instance['name']] = {
                    "avg_distance": sum(distances) / len(distances),
                    "min_distance": min(distances),
                    "avg_time": sum(times) / len(times),
                    "successful_runs": len(distances)
                }
                print(f"  {instance['name']:12} -> Avg: {instance_results[instance['name']]['avg_distance']:7.2f}, "
                      f"Min: {instance_results[instance['name']]['min_distance']:7.2f}, "
                      f"Time: {instance_results[instance['name']]['avg_time']:5.2f}s")
            else:
                print(f"  {instance['name']:12} -> FAILED")
        
        if instance_results:
            avg_distances = [r['avg_distance'] for r in instance_results.values()]
            avg_times = [r['avg_time'] for r in instance_results.values()]
            
            result = {
                "params": params,
                "instance_results": instance_results,
                "overall_avg_distance": sum(avg_distances) / len(avg_distances),
                "overall_avg_time": sum(avg_times) / len(avg_times),
                "total_successful": sum(r['successful_runs'] for r in instance_results.values()),
                "total_possible": len(instances) * num_runs
            }
            results.append(result)
        else:
            print(f"  ✗ All instances failed")
    
    return results


def generate_aco_grid_search_combinations():
    """Generate ACO parameter combinations - reduced grid for faster tuning."""
    param_grid = {
        "num_ants": [50, 70],
        "alpha": [1.0, 1.5],
        "beta": [2.0, 3.0],
        "evaporation_rate": [0.3, 0.5],
        "max_iterations": [100, 150],
        "elitist_weight": [1.0, 2.0]
    }
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def generate_sa_grid_search_combinations():
    """Generate SA parameter combinations - balanced grid for multi-instance testing."""
    param_grid = {
        "initial_temperature": [1000.0, 2000.0, 3000.0],
        "cooling_rate": [0.995, 0.998],
        "min_temperature": [0.1, 0.5, 1.0],
        "max_iterations": [5000, 10000, 15000]
    }
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def main():
    """Main tuning function."""
    print("=" * 70)
    print("MULTI-INSTANCE PARAMETER TUNING FOR TSP SOLVERS")
    print("=" * 70)
    
    # Load existing instances
    print("\nLoading existing instances from data/ folder...")
    instances = load_existing_instances()
    print(f"\nUsing {len(instances)} test instances:")
    for inst in instances:
        print(f"  - {inst['name']}: {inst['size']} cities (from {inst['file']})")
    
    # Generate parameter combinations
    print("\nGenerating ACO parameter combinations...")
    aco_combinations = generate_aco_grid_search_combinations()
    print(f"Total ACO combinations: {len(aco_combinations)}")
    
    print("\nGenerating SA parameter combinations...")
    sa_combinations = generate_sa_grid_search_combinations()
    print(f"Total SA combinations: {len(sa_combinations)}")
    
    # Estimate time
    est_aco_time = len(aco_combinations) * len(instances) * 3 * 1.0  # ~1s per run
    est_sa_time = len(sa_combinations) * len(instances) * 3 * 0.5   # ~0.5s per run
    total_est = est_aco_time + est_sa_time
    
    print("\n" + "=" * 70)
    print("EXECUTION PLAN")
    print("=" * 70)
    print(f"ACO: {len(aco_combinations)} combinations × {len(instances)} instances × 3 runs")
    print(f"SA:  {len(sa_combinations)} combinations × {len(instances)} instances × 3 runs")
    print(f"Estimated time: ~{total_est/60:.1f} minutes")
    print("=" * 70)
    
    user_input = input("\nProceed with tuning? (yes/no): ")
    if user_input.lower() not in ['yes', 'y']:
        print("Tuning cancelled.")
        return
    
    # Run tuning
    start_time = time.time()
    aco_results = test_aco_on_instances(instances, aco_combinations, num_runs=3)
    sa_results = test_sa_on_instances(instances, sa_combinations, num_runs=3)
    total_time = time.time() - start_time
    
    # Analyze results
    if aco_results:
        # Sort by overall average distance (normalized across all instances)
        best_aco = min(aco_results, key=lambda x: x["overall_avg_distance"])
        print("\n" + "=" * 70)
        print("BEST ACO PARAMETERS (Multi-Instance)")
        print("=" * 70)
        print(f"Parameters:")
        for key, value in best_aco['params'].items():
            print(f"  {key}: {value}")
        print(f"\nOverall Performance:")
        print(f"  Average Distance (normalized): {best_aco['overall_avg_distance']:.2f}")
        print(f"  Average Time: {best_aco['overall_avg_time']:.2f}s")
        print(f"  Success Rate: {best_aco['total_successful']}/{best_aco['total_possible']}")
        print(f"\nPer-Instance Results:")
        for inst_name, inst_result in best_aco['instance_results'].items():
            print(f"  {inst_name:12} -> Avg: {inst_result['avg_distance']:7.2f}, "
                  f"Min: {inst_result['min_distance']:7.2f}")
        
        # Top 5
        sorted_aco = sorted(aco_results, key=lambda x: x["overall_avg_distance"])[:5]
        print(f"\nTop 5 ACO Parameter Sets:")
        for i, result in enumerate(sorted_aco, 1):
            print(f"  {i}. Overall Avg: {result['overall_avg_distance']:.2f}, "
                  f"Success: {result['total_successful']}/{result['total_possible']}")
            print(f"     Params: {result['params']}")
    
    if sa_results:
        best_sa = min(sa_results, key=lambda x: x["overall_avg_distance"])
        print("\n" + "=" * 70)
        print("BEST SA PARAMETERS (Multi-Instance)")
        print("=" * 70)
        print(f"Parameters:")
        for key, value in best_sa['params'].items():
            print(f"  {key}: {value}")
        print(f"\nOverall Performance:")
        print(f"  Average Distance (normalized): {best_sa['overall_avg_distance']:.2f}")
        print(f"  Average Time: {best_sa['overall_avg_time']:.2f}s")
        print(f"  Success Rate: {best_sa['total_successful']}/{best_sa['total_possible']}")
        print(f"\nPer-Instance Results:")
        for inst_name, inst_result in best_sa['instance_results'].items():
            print(f"  {inst_name:12} -> Avg: {inst_result['avg_distance']:7.2f}, "
                  f"Min: {inst_result['min_distance']:7.2f}")
        
        # Top 5
        sorted_sa = sorted(sa_results, key=lambda x: x["overall_avg_distance"])[:5]
        print(f"\nTop 5 SA Parameter Sets:")
        for i, result in enumerate(sorted_sa, 1):
            print(f"  {i}. Overall Avg: {result['overall_avg_distance']:.2f}, "
                  f"Success: {result['total_successful']}/{result['total_possible']}")
            print(f"     Params: {result['params']}")
    
    # Save results
    os.makedirs("results/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {
        
        "instances": [{"name": i['name'], "size": i['size']} for i in instances],
        "aco_results": aco_results,
        "sa_results": sa_results,
        "best_aco": best_aco if aco_results else None,
        "best_sa": best_sa if sa_results else None,
        "total_time": total_time
    }
    
    filepath = f"results/logs/multi_instance_tuning_{timestamp}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print(f"✓ Tuning completed in {total_time/60:.1f} minutes")
    print(f"✓ Results saved to {filepath}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    
    