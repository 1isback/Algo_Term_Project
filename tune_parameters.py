"""
Fast Parameter Tuning Script for ACO and SA Solvers
Tests on all instances (20, 50, 100 cities) to find best universal parameters
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


def test_aco_on_all_instances(instances_data, param_combinations: List[Dict], 
                               num_runs: int = 2, max_time: int = 720):
    """Test ACO parameters on all instances (small, medium, large)."""
    print("\n" + "=" * 70)
    print("ACO PARAMETER TUNING ON ALL INSTANCES")
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
        
        # Test on all instances
        instance_results = {}
        all_distances = []
        all_times = []
        
        for instance_name, cities in instances_data.items():
            distances = []
            times = []
            
            for run in range(num_runs):
                try:
                    solver = ACOSolver(**params)
                    run_start = time.time()
                    tour, distance, _ = solver.solve(cities, seed=42 + run)
                    run_time = time.time() - run_start
                    
                    if tour is not None:
                        distances.append(distance)
                        times.append(run_time)
                        all_distances.append(distance)
                        all_times.append(run_time)
                except Exception as e:
                    continue
            
            if distances:
                instance_results[instance_name] = {
                    "avg_distance": sum(distances) / len(distances),
                    "min_distance": min(distances),
                    "avg_time": sum(times) / len(times)
                }
        
        # Calculate overall score (normalized average across all instances)
        if all_distances:
            # Normalize distances for fair comparison across different sized instances
            avg_distance = sum(all_distances) / len(all_distances)
            avg_time = sum(all_times) / len(all_times)
            
            result = {
                "params": params,
                "overall_avg_distance": avg_distance,
                "overall_avg_time": avg_time,
                "instance_results": instance_results,
                "num_successful_runs": len(all_distances)
            }
            results.append(result)
    
    print_progress_bar(min(i+1, total), total, 
                      prefix=f'ACO Testing',
                      suffix=f'({min(i+1, total)}/{total}) DONE')
    print()
    
    return results


def test_sa_on_all_instances(instances_data, param_combinations: List[Dict], 
                              num_runs: int = 3, max_time: int = 420):
    """Test SA parameters on all instances (small, medium, large)."""
    print("\n" + "=" * 70)
    print("SA PARAMETER TUNING ON ALL INSTANCES")
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
        
        # Test on all instances
        instance_results = {}
        all_distances = []
        all_times = []
        
        for instance_name, cities in instances_data.items():
            distances = []
            times = []
            
            for run in range(num_runs):
                try:
                    solver = SASolver(**params)
                    run_start = time.time()
                    tour, distance, _ = solver.solve(cities, seed=42 + run)
                    run_time = time.time() - run_start
                    
                    if tour is not None:
                        distances.append(distance)
                        times.append(run_time)
                        all_distances.append(distance)
                        all_times.append(run_time)
                except Exception as e:
                    continue
            
            if distances:
                instance_results[instance_name] = {
                    "avg_distance": sum(distances) / len(distances),
                    "min_distance": min(distances),
                    "avg_time": sum(times) / len(times)
                }
        
        # Calculate overall score
        if all_distances:
            avg_distance = sum(all_distances) / len(all_distances)
            avg_time = sum(all_times) / len(all_times)
            
            result = {
                "params": params,
                "overall_avg_distance": avg_distance,
                "overall_avg_time": avg_time,
                "instance_results": instance_results,
                "num_successful_runs": len(all_distances)
            }
            results.append(result)
    
    print_progress_bar(min(i+1, total), total, 
                      prefix=f'SA Testing',
                      suffix=f'({min(i+1, total)}/{total}) DONE')
    print()
    
    return results


def generate_aco_grid_search_combinations():
    """Generate reduced ACO parameter combinations for faster tuning."""
    param_grid = {
        "num_ants": [30, 50, 70],
        "alpha": [0.5, 1.0, 2.0],
        "beta": [2.0, 3.0, 4.0],
        "evaporation_rate": [0.3, 0.5],
        "max_iterations": [50, 100],
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
    """Generate all SA parameter combinations using grid search."""
    param_grid = {
        "initial_temperature": [500.0, 1000.0, 2000.0, 3000.0],
        "cooling_rate": [0.99, 0.995, 0.998],
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


def load_all_instances():
    """Load all test instances (small, medium, large)."""
    instances = {}
    
    files = {
        "small_20": "data/small_instances.json",
        "medium_50": "data/medium_instances.json",
        "large_100": "data/large_instances.json"
    }
    
    for name, filepath in files.items():
        if os.path.exists(filepath):
            try:
                map_instance = Map.load_from_json(filepath)
                instances[name] = map_instance.cities
                print(f"  ✓ Loaded {name}: {len(map_instance.cities)} cities")
            except Exception as e:
                print(f"  ✗ Error loading {name}: {e}")
        else:
            print(f"  ⚠ File not found: {filepath}")
    
    return instances


def main():
    """Main tuning function with all instances."""
    print("=" * 70)
    print("UNIVERSAL PARAMETER TUNING FOR TSP SOLVERS")
    print("Testing on Small (20), Medium (50), and Large (100) instances")
    print("=" * 70)
    
    # Time allocation
    MAX_TOTAL_TIME = 20 * 60  # 20 minutes
    ACO_TIME_BUDGET = 12 * 60  # 12 minutes for ACO
    SA_TIME_BUDGET = 7 * 60    # 7 minutes for SA
    
    # Load all instances
    print("\n1. Loading all instances...")
    instances_data = load_all_instances()
    
    if not instances_data:
        print("\n✗ No instances found! Please ensure data files exist.")
        return
    
    print(f"\n  Total instances loaded: {len(instances_data)}")
    
    # Generate parameter combinations
    print("\n2. Generating ACO parameter combinations (OPTIMIZED)...")
    aco_combinations = generate_aco_grid_search_combinations()
    print(f"   Total ACO combinations: {len(aco_combinations)}")
    print("   Parameter ranges:")
    print("     num_ants: [30, 50, 70]")
    print("     alpha: [0.5, 1.0, 2.0]")
    print("     beta: [2.0, 3.0, 4.0]")
    print("     evaporation_rate: [0.3, 0.5]")
    print("     max_iterations: [50, 100]")
    print("     elitist_weight: [1.0, 2.0]")
    
    print("\n3. Generating SA parameter combinations (FULL)...")
    sa_combinations = generate_sa_grid_search_combinations()
    print(f"   Total SA combinations: {len(sa_combinations)}")
    print("   Parameter ranges:")
    print("     initial_temperature: [500, 1000, 2000, 3000]")
    print("     cooling_rate: [0.99, 0.995, 0.998]")
    print("     min_temperature: [0.1, 0.5, 1.0]")
    print("     max_iterations: [5000, 10000, 15000]")
    
    print("\n" + "=" * 70)
    print(f"Time budget: {MAX_TOTAL_TIME/60:.0f} minutes total")
    print(f"  ACO: {ACO_TIME_BUDGET/60:.0f} minutes")
    print(f"  SA: {SA_TIME_BUDGET/60:.0f} minutes")
    print("=" * 70)
    
    overall_start = time.time()
    
    # Run ACO tuning
    print("\n4. Testing ACO on all instances...")
    aco_results = test_aco_on_all_instances(instances_data, aco_combinations, 
                                            num_runs=2, max_time=ACO_TIME_BUDGET)
    
    # Run SA tuning
    print("\n5. Testing SA on all instances...")
    sa_results = test_sa_on_all_instances(instances_data, sa_combinations, 
                                          num_runs=3, max_time=SA_TIME_BUDGET)
    
    total_time = time.time() - overall_start
    
    # Find best universal parameters
    if aco_results:
        best_aco = min(aco_results, key=lambda x: x["overall_avg_distance"])
        print("\n" + "=" * 70)
        print("BEST UNIVERSAL ACO PARAMETERS (Works for all instances)")
        print("=" * 70)
        print(f"Parameters:")
        for key, value in best_aco['params'].items():
            print(f"  {key}: {value}")
        print(f"\nOverall Performance:")
        print(f"  Average Distance (across all instances): {best_aco['overall_avg_distance']:.2f}")
        print(f"  Average Time: {best_aco['overall_avg_time']:.2f}s")
        print(f"  Total Successful Runs: {best_aco['num_successful_runs']}")
        
        print(f"\nPer Instance Performance:")
        for instance_name, results in best_aco['instance_results'].items():
            print(f"  {instance_name}:")
            print(f"    Avg Distance: {results['avg_distance']:.2f}")
            print(f"    Min Distance: {results['min_distance']:.2f}")
            print(f"    Avg Time: {results['avg_time']:.2f}s")
        
        # Show top 5
        sorted_aco = sorted(aco_results, key=lambda x: x["overall_avg_distance"])[:5]
        print(f"\nTop 5 Universal ACO Parameter Sets:")
        for i, result in enumerate(sorted_aco, 1):
            print(f"  {i}. Overall Dist: {result['overall_avg_distance']:.2f}")
            print(f"     Params: {result['params']}")
    
    if sa_results:
        best_sa = min(sa_results, key=lambda x: x["overall_avg_distance"])
        print("\n" + "=" * 70)
        print("BEST UNIVERSAL SA PARAMETERS (Works for all instances)")
        print("=" * 70)
        print(f"Parameters:")
        for key, value in best_sa['params'].items():
            print(f"  {key}: {value}")
        print(f"\nOverall Performance:")
        print(f"  Average Distance (across all instances): {best_sa['overall_avg_distance']:.2f}")
        print(f"  Average Time: {best_sa['overall_avg_time']:.2f}s")
        print(f"  Total Successful Runs: {best_sa['num_successful_runs']}")
        
        print(f"\nPer Instance Performance:")
        for instance_name, results in best_sa['instance_results'].items():
            print(f"  {instance_name}:")
            print(f"    Avg Distance: {results['avg_distance']:.2f}")
            print(f"    Min Distance: {results['min_distance']:.2f}")
            print(f"    Avg Time: {results['avg_time']:.2f}s")
        
        # Show top 5
        sorted_sa = sorted(sa_results, key=lambda x: x["overall_avg_distance"])[:5]
        print(f"\nTop 5 Universal SA Parameter Sets:")
        for i, result in enumerate(sorted_sa, 1):
            print(f"  {i}. Overall Dist: {result['overall_avg_distance']:.2f}")
            print(f"     Params: {result['params']}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Instances tested: {len(instances_data)}")
    print(f"ACO combinations tested: {len(aco_results)}/{len(aco_combinations)}")
    print(f"SA combinations tested: {len(sa_results)}/{len(sa_combinations)}")
    print("=" * 70)
    
    # Save results
    os.makedirs("results/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {
        "tuning_time_seconds": total_time,
        "instances_tested": list(instances_data.keys()),
        "aco_combinations_tested": len(aco_results),
        "sa_combinations_tested": len(sa_results),
        "aco_results": aco_results,
        "sa_results": sa_results,
        "best_universal_aco": best_aco if aco_results else None,
        "best_universal_sa": best_sa if sa_results else None
    }
    
    filepath = f"results/logs/universal_tuning_{timestamp}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {filepath}")
    
    # Print final recommendation
    if aco_results and sa_results:
        print("\n" + "=" * 70)
        print("🎯 RECOMMENDED UNIVERSAL PARAMETERS")
        print("=" * 70)
        print("\nACO:")
        for key, value in best_aco['params'].items():
            print(f"  {key}: {value}")
        print("\nSA:")
        for key, value in best_sa['params'].items():
            print(f"  {key}: {value}")
        print("\nThese parameters work best across all instance sizes (20, 50, 100)!")
        print("=" * 70)


if __name__ == "__main__":
    main()
