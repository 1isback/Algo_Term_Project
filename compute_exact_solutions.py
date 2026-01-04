"""
Compute Exact Solutions for All Instances
This script calculates optimal solutions using brute force for all instances
and saves them for later use in approximation ratio calculations.
"""

import os
import json
import time
from src.models import Map
from src.solvers.exact_solver import ExactSolver


def compute_exact_solutions():
    """Compute exact solutions for all instances."""
    print("=" * 60)
    print("COMPUTING EXACT SOLUTIONS FOR ALL INSTANCES")
    print("=" * 60)
    
    instances = {
        "small": ("data/small_instances.json", "Small Instance"),
        "medium": ("data/medium_instances.json", "Medium Instance"),
        "large": ("data/large_instances.json", "Large Instance")
    }
    
    exact_solutions = {}
    
    for instance_key, (filepath, instance_name) in instances.items():
        if not os.path.exists(filepath):
            print(f"\n⚠ Warning: {filepath} not found. Skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {instance_name}")
        print(f"{'='*60}")
        
        map_instance = Map.load_from_json(filepath)
        cities = map_instance.cities
        n = len(cities)
        
        print(f"  Loaded {n} cities")
        
        # Check if instance is too large for exact solver
        if n > 21:
            print(f"  ⚠ Instance too large ({n} cities) for exact solver (max 21)")
            print(f"  → Exact solution will not be computed")
            exact_solutions[instance_key] = {
                "instance": instance_name,
                "num_cities": n,
                "optimal_distance": None,
                "computation_time": None,
                "status": "too_large"
            }
            continue
        
        print(f"  Computing exact solution (brute force)...")
        solver = ExactSolver()
        start_time = time.time()
        result = solver.solve(cities)
        elapsed_time = time.time() - start_time
        
        if len(result) == 3:
            tour, distance, _ = result
        else:
            tour, distance = result
        
        if tour is not None:
            exact_solutions[instance_key] = {
                "instance": instance_name,
                "num_cities": n,
                "optimal_distance": distance,
                "optimal_tour": tour,
                "computation_time": elapsed_time,
                "status": "success"
            }
            print(f"  ✓ Optimal distance: {distance:.2f}")
            print(f"  ✓ Computation time: {elapsed_time:.2f}s")
        else:
            exact_solutions[instance_key] = {
                "instance": instance_name,
                "num_cities": n,
                "optimal_distance": None,
                "computation_time": None,
                "status": "failed"
            }
            print(f"  ✗ Failed to compute exact solution")
    
    # Save exact solutions
    os.makedirs("data", exist_ok=True)
    output_file = "data/exact_solutions.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(exact_solutions, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, solution in exact_solutions.items():
        status = solution["status"]
        if status == "success":
            print(f"{solution['instance']:20s} | Optimal: {solution['optimal_distance']:8.2f} | "
                  f"Time: {solution['computation_time']:6.2f}s")
        elif status == "too_large":
            print(f"{solution['instance']:20s} | Status: Too large for exact solver")
        else:
            print(f"{solution['instance']:20s} | Status: Failed")
    
    print(f"\n✓ Exact solutions saved to {output_file}")
    return exact_solutions


if __name__ == "__main__":
    compute_exact_solutions()

