"""
Neuro Courier Project - Main Entry Point
This is the main control script for running TSP experiments.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

from src.models import Map
from src.generator import generate_map
from src.utils import plot_tour, calculate_tour_distance
from src.solvers.exact_solver import ExactSolver
from src.solvers.aco_solver import ACOSolver
from src.solvers.sa_solver import SASolver


def generate_instances():
    """Generate small, medium, and large TSP instances."""
    print("=" * 60)
    print("Generating TSP Instances")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Small instance: 10-12 cities
    print("\nGenerating small instance (10 cities)...")
    small_map = generate_map(num_cities=10, seed=42, name="Small_Instance")
    small_map.save_to_json("data/small_instances.json")
    print(f"  Saved: data/small_instances.json ({len(small_map)} cities)")
    
    # Medium instance: 50 cities
    print("\nGenerating medium instance (50 cities)...")
    medium_map = generate_map(num_cities=50, seed=42, name="Medium_Instance")
    medium_map.save_to_json("data/medium_instances.json")
    print(f"  Saved: data/medium_instances.json ({len(medium_map)} cities)")
    
    # Large instance: 100 cities
    print("\nGenerating large instance (100 cities)...")
    large_map = generate_map(num_cities=100, seed=42, name="Large_Instance")
    large_map.save_to_json("data/large_instances.json")
    print(f"  Saved: data/large_instances.json ({len(large_map)} cities)")
    
    print("\n✓ All instances generated successfully!")


def run_solver(solver_name: str, solver, cities, instance_name: str):
    """Run a solver and return results."""
    print(f"\n{'='*60}")
    print(f"Running {solver_name} on {instance_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    tour, distance = solver.solve(cities)
    elapsed_time = time.time() - start_time
    
    if tour is None:
        print(f"  ✗ {solver_name} failed to find a solution")
        return None
    
    print(f"\n  ✓ Solution found!")
    print(f"  Distance: {distance:.2f}")
    print(f"  Time: {elapsed_time:.2f} seconds")
    
    return {
        "solver": solver_name,
        "instance": instance_name,
        "tour": tour,
        "distance": distance,
        "time": elapsed_time,
        "num_cities": len(cities)
    }


def save_results(results: list, filename: str):
    """Save results to JSON file."""
    os.makedirs("results/logs", exist_ok=True)
    filepath = f"results/logs/{filename}"
    
    # Convert to serializable format
    serializable_results = []
    for r in results:
        if r is not None:
            serializable_results.append({
                "solver": r["solver"],
                "instance": r["instance"],
                "distance": r["distance"],
                "time": r["time"],
                "num_cities": r["num_cities"],
                "tour_length": len(r["tour"])
            })
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to {filepath}")


def run_experiments():
    """Run experiments on all instances with all solvers."""
    print("\n" + "=" * 60)
    print("NEURO COURIER PROJECT - TSP SOLVER EXPERIMENTS")
    print("=" * 60)
    
    # Load instances
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
        print(f"Loading {instance_name}")
        print(f"{'='*60}")
        
        map_instance = Map.load_from_json(filepath)
        cities = map_instance.cities
        print(f"  Loaded {len(cities)} cities")
        
        # Run Exact Solver (only for small instances)
        if instance_key == "small":
            exact_solver = ExactSolver()
            result = run_solver("Exact (Brute Force)", exact_solver, cities, instance_name)
            if result:
                all_results.append(result)
                
                # Plot result
                os.makedirs("results/plots", exist_ok=True)
                plot_path = f"results/plots/exact_{instance_key}.png"
                plot_tour(cities, result["tour"], 
                         title=f"Exact Solver - {instance_name}\nDistance: {result['distance']:.2f}",
                         save_path=plot_path, show=False)
        
        # Run ACO Solver
        aco_solver = ACOSolver(num_ants=50, max_iterations=100)
        result = run_solver("ACO (Ant Colony Optimization)", aco_solver, cities, instance_name)
        if result:
            all_results.append(result)
            
            # Plot result
            os.makedirs("results/plots", exist_ok=True)
            plot_path = f"results/plots/aco_{instance_key}.png"
            plot_tour(cities, result["tour"],
                     title=f"ACO Solver - {instance_name}\nDistance: {result['distance']:.2f}",
                     save_path=plot_path, show=False)
        
        # Run SA Solver
        sa_solver = SASolver(initial_temperature=1000.0, max_iterations=10000)
        result = run_solver("SA (Simulated Annealing)", sa_solver, cities, instance_name)
        if result:
            all_results.append(result)
            
            # Plot result
            os.makedirs("results/plots", exist_ok=True)
            plot_path = f"results/plots/sa_{instance_key}.png"
            plot_tour(cities, result["tour"],
                     title=f"SA Solver - {instance_name}\nDistance: {result['distance']:.2f}",
                     save_path=plot_path, show=False)
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(all_results, f"results_{timestamp}.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for result in all_results:
        print(f"{result['solver']:30s} | {result['instance']:20s} | "
              f"Distance: {result['distance']:8.2f} | Time: {result['time']:6.2f}s")
    print("=" * 60)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neuro Courier Project - TSP Solver")
    parser.add_argument("--generate", action="store_true", 
                       help="Generate TSP instances only")
    parser.add_argument("--run", action="store_true",
                       help="Run experiments on all instances")
    
    args = parser.parse_args()
    
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

