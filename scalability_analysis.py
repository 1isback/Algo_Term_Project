"""
Scalability Analysis Script
Analyzes how algorithms scale with problem size (runtime vs number of cities).
"""

import os
import time
import json
import csv
from datetime import datetime
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

from src.models import Map
from src.generator import generate_map
from src.solvers.optimized_exact_solver import calculate_distance_matrix, solve_tsp_ilp
from src.solvers.aco_solver import ACOSolver
from src.solvers.sa_solver import SASolver


def analyze_scalability(sizes: List[int], num_runs: int = 3):
    """Analyze scalability for different problem sizes."""
    print("=" * 60)
    print("SCALABILITY ANALYSIS")
    print("=" * 60)
    
    results = {
        "sizes": [],
        "exact": {"times": [], "distances": []},
        "aco": {"times": [], "distances": []},
        "sa": {"times": [], "distances": []}
    }
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {size} cities")
        print(f"{'='*60}")
        
        # Generate instance
        test_map = generate_map(num_cities=size, seed=42, name=f"Scalability_{size}")
        cities = test_map.cities
        
        results["sizes"].append(size)
        
        # Test Exact Solver (Optimized ILP - only for small sizes, max 21 cities)
        if size <= 21:
            print("\nTesting Exact Solver (Optimized ILP)...")
            exact_times = []
            exact_distances = []
            
            # Convert City objects to dict format for optimized solver
            cities_dict = [city.to_dict() for city in cities]
            dist_matrix = calculate_distance_matrix(cities_dict)
            
            for run in range(num_runs):
                start_time = time.time()
                tour, distance, status, model = solve_tsp_ilp(dist_matrix)
                elapsed_time = time.time() - start_time
                
                if tour is not None and distance is not None:
                    exact_times.append(elapsed_time)
                    exact_distances.append(distance)
            
            if exact_times:
                avg_time = sum(exact_times) / len(exact_times)
                avg_distance = sum(exact_distances) / len(exact_distances)
                results["exact"]["times"].append(avg_time)
                results["exact"]["distances"].append(avg_distance)
                print(f"  ✓ Avg Time: {avg_time:.4f}s, Avg Distance: {avg_distance:.2f}")
            else:
                results["exact"]["times"].append(None)
                results["exact"]["distances"].append(None)
        else:
            results["exact"]["times"].append(None)
            results["exact"]["distances"].append(None)
        
        # Test ACO Solver
        print("\nTesting ACO Solver...")
        aco_times = []
        aco_distances = []
        
        for run in range(num_runs):
            solver = ACOSolver(num_ants=50, max_iterations=50)  # Fixed iterations for fair comparison
            start_time = time.time()
            tour, distance, _ = solver.solve(cities, seed=42 + run)
            elapsed_time = time.time() - start_time
            
            if tour is not None:
                aco_times.append(elapsed_time)
                aco_distances.append(distance)
        
        if aco_times:
            avg_time = sum(aco_times) / len(aco_times)
            avg_distance = sum(aco_distances) / len(aco_distances)
            results["aco"]["times"].append(avg_time)
            results["aco"]["distances"].append(avg_distance)
            print(f"  ✓ Avg Time: {avg_time:.4f}s, Avg Distance: {avg_distance:.2f}")
        else:
            results["aco"]["times"].append(None)
            results["aco"]["distances"].append(None)
        
        # Test SA Solver
        print("\nTesting SA Solver...")
        sa_times = []
        sa_distances = []
        
        for run in range(num_runs):
            solver = SASolver(initial_temperature=1000.0, max_iterations=5000)  # Fixed iterations
            start_time = time.time()
            tour, distance, _ = solver.solve(cities, seed=42 + run)
            elapsed_time = time.time() - start_time
            
            if tour is not None:
                sa_times.append(elapsed_time)
                sa_distances.append(distance)
        
        if sa_times:
            avg_time = sum(sa_times) / len(sa_times)
            avg_distance = sum(sa_distances) / len(sa_distances)
            results["sa"]["times"].append(avg_time)
            results["sa"]["distances"].append(avg_distance)
            print(f"  ✓ Avg Time: {avg_time:.4f}s, Avg Distance: {avg_distance:.2f}")
        else:
            results["sa"]["times"].append(None)
            results["sa"]["distances"].append(None)
    
    return results


def plot_scalability(results: Dict):
    """Plot scalability graphs."""
    sizes = results["sizes"]
    
    # Runtime plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot runtime
    if any(results["exact"]["times"]):
        exact_times = [t if t is not None else 0 for t in results["exact"]["times"]]
        ax1.plot(sizes[:len(exact_times)], exact_times, 'ro-', label='Exact', linewidth=2, markersize=8)
    
    aco_times = [t if t is not None else 0 for t in results["aco"]["times"]]
    sa_times = [t if t is not None else 0 for t in results["sa"]["times"]]
    
    ax1.plot(sizes, aco_times, 'bo-', label='ACO', linewidth=2, markersize=8)
    ax1.plot(sizes, sa_times, 'go-', label='SA', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Cities', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Scalability: Runtime vs Problem Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot distance (solution quality)
    if any(results["exact"]["distances"]):
        exact_distances = [d if d is not None else 0 for d in results["exact"]["distances"]]
        ax2.plot(sizes[:len(exact_distances)], exact_distances, 'ro-', label='Exact (Optimal)', 
                linewidth=2, markersize=8)
    
    aco_distances = [d if d is not None else 0 for d in results["aco"]["distances"]]
    sa_distances = [d if d is not None else 0 for d in results["sa"]["distances"]]
    
    ax2.plot(sizes, aco_distances, 'bo-', label='ACO', linewidth=2, markersize=8)
    ax2.plot(sizes, sa_distances, 'go-', label='SA', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of Cities', fontsize=12)
    ax2.set_ylabel('Tour Distance', fontsize=12)
    ax2.set_title('Scalability: Solution Quality vs Problem Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    os.makedirs("results/plots", exist_ok=True)
    save_path = "results/plots/scalability_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Scalability plot saved to {save_path}")
    plt.close()


def save_scalability_results(results: Dict):
    """Save scalability results to CSV and JSON."""
    os.makedirs("results/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV
    csv_path = f"results/logs/scalability_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Size', 'Exact_Time', 'Exact_Distance', 'ACO_Time', 'ACO_Distance', 
                        'SA_Time', 'SA_Distance'])
        
        for i, size in enumerate(results["sizes"]):
            exact_time = results["exact"]["times"][i] if i < len(results["exact"]["times"]) else None
            exact_dist = results["exact"]["distances"][i] if i < len(results["exact"]["distances"]) else None
            aco_time = results["aco"]["times"][i] if i < len(results["aco"]["times"]) else None
            aco_dist = results["aco"]["distances"][i] if i < len(results["aco"]["distances"]) else None
            sa_time = results["sa"]["times"][i] if i < len(results["sa"]["times"]) else None
            sa_dist = results["sa"]["distances"][i] if i < len(results["sa"]["distances"]) else None
            
            writer.writerow([
                size,
                f"{exact_time:.4f}" if exact_time else "N/A",
                f"{exact_dist:.2f}" if exact_dist else "N/A",
                f"{aco_time:.4f}" if aco_time else "N/A",
                f"{aco_dist:.2f}" if aco_dist else "N/A",
                f"{sa_time:.4f}" if sa_time else "N/A",
                f"{sa_dist:.2f}" if sa_dist else "N/A"
            ])
    
    print(f"✓ CSV results saved to {csv_path}")
    
    # Save JSON
    json_path = f"results/logs/scalability_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ JSON results saved to {json_path}")


def main():
    """Main function."""
    # Test sizes: small to medium (exact solver works up to 21 cities)
    sizes = [5, 10, 15, 20, 21, 30, 40, 50]
    
    print("\nStarting scalability analysis...")
    print(f"Testing sizes: {sizes}")
    print("This may take a while...\n")
    
    results = analyze_scalability(sizes, num_runs=3)
    
    # Plot results
    plot_scalability(results)
    
    # Save results
    save_scalability_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SCALABILITY SUMMARY")
    print("=" * 60)
    print(f"{'Size':<8} | {'Exact Time':<12} | {'ACO Time':<12} | {'SA Time':<12}")
    print("-" * 60)
    
    for i, size in enumerate(results["sizes"]):
        exact_time = results["exact"]["times"][i] if i < len(results["exact"]["times"]) and results["exact"]["times"][i] else "N/A"
        aco_time = results["aco"]["times"][i] if i < len(results["aco"]["times"]) and results["aco"]["times"][i] else "N/A"
        sa_time = results["sa"]["times"][i] if i < len(results["sa"]["times"]) and results["sa"]["times"][i] else "N/A"
        
        exact_str = f"{exact_time:.4f}s" if isinstance(exact_time, float) else "N/A"
        aco_str = f"{aco_time:.4f}s" if isinstance(aco_time, float) else "N/A"
        sa_str = f"{sa_time:.4f}s" if isinstance(sa_time, float) else "N/A"
        
        print(f"{size:<8} | {exact_str:<12} | {aco_str:<12} | {sa_str:<12}")


if __name__ == "__main__":
    main()

