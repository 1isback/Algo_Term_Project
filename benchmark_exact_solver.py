"""
Benchmark script to test ExactSolver performance on different instance sizes.
"""

import time
import math
from src.generator import generate_map
from src.solvers.exact_solver import ExactSolver


def benchmark_exact_solver(sizes=[10, 12, 15]):
    """Benchmark exact solver on different instance sizes."""
    print("=" * 60)
    print("EXACT SOLVER PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    solver = ExactSolver()
    results = []
    
    for n in sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n} cities...")
        print(f"{'='*60}")
        
        # Generate test instance
        test_map = generate_map(num_cities=n, seed=42, name=f"Benchmark_{n}")
        cities = test_map.cities
        
        # Calculate permutations
        num_permutations = math.factorial(n - 1)
        print(f"  Total permutations to check: {num_permutations:,}")
        
        # Run solver
        print(f"  Running exact solver...")
        start_time = time.time()
        try:
            tour, distance, _ = solver.solve(cities)
            elapsed_time = time.time() - start_time
            
            if tour is not None:
                results.append({
                    'n': n,
                    'permutations': num_permutations,
                    'time': elapsed_time,
                    'distance': distance,
                    'success': True
                })
                print(f"  ✓ Completed in {elapsed_time:.2f} seconds")
                print(f"  ✓ Optimal distance: {distance:.2f}")
                print(f"  ✓ Time per permutation: {(elapsed_time/num_permutations)*1e6:.4f} microseconds")
            else:
                print(f"  ✗ Failed (instance too large)")
                results.append({
                    'n': n,
                    'permutations': num_permutations,
                    'time': None,
                    'success': False
                })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'n': n,
                'permutations': num_permutations,
                'time': None,
                'success': False
            })
    
    # Summary and extrapolation
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Cities':<10} | {'Permutations':<20} | {'Time (s)':<15} | {'μs/perm':<15}")
    print("-" * 60)
    
    for r in results:
        if r['success']:
            time_str = f"{r['time']:.2f}"
            us_per_perm = (r['time'] / r['permutations']) * 1e6
            print(f"{r['n']:<10} | {r['permutations']:<20,} | {time_str:<15} | {us_per_perm:<15.4f}")
        else:
            print(f"{r['n']:<10} | {r['permutations']:<20,} | {'FAILED':<15} | {'-':<15}")
    
    # Estimate for 20 cities
    print("\n" + "=" * 60)
    print("EXTRAPOLATION FOR 20 CITIES")
    print("=" * 60)
    
    successful_results = [r for r in results if r['success']]
    if len(successful_results) >= 2:
        # Calculate average time per permutation
        total_time = sum(r['time'] for r in successful_results)
        total_perms = sum(r['permutations'] for r in successful_results)
        avg_us_per_perm = (total_time / total_perms) * 1e6
        
        n20_permutations = math.factorial(20 - 1)
        estimated_time = (avg_us_per_perm * n20_permutations) / 1e6
        
        print(f"Average time per permutation: {avg_us_per_perm:.4f} microseconds")
        print(f"20 cities permutations (19!): {n20_permutations:,}")
        print(f"\nEstimated time for 20 cities: {estimated_time:.2f} seconds")
        print(f"  = {estimated_time/60:.2f} minutes")
        print(f"  = {estimated_time/3600:.2f} hours")
    else:
        print("Not enough successful results to extrapolate.")
    
    return results


if __name__ == "__main__":
    # Test with progressively larger instances
    # Start with smaller ones to get baseline
    benchmark_exact_solver(sizes=[10, 12, 15])

