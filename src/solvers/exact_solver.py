"""
Brute Force (Exact) Solver for TSP.
Only suitable for small instances (typically < 12 cities).
"""

import itertools
from typing import List, Tuple
from ..models import City
from ..utils import calculate_tour_distance


class ExactSolver:
    """Brute force solver that tries all possible tours."""
    
    def __init__(self):
        self.best_tour = None
        self.best_distance = float('inf')
    
    def solve(self, cities: List[City]) -> Tuple[List[int], float, List[float]]:
        """
        Solve TSP using brute force (tries all permutations).
        
        Args:
            cities: List of cities to visit
        
        Returns:
            Tuple of (best_tour, best_distance, history)
            history: Empty list (brute force doesn't have iterations)
        """
        n = len(cities)
        
        if n < 2:
            return [0], 0.0, [0.0]
        
        if n > 101:
            print(f"Warning: Brute force is too slow for {n} cities. Consider using heuristic methods.")
            return None, float('inf'), []
        
        # Generate all permutations (excluding the first city to reduce redundancy)
        # We fix the first city and permute the rest
        indices = list(range(1, n))
        best_tour = None
        best_distance = float('inf')
        history = []  # Empty for brute force (no iterations)
        
        for perm in itertools.permutations(indices):
            tour = [0] + list(perm)  # Start from city 0
            distance = calculate_tour_distance(cities, tour)
            
            if distance < best_distance:
                best_distance = distance
                best_tour = tour
        
        self.best_tour = best_tour
        self.best_distance = best_distance
        
        return best_tour, best_distance, history

