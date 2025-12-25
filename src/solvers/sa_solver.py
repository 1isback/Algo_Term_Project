"""
Simulated Annealing (SA) Solver for TSP.
"""

import random
import math
from typing import List, Tuple
from ..models import City
from ..utils import calculate_tour_distance


class SASolver:
    """Simulated Annealing solver for TSP."""
    
    def __init__(self,
                 initial_temperature: float = 1000.0,
                 cooling_rate: float = 0.995,
                 min_temperature: float = 0.1,
                 max_iterations: int = 10000):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        
        self.best_tour = None
        self.best_distance = float('inf')
    
    def _generate_initial_solution(self, n: int) -> List[int]:
        """Generate initial random tour."""
        tour = list(range(n))
        random.shuffle(tour)
        return tour
    
    def _swap_two_cities(self, tour: List[int]) -> List[int]:
        """Swap two random cities in the tour."""
        new_tour = tour.copy()
        i, j = random.sample(range(len(tour)), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour
    
    def _reverse_segment(self, tour: List[int]) -> List[int]:
        """Reverse a random segment of the tour."""
        new_tour = tour.copy()
        n = len(tour)
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        new_tour[i:j+1] = reversed(new_tour[i:j+1])
        return new_tour
    
    def _get_neighbor(self, tour: List[int]) -> List[int]:
        """Generate a neighbor solution using random mutation."""
        if random.random() < 0.5:
            return self._swap_two_cities(tour)
        else:
            return self._reverse_segment(tour)
    
    def solve(self, cities: List[City], seed: int = None) -> Tuple[List[int], float, List[float]]:
        """
        Solve TSP using Simulated Annealing.
        
        Args:
            cities: List of cities to visit
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (best_tour, best_distance, history)
            history: List of best_distance values at each iteration
        """
        if seed is not None:
            random.seed(seed)
        
        n = len(cities)
        if n < 2:
            return [0], 0.0, [0.0]
        
        # Initialize
        current_tour = self._generate_initial_solution(n)
        current_distance = calculate_tour_distance(cities, current_tour)
        
        self.best_tour = current_tour.copy()
        self.best_distance = current_distance
        
        temperature = self.initial_temperature
        iteration = 0
        history = []  # Track best distance at each iteration
        
        while temperature > self.min_temperature and iteration < self.max_iterations:
            # Generate neighbor
            neighbor_tour = self._get_neighbor(current_tour)
            neighbor_distance = calculate_tour_distance(cities, neighbor_tour)
            
            # Calculate acceptance probability
            delta = neighbor_distance - current_distance
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_tour = neighbor_tour
                current_distance = neighbor_distance
                
                # Update best solution
                if current_distance < self.best_distance:
                    self.best_tour = current_tour.copy()
                    self.best_distance = current_distance
            
            # Record best distance for this iteration
            history.append(self.best_distance)
            
            # Cool down
            temperature *= self.cooling_rate
            iteration += 1
            
            if iteration % 1000 == 0:
                print(f"  Iteration {iteration}, Temperature: {temperature:.2f}, Best distance: {self.best_distance:.2f}")
        
        return self.best_tour, self.best_distance, history

