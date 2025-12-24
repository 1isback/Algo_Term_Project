"""
Ant Colony Optimization (ACO) Solver for TSP.
"""

import random
import math
from typing import List, Tuple
from ..models import City
from ..utils import euclidean_distance, calculate_tour_distance


class ACOSolver:
    """Ant Colony Optimization solver for TSP."""
    
    def __init__(self, 
                 num_ants: int = 50,
                 alpha: float = 1.0,  # Pheromone importance
                 beta: float = 2.0,   # Heuristic importance
                 evaporation_rate: float = 0.5,
                 q: float = 100.0,   # Pheromone deposit constant
                 max_iterations: int = 100):
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.max_iterations = max_iterations
        
        self.pheromone_matrix = None
        self.distance_matrix = None
        self.best_tour = None
        self.best_distance = float('inf')
    
    def _initialize_matrices(self, cities: List[City]):
        """Initialize distance and pheromone matrices."""
        n = len(cities)
        self.distance_matrix = [[0.0] * n for _ in range(n)]
        self.pheromone_matrix = [[1.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = euclidean_distance(cities[i], cities[j])
                self.distance_matrix[i][j] = dist
                self.distance_matrix[j][i] = dist
    
    def _calculate_probability(self, current_city: int, unvisited: List[int]) -> List[float]:
        """Calculate probability of moving to each unvisited city."""
        probabilities = []
        total = 0.0
        
        for city in unvisited:
            pheromone = self.pheromone_matrix[current_city][city]
            heuristic = 1.0 / (self.distance_matrix[current_city][city] + 1e-10)
            value = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(value)
            total += value
        
        # Normalize
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / len(unvisited)] * len(unvisited)
        
        return probabilities
    
    def _select_next_city(self, current_city: int, unvisited: List[int]) -> int:
        """Select next city based on probability distribution."""
        probabilities = self._calculate_probability(current_city, unvisited)
        return random.choices(unvisited, weights=probabilities)[0]
    
    def _construct_tour(self, start_city: int, n: int) -> List[int]:
        """Construct a tour using ant behavior."""
        tour = [start_city]
        unvisited = list(range(n))
        unvisited.remove(start_city)
        current_city = start_city
        
        while unvisited:
            next_city = self._select_next_city(current_city, unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        return tour
    
    def _update_pheromones(self, tours: List[List[int]], distances: List[float], cities: List[City]):
        """Update pheromone matrix based on ant tours."""
        n = len(cities)
        
        # Evaporation
        for i in range(n):
            for j in range(n):
                self.pheromone_matrix[i][j] *= (1.0 - self.evaporation_rate)
        
        # Deposit pheromones
        for tour, distance in zip(tours, distances):
            if distance > 0:
                pheromone_deposit = self.q / distance
                for k in range(len(tour)):
                    i = tour[k]
                    j = tour[(k + 1) % len(tour)]
                    self.pheromone_matrix[i][j] += pheromone_deposit
                    self.pheromone_matrix[j][i] += pheromone_deposit
    
    def solve(self, cities: List[City], seed: int = None) -> Tuple[List[int], float]:
        """
        Solve TSP using Ant Colony Optimization.
        
        Args:
            cities: List of cities to visit
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (best_tour, best_distance)
        """
        if seed is not None:
            random.seed(seed)
        
        n = len(cities)
        if n < 2:
            return [0], 0.0
        
        self._initialize_matrices(cities)
        self.best_tour = None
        self.best_distance = float('inf')
        
        for iteration in range(self.max_iterations):
            tours = []
            distances = []
            
            # Each ant constructs a tour
            for ant in range(self.num_ants):
                start_city = random.randint(0, n - 1)
                tour = self._construct_tour(start_city, n)
                distance = calculate_tour_distance(cities, tour)
                
                tours.append(tour)
                distances.append(distance)
                
                # Update best solution
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_tour = tour.copy()
            
            # Update pheromones
            self._update_pheromones(tours, distances, cities)
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}/{self.max_iterations}, Best distance: {self.best_distance:.2f}")
        
        return self.best_tour, self.best_distance

