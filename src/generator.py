"""
Random city generator for creating TSP instances.
"""

import random
from typing import List, Tuple
from .models import City, Map


def generate_random_cities(num_cities: int, 
                          x_range: Tuple[float, float] = (0, 100),
                          y_range: Tuple[float, float] = (0, 100),
                          seed: int = None) -> List[City]:
    """
    Generate random cities with coordinates.
    
    Args:
        num_cities: Number of cities to generate
        x_range: Tuple of (min_x, max_x) for x coordinates
        y_range: Tuple of (min_y, max_y) for y coordinates
        seed: Random seed for reproducibility
    
    Returns:
        List of City objects
    """
    if seed is not None:
        random.seed(seed)
    
    cities = []
    for i in range(num_cities):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        city = City(f"City_{i+1}", x, y)
        cities.append(city)
    
    return cities


def generate_map(num_cities: int,
                x_range: Tuple[float, float] = (0, 100),
                y_range: Tuple[float, float] = (0, 100),
                seed: int = None,
                name: str = None) -> Map:
    """
    Generate a Map with random cities.
    
    Args:
        num_cities: Number of cities to generate
        x_range: Tuple of (min_x, max_x) for x coordinates
        y_range: Tuple of (min_y, max_y) for y coordinates
        seed: Random seed for reproducibility
        name: Name of the map (default: "Map_{num_cities}")
    
    Returns:
        Map object
    """
    cities = generate_random_cities(num_cities, x_range, y_range, seed)
    map_name = name or f"Map_{num_cities}"
    return Map(cities, map_name)

