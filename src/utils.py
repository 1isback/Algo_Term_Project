"""
Utility functions for distance calculation and visualization.
"""

import math
import matplotlib.pyplot as plt
from typing import List, Tuple
from .models import City


def euclidean_distance(city1: City, city2: City) -> float:
    """
    Calculate Euclidean distance between two cities.
    
    Args:
        city1: First city
        city2: Second city
    
    Returns:
        Euclidean distance
    """
    dx = city1.x - city2.x
    dy = city1.y - city2.y
    return math.sqrt(dx * dx + dy * dy)


def calculate_tour_distance(cities: List[City], tour: List[int]) -> float:
    """
    Calculate total distance of a tour.
    
    Args:
        cities: List of all cities
        tour: List of city indices representing the tour
    
    Returns:
        Total distance of the tour
    
    Raises:
        ValueError: If tour contains invalid indices or cities list is empty
    """
    if not cities:
        raise ValueError("Cities list cannot be empty")
    
    if not tour:
        return 0.0
    
    if len(tour) < 2:
        return 0.0
    
    # Validate indices
    for idx in tour:
        if idx < 0 or idx >= len(cities):
            raise ValueError(f"Invalid city index {idx}. Must be between 0 and {len(cities)-1}")
    
    total_distance = 0.0
    for i in range(len(tour)):
        current_idx = tour[i]
        next_idx = tour[(i + 1) % len(tour)]
        total_distance += euclidean_distance(cities[current_idx], cities[next_idx])
    
    return total_distance


def plot_tour(cities: List[City], 
              tour: List[int],
              title: str = "TSP Tour",
              save_path: str = None,
              show: bool = True):
    """
    Plot a TSP tour visualization.
    
    Args:
        cities: List of all cities
        tour: List of city indices representing the tour
        title: Plot title
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    
    Raises:
        ValueError: If cities list is empty or tour is invalid
    """
    if not cities:
        raise ValueError("Cities list cannot be empty")
    
    if not tour:
        raise ValueError("Tour cannot be empty")
    
    if len(tour) != len(cities):
        raise ValueError(f"Tour length ({len(tour)}) must match number of cities ({len(cities)})")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract coordinates
    x_coords = [cities[i].x for i in tour]
    y_coords = [cities[i].y for i in tour]
    
    # Close the tour (connect last to first)
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    
    # Plot the tour path
    ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.6, label='Tour Path')
    
    # Plot cities
    for i, city_idx in enumerate(tour):
        city = cities[city_idx]
        ax.plot(city.x, city.y, 'ro', markersize=8)
        ax.annotate(f"{i+1}", (city.x, city.y), fontsize=8, ha='center', va='center', color='white', weight='bold')
        ax.annotate(city.name, (city.x, city.y + 2), fontsize=7, ha='center', va='bottom')
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

