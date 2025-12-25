"""
Unit tests for utils.py
"""

import unittest
from src.models import City
from src.utils import euclidean_distance, calculate_tour_distance


class TestUtils(unittest.TestCase):
    def test_euclidean_distance(self):
        c1 = City("A", 0, 0)
        c2 = City("B", 3, 4)
        dist = euclidean_distance(c1, c2)
        self.assertAlmostEqual(dist, 5.0, places=5)
    
    def test_euclidean_distance_same_point(self):
        c1 = City("A", 5, 5)
        c2 = City("B", 5, 5)
        dist = euclidean_distance(c1, c2)
        self.assertAlmostEqual(dist, 0.0, places=5)
    
    def test_tour_distance_square(self):
        # Create a square: (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
        cities = [
            City("A", 0, 0),
            City("B", 1, 0),
            City("C", 1, 1),
            City("D", 0, 1)
        ]
        tour = [0, 1, 2, 3]  # A -> B -> C -> D -> A
        dist = calculate_tour_distance(cities, tour)
        self.assertAlmostEqual(dist, 4.0, places=5)
    
    def test_tour_distance_two_cities(self):
        cities = [
            City("A", 0, 0),
            City("B", 3, 4)
        ]
        tour = [0, 1]
        dist = calculate_tour_distance(cities, tour)
        # Distance A->B + B->A = 5 + 5 = 10
        self.assertAlmostEqual(dist, 10.0, places=5)
    
    def test_tour_distance_empty(self):
        cities = [City("A", 0, 0)]
        tour = [0]
        dist = calculate_tour_distance(cities, tour)
        self.assertEqual(dist, 0.0)


if __name__ == '__main__':
    unittest.main()

