"""
Unit tests for generator.py
"""

import unittest
from src.generator import generate_random_cities, generate_map
from src.models import City


class TestGenerator(unittest.TestCase):
    def test_generate_random_cities(self):
        cities = generate_random_cities(5, seed=42)
        self.assertEqual(len(cities), 5)
        self.assertIsInstance(cities[0], City)
        self.assertEqual(cities[0].name, "City_1")
    
    def test_generate_random_cities_reproducibility(self):
        cities1 = generate_random_cities(5, seed=42)
        cities2 = generate_random_cities(5, seed=42)
        self.assertEqual(len(cities1), len(cities2))
        # Coordinates should be the same with same seed
        for c1, c2 in zip(cities1, cities2):
            self.assertAlmostEqual(c1.x, c2.x, places=5)
            self.assertAlmostEqual(c1.y, c2.y, places=5)
    
    def test_generate_map(self):
        map_obj = generate_map(num_cities=5, seed=42)
        self.assertEqual(len(map_obj), 5)
        self.assertEqual(map_obj.name, "Map_5")
    
    def test_generate_map_custom_name(self):
        map_obj = generate_map(num_cities=5, seed=42, name="Custom")
        self.assertEqual(map_obj.name, "Custom")
    
    def test_generate_map_custom_range(self):
        map_obj = generate_map(num_cities=5, x_range=(0, 50), y_range=(0, 50), seed=42)
        for city in map_obj.cities:
            self.assertGreaterEqual(city.x, 0)
            self.assertLessEqual(city.x, 50)
            self.assertGreaterEqual(city.y, 0)
            self.assertLessEqual(city.y, 50)


if __name__ == '__main__':
    unittest.main()

