"""
Unit tests for models.py
"""

import unittest
import os
import tempfile
from src.models import City, Map


class TestCity(unittest.TestCase):
    def test_city_creation(self):
        city = City("Test", 10.0, 20.0)
        self.assertEqual(city.name, "Test")
        self.assertEqual(city.x, 10.0)
        self.assertEqual(city.y, 20.0)
    
    def test_city_to_dict(self):
        city = City("Test", 10.0, 20.0)
        data = city.to_dict()
        self.assertEqual(data["name"], "Test")
        self.assertEqual(data["x"], 10.0)
        self.assertEqual(data["y"], 20.0)
    
    def test_city_from_dict(self):
        data = {"name": "Test", "x": 10.0, "y": 20.0}
        city = City.from_dict(data)
        self.assertEqual(city.name, "Test")
        self.assertEqual(city.x, 10.0)
        self.assertEqual(city.y, 20.0)
    
    def test_city_equality(self):
        city1 = City("Test", 10.0, 20.0)
        city2 = City("Test", 10.0, 20.0)
        city3 = City("Other", 10.0, 20.0)
        self.assertEqual(city1, city2)
        self.assertNotEqual(city1, city3)


class TestMap(unittest.TestCase):
    def test_map_creation(self):
        cities = [City(f"C{i}", i, i) for i in range(5)]
        map_obj = Map(cities, "TestMap")
        self.assertEqual(len(map_obj), 5)
        self.assertEqual(map_obj.name, "TestMap")
    
    def test_map_save_load(self):
        cities = [City(f"C{i}", i, i) for i in range(3)]
        map_obj = Map(cities, "TestMap")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            map_obj.save_to_json(temp_path)
            loaded_map = Map.load_from_json(temp_path)
            
            self.assertEqual(len(loaded_map), 3)
            self.assertEqual(loaded_map.name, "TestMap")
            self.assertEqual(loaded_map.cities[0].name, "C0")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_map_get_city(self):
        cities = [City(f"C{i}", i, i) for i in range(5)]
        map_obj = Map(cities, "TestMap")
        self.assertEqual(map_obj.get_city(0).name, "C0")
        self.assertEqual(map_obj.get_city(2).name, "C2")


if __name__ == '__main__':
    unittest.main()

