"""
City and Map classes for the Neuro Courier Project.
"""

from typing import List, Tuple
import json


class City:
    """Represents a city with coordinates."""
    
    def __init__(self, name: str, x: float, y: float):
        self.name = name
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"City({self.name}, x={self.x}, y={self.y})"
    
    def __eq__(self, other):
        if not isinstance(other, City):
            return False
        return self.name == other.name and self.x == other.x and self.y == other.y
    
    def to_dict(self):
        """Convert city to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create City from dictionary."""
        return cls(data["name"], data["x"], data["y"])


class Map:
    """Represents a map with multiple cities."""
    
    def __init__(self, cities: List[City], name: str = "Map"):
        self.cities = cities
        self.name = name
    
    def __len__(self):
        return len(self.cities)
    
    def __repr__(self):
        return f"Map({self.name}, {len(self.cities)} cities)"
    
    def get_city(self, index: int) -> City:
        """Get city by index."""
        return self.cities[index]
    
    def save_to_json(self, filepath: str):
        """Save map to JSON file."""
        data = {
            "name": self.name,
            "cities": [city.to_dict() for city in self.cities]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str):
        """Load map from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cities = [City.from_dict(city_data) for city_data in data["cities"]]
        return cls(cities, data.get("name", "Map"))

