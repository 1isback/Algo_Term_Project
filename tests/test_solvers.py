"""
Unit tests for solvers
"""

import unittest
from src.models import City
from src.generator import generate_map
from src.solvers.exact_solver import ExactSolver
from src.solvers.aco_solver import ACOSolver
from src.solvers.sa_solver import SASolver


class TestExactSolver(unittest.TestCase):
    def test_exact_solver_small(self):
        cities = [City(f"C{i}", i, i) for i in range(5)]
        solver = ExactSolver()
        tour, dist, history = solver.solve(cities)
        self.assertIsNotNone(tour)
        self.assertGreater(dist, 0)
        self.assertEqual(len(tour), 5)
        self.assertEqual(len(history), 0)  # No iterations for brute force
    
    def test_exact_solver_too_large(self):
        cities = [City(f"C{i}", i, i) for i in range(22)]  # 22 > 21 limit
        solver = ExactSolver()
        tour, dist, history = solver.solve(cities)
        self.assertIsNone(tour)
        self.assertEqual(dist, float('inf'))
    
    def test_exact_solver_single_city(self):
        cities = [City("A", 0, 0)]
        solver = ExactSolver()
        tour, dist, history = solver.solve(cities)
        self.assertEqual(tour, [0])
        self.assertEqual(dist, 0.0)


class TestACOSolver(unittest.TestCase):
    def test_aco_solver_basic(self):
        cities = [City(f"C{i}", i, i) for i in range(5)]
        solver = ACOSolver(num_ants=5, max_iterations=2)
        tour, dist, history = solver.solve(cities, seed=42)
        self.assertIsNotNone(tour)
        self.assertGreater(dist, 0)
        self.assertEqual(len(tour), 5)
        self.assertEqual(len(history), 2)  # 2 iterations
    
    def test_aco_solver_single_city(self):
        cities = [City("A", 0, 0)]
        solver = ACOSolver()
        tour, dist, history = solver.solve(cities, seed=42)
        self.assertEqual(tour, [0])
        self.assertEqual(dist, 0.0)
    
    def test_aco_solver_elitist(self):
        # Test that elitist strategy is working
        cities = [City(f"C{i}", i, i) for i in range(5)]
        solver = ACOSolver(num_ants=5, max_iterations=5, elitist_weight=2.0)
        tour, dist, history = solver.solve(cities, seed=42)
        self.assertIsNotNone(tour)
        # History should show improvement
        if len(history) > 1:
            self.assertLessEqual(history[-1], history[0])


class TestSASolver(unittest.TestCase):
    def test_sa_solver_basic(self):
        cities = [City(f"C{i}", i, i) for i in range(5)]
        solver = SASolver(max_iterations=10)
        tour, dist, history = solver.solve(cities, seed=42)
        self.assertIsNotNone(tour)
        self.assertGreater(dist, 0)
        self.assertEqual(len(tour), 5)
        self.assertEqual(len(history), 10)  # 10 iterations
    
    def test_sa_solver_single_city(self):
        cities = [City("A", 0, 0)]
        solver = SASolver()
        tour, dist, history = solver.solve(cities, seed=42)
        self.assertEqual(tour, [0])
        self.assertEqual(dist, 0.0)
    
    def test_sa_solver_convergence(self):
        cities = [City(f"C{i}", i, i) for i in range(5)]
        solver = SASolver(max_iterations=20)
        tour, dist, history = solver.solve(cities, seed=42)
        self.assertIsNotNone(tour)
        # Best distance should be in history
        self.assertEqual(dist, min(history))


if __name__ == '__main__':
    unittest.main()

