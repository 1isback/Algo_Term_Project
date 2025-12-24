import numpy as np
import matplotlib.pyplot as plt
import math
import itertools

class TSPInstance:
    def __init__(self, num_cities, seed=42):
        self.num_cities = num_cities
        self.cities = [] 
        self.distance_matrix = [] 
        
        np.random.seed(seed)
        self._generate_instance()
        
    def _generate_instance(self):
        self.cities = np.random.rand(self.num_cities, 2) * 1000
        self.distance_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.distance_matrix[i][j] = np.linalg.norm(self.cities[i] - self.cities[j])
                else:
                    self.distance_matrix[i][j] = float('inf') 

    def calculate_tour_distance(self, tour):
        """Unified method to calculate total tour distance."""
        return sum(self.distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1))

    def plot_cities(self, title="TSP Cities"):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red')
        for i, (txt_x, txt_y) in enumerate(self.cities):
            plt.annotate(str(i), (txt_x, txt_y), xytext=(5, 5), textcoords='offset points')
        plt.title(f"{title} (n={self.num_cities})")
        plt.grid(True)
        plt.show()

def solve_exact_brute_force(instance):
    if instance.num_cities > 12:
        return None, float('inf')

    start_node = 0 
    other_nodes = list(range(1, instance.num_cities))
    best_distance = float('inf')
    best_tour = []

    for perm in itertools.permutations(other_nodes):
        current_tour = [start_node] + list(perm) + [start_node]
        current_dist = instance.calculate_tour_distance(current_tour)
            
        if current_dist < best_distance:
            best_distance = current_dist
            best_tour = current_tour
            
    return best_tour, best_distance

class ACOSolver:
    def __init__(self, instance, num_ants=20, max_iter=100, alpha=1.0, beta=3.0, rho=0.1, q=100.0):
        self.instance = instance
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.pheromone = np.ones((instance.num_cities, instance.num_cities)) * 0.1

    def solve(self):
        best_tour, best_dist = None, float('inf')
        
        for iteration in range(self.max_iter):
            all_tours, all_dists = [], []
            for _ in range(self.num_ants):
                tour = self._generate_tour()
                d = self.instance.calculate_tour_distance(tour)
                all_tours.append(tour)
                all_dists.append(d)
                
                if d < best_dist:
                    best_dist, best_tour = d, tour
            
            self._update_pheromones(all_tours, all_dists)
            if (iteration + 1) % 10 == 0:
                print(f"ACO Iter {iteration+1}/{self.max_iter}, Best: {best_dist:.2f}")
                
        return best_tour, best_dist

    def _generate_tour(self):
        n = self.instance.num_cities
        current_node = np.random.randint(n)
        start_node = current_node
        tour = [start_node]
        unvisited = set(range(n)) - {start_node}
        
        while unvisited:
            possible_nodes = list(unvisited)
            tau = self.pheromone[current_node][possible_nodes] ** self.alpha
            eta = (1.0 / self.instance.distance_matrix[current_node][possible_nodes]) ** self.beta
            probs = tau * eta
            
            probs_sum = probs.sum()
            probs = probs / probs_sum if probs_sum > 0 else np.ones(len(probs)) / len(probs)
            
            next_node = np.random.choice(possible_nodes, p=probs)
            tour.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
            
        tour.append(start_node)
        return tour

    def _update_pheromones(self, all_tours, all_dists):
        self.pheromone *= (1.0 - self.rho)
        for tour, dist in zip(all_tours, all_dists):
            deposit = self.q / dist 
            for i in range(len(tour)-1):
                u, v = tour[i], tour[i+1]
                self.pheromone[u][v] += deposit
                self.pheromone[v][u] += deposit

class SASolver:
    def __init__(self, instance, initial_temp=1000, cooling_rate=0.995, max_iter=10000):
        self.instance = instance
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter

    def solve(self):
        n = self.instance.num_cities
        current_tour = list(range(n))
        np.random.shuffle(current_tour)
        current_tour.append(current_tour[0]) 
        
        current_dist = self.instance.calculate_tour_distance(current_tour)
        best_tour, best_dist = list(current_tour), current_dist
        temp = self.initial_temp
        
        for _ in range(self.max_iter):
            new_tour = self._get_neighbor(current_tour)
            new_dist = self.instance.calculate_tour_distance(new_tour)
            
            delta = new_dist - current_dist
            if delta < 0 or (temp > 1e-9 and np.random.rand() < math.exp(-delta / temp)):
                current_tour, current_dist = new_tour, new_dist
                if new_dist < best_dist:
                    best_dist, best_tour = new_dist, list(new_tour)
            
            temp *= self.cooling_rate
        return best_tour, best_dist

    def _get_neighbor(self, tour):
        new_tour = list(tour)
        n = len(tour) - 1
        if n > 2:
            i = np.random.randint(1, n-1)
            j = np.random.randint(i+1, n)
            new_tour[i:j+1] = reversed(new_tour[i:j+1])
        return new_tour

if __name__ == "__main__":
    n_cities = 10
    inst = TSPInstance(n_cities)
    
    print(f"\n------------------------------------------")
    print(f"   TSP SOLVER TEST: {n_cities} Cities")
    print(f"------------------------------------------\n")
    
    # 1. Brute Force
    print("--- [1] Exact Solution (Brute Force) ---")
    _, target = solve_exact_brute_force(inst)
    print(f"Target Optimum Distance: {target:.2f}\n")

    # 2. ACO
    print("--- [2] Ant Colony Optimization (ACO) ---")
    aco_solver = ACOSolver(inst, max_iter=50)
    aco_tour, aco_d = aco_solver.solve()
    aco_error = ((aco_d - target) / target) * 100
    print(f"ACO Final Result: {aco_d:.2f} (Error: {aco_error:.2f}%)\n")

    # 3. SA
    print("--- [3] Simulated Annealing (SA) ---")
    sa_solver = SASolver(inst, max_iter=5000)
    sa_tour, sa_d = sa_solver.solve()
    sa_error = ((sa_d - target) / target) * 100
    print(f"SA Final Result : {sa_d:.2f} (Error: {sa_error:.2f}%)")
    print(f"\n------------------------------------------")
