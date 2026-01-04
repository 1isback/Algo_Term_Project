import numpy as np
from pulp import *
import time
import matplotlib.pyplot as plt

def generate_cities(n_cities, seed=42):
    """Rastgele şehir koordinatları oluştur"""
    np.random.seed(seed)
    cities = np.random.rand(n_cities, 2) * 100
    return cities

def calculate_distance_matrix(cities):
    """Şehirler arası mesafe matrisini hesapla"""
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = np.sqrt(
                    (cities[i][0] - cities[j][0])**2 + 
                    (cities[i][1] - cities[j][1])**2
                )
    return dist

def solve_tsp_ilp(dist_matrix):
    """ILP ile TSP'yi exact olarak çöz"""
    n = len(dist_matrix)
    
    # Model oluştur
    prob = LpProblem("TSP", LpMinimize)
    
    # Karar değişkenleri: x[i][j] = 1 eğer i'den j'ye gidiliyorsa
    x = [[LpVariable(f"x_{i}_{j}", cat='Binary') 
          for j in range(n)] for i in range(n)]
    
    # Pozisyon değişkenleri (subtour elimination için)
    u = [LpVariable(f"u_{i}", lowBound=0, upBound=n-1, cat='Continuous') 
         for i in range(n)]
    
    # Amaç fonksiyonu: Toplam mesafeyi minimize et
    prob += lpSum(dist_matrix[i][j] * x[i][j] 
                  for i in range(n) for j in range(n) if i != j)
    
    # Kısıtlar
    # 1. Her şehirden tam olarak bir kez çıkılmalı
    for i in range(n):
        prob += lpSum(x[i][j] for j in range(n) if i != j) == 1
    
    # 2. Her şehre tam olarak bir kez girilmeli
    for j in range(n):
        prob += lpSum(x[i][j] for i in range(n) if i != j) == 1
    
    # 3. Subtour elimination (MTZ formülasyonu)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i][j] <= n - 1
    
    # Çöz
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=300))
    
    # Sonuçları al
    tour = []
    current = 0
    visited = {0}
    tour.append(current)
    
    while len(visited) < n:
        for j in range(n):
            if j != current and value(x[current][j]) > 0.5:
                tour.append(j)
                visited.add(j)
                current = j
                break
    
    total_distance = sum(dist_matrix[tour[i]][tour[i+1]] 
                        for i in range(len(tour)-1))
    total_distance += dist_matrix[tour[-1]][tour[0]]
    
    return tour, total_distance, LpStatus[prob.status]

def plot_tour(cities, tour, total_distance):
    """Tour'u görselleştir"""
    plt.figure(figsize=(10, 8))
    
    # Şehirleri çiz
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=200, zorder=3)
    
    # Şehir numaralarını ekle
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x, y), fontsize=12, ha='center', va='center',
                    color='white', weight='bold')
    
    # Tour'u çiz
    for i in range(len(tour)):
        start = tour[i]
        end = tour[(i + 1) % len(tour)]
        plt.plot([cities[start][0], cities[end][0]], 
                [cities[start][1], cities[end][1]], 
                'b-', linewidth=2, alpha=0.7)
        
        # Ok işaretleri ekle
        dx = cities[end][0] - cities[start][0]
        dy = cities[end][1] - cities[start][1]
        plt.arrow(cities[start][0], cities[start][1], 
                 dx*0.85, dy*0.85,
                 head_width=2, head_length=2, 
                 fc='blue', ec='blue', alpha=0.5)
    
    plt.title(f'TSP Optimal Tour (Toplam Mesafe: {total_distance:.2f})', 
             fontsize=14, weight='bold')
    plt.xlabel('X Koordinatı')
    plt.ylabel('Y Koordinatı')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """Ana fonksiyon"""
    n_cities = 20
    
    print("="*60)
    print(f"TSP ILP EXACT SOLVER - {n_cities} Şehir")
    print("="*60)
    
    # Şehirleri oluştur
    print("\n1. Şehirler oluşturuluyor...")
    cities = generate_cities(n_cities)
    print(f"   ✓ {n_cities} şehir oluşturuldu")
    
    # Mesafe matrisini hesapla
    print("\n2. Mesafe matrisi hesaplanıyor...")
    dist_matrix = calculate_distance_matrix(cities)
    print(f"   ✓ {n_cities}x{n_cities} mesafe matrisi hazır")
    
    # TSP'yi çöz
    print("\n3. ILP ile TSP çözülüyor...")
    print("   (Bu işlem birkaç dakika sürebilir...)")
    start_time = time.time()
    
    tour, total_distance, status = solve_tsp_ilp(dist_matrix)
    
    solve_time = time.time() - start_time
    
    # Sonuçları göster
    print(f"\n{'='*60}")
    print("SONUÇLAR")
    print(f"{'='*60}")
    print(f"Status: {status}")
    print(f"Çözüm Süresi: {solve_time:.2f} saniye")
    print(f"Optimal Toplam Mesafe: {total_distance:.4f}")
    print(f"\nOptimal Tour:")
    print(" → ".join(map(str, tour)) + f" → {tour[0]}")
    print(f"{'='*60}")
    
    # Tour'u görselleştir
    print("\n4. Tour görselleştiriliyor...")
    plot_tour(cities, tour, total_distance)

if __name__ == "__main__":
    main()