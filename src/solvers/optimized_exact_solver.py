import json
import time as time_module
import numpy as np
from pulp import *

def load_instance(filepath):
    """JSON dosyasından instance'ı yükle"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def calculate_distance_matrix(cities):
    """Şehirler arası mesafe matrisini hesapla"""
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = np.sqrt(
                    (cities[i]['x'] - cities[j]['x'])**2 + 
                    (cities[i]['y'] - cities[j]['y'])**2
                )
    return dist

def solve_tsp_ilp(dist_matrix):
    """ILP ile TSP'yi exact olarak çöz - MTZ formülasyonu"""
    n = len(dist_matrix)
    
    print(f"   Model kurgulanıyor ({n}x{n} karar değişkeni)...")
    
    # Model oluştur
    prob = LpProblem("TSP", LpMinimize)
    
    # Karar değişkenleri: x[i][j] = 1 eğer i'den j'ye gidiliyorsa
    print(f"   [████████████████████] 100% - Karar değişkenleri oluşturuluyor...")
    x = [[LpVariable(f"x_{i}_{j}", cat='Binary') 
          for j in range(n)] for i in range(n)]
    
    # Pozisyon değişkenleri (subtour elimination için MTZ)
    u = [LpVariable(f"u_{i}", lowBound=0, upBound=n-1, cat='Continuous') 
         for i in range(n)]
    
    # Amaç fonksiyonu: Toplam mesafeyi minimize et
    prob += lpSum(dist_matrix[i][j] * x[i][j] 
                  for i in range(n) for j in range(n) if i != j)
    
    print("   Kısıtlar ekleniyor...")
    
    # Kısıt 1: Her şehirden tam olarak bir kez çıkılmalı
    total_constraints = n + n + (n-1)*(n-2)
    current = 0
    
    for i in range(n):
        prob += lpSum(x[i][j] for j in range(n) if i != j) == 1, f"out_{i}"
        current += 1
        if i % max(1, n//10) == 0:
            progress = (current / total_constraints) * 100
            bar_length = int(progress / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   [{bar}] {progress:.1f}% - Giden kenar kısıtları", end='\r')
    
    # Kısıt 2: Her şehre tam olarak bir kez girilmeli
    for j in range(n):
        prob += lpSum(x[i][j] for i in range(n) if i != j) == 1, f"in_{j}"
        current += 1
        if j % max(1, n//10) == 0:
            progress = (current / total_constraints) * 100
            bar_length = int(progress / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   [{bar}] {progress:.1f}% - Gelen kenar kısıtları", end='\r')
    
    # Kısıt 3: Subtour elimination (MTZ formülasyonu)
    print(f"\n   [{('░'*20)}] 0.0% - MTZ kısıtları ekleniyor...", end='\r')
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i][j] <= n - 1, f"mtz_{i}_{j}"
                current += 1
                if current % max(1, total_constraints//50) == 0:
                    progress = (current / total_constraints) * 100
                    bar_length = int(progress / 5)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"   [{bar}] {progress:.1f}% - MTZ kısıtları ekleniyor...", end='\r')
    
    print(f"\n   [████████████████████] 100% - Toplam {len(prob.constraints)} kısıt eklendi")
    print(f"   Çözücü başlatılıyor (zaman sınırı yok)...")
    
    # Çöz - CBC solver parametreleri optimize edildi, zaman sınırı kaldırıldı
    solver = PULP_CBC_CMD(
        msg=1,
        threads=None,  # Tüm CPU çekirdeklerini kullan
        options=[
            'cuts', 'on',
            'presolve', 'on',
            'heuristics', 'on',
            'strongBranching', '10'
        ]
    )
    
    prob.solve(solver)
    
    # Sonuçları al
    if LpStatus[prob.status] == 'Optimal' or LpStatus[prob.status] == 'Feasible':
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
        
        return tour, total_distance, LpStatus[prob.status], prob
    else:
        return None, None, LpStatus[prob.status], prob

def save_solution(instance_name, tour, distance, solve_time, filepath):
    """Çözümü JSON olarak kaydet"""
    solution = {
        'instance': instance_name,
        'optimal_tour': tour,
        'optimal_distance': distance,
        'solve_time_seconds': solve_time,
        'tour_sequence': ' -> '.join(map(str, tour)) + f' -> {tour[0]}'
    }
    
    with open(filepath, 'w') as f:
        json.dump(solution, f, indent=2)
    
    return solution

def main():
    """Ana fonksiyon"""
    
    print("="*70)
    print("TSP ILP EXACT SOLVER - Small Instance")
    print("="*70)
    
    # Small instance dosyasını yükle
    data_path = 'data/small_instances.json'
    
    print(f"\n1. Instance yükleniyor: {data_path}")
    try:
        instance_data = load_instance(data_path)
        instance_name = instance_data['name']
        cities = instance_data['cities']
        n_cities = len(cities)
        print(f"   ✓ {instance_name} yüklendi ({n_cities} şehir)")
    except Exception as e:
        print(f"   ✗ Hata: {e}")
        return
    
    print(f"\n{'='*70}")
    print(f"Instance: {instance_name} ({n_cities} şehir)")
    print(f"{'='*70}")
    
    # Mesafe matrisini hesapla
    print("\n2. Mesafe matrisi hesaplanıyor...")
    dist_matrix = calculate_distance_matrix(cities)
    print(f"   ✓ {n_cities}x{n_cities} mesafe matrisi hazır")
    
    # TSP'yi çöz
    print("\n3. ILP ile exact solution bulunuyor...")
    start_time = time_module.time()
    
    tour, total_distance, status, model = solve_tsp_ilp(dist_matrix)
    
    solve_time = time_module.time() - start_time
    
    # Sonuçları göster
    print(f"\n{'='*70}")
    print("SONUÇLAR")
    print(f"{'='*70}")
    print(f"Status: {status}")
    print(f"Çözüm Süresi: {solve_time:.2f} saniye ({solve_time/60:.2f} dakika)")
    
    if tour is not None:
        print(f"Optimal Toplam Mesafe: {total_distance:.6f}")
        print(f"\nOptimal Tour (şehir indeksleri):")
        print(" → ".join(map(str, tour)) + f" → {tour[0]}")
        
        print(f"\nOptimal Tour (şehir isimleri):")
        city_names = [cities[i]['name'] for i in tour]
        print(" → ".join(city_names) + f" → {city_names[0]}")
        
        # Çözümü kaydet
        solution_path = f'results/{instance_name}_exact_solution.json'
        solution = save_solution(instance_name, tour, total_distance, solve_time, solution_path)
        
        # Şehir isimlerini de ekle
        solution['tour_with_names'] = city_names + [city_names[0]]
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        
        print(f"\n✓ Çözüm kaydedildi: {solution_path}")
        
        # Özet istatistikler
        print(f"\n{'='*70}")
        print("ÖZET İSTATİSTİKLER")
        print(f"{'='*70}")
        print(f"Şehir Sayısı: {n_cities}")
        print(f"Optimal Mesafe: {total_distance:.6f}")
        print(f"Ortalama Şehir Arası Mesafe: {total_distance/n_cities:.6f}")
        print(f"Çözüm Süresi: {solve_time:.2f} saniye")
    else:
        print(f"✗ Çözüm bulunamadı! Status: {status}")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
    