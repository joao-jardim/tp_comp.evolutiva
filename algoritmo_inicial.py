import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import requests
from io import StringIO


# Função para carregar instância da CVRPLIB
def load_cvrp_instance(url):
    response = requests.get(url)
    data = response.text
    lines = data.split('\n')
    
    coords = []
    demands = []
    capacity = None
    reading_coords = False
    reading_demands = False
    
    for line in lines:
        if line.startswith("CAPACITY"):
            capacity = int(line.split()[-1])
        elif line.startswith("NODE_COORD_SECTION"):
            reading_coords = True
            continue
        elif line.startswith("DEMAND_SECTION"):
            reading_coords = False
            reading_demands = True
            continue
        elif line.startswith("DEPOT_SECTION"):
            break
        elif reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
        elif reading_demands:
            parts = line.split()
            if len(parts) >= 2:
                demands.append(int(parts[1]))
    
    return coords, demands, capacity

# Calcular distância euclidiana entre dois pontos
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def split_routes(individual, demands, capacity):
    routes = []
    current_route = []
    current_load = 0
    
    for customer in individual:
        demand = demands[customer + 1]  # +1 porque indivíduos são 0 a 30, mas demands é 0 a 31
        if current_load + demand <= capacity:
            current_route.append(customer + 1)  # Retorna índices 1 a 31 para consistência
            current_load += demand
        else:
            if current_route:
                routes.append(current_route)
            current_route = [customer + 1]
            current_load = demand
    
    if current_route:
        routes.append(current_route)
    
    return routes

# Função de avaliação (fitness)
def evaluate(individual, coords, demands, capacity):
    routes = split_routes(individual, demands, capacity)
    total_distance = 0
    
    for route in routes:
        if not route:
            continue
        total_distance += euclidean_distance(coords[0], coords[route[0]])  # Depósito ao primeiro
        for i in range(len(route) - 1):
            total_distance += euclidean_distance(coords[route[i]], coords[route[i + 1]])
        total_distance += euclidean_distance(coords[route[-1]], coords[0])  # Último ao depósito
    
    return (total_distance,)  # DEAP exige tupla

def repair_individual(ind, n_customers):
    valid_indices = set(range(0, n_customers))  # 0 a 30
    current_indices = set(ind)
    ind[:] = [x for x in ind if 0 <= x < n_customers]
    missing = list(valid_indices - current_indices)
    random.shuffle(missing)
    ind.extend(missing[:n_customers - len(ind)])
    ind[:] = ind[:n_customers]
    return ind

def plot_routes(coords, routes):
    plt.figure(figsize=(10, 6))
    plt.scatter([coords[0][0]], [coords[0][1]], c='red', label='Depósito', s=100)
    for i in range(1, len(coords)):
        plt.scatter([coords[i][0]], [coords[i][1]], c='blue', s=50)
        plt.text(coords[i][0], coords[i][1], str(i), fontsize=8, ha='right')  # Adiciona o índice
    colors = ['green', 'blue', 'cyan', 'purple', 'yellow']
    for i, route in enumerate(routes):
        full_route = [0] + route + [0]
        x = [coords[c][0] for c in full_route]
        y = [coords[c][1] for c in full_route]
        plt.plot(x, y, color=colors[i % len(colors)], label=f'Rota {i+1}', linewidth=2)
    plt.title("Rotas Otimizadas para o CVRP")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.legend()
    plt.grid(True)
    plt.show()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

url = "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n32-k5.vrp"
coords, demands, capacity = load_cvrp_instance(url)

# Verificar consistência dos dados
print(f"Len coords: {len(coords)}, Len demands: {len(demands)}, Capacity: {capacity}")
n_customers = len(coords) - 1  # Deve ser 31 para A-n32-k5
print(f"n_customers: {n_customers}")
# Ajustar a geração de índices para base 0
def generate_indices():
    indices = random.sample(range(0, n_customers), n_customers)  # 0 a 30
    if len(indices) != n_customers or max(indices) >= n_customers or min(indices) < 0:
        raise ValueError(f"Índices inválidos gerados: {indices}")
    return indices

toolbox.register("indices", generate_indices)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate, coords=coords, demands=demands, capacity=capacity)

def repair_individual(ind, n_customers):
    valid_indices = set(range(1, n_customers + 1))
    current_indices = set(ind)
    ind[:] = [x for x in ind if 1 <= x <= n_customers]
    missing = list(valid_indices - current_indices)
    random.shuffle(missing)
    ind.extend(missing[:n_customers - len(ind)])
    ind[:] = ind[:n_customers]
    return ind

def main(toolbox):
    random.seed(42)
    pop = toolbox.population(n=100)
    
    print("Verificando população inicial:")
    for i, ind in enumerate(pop):
        if len(ind) != n_customers or max(ind) >= n_customers or min(ind) < 0:
            print(f"Indivíduo inicial {i} inválido: {ind}")
            ind[:] = repair_individual(ind, n_customers)
            print(f"Indivíduo {i} corrigido: {ind}")
        #else:
            #print(f"Indivíduo {i} OK: {ind[:5]}... (tamanho: {len(ind)})")
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    # Listas para armazenar as estatísticas por geração
    min_fitnesses = []
    avg_fitnesses = []
    
    for gen in range(50):
        print(f"\nGeração {gen}")
        pop = toolbox.select(pop, len(pop))
        
        offspring = []
        for i in range(0, len(pop), 2):
            if i + 1 < len(pop) and random.random() < 0.7:
                ind1, ind2 = toolbox.clone(pop[i]), toolbox.clone(pop[i + 1])
                #print(f"Antes crossover - ind1: {ind1[:5]}..., ind2: {ind2[:5]}...")
                toolbox.mate(ind1, ind2)
                if len(ind1) != n_customers or max(ind1) >= n_customers or min(ind1) < 0:
                    print(f"Erro após crossover em ind1 (antes da correção): {ind1}")
                    ind1[:] = repair_individual(ind1, n_customers)
                    print(f"Ind1 corrigido: {ind1}")
                if len(ind2) != n_customers or max(ind2) >= n_customers or min(ind2) < 0:
                    print(f"Erro após crossover em ind2 (antes da correção): {ind2}")
                    ind2[:] = repair_individual(ind2, n_customers)
                    print(f"Ind2 corrigido: {ind2}")
                del ind1.fitness.values, ind2.fitness.values
                offspring.extend([ind1, ind2])
            else:
                offspring.append(toolbox.clone(pop[i]))
                if i + 1 < len(pop):
                    offspring.append(toolbox.clone(pop[i + 1]))
        
        for ind in offspring:
            if random.random() < 0.2:
                #print(f"Antes mutação: {ind[:5]}...")
                toolbox.mutate(ind)
                if len(ind) != n_customers or max(ind) >= n_customers or min(ind) < 0:
                    print(f"Erro após mutação (antes da correção): {ind}")
                    ind[:] = repair_individual(ind, n_customers)
                    print(f"Indivíduo corrigido: {ind}")
                del ind.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop = offspring
        hof.update(pop)
        
        # Calcular e armazenar estatísticas
        stats_data = stats.compile(pop)
        min_fitnesses.append(stats_data["min"])
        avg_fitnesses.append(stats_data["avg"])
        print(f"Estatísticas - Avg: {stats_data['avg']:.2f}, Min: {stats_data['min']:.2f}")
    
    best_individual = hof[0]
    best_routes = split_routes(best_individual, demands, capacity)
    print(f"Melhor distância: {hof[0].fitness.values[0]}")
    print(f"Rotas: {best_routes}")
    
    plot_routes(coords, best_routes)
    
    gen = range(50)
    plt.figure(figsize=(10, 6))
    plt.plot(gen, min_fitnesses, label="Distância Mínima")
    plt.plot(gen, avg_fitnesses, label="Distância Média")
    plt.title("Convergência do Algoritmo Genético")
    plt.xlabel("Geração")
    plt.ylabel("Distância")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return pop, None, hof

if __name__ == "__main__":
    pop, log, hof = main(toolbox)
    