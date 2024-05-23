import itertools
import random
import time
import math
import networkx as nx
from scipy.optimize import dual_annealing
from deap import base, creator, tools, algorithms
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import acopy
import matplotlib.pyplot as plt
import pandas as pd

# Generate random distances
def generate_distances(n):
    dists = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dists[i][j] = dists[j][i] = random.randint(1, 99)
    return dists

# Held-Karp algorithm (Dynamic Programming)
def held_karp(dists):
    n = len(dists)
    C = {}
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            for k in subset:
                prev = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)
    bits = (2 ** n - 1) - 1
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits
    path.append(0)
    return opt, list(reversed(path))

# Nearest Neighbor Heuristic
def nearest_neighbor(dists):
    n = len(dists)
    start = 0
    path = [start]
    total_weight = 0
    while len(path) < n:
        last = path[-1]
        next_node = min((dists[last][i], i) for i in range(n) if i not in path)[1]
        path.append(next_node)
        total_weight += dists[last][next_node]
    total_weight += dists[path[-1]][start]
    return total_weight

# Genetic Algorithm using DEAP
def genetic_algorithm(dists):
    n = len(dists)
    distance_matrix = np.array(dists)

    def eval_tsp(individual):
        distance = distance_matrix[individual[-1], individual[0]]
        for gene1, gene2 in zip(individual[0:-1], individual[1:]):
            distance += distance_matrix[gene1, gene2]
        return distance,

    if not hasattr(genetic_algorithm, "creator_setup"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        genetic_algorithm.creator_setup = True

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", eval_tsp)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 50, stats=stats, halloffame=hof, verbose=False)

    return hof[0].fitness.values[0]

# Simulated Annealing using scipy
def simulated_annealing_alg(dists):
    n = len(dists)
    distance_matrix = np.array(dists)

    def tsp_objective(permutation):
        idx = np.argsort(permutation)
        distance = distance_matrix[idx[-1], idx[0]]
        for i in range(len(idx) - 1):
            distance += distance_matrix[idx[i], idx[i + 1]]
        return distance

    bounds = [(0, n - 1)] * n

    result = dual_annealing(tsp_objective, bounds)
    return result.fun

# Branch and Bound Algorithm
maxsize = float('inf')

def copyToFinal(curr_path, final_path):
    final_path[:len(curr_path)] = curr_path[:]
    final_path[len(curr_path) - 1] = curr_path[0]

def firstMin(adj, i):
    min_cost = maxsize
    for k in range(len(adj)):
        if adj[i][k] < min_cost and i != k:
            min_cost = adj[i][k]
    return min_cost

def secondMin(adj, i):
    first, second = maxsize, maxsize
    for j in range(len(adj)):
        if i == j:
            continue
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]
        elif (adj[i][j] <= second and adj[i][j] != first):
            second = adj[i][j]
    return second

def TSPRec(adj, curr_bound, curr_weight, level, curr_path, visited, final_path, final_res):
    if level == len(adj):
        if adj[curr_path[level - 1]][curr_path[0]] != 0:
            curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]]
            if curr_res < final_res[0]:
                copyToFinal(curr_path, final_path)
                final_res[0] = curr_res
        return

    for i in range(len(adj)):
        if (adj[curr_path[level - 1]][i] != 0 and not visited[i]):
            temp = curr_bound
            curr_weight += adj[curr_path[level - 1]][i]

            if level == 1:
                curr_bound -= ((firstMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2)
            else:
                curr_bound -= ((secondMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2)

            if curr_bound + curr_weight < final_res[0]:
                curr_path[level] = i
                visited[i] = True
                TSPRec(adj, curr_bound, curr_weight, level + 1, curr_path, visited, final_path, final_res)

            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound = temp

            visited = [False] * len(visited)
            for j in range(level):
                if curr_path[j] != -1:
                    visited[curr_path[j]] = True

def branch_and_bound(adj):
    n = len(adj)
    curr_bound = 0
    curr_path = [-1] * (n + 1)
    visited = [False] * n
    final_path = [None] * (n + 1)
    final_res = [maxsize]

    for i in range(n):
        curr_bound += (firstMin(adj, i) + secondMin(adj, i))
    curr_bound = math.ceil(curr_bound / 2)

    visited[0] = True
    curr_path[0] = 0

    TSPRec(adj, curr_bound, 0, 1, curr_path, visited, final_path, final_res)
    return final_res[0]

# Google OR-Tools
def ortools_tsp(dists):
    n = len(dists)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dists[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        return solution.ObjectiveValue()
    else:
        return None

# Ant Colony Optimization using acopy
def ant_colony_optimization(dists):
    G = nx.Graph()
    n = len(dists)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dists[i][j])

    colony = acopy.Colony(alpha=1, beta=3)
    solver = acopy.Solver(rho=0.1, q=1)

    tour = solver.solve(G, colony, limit=100)
    return tour.cost

# Christofides' Algorithm
def christofides_tsp(dists):
    G = nx.Graph()
    n = len(dists)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dists[i][j])

    # 1. Minimum Spanning Tree (MST)
    T = nx.minimum_spanning_tree(G)

    # 2. Find odd degree vertices in the MST
    odd_degree_vertices = [v for v, degree in T.degree() if degree % 2 == 1]

    # 3. Minimum weight perfect matching in the subgraph induced by odd degree vertices
    M = nx.Graph()
    M.add_nodes_from(odd_degree_vertices)
    for u, v in itertools.combinations(odd_degree_vertices, 2):
        M.add_edge(u, v, weight=G[u][v]['weight'])
    matching = nx.algorithms.matching.min_weight_matching(M, maxcardinality=True)

    # 4. Combine the edges of MST and matching to form an Eulerian circuit
    multigraph = nx.MultiGraph(T)
    multigraph.add_edges_from(matching)
    eulerian_circuit = list(nx.eulerian_circuit(multigraph, source=0))

    # 5. Form the Hamiltonian circuit by skipping repeated vertices
    path = []
    visited = set()
    for u, v in eulerian_circuit:
        if u not in visited:
            path.append(u)
            visited.add(u)
        if v not in visited:
            path.append(v)
            visited.add(v)
    path.append(path[0])  # To make it a cycle

    # Calculate the total weight of the Hamiltonian circuit
    total_weight = sum(dists[path[i]][path[i + 1]] for i in range(n))

    return total_weight

def run_experiments():
    vertex_counts = [5, 6, 7, 8, 9, 10, 11, 12]
    num_trials = 10

    results = {
        "Dynamic Programming": [],
        "Nearest Neighbor": [],
        "Genetic Algorithm": [],
        "Simulated Annealing": [],
        "Branch and Bound": [],
        "Google OR-Tools": [],
        "Ant Colony Optimization": [],
        "Christofides' Algorithm": []
    }

    errors = {
        "Nearest Neighbor": [],
        "Genetic Algorithm": [],
        "Simulated Annealing": [],
        "Branch and Bound": [],
        "Google OR-Tools": [],
        "Ant Colony Optimization": [],
        "Christofides' Algorithm": []
    }

    std_devs = {
        "Nearest Neighbor": [],
        "Genetic Algorithm": [],
        "Simulated Annealing": [],
        "Branch and Bound": [],
        "Google OR-Tools": [],
        "Ant Colony Optimization": [],
        "Christofides' Algorithm": []
    }

    times = {
        "Dynamic Programming": [],
        "Nearest Neighbor": [],
        "Genetic Algorithm": [],
        "Simulated Annealing": [],
        "Branch and Bound": [],
        "Google OR-Tools": [],
        "Ant Colony Optimization": [],
        "Christofides' Algorithm": []
    }

    for n in vertex_counts:
        print(f"{n} vertices")
        dp_times = []
        nn_errors = []
        nn_times = []
        ga_errors = []
        ga_times = []
        sa_errors = []
        sa_times = []
        bb_errors = []
        bb_times = []
        ortools_errors = []
        ortools_times = []
        aco_errors = []
        aco_times = []
        christofides_errors = []
        christofides_times = []

        for j in range(num_trials):
            print(f"{j} trial")
            dists = generate_distances(n)

            start = time.time()
            dp_result, _ = held_karp(dists)
            dp_time = time.time() - start
            dp_times.append(dp_time)

            start = time.time()
            nn_result = nearest_neighbor(dists)
            nn_time = time.time() - start
            nn_error = abs(dp_result - nn_result) / dp_result
            nn_errors.append(nn_error)
            nn_times.append(nn_time)

            start = time.time()
            ga_result = genetic_algorithm(dists)
            ga_time = time.time() - start
            ga_error = abs(dp_result - ga_result) / dp_result
            ga_errors.append(ga_error)
            ga_times.append(ga_time)

            start = time.time()
            sa_result = simulated_annealing_alg(dists)
            sa_time = time.time() - start
            sa_error = abs(dp_result - sa_result) / dp_result
            sa_errors.append(sa_error)
            sa_times.append(sa_time)

            start = time.time()
            bb_result = branch_and_bound(dists)
            bb_time = time.time() - start
            bb_error = abs(dp_result - bb_result) / dp_result
            bb_errors.append(bb_error)
            bb_times.append(bb_time)

            start = time.time()
            ortools_result = ortools_tsp(dists)
            ortools_time = time.time() - start
            ortools_error = abs(dp_result - ortools_result) / dp_result
            ortools_errors.append(ortools_error)
            ortools_times.append(ortools_time)

            start = time.time()
            aco_result = ant_colony_optimization(dists)
            aco_time = time.time() - start
            aco_error = abs(dp_result - aco_result) / dp_result
            aco_errors.append(aco_error)
            aco_times.append(aco_time)

            start = time.time()
            christofides_result = christofides_tsp(dists)
            christofides_time = time.time() - start
            christofides_error = abs(dp_result - christofides_result) / dp_result
            christofides_errors.append(christofides_error)
            christofides_times.append(christofides_time)

        results["Dynamic Programming"].append(np.mean(dp_times))
        results["Nearest Neighbor"].append(np.mean(nn_errors))
        results["Genetic Algorithm"].append(np.mean(ga_errors))
        results["Simulated Annealing"].append(np.mean(sa_errors))
        results["Branch and Bound"].append(np.mean(bb_errors))
        results["Google OR-Tools"].append(np.mean(ortools_errors))
        results["Ant Colony Optimization"].append(np.mean(aco_errors))
        results["Christofides' Algorithm"].append(np.mean(christofides_errors))

        times["Dynamic Programming"].append(np.mean(dp_times))
        times["Nearest Neighbor"].append(np.mean(nn_times))
        times["Genetic Algorithm"].append(np.mean(ga_times))
        times["Simulated Annealing"].append(np.mean(sa_times))
        times["Branch and Bound"].append(np.mean(bb_times))
        times["Google OR-Tools"].append(np.mean(ortools_times))
        times["Ant Colony Optimization"].append(np.mean(aco_times))
        times["Christofides' Algorithm"].append(np.mean(christofides_times))

        std_devs["Nearest Neighbor"].append(np.std(nn_errors))
        std_devs["Genetic Algorithm"].append(np.std(ga_errors))
        std_devs["Simulated Annealing"].append(np.std(sa_errors))
        std_devs["Branch and Bound"].append(np.std(bb_errors))
        std_devs["Google OR-Tools"].append(np.std(ortools_errors))
        std_devs["Ant Colony Optimization"].append(np.std(aco_errors))
        std_devs["Christofides' Algorithm"].append(np.std(christofides_errors))

    return vertex_counts, results, times, std_devs

# Функция для построения таблицы
def build_table(vertex_counts, results, times, std_devs):
    data = {
        "Number of Vertices": vertex_counts,
        "Dynamic Programming (Time)": times["Dynamic Programming"],
        "Nearest Neighbor (Error)": results["Nearest Neighbor"],
        "Nearest Neighbor (Time)": times["Nearest Neighbor"],
        "Nearest Neighbor (Std Dev)": std_devs["Nearest Neighbor"],
        "Genetic Algorithm (Error)": results["Genetic Algorithm"],
        "Genetic Algorithm (Time)": times["Genetic Algorithm"],
        "Genetic Algorithm (Std Dev)": std_devs["Genetic Algorithm"],
        "Simulated Annealing (Error)": results["Simulated Annealing"],
        "Simulated Annealing (Time)": times["Simulated Annealing"],
        "Simulated Annealing (Std Dev)": std_devs["Simulated Annealing"],
        "Branch and Bound (Error)": results["Branch and Bound"],
        "Branch and Bound (Time)": times["Branch and Bound"],
        "Branch and Bound (Std Dev)": std_devs["Branch and Bound"],
        "Google OR-Tools (Error)": results["Google OR-Tools"],
        "Google OR-Tools (Time)": times["Google OR-Tools"],
        "Google OR-Tools (Std Dev)": std_devs["Google OR-Tools"],
        "Ant Colony Optimization (Error)": results["Ant Colony Optimization"],
        "Ant Colony Optimization (Time)": times["Ant Colony Optimization"],
        "Ant Colony Optimization (Std Dev)": std_devs["Ant Colony Optimization"],
        "Christofides' Algorithm (Error)": results["Christofides' Algorithm"],
        "Christofides' Algorithm (Time)": times["Christofides' Algorithm"],
        "Christofides' Algorithm (Std Dev)": std_devs["Christofides' Algorithm"]
    }
    df = pd.DataFrame(data)
    return df

# Функция для сохранения таблицы
def save_table(df, filename):
    df.to_csv(filename, index=False)

# Функция для построения графика
def plot_results(vertex_counts, results):
    plt.figure(figsize=(10, 8))
    for alg, errors in results.items():
        if alg != "Dynamic Programming" and alg != "Branch and Bound":
            plt.plot(vertex_counts, errors, label=f"{alg} (Error)")

    plt.xlabel('Number of Vertices')
    plt.ylabel('Average Relative Error')
    plt.title('Relative Error vs. Number of Vertices for TSP Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main
def main():
    vertex_counts, results, times, std_devs = run_experiments()
    df = build_table(vertex_counts, results, times, std_devs)
    print(df)
    save_table(df, "tsp_results.csv")
    plot_results(vertex_counts, results)

if __name__ == "__main__":
    main()
