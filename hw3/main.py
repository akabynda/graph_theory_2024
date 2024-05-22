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

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

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


# Function to run experiments and collect data
def run_experiments():
    vertex_counts = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
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

    for n in vertex_counts:
        print(f"{n} vertices")
        dp_times = []
        nn_times = []
        ga_times = []
        sa_times = []
        bb_times = []
        ortools_times = []
        aco_times = []
        christofides_times = []

        for j in range(num_trials):
            print(f"{j} trial")
            dists = generate_distances(n)

            start = time.time()
            dp_result, _ = held_karp(dists)
            end = time.time()
            dp_time = end - start
            dp_times.append(dp_time)

            start = time.time()
            nn_result = nearest_neighbor(dists)
            end = time.time()
            nn_time = end - start
            nn_times.append(nn_time)

            start = time.time()
            ga_result = genetic_algorithm(dists)
            end = time.time()
            ga_time = end - start
            ga_times.append(ga_time)

            start = time.time()
            sa_result = simulated_annealing_alg(dists)
            end = time.time()
            sa_time = end - start
            sa_times.append(sa_time)

            start = time.time()
            bb_result = branch_and_bound(dists)
            end = time.time()
            bb_time = end - start
            bb_times.append(bb_time)

            start = time.time()
            ortools_result = ortools_tsp(dists)
            end = time.time()
            ortools_time = end - start
            ortools_times.append(ortools_time)

            start = time.time()
            aco_result = ant_colony_optimization(dists)
            end = time.time()
            aco_time = end - start
            aco_times.append(aco_time)

            start = time.time()
            christofides_result = christofides_tsp(dists)
            end = time.time()
            christofides_time = end - start
            christofides_times.append(christofides_time)

            local_results = {
                "Dynamic Programming": (dp_result, dp_time),
                "Nearest Neighbor": (nn_result, nn_time),
                "Genetic Algorithm": (ga_result, ga_time),
                "Simulated Annealing": (sa_result, sa_time),
                "Branch and Bound": (bb_result, bb_time),
                "Google OR-Tools": (ortools_result, ortools_time),
                "Ant Colony Optimization": (aco_result, aco_time),
                "Christofides' Algorithm": (christofides_result, christofides_time)
            }

            for alg, (res, time_taken) in local_results.items():
                print(f"{alg}: Result = {res}, Time = {time_taken:.8f} seconds")

        results["Dynamic Programming"].append(np.mean(dp_times))
        results["Nearest Neighbor"].append(np.mean(nn_times))
        results["Genetic Algorithm"].append(np.mean(ga_times))
        results["Simulated Annealing"].append(np.mean(sa_times))
        results["Branch and Bound"].append(np.mean(bb_times))
        results["Google OR-Tools"].append(np.mean(ortools_times))
        results["Ant Colony Optimization"].append(np.mean(aco_times))
        results["Christofides' Algorithm"].append(np.mean(christofides_times))

    return vertex_counts, results


# Function to plot the results
def plot_results(vertex_counts, results):
    plt.figure(figsize=(10, 8))

    for alg, times in results.items():
        plt.plot(vertex_counts, times, label=alg)

    plt.xlabel('Number of Vertices')
    plt.ylabel('Average Running Time (seconds)')
    plt.title('Running Time vs. Number of Vertices for TSP Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()


# Main
def main():
    vertex_counts, results = run_experiments()
    plot_results(vertex_counts, results)


if __name__ == "__main__":
    main()
