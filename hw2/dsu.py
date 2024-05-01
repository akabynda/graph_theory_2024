import time
import random

import csv

def write_results_to_csv(results, filename='dsu_performance.csv'):
    headers = ['Scenario', 'Nodes', 'Operations', 'Weights', 'DSU Time', 'Recalculation Time']
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for result in results:
            writer.writerow(result)

class DSU:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)

def simulate_network_operations(dsu, num_nodes, num_operations, op_weights, recalculate):
    edges = {}
    operation_times_dsu = []
    operation_times_recalc = []

    for _ in range(num_operations):
        operation = random.choices(['add', 'fail', 'repair', 'check'], weights=op_weights, k=1)[0]
        a, b = random.sample(range(num_nodes), 2)

        start_time = time.time()
        if operation == 'add':
            if (a, b) not in edges:
                dsu.union(a, b)
                edges[(a, b)] = 'active'
        elif operation == 'fail':
            if (a, b) in edges:
                edges[(a, b)] = 'failed'
        elif operation == 'repair':
            if (a, b) in edges and edges[(a, b)] == 'failed':
                dsu.union(a, b)
                edges[(a, b)] = 'active'
        elif operation == 'check':
            dsu.connected(a, b)
        operation_times_dsu.append(time.time() - start_time)

        # Recalculation from scratch
        if recalculate:
            start_time_scratch = time.time()
            recalculate_components(edges, num_nodes)
            operation_times_recalc.append(time.time() - start_time_scratch)

    return sum(operation_times_dsu), sum(operation_times_recalc)

def recalculate_components(edges, num_nodes):
    adjacency_list = {i: set() for i in range(num_nodes)}
    for (a, b), status in edges.items():
        if status == 'active':
            adjacency_list[a].add(b)
            adjacency_list[b].add(a)
    components = []
    visited = set()

    def dfs(node):
        stack = [node]
        component = set()
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                component.add(n)
                stack.extend(adjacency_list[n] - visited)
        return component

    for node in range(num_nodes):
        if node not in visited:
            components.append(dfs(node))

def main():
    results = []
    nums_nodes = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    nums_operations = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
    for num_nodes in nums_nodes:
        for num_operations in nums_operations:
            dsu = DSU(num_nodes)

            scenarios = [
                (1, 1, 1, 1),  # Balanced scenario
                (5, 1, 1, 1),  # High frequency of additions
                (1, 5, 1, 1),  # High frequency of failures
                (1, 1, 5, 1),  # High frequency of repairs
                (1, 1, 1, 5)  # High frequency of connectivity checks
            ]

            for i, weights  in enumerate(scenarios, 1):
                dsu = DSU(num_nodes)
                time_dsu, time_recalc = simulate_network_operations(dsu, num_nodes, num_operations, weights, True)
                print(f"Scenario {i}: Nodes {num_nodes}, Ops {num_operations}, Weights {weights}")
                print(f"DSU Total time: {time_dsu:.4f} seconds")
                print(f"Recalculation Total time: {time_recalc:.4f} seconds\n")
                results.append([f"Scenario {i}", num_nodes, num_operations, weights, f"{time_dsu:.4f}", f"{time_recalc:.4f}"])
    write_results_to_csv(results)


if __name__ == "__main__":
    main()
