import time
import random
import jgrapht
from jgrapht.algorithms.shortestpaths import delta_stepping
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def delta_stepping_linear_algebra(A, source, delta):
    num_vertices = A.shape[0]
    inf = np.inf

    # Initialize distances and the active mask
    dist = np.full(num_vertices, inf)
    dist[source] = 0
    active = np.zeros(num_vertices, dtype=bool)
    active[source] = True

    # Create matrices for light and heavy edges
    light_edges = np.where(A <= delta, A, inf)
    heavy_edges = np.where(A > delta, A, inf)

    while np.any(active):
        # Get current active vertices
        active_indices = np.nonzero(active)[0]
        active[:] = False  # Reset active mask for the next iteration

        # Process all vertices that are currently active
        for u in active_indices:
            # Update distances using light edges
            relax_distances = dist[u] + light_edges[u]
            improvement = relax_distances < dist
            dist = np.where(improvement, relax_distances, dist)
            active |= improvement  # Add improved vertices to active set

            # Update distances using heavy edges
            relax_distances = dist[u] + heavy_edges[u]
            improvement = relax_distances < dist
            dist = np.where(improvement, relax_distances, dist)
            active |= improvement  # Add improved vertices to active set

    return dist


def create_random_graph(num_vertices, edge_probability, weight_range, visualize=False):
    g = nx.gnp_random_graph(num_vertices, edge_probability, directed=True)
    for (u, v) in g.edges():
        g.edges[u, v]['weight'] = np.random.randint(*weight_range)
    print(f"Graph with {num_vertices} vertices and edge probability {edge_probability}:")
    if visualize:
        for u, v, data in g.edges(data=True):
            print(f"{u} -> {v} with weight {data['weight']}")
    return g


def adapt_nx_to_jgrapht(nx_graph):
    g = jgrapht.create_graph(directed=True, weighted=True)
    for node in nx_graph.nodes:
        g.add_vertex(node)
    for u, v, data in nx_graph.edges(data=True):
        g.add_edge(u, v, weight=data['weight'])
    return g


def test_algorithms(nx_graph, source):
    num_vertices = nx_graph.number_of_nodes()
    adj_matrix = nx.to_numpy_array(nx_graph, weight='weight', nonedge=np.inf)

    times = {}

    # Our Implementation
    start_time = time.time()
    distances = delta_stepping_linear_algebra(adj_matrix, source, delta=5)
    times['Delta-Stepping (Custom)'] = time.time() - start_time

    # JGraphT Delta-Stepping
    jg_graph = adapt_nx_to_jgrapht(nx_graph)
    start_time = time.time()
    jg_result = delta_stepping(jg_graph, source_vertex=source, delta=5)
    times['Delta-Stepping (JGraphT)'] = time.time() - start_time
    jg_distances = {v: jg_result.get_path(v).weight if jg_result.get_path(v) is not None else np.inf for v in
                    jg_graph.vertices}

    # NetworkX Dijkstra
    start_time = time.time()
    nx_dijkstra_distances = nx.single_source_dijkstra_path_length(nx_graph, source)
    times['Dijkstra (NetworkX)'] = time.time() - start_time

    # NetworkX Bellman-Ford
    start_time = time.time()
    nx_bellman_distances = nx.single_source_bellman_ford_path_length(nx_graph, source)
    times['Bellman-Ford (NetworkX)'] = time.time() - start_time

    # print("Computed distances using custom method:", distances)
    # print("Distances from JGraphT:", jg_distances)
    # print("Distances from NetworkX Dijkstra:", nx_dijkstra_distances)
    # print("Distances from NetworkX Bellman-Ford:", nx_bellman_distances)

    discrepancies = []
    for node in nx_graph.nodes():
        distances_list = [
            distances[node],
            jg_distances.get(node, np.inf),
            nx_dijkstra_distances.get(node, np.inf),
            nx_bellman_distances.get(node, np.inf)
        ]
        if not np.allclose(distances_list, distances_list[0], atol=1e-6, rtol=0):
            discrepancies.append((node, distances_list))
    if discrepancies:
        print("Discrepancies found:")
        for node, dists in discrepancies:
            print(f"Node {node}: {dists}")
    return times


def main():
    num_vertices = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                    1000]  # Various graph sizes
    weight_range = (1, 10)
    results = {}

    for n in num_vertices:
        local_results = dict()
        m = 100
        for i in range(m):
            graph = create_random_graph(n, edge_probability=random.uniform(0.1, 0.5), weight_range=weight_range,
                                        visualize=False)
            local_result = test_algorithms(graph, source=0)
            for key, value in local_result.items():
                if key in local_results.keys():
                    local_results[key] += value
                else:
                    local_results[key] = value

        for key, value in local_results.items():
            local_results[key] = value / m
        results[n] = local_results

    # Plotting results
    plt.figure(figsize=(10, 6))
    for algo in results[num_vertices[0]].keys():
        plt.plot(num_vertices, [results[n][algo] for n in num_vertices], label=algo, marker='o')

    plt.title('Algorithm Performance Comparison')
    plt.xlabel('Number of vertices')
    plt.ylabel('Time taken (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig("result2.png")
    # plt.show()


if __name__ == '__main__':
    main()
