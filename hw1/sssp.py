import numpy as np


def dijkstra(graph, start):
    n_vertices = graph.shape[0]
    shortest_paths = np.full(n_vertices, np.inf)
    shortest_paths[start] = 0
    visited = np.zeros(n_vertices, dtype=bool)

    for _ in range(n_vertices):
        # Выбираем ближайший непосещённый узел
        min_distance = np.inf
        for v in range(n_vertices):
            if not visited[v] and shortest_paths[v] < min_distance:
                min_distance = shortest_paths[v]
                current = v
        visited[current] = True

        # Обновляем расстояния до соседей выбранного узла
        for v in range(n_vertices):
            if not visited[v] and graph[current, v] > 0:
                new_distance = shortest_paths[current] + graph[current, v]
                if new_distance < shortest_paths[v]:
                    shortest_paths[v] = new_distance

    return shortest_paths


def bellman_ford(graph, start):
    n_vertices = graph.shape[0]
    distance = np.full(n_vertices, np.inf)
    distance[start] = 0

    for _ in range(n_vertices - 1):
        for i in range(n_vertices):
            for j in range(n_vertices):
                if graph[i, j] and distance[i] + graph[i, j] < distance[j]:
                    distance[j] = distance[i] + graph[i, j]

    # Проверка на наличие циклов отрицательного веса
    for i in range(n_vertices):
        for j in range(n_vertices):
            if graph[i, j] and distance[i] + graph[i, j] < distance[j]:
                return None

    return distance
