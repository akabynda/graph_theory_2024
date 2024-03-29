from sssp import dijkstra, bellman_ford
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import bellman_ford as scipy_bellman_ford
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
from scipy.sparse import csr_matrix
import time

# Настройки графа
n_vertices = 100  # Количество вершин
density = 0.2  # Плотность связей
weight_range = (1, 10)  # Диапазон весов рёбер

# Генерация случайного взвешенного графа
np.random.seed(42)
mask = np.random.rand(n_vertices, n_vertices) < density
weights = np.random.randint(*weight_range, size=(n_vertices, n_vertices))
graph = np.where(mask, weights, 0)

# Создание разреженной матрицы для SciPy
sparse_graph = csr_matrix(graph)

# Преобразование в формат NetworkX
G_nx = nx.from_numpy_array(graph, create_using=nx.DiGraph)

# Тестирование производительности
start_vertex = 0

# Наша dijkstra
start_time = time.time()
our_dijkstra_paths = dijkstra(graph, start_vertex)
our_dijkstra_time = time.time() - start_time

# Наш bellman-ford
start_time = time.time()
our_bellman_ford_paths = bellman_ford(graph, start_vertex)
our_bellman_ford_time = time.time() - start_time

# NetworkX dijkstra
start_time = time.time()
nx_paths = nx.single_source_dijkstra_path_length(G_nx, start_vertex)
nx_dijkstra_time = time.time() - start_time

# NetworkX bellman-ford
start_time = time.time()
nx_bf_distances = nx.single_source_bellman_ford_path_length(G_nx, start_vertex)
nx_bf_time = time.time() - start_time

# SciPy dijkstra
start_time = time.time()
scipy_distances, _ = scipy_dijkstra(csgraph=sparse_graph, directed=True, indices=start_vertex, return_predecessors=True)
scipy_dijkstra_time = time.time() - start_time

# SciPy bellman-ford
start_time = time.time()
scipy_bf_distances, _ = scipy_bellman_ford(csgraph=sparse_graph, directed=True, indices=start_vertex,
                                           return_predecessors=True)
scipy_bf_time = time.time() - start_time

print("Our dijkstra:", our_dijkstra_time)
print("Our bellman ford:", our_bellman_ford_time)
print("SciPy dijkstra:", scipy_dijkstra_time)
print("SciPy bellman ford:", scipy_bf_time)
print("NetworkX dijkstra:", nx_dijkstra_time)
print("NetworkX bellman ford:", nx_bf_time)
print("Best time:",
      min(our_dijkstra_time, our_bellman_ford_time, scipy_dijkstra_time, nx_dijkstra_time, nx_bf_time, scipy_bf_time))
