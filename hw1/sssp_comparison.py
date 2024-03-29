from sssp import dijkstra
import numpy as np
import networkx as nx
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

# Наша реализация
start_time = time.time()
our_dijkstra_paths = dijkstra(graph, start_vertex)
our_time = time.time() - start_time

# NetworkX
start_time = time.time()
nx_paths = nx.single_source_dijkstra_path_length(G_nx, start_vertex)
nx_time = time.time() - start_time

# SciPy
start_time = time.time()
scipy_distances, _ = scipy_dijkstra(csgraph=sparse_graph, directed=True, indices=start_vertex, return_predecessors=True)
scipy_time = time.time() - start_time

print("Our dijkstra:", our_time)
print("SciPy dijkstra:", scipy_time)
print("NetworkX dijkstra:", nx_time)
print("Best time:", min(our_time, scipy_time, nx_time))
