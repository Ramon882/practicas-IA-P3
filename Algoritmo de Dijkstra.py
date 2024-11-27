import networkx as nx  # Para crear y manejar grafos
import matplotlib.pyplot as plt  # Para la visualización gráfica
import heapq  # Para manejar la cola de prioridad en Dijkstra
import time  # Para pausar entre visualizaciones

def visualize_graph(graph, pos, shortest_path=None, visited_nodes=None):
    """
    Función para visualizar el grafo usando NetworkX.
    - Resalta nodos visitados y el camino más corto encontrado hasta ahora.
    """
    plt.figure(figsize=(8, 6))  # Tamaño de la figura
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10)

    # Dibujar los pesos de las aristas
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

    # Resaltar nodos visitados (en naranja)
    if visited_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=visited_nodes, node_color='orange')

    # Resaltar el camino más corto (en rojo)
    if shortest_path:
        path_edges = list(zip(shortest_path, shortest_path[1:]))  # Crear pares de nodos consecutivos
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2)

    plt.show()  # Mostrar la gráfica
    time.sleep(1)  # Pausa para observar cada paso

def dijkstra_with_visualization(graph, start):
    """
    Algoritmo de Dijkstra con visualización paso a paso.
    """
    # Crear un grafo de NetworkX
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)

    # Generar posiciones de los nodos para la visualización
    pos = nx.spring_layout(G)

    # Inicializar las distancias y la cola de prioridad
    distances = {node: float('inf') for node in graph}  # Todas las distancias son infinitas al inicio
    distances[start] = 0  # La distancia al nodo inicial es 0
    priority_queue = [(0, start)]  # Cola de prioridad (distancia, nodo)
    path = {}  # Para rastrear los caminos
    visited_nodes = []  # Lista de nodos visitados

    while priority_queue:
        # Obtener el nodo con la menor distancia acumulada
        current_distance, current_node = heapq.heappop(priority_queue)

        # Si el nodo ya fue visitado, lo omitimos
        if current_node in visited_nodes:
            continue

        # Marcar el nodo como visitado
        visited_nodes.append(current_node)
        print(f"Visitando nodo: {current_node}, distancia acumulada: {current_distance}")

        # Visualizar el grafo con los nodos visitados hasta el momento
        visualize_graph(G, pos, visited_nodes=visited_nodes)

        # Explorar los vecinos del nodo actual
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight  # Distancia acumulada al vecino

            # Si encontramos un camino más corto, lo actualizamos
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))  # Añadir a la cola de prioridad
                path[neighbor] = current_node  # Actualizar el camino
                print(f"Actualizando distancia de {neighbor} a {distance}")

    # Visualizar el camino más corto completo al final
    print("\nDistancias finales:", distances)
    shortest_path = reconstruct_path(path, start, visited_nodes[-1])  # Reconstruir el camino
    visualize_graph(G, pos, shortest_path=shortest_path)

    return distances, path

def reconstruct_path(path, start, end):
    """
    Reconstruir el camino más corto desde el nodo inicial hasta un nodo final.
    """
    current = end
    shortest_path = [current]

    while current in path:
        current = path[current]
        shortest_path.insert(0, current)

    if shortest_path[0] != start:
        print("No existe un camino válido.")
        return []
    return shortest_path

# Ejemplo de grafo
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

# Nodo inicial
start_node = 'A'

# Ejecutar el algoritmo de Dijkstra con visualización
distances, path = dijkstra_with_visualization(graph, start_node)
