import networkx as nx
import matplotlib.pyplot as plt
import heapq
import time

def visualize_graph(graph, pos, shortest_path=None, visited_nodes=None):
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
    if visited_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=visited_nodes, node_color='orange')
    if shortest_path:
        path_edges = list(zip(shortest_path, shortest_path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.show()
    time.sleep(1)

def dijkstra_with_visualization(graph, start, end):
    G = nx.Graph()
    for place, connections in graph.items():
        for neighbor, distance in connections.items():
            G.add_edge(place, neighbor, weight=distance)
    pos = nx.spring_layout(G)
    
    distances = {place: float('inf') for place in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    path = {}
    visited_nodes = []
    
    while priority_queue:
        current_distance, current_place = heapq.heappop(priority_queue)
        if current_place in visited_nodes:
            continue
        visited_nodes.append(current_place)
        visualize_graph(G, pos, visited_nodes=visited_nodes)
        
        for neighbor, distance in graph[current_place].items():
            new_distance = current_distance + distance
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor))
                path[neighbor] = current_place
    
    shortest_path = reconstruct_path(path, start, end)
    visualize_graph(G, pos, shortest_path=shortest_path)
    print(f"Distancia más corta de {start} a {end}: {distances[end]} minutos")
    print(f"Ruta óptima: {' -> '.join(shortest_path)}")
    return distances, path

def reconstruct_path(path, start, end):
    current = end
    shortest_path = [current]
    while current in path:
        current = path[current]
        shortest_path.insert(0, current)
    return shortest_path if shortest_path[0] == start else []

# Ejemplo de rutas desde casa hasta la escuela con tiempos en minutos
school_route = {
    'Casa': {'Ruta 1': 55, 'Ruta 2': 58, 'Ruta 3': 60},
    'Ruta 1': {'Casa': 55, 'Escuela': 5},
    'Ruta 2': {'Casa': 58, 'Escuela': 3},
    'Ruta 3': {'Casa': 60, 'Escuela': 2},
    'Escuela': {'Ruta 1': 5, 'Ruta 2': 3, 'Ruta 3': 2}
}

# Definir origen y destino
start_place = 'Casa'
end_place = 'Escuela'

distances, path = dijkstra_with_visualization(school_route, start_place, end_place)
