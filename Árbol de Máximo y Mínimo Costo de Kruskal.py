# Función para encontrar el representante de un conjunto (conjunto disjunto)
def find(parent, i):
    # Si el nodo i es su propio representante, lo retornamos
    if parent[i] == i:
        return i
    # Si no, seguimos buscando el representante de su conjunto
    return find(parent, parent[i])

# Función para realizar la unión de dos conjuntos
def union(parent, rank, x, y):
    # Encontramos los representantes de los conjuntos de x e y
    xroot = find(parent, x)
    yroot = find(parent, y)
    
    # Unimos los conjuntos según su rango (para mantener balance)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot  # El conjunto de yroot se convierte en el padre de xroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot  # El conjunto de xroot se convierte en el padre de yroot
    else:
        parent[yroot] = xroot  # En caso de igualdad, uno se convierte en padre del otro
        rank[xroot] += 1      # Aumentamos el rango del conjunto padre

# Función de Kruskal para encontrar el Árbol de Expansión Mínima
def kruskal_min(grafo, num_nodos):
    result = []  # Lista para almacenar el árbol de expansión mínima
    i, e = 0, 0  # i recorre las aristas ordenadas; e cuenta las aristas en el árbol
    grafo = sorted(grafo, key=lambda item: item[2])  # Ordenamos las aristas por peso ascendente
    parent = []  # Lista para almacenar el padre de cada nodo
    rank = []    # Lista para almacenar el rango de cada conjunto

    # Inicializamos cada nodo como su propio conjunto
    for node in range(num_nodos):
        parent.append(node)
        rank.append(0)

    # Recorremos las aristas hasta formar un árbol de expansión
    while e < num_nodos - 1:  # Continuamos hasta tener n-1 aristas
        u, v, w = grafo[i]    # Seleccionamos la arista con el menor peso
        i = i + 1             # Avanzamos al siguiente conjunto de aristas
        x = find(parent, u)   # Encontramos el representante del conjunto de u
        y = find(parent, v)   # Encontramos el representante del conjunto de v

        # Si no forman un ciclo, añadimos la arista al árbol de expansión
        if x != y:
            e = e + 1         # Incrementamos el contador de aristas en el árbol
            result.append([u, v, w])  # Agregamos la arista al resultado
            union(parent, rank, x, y) # Unimos los dos conjuntos

    # Imprimimos el resultado del Árbol de Expansión Mínima
    print("Árbol de Expansión Mínima (Kruskal):")
    total_cost = 0  # Inicializamos el costo total del árbol
    for u, v, weight in result:
        total_cost += weight  # Sumamos el peso de cada arista seleccionada
        print(f"Arista ({u}-{v}) con peso {weight}")  # Mostramos cada arista
    print(f"Costo Total: {total_cost}")  # Mostramos el costo total

# Función de Kruskal para encontrar el Árbol de Expansión Máxima
def kruskal_max(grafo, num_nodos):
    result = []  # Lista para almacenar el árbol de expansión máxima
    i, e = 0, 0  # i recorre las aristas ordenadas; e cuenta las aristas en el árbol
    grafo = sorted(grafo, key=lambda item: item[2], reverse=True)  # Ordenamos por peso descendente
    parent = []  # Lista para almacenar el padre de cada nodo
    rank = []    # Lista para almacenar el rango de cada conjunto

    # Inicializamos cada nodo como su propio conjunto
    for node in range(num_nodos):
        parent.append(node)
        rank.append(0)

    # Recorremos las aristas hasta formar un árbol de expansión
    while e < num_nodos - 1:  # Continuamos hasta tener n-1 aristas
        u, v, w = grafo[i]    # Seleccionamos la arista con el mayor peso
        i = i + 1             # Avanzamos al siguiente conjunto de aristas
        x = find(parent, u)   # Encontramos el representante del conjunto de u
        y = find(parent, v)   # Encontramos el representante del conjunto de v

        # Si no forman un ciclo, añadimos la arista al árbol de expansión
        if x != y:
            e = e + 1         # Incrementamos el contador de aristas en el árbol
            result.append([u, v, w])  # Agregamos la arista al resultado
            union(parent, rank, x, y) # Unimos los dos conjuntos

    # Imprimimos el resultado del Árbol de Expansión Máxima
    print("Árbol de Expansión Máxima (Kruskal):")
    total_cost = 0  # Inicializamos el costo total del árbol
    for u, v, weight in result:
        total_cost += weight  # Sumamos el peso de cada arista seleccionada
        print(f"Arista ({u}-{v}) con peso {weight}")  # Mostramos cada arista
    print(f"Costo Total: {total_cost}")  # Mostramos el costo total

# Grafo representado como una lista de aristas (u, v, peso)
grafo = [
    (0, 1, 10),  # Arista entre el nodo 0 y 1 con peso 10
    (0, 2, 20),  # Arista entre el nodo 0 y 2 con peso 20
    (1, 2, 5),   # Arista entre el nodo 1 y 2 con peso 5
    (1, 3, 15),  # Arista entre el nodo 1 y 3 con peso 15
    (2, 3, 30),  # Arista entre el nodo 2 y 3 con peso 30
    (3, 4, 10)   # Arista entre el nodo 3 y 4 con peso 10
]

num_nodos = 5  # Número de nodos en el grafo

# Ejecutamos el algoritmo de Kruskal para el AEM y AEM máximo
kruskal_min(grafo, num_nodos)
kruskal_max(grafo, num_nodos)
