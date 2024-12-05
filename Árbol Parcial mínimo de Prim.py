import heapq

# Clase para representar un grafo
class Grafo:
    def __init__(self, num_nodos):
        # Inicializamos el grafo con el número de nodos y un diccionario vacío para las aristas
        self.num_nodos = num_nodos
        self.grafo = {i: [] for i in range(num_nodos)}

    def agregar_arista(self, nodo1, nodo2, peso):
        # Agregamos una arista entre dos nodos con un peso especificado
        self.grafo[nodo1].append((peso, nodo2))
        self.grafo[nodo2].append((peso, nodo1))  # Como es un grafo no dirigido, agregamos también la arista en el sentido contrario

    def prim(self):
        # Inicializamos una lista de visitados para marcar los nodos que ya han sido procesados
        visitados = [False] * self.num_nodos
        # Usamos un heap (min-heap) para seleccionar siempre la arista de menor peso
        min_heap = [(0, 0)]  # Comenzamos desde el nodo 0 con peso 0
        apm = []  # Aquí guardaremos las aristas seleccionadas para el Árbol Parcial Mínimo
        costo_total = 0  # Variable para acumular el costo total del APM

        print("Paso a paso del algoritmo de Prim:\n")
        
        while min_heap:
            # Extraemos el nodo con el menor peso
            peso, nodo = heapq.heappop(min_heap)
            
            # Si el nodo ya ha sido visitado, lo saltamos
            if visitados[nodo]:
                continue
            
            # Marcamos el nodo como visitado
            visitados[nodo] = True
            # Si el peso no es 0, significa que esta arista es parte del APM
            costo_total += peso

            # Si el peso no es 0, añadimos esta arista al APM
            if peso != 0:
                apm.append((peso, nodo))
                print(f"Se selecciona el nodo {nodo} con peso {peso}")

            # Ahora, consideramos las aristas del nodo actual
            for arista in self.grafo[nodo]:
                peso_arista, vecino = arista
                # Si el nodo vecino no ha sido visitado, lo agregamos al heap para procesarlo
                if not visitados[vecino]:
                    heapq.heappush(min_heap, (peso_arista, vecino))
                    print(f"Se considera la arista ({nodo}-{vecino}) con peso {peso_arista}")

        # Una vez que se ha construido el APM, mostramos las aristas seleccionadas
        print("\nÁrbol de Expansión Mínimo generado:")
        for peso, nodo in apm:
            print(f"Peso: {peso}, Nodo: {nodo}")
        
        # Finalmente, mostramos el costo total del APM
        print(f"\nCosto Total del Árbol de Expansión Mínimo: {costo_total}")


# Crear el grafo
num_nodos = 5  # Número de nodos en el grafo
grafo = Grafo(num_nodos)

# Agregar aristas (ejemplo)
grafo.agregar_arista(0, 1, 10)  # Nodo 0 se conecta con Nodo 1 con peso 10
grafo.agregar_arista(0, 2, 20)  # Nodo 0 se conecta con Nodo 2 con peso 20
grafo.agregar_arista(1, 2, 5)   # Nodo 1 se conecta con Nodo 2 con peso 5
grafo.agregar_arista(1, 3, 15)  # Nodo 1 se conecta con Nodo 3 con peso 15
grafo.agregar_arista(2, 3, 30)  # Nodo 2 se conecta con Nodo 3 con peso 30
grafo.agregar_arista(3, 4, 10)  # Nodo 3 se conecta con Nodo 4 con peso 10

# Ejecutar el algoritmo de Prim para encontrar el Árbol Parcial Mínimo
grafo.prim()
