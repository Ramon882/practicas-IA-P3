def balanced_multiway_merge(runs):
    # Simula la combinación de múltiples bloques balanceados
    import heapq  # Usamos una cola de prioridad para mantener el balance

    # Inicializamos un heap (cola de prioridad) para almacenar los primeros elementos de cada bloque
    heap = []
    for i, run in enumerate(runs):
        if run:  # Si el bloque no está vacío
            heapq.heappush(heap, (run[0], i, 0))  # (valor, índice del bloque, índice dentro del bloque)

    result = []  # Lista para almacenar el resultado final

    # Procesamos el heap hasta vaciar todos los bloques
    while heap:
        value, run_idx, elem_idx = heapq.heappop(heap)  # Obtenemos el elemento más pequeño
        result.append(value)  # Lo agregamos al resultado

        # Si hay más elementos en el mismo bloque, los añadimos al heap
        if elem_idx + 1 < len(runs[run_idx]):
            heapq.heappush(heap, (runs[run_idx][elem_idx + 1], run_idx, elem_idx + 1))

    return result  # Retornamos el arreglo ordenado

# Ejemplo de uso
runs = [[11, 25], [12, 22], [34, 64]]  # Simulamos bloques ya ordenados
print("Ordenado:", balanced_multiway_merge(runs))  # Salida ordenada
