def selection_sort(arr):
    # Recorremos cada elemento del arreglo
    for i in range(len(arr)):
        min_idx = i  # Suponemos que el primer elemento es el menor
        # Buscamos el menor en el resto del arreglo
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        # Intercambiamos el menor encontrado con el primer elemento del segmento
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

# Ejemplo de uso
arr = [64, 25, 12, 22, 11]
selection_sort(arr)
print("Ordenado:", arr)
