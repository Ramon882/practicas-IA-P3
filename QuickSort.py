def quicksort(arr):
    if len(arr) <= 1:
        return arr  # Un arreglo de 0 o 1 elemento ya está ordenado
    pivot = arr[len(arr) // 2]  # Seleccionamos el pivote
    left = [x for x in arr if x < pivot]  # Elementos menores que el pivote
    middle = [x for x in arr if x == pivot]  # Elementos iguales al pivote
    right = [x for x in arr if x > pivot]  # Elementos mayores que el pivote
    return quicksort(left) + middle + quicksort(right)

# Ejemplo de uso
arr = [64, 34, 25, 12, 22]
print("Ordenado:", quicksort(arr))
