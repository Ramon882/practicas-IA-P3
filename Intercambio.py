def bubble_sort(arr):
    n = len(arr)
    # Recorremos el arreglo varias veces
    for i in range(n):
        # Comparamos elementos adyacentes
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                # Intercambiamos si están en el orden incorrecto
                arr[j], arr[j+1] = arr[j+1], arr[j]

# Ejemplo de uso
arr = [64, 34, 25, 12, 22]
bubble_sort(arr)
print("Ordenado:", arr)
