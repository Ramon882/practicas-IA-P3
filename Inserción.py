def insertion_sort(arr):
    # Recorremos el arreglo desde el segundo elemento hasta el final
    for i in range(1, len(arr)):
        key = arr[i]  # Guardamos el valor actual como clave
        j = i - 1
        # Desplazamos los elementos mayores que la clave una posición adelante
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        # Insertamos la clave en su posición correcta
        arr[j + 1] = key

# Ejemplo de uso
arr = [64, 34, 25, 12, 22]
insertion_sort(arr)
print("Ordenado:", arr)
