def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2  # Dividimos el arreglo en dos mitades
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)  # Ordenamos la primera mitad
        merge_sort(R)  # Ordenamos la segunda mitad

        # Combinamos ambas mitades
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

# Ejemplo de uso
arr = [64, 34, 25, 12, 22]
merge_sort(arr)
print("Ordenado:", arr)
