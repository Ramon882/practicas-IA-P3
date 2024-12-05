def natural_merge_sort(arr):
    # Dividimos el arreglo en subsecuencias ordenadas naturalmente
    runs = []  # Lista para almacenar subsecuencias ordenadas
    temp = [arr[0]]  # Inicializamos la primera subsecuencia

    # Recorremos el arreglo para identificar subsecuencias ordenadas
    for i in range(1, len(arr)):
        if arr[i] >= arr[i - 1]:
            temp.append(arr[i])  # Continuamos agregando a la subsecuencia actual
        else:
            runs.append(temp)  # Guardamos la subsecuencia en "runs"
            temp = [arr[i]]  # Iniciamos una nueva subsecuencia

    runs.append(temp)  # Agregamos la última subsecuencia

    # Combinamos las subsecuencias hasta obtener un arreglo completamente ordenado
    while len(runs) > 1:
        temp_runs = []

        # Fusionamos las subsecuencias de dos en dos
        for i in range(0, len(runs), 2):
            if i + 1 < len(runs):
                temp_runs.append(sorted(runs[i] + runs[i + 1]))
            else:
                temp_runs.append(runs[i])  # Si sobra una subsecuencia, la dejamos como está

        runs = temp_runs  # Actualizamos las subsecuencias

    return runs[0] if runs else []

# Ejemplo de uso
arr = [64, 34, 25, 12, 22]
print("Ordenado:", natural_merge_sort(arr))  # Salida ordenada
