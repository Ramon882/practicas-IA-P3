def straight_merge_sort(files):
    # Simula el manejo de archivos grandes dividiéndolos en bloques iniciales (runs)
    runs = [[num] for num in files]  # Cada elemento del arreglo se trata como un bloque independiente

    # Continuamos fusionando bloques hasta que quede solo uno
    while len(runs) > 1:
        temp_runs = []  # Lista temporal para guardar los bloques fusionados

        # Iteramos sobre los bloques en pares
        for i in range(0, len(runs), 2):
            if i + 1 < len(runs):
                # Fusionamos dos bloques consecutivos y los ordenamos
                temp_runs.append(sorted(runs[i] + runs[i + 1]))
            else:
                # Si hay un bloque sin pareja, lo dejamos intacto
                temp_runs.append(runs[i])

        runs = temp_runs  # Actualizamos los bloques para la siguiente iteración

    # Retornamos el bloque final ordenado, si existe
    return runs[0] if runs else []

# Ejemplo de uso
files = [64, 25, 12, 22, 11]  # Simulamos un archivo de datos
print("Ordenado:", straight_merge_sort(files))  # Salida ordenada
