def distribution_of_initial_runs(arr):
    # Distribuimos el arreglo en corridas iniciales pequeñas
    runs = []
    run_size = 2  # Tamaño de las corridas iniciales

    for i in range(0, len(arr), run_size):
        runs.append(sorted(arr[i:i + run_size]))  # Generamos bloques ordenados

    return runs  # Retornamos las corridas iniciales

# Ejemplo de uso
arr = [64, 34, 25, 12, 22, 11]
print("Corridas iniciales:", distribution_of_initial_runs(arr))
