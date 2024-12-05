def polyphase_sort(runs):
    # Simula el ordenamiento por fases usando particiones desbalanceadas
    while len(runs) > 1:
        temp_runs = []

        # Fusionamos todos los bloques en uno solo ordenado
        merged = []
        for run in runs:
            merged.extend(run)

        temp_runs.append(sorted(merged))  # Ordenamos la combinación completa
        runs = temp_runs  # Actualizamos los bloques para la siguiente fase

    return runs[0] if runs else []

# Ejemplo de uso
runs = [[34, 64], [12, 22], [11, 25]]  # Bloques iniciales
print("Ordenado:", polyphase_sort(runs))  # Salida ordenada
