import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def seleccion_mrmr(X, y, n_seleccion):
    """
    Implementación básica de mRMR (Mínima Redundancia Máxima Relevancia).
    Selecciona n_seleccion características maximizando (Relevancia - Redundancia).
    """
    n_caracteristicas = X.shape[1]
    nombres_caracteristicas = list(X.columns)
    
    indices_seleccionados = []
    indices_candidatos = list(range(n_caracteristicas))
    
    print(f"   - Calculando Relevancia inicial (MI) para {n_caracteristicas} características...")
    relevancia_inicial = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    
    # Inicializar redundancia acumulada
    redundancia_acumulada = np.zeros(n_caracteristicas)
    
    # --- Paso 1: Seleccionar la primera característica (Máxima Relevancia) ---
    primera_mejor = np.argmax(relevancia_inicial)
    indices_seleccionados.append(primera_mejor)
    indices_candidatos.remove(primera_mejor)
    
    print(f"   - 1/{n_seleccion}: {nombres_caracteristicas[primera_mejor]} (Puntaje: {relevancia_inicial[primera_mejor]:.4f})")
    
    # --- Paso 2: Seleccionar las siguientes características iterativamente ---
    for i in range(1, n_seleccion):
        idx_ultimo_seleccionado = indices_seleccionados[-1]
        datos_ultimo_seleccionado = X.iloc[:, idx_ultimo_seleccionado].values.reshape(-1, 1)
        
        # Actualizar Redundancia
        for idx_candidato in indices_candidatos:
            datos_candidato = X.iloc[:, idx_candidato].values.reshape(-1, 1)
            # MI entre dos variables continuas (Característica vs Característica)
            mi_redundancia = mutual_info_regression(datos_candidato, datos_ultimo_seleccionado.ravel(), discrete_features=False, random_state=42)[0]
            redundancia_acumulada[idx_candidato] += mi_redundancia
            
        # Calcular Puntaje mRMR = Relevancia - (Redundancia Promedio)
        puntajes_mrmr = -np.inf * np.ones(n_caracteristicas)
        
        for idx_candidato in indices_candidatos:
            promedio_redundancia = redundancia_acumulada[idx_candidato] / len(indices_seleccionados)
            puntajes_mrmr[idx_candidato] = relevancia_inicial[idx_candidato] - promedio_redundancia
            
        # Seleccionar el mejor
        idx_mejor_siguiente = np.argmax(puntajes_mrmr)
        indices_seleccionados.append(idx_mejor_siguiente)
        indices_candidatos.remove(idx_mejor_siguiente)
        
        # Imprimir progreso
        if (i+1) % 5 == 0 or (i+1) == n_seleccion:
             print(f"   - {i+1}/{n_seleccion}: {nombres_caracteristicas[idx_mejor_siguiente]}")
             
    # Retornar DataFrame ordenado
    nombres_seleccionados = [nombres_caracteristicas[i] for i in indices_seleccionados]
    relevancia_seleccionada = [relevancia_inicial[i] for i in indices_seleccionados]
    
    return pd.DataFrame({
        'Orden_Seleccion': range(1, n_seleccion + 1),
        'Caracteristica': nombres_seleccionados,
        'Relevancia_Original': relevancia_seleccionada
    })

def ejecutar_mrmr_por_bandas():
    # Rutas
    archivo_entrada = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    directorio_salida = os.path.join('Resultados', 'Ranking')
    
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        
    if not os.path.exists(archivo_entrada):
        archivo_entrada = os.path.join('Resultados', 'datos_bandas.parquet')
        if not os.path.exists(archivo_entrada):
            print("Error: No se encontró el archivo de datos.")
            return

    print(f"Cargando datos: {archivo_entrada}")
    df = pd.read_parquet(archivo_entrada)
    
    # Filtrar Clases: Relajación (0) vs Ansiedad (>=5)
    print("Filtrando: Relajación (0) vs Ansiedad (>=5)...")
    if 'Puntaje' in df.columns:
        df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
        y = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
    else:
        print("Advertencia: No se pudo filtrar por puntaje.")
        return

    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    todas_caracteristicas = [c for c in df.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df[c])]

    for banda in bandas:
        print(f"Banda: {banda}")
        
        cols_banda = [c for c in todas_caracteristicas if c.endswith(f"_{banda}")]
        
        if not cols_banda:
            continue
            
        n_total = len(cols_banda)
        n_seleccion = int(n_total * 0.25) # 25%
        if n_seleccion < 1: n_seleccion = 1
        
        print(f"Total características: {n_total}")
        print(f"Objetivo selección (25%): {n_seleccion} características")
        
        X_banda = df[cols_banda]
        
        # Ejecutar mRMR
        inicio = time.time()
        df_mrmr = seleccion_mrmr(X_banda, y, n_seleccion)
        tiempo_transcurrido = time.time() - inicio
        print(f"Tiempo de ejecución: {tiempo_transcurrido:.2f} segundos")
        
        # Guardar
        archivo_salida = os.path.join(directorio_salida, f'mRMR_{banda}.csv')
        df_mrmr.to_csv(archivo_salida, index=False)
        print(f"Guardado en: {archivo_salida}")


    print(f" Procesando todas las bandas")

    
    if not todas_caracteristicas:
        print("No se encontraron características para el análisis global.")
    else:
        n_total = len(todas_caracteristicas)
        n_seleccion = int(n_total * 0.25) 
        
        if n_seleccion < 1: n_seleccion = 1
        
        print(f"Total características globales: {n_total}")
        print(f"Objetivo selección (25%): {n_seleccion} características")
        
        X_todo = df[todas_caracteristicas]
        
        # Ejecutar mRMR Global
        inicio = time.time()
        df_mrmr_todo = seleccion_mrmr(X_todo, y, n_seleccion)
        tiempo_transcurrido = time.time() - inicio
        print(f"Tiempo de ejecución Global: {tiempo_transcurrido:.2f} segundos")
        
        # Guardar Global
        archivo_salida_todo = os.path.join(directorio_salida, 'mRMR_Global.csv')
        df_mrmr_todo.to_csv(archivo_salida_todo, index=False)
        print(f"Guardado en: {archivo_salida_todo}")

if __name__ == "__main__":
    ejecutar_mrmr_por_bandas()