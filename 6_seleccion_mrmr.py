import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def seleccion_mrmr(X, y, n_seleccion):
    """
    Selecciona n_seleccion características maximizando (Relevancia - Redundancia).
    """
    n_caracteristicas = X.shape[1]
    nombres_caracteristicas = list(X.columns)
    
    indices_seleccionados = []
    indices_candidatos = list(range(n_caracteristicas))
    
    relevancia_inicial = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    
    redundancia_acumulada = np.zeros(n_caracteristicas)
    
    primera_mejor = np.argmax(relevancia_inicial)
    indices_seleccionados.append(primera_mejor)
    indices_candidatos.remove(primera_mejor)
    
    for i in range(1, n_seleccion):
        idx_ultimo_seleccionado = indices_seleccionados[-1]
        datos_ultimo_seleccionado = X.iloc[:, idx_ultimo_seleccionado].values.reshape(-1, 1)
        
        for idx_candidato in indices_candidatos:
            datos_candidato = X.iloc[:, idx_candidato].values.reshape(-1, 1)
            mi_redundancia = mutual_info_regression(datos_candidato, datos_ultimo_seleccionado.ravel(), discrete_features=False, random_state=42)[0]
            redundancia_acumulada[idx_candidato] += mi_redundancia
            
        puntajes_mrmr = -np.inf * np.ones(n_caracteristicas)
        
        for idx_candidato in indices_candidatos:
            promedio_redundancia = redundancia_acumulada[idx_candidato] / len(indices_seleccionados)
            puntajes_mrmr[idx_candidato] = relevancia_inicial[idx_candidato] - promedio_redundancia
            
        idx_mejor_siguiente = np.argmax(puntajes_mrmr)
        
        print(f"   - {i+1}/{n_seleccion}: {nombres_caracteristicas[idx_mejor_siguiente]}")
        
        indices_seleccionados.append(idx_mejor_siguiente)
        indices_candidatos.remove(idx_mejor_siguiente)
             
    nombres_seleccionados = [nombres_caracteristicas[i] for i in indices_seleccionados]
    relevancia_seleccionada = [relevancia_inicial[i] for i in indices_seleccionados]
    
    return pd.DataFrame({
        'Orden_Seleccion': range(1, n_seleccion + 1),
        'Caracteristica': nombres_seleccionados,
        'Relevancia_Original': relevancia_seleccionada
    })

def ejecutar_mrmr_por_bandas():
    # Análisis por Bandas 
    archivo_entrada = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    directorio_salida = os.path.join('Resultados', 'Ranking')
    
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        
    if os.path.exists(archivo_entrada):
        df = pd.read_parquet(archivo_entrada)
        
        if 'Puntaje' in df.columns:
            # Filtrar clases extremas
            df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
            y = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
            
            bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
            cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
            todas_caracteristicas = [c for c in df.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df[c])]

            for banda in bandas:
                cols_banda = [c for c in todas_caracteristicas if c.endswith(f"_{banda}")]
                
                if not cols_banda:
                    continue
                    
                n_total = len(cols_banda)
                n_seleccion = int(n_total * 0.25)
                if n_seleccion < 1: n_seleccion = 1
                
                X_banda = df[cols_banda]
                
                print(f"Procesando mRMR banda: {banda}")
                df_mrmr = seleccion_mrmr(X_banda, y, n_seleccion)
                
                archivo_salida = os.path.join(directorio_salida, f'mRMR_{banda}.csv')
                df_mrmr.to_csv(archivo_salida, index=False)

    # Análisis Global Completo 
    archivo_completo = os.path.join('Resultados', 'datos_completos_normalizados.parquet')
    if os.path.exists(archivo_completo):
        print("Iniciando mRMR Global Completo (Todas las características)...")
        df_completo = pd.read_parquet(archivo_completo)
        
        # Filtrar clases
        df_completo = df_completo[ (df_completo['Puntaje'] == 0) | (df_completo['Puntaje'] >= 5) ].copy()
        y_completo = df_completo['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
        
        cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
        feats_completo = [c for c in df_completo.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df_completo[c])]
        
        n_total = len(feats_completo)
        n_seleccion = int(n_total * 0.25) # 25% de todo el conjunto
        if n_seleccion < 5: n_seleccion = 5
        
        X_completo = df_completo[feats_completo]
        
        df_mrmr_completo = seleccion_mrmr(X_completo, y_completo, n_seleccion)
        
        archivo_salida_completo = os.path.join(directorio_salida, 'mRMR_Universal_Completo.csv')
        df_mrmr_completo.to_csv(archivo_salida_completo, index=False)
        print(f"Ranking Global Completo guardado en: {archivo_salida_completo}")

if __name__ == "__main__":
    ejecutar_mrmr_por_bandas()