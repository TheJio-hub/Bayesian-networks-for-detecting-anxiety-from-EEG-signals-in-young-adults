import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def columnas_ratio_por_banda(columnas, banda):
    seleccionadas = []
    for col in columnas:
        if not col.startswith('Ratio_'):
            continue
        partes = col.split('_')
        if len(partes) < 3:
            continue
        tipo_ratio = partes[1]
        if banda in tipo_ratio:
            seleccionadas.append(col)
    return seleccionadas

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
    archivo_entrada = os.path.join('Resultados', 'Exploratorio', 'datos_bandas_normalizados.parquet')
    directorio_salida = os.path.join('Resultados', 'Análisis por bandas', 'Ranking de características')
    
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        
    archivo_asimetria = os.path.join('Resultados', 'Exploratorio', 'datos_asimetria_normalizados.parquet')
    archivo_ratios = os.path.join('Resultados', 'Exploratorio', 'datos_ratios_normalizados.parquet')

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
                os.makedirs(os.path.join(directorio_salida, banda), exist_ok=True)
                
                print(f"Procesando mRMR banda: {banda}")
                df_mrmr = seleccion_mrmr(X_banda, y, n_seleccion)
                
                archivo_salida = os.path.join(directorio_salida, banda, f'mRMR_{banda}.csv')
                df_mrmr.to_csv(archivo_salida, index=False)

                # Asimetria por banda
                dir_asim = os.path.join(directorio_salida, banda, 'Asimetria')
                os.makedirs(dir_asim, exist_ok=True)
                if os.path.exists(archivo_asimetria):
                    df_asim = pd.read_parquet(archivo_asimetria)
                    df_asim = df_asim[(df_asim['Puntaje'] == 0) | (df_asim['Puntaje'] >= 5)].copy()
                    cols_asim = [c for c in df_asim.columns if c.startswith('Asym_') and c.endswith(f'_{banda}')]
                    if cols_asim:
                        X_asim = df_asim[cols_asim]
                        df_mrmr_asim = seleccion_mrmr(X_asim, y, len(cols_asim))
                        df_mrmr_asim.to_csv(os.path.join(dir_asim, f'mRMR_Asimetria_{banda}.csv'), index=False)
                    else:
                        pd.DataFrame(columns=['Orden_Seleccion', 'Caracteristica', 'Relevancia_Original']).to_csv(
                            os.path.join(dir_asim, f'mRMR_Asimetria_{banda}.csv'), index=False
                        )

                # Ratios por banda
                dir_ratios = os.path.join(directorio_salida, banda, 'Ratios')
                os.makedirs(dir_ratios, exist_ok=True)
                if os.path.exists(archivo_ratios):
                    df_rat = pd.read_parquet(archivo_ratios)
                    df_rat = df_rat[(df_rat['Puntaje'] == 0) | (df_rat['Puntaje'] >= 5)].copy()
                    cols_ratio = columnas_ratio_por_banda(df_rat.columns.tolist(), banda)
                    if cols_ratio:
                        X_ratio = df_rat[cols_ratio]
                        df_mrmr_ratio = seleccion_mrmr(X_ratio, y, len(cols_ratio))
                        df_mrmr_ratio.to_csv(os.path.join(dir_ratios, f'mRMR_Ratios_{banda}.csv'), index=False)
                    else:
                        pd.DataFrame(columns=['Orden_Seleccion', 'Caracteristica', 'Relevancia_Original']).to_csv(
                            os.path.join(dir_ratios, f'mRMR_Ratios_{banda}.csv'), index=False
                        )

    return None

if __name__ == "__main__":
    ejecutar_mrmr_por_bandas()