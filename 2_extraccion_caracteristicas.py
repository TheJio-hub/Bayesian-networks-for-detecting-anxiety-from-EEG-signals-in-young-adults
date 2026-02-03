import pandas as pd
import numpy as np
from scipy.signal import welch
import os

def calcular_bandas_potencia(dataset_path):
    print(f"Cargando dataset desde {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    
    # Definir columnas de metadatos 
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    # El resto son datos de señal
    cols_senal = [c for c in df.columns if c not in cols_meta]
    
    canales_set = set()
    for col in cols_senal:
        if '_' in col:
            nombre_canal = col.rsplit('_', 1)[0]
            canales_set.add(nombre_canal)

    lista_canales = sorted(list(canales_set))
    print(f"Canales identificados ({len(lista_canales)}): {lista_canales}")
    
    # Definimos bandas de frecuencia (en Hz)
    bandas = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45) # Nyquist es 64Hz
    }
    
    fs = 128 # Frecuencia de muestreo
    
    # Creamos un DataFrame para los resultados de características
    df_features = df[cols_meta].copy()
    
    print("Calculando potencia espectral (PSD) por canal...")
    
    for canal in lista_canales:

        cols_este_canal = [f"{canal}_{i+1}" for i in range(128)]
        
        # Verificamos que existan
        cols_validas = [c for c in cols_este_canal if c in df.columns]
        
        if len(cols_validas) < 128:
            print(f"Advertencia: Canal {canal} tiene menos de 128 columnas. Saltando.")
            continue
            
        # Extraer matriz de datos para este canal: (N_epocas, 128)
        datos_canal = df[cols_validas].values
        
        # Calcular PSD usando Welch
        # nperseg=128 nos da resolución de 1Hz. window='hann' es default.
        freqs, psd = welch(datos_canal, fs=fs, nperseg=128, axis=1)
        
        # Calcular poder promedio para cada banda
        for nombre_banda, (baja, alta) in bandas.items():
            # Encontrar índices de frecuencias dentro de la banda
            idx_banda = np.logical_and(freqs >= baja, freqs <= alta)
            
            # Promedio de potencia en esos índices
            # psd shape es (N_epocas, N_freqs)
            potencia_banda = psd[:, idx_banda].mean(axis=1)
            
            # Agregar al dataframe de resultados
            col_feature = f"{canal}_{nombre_banda}"
            df_features[col_feature] = potencia_banda
            
    print(f"Extracción inicial completada. Dimensiones: {df_features.shape}")
    
    # CORRECCIÓN DE LÍNEA BASE (BASELINE CORRECTION)
    # Restar la potencia promedio del Baseline a la potencia de la Actividad
    print("Aplicando corrección de línea base (Resta de Potencia)...")
    
    # Identificar columnas de características (las que terminan en _Delta, _Theta, etc.)
    feature_cols = [c for c in df_features.columns if c not in cols_meta]
    
    # Separar Baselines y Actividad
    # Baselines tienen 'Baseline_' en la columna Tarea
    is_baseline = df_features['Tarea'].str.startswith('Baseline_')
    df_baselines = df_features[is_baseline].copy()
    df_activity = df_features[~is_baseline].copy()
    
    # Calcular promedio de baseline por (Sujeto, Trial)
    # Agrupamos por Sujeto y Trial (la Tarea de baseline es algo como 'Baseline_Aritmetica', 
    # pero corresponde al mismo Sujeto y Trial que 'Aritmetica')
    # Necesitamos una columna que vincule Baseline con Actividad. 'Trial' y 'Sujeto'.
    # 'Tarea' en baseline es 'Baseline_Aritmetica'. En actividad es 'Aritmetica'.
    # Trial 1 de Aritmetica NO es el mismo momento que Trial 1 de Espejo.
    # necesitamos mapear 'Baseline_X' a 'X'.
    
    df_baselines['Tarea_Real'] = df_baselines['Tarea'].str.replace('Baseline_', '')
    
    # Calcular vector promedio de características para cada (Sujeto, Tarea_Real, Trial)
    # Usamos feature_cols para promediar solo las numéricas
    baseline_means = df_baselines.groupby(['Sujeto', 'Tarea_Real', 'Trial'])[feature_cols].mean().reset_index()
    
    # Renombrar columnas de baseline para el merge (add suffix _base)
    baseline_means = baseline_means.rename(columns={c: c + '_base' for c in feature_cols})
    baseline_means = baseline_means.rename(columns={'Tarea_Real': 'Tarea'}) # Alinear clave para merge
    
    # Merge activity con sus baselines
    df_corrected = pd.merge(df_activity, baseline_means, on=['Sujeto', 'Tarea', 'Trial'], how='left')
    
    # Aplicar resta solo donde hubo match (Relajación no tiene baseline, quedará NaN en _base)
    # Tareas con Baseline: Aritmetica, Espejo, Stroop
    for col in feature_cols:
        col_base = col + '_base'
        # Si col_base es no-nulo, restar. Si es nulo (Relajacion), mantener original.
        # fillna(0) en la base para restar 0 a Relajacion
        df_corrected[col] = df_corrected[col] - df_corrected[col_base].fillna(0)
    
    # Limpiar columnas temporales
    cols_to_drop = [c + '_base' for c in feature_cols]
    df_corrected.drop(columns=cols_to_drop, inplace=True)
    
    df_features = df_corrected
    print(f"Corrección finalizada. Registros finales (sin baselines): {df_features.shape}")
    
    # Guardar
    output_dir = 'Resultados'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_parquet = os.path.join(output_dir, 'datos_bandas.parquet')
    output_csv = os.path.join(output_dir, 'datos_bandas.csv')
    
    print(f"Guardando en {output_parquet}...")
    df_features.to_parquet(output_parquet, index=False)
    
    print(f"Guardando en {output_csv}...")
    df_features.to_csv(output_csv, index=False)
    return df_features

if __name__ == "__main__":
    archivo_entrada = os.path.join('Resultados', 'datos_completo_epocas.parquet')
    if os.path.exists(archivo_entrada):
        df_final = calcular_bandas_potencia(archivo_entrada)

        print("\nVista previa de las características:")
        print(df_final.head())
    else:
        print(f"Error: No se encuentra el archivo {archivo_entrada}")
