import pandas as pd
import numpy as np
from scipy.signal import welch
import os

def calcular_bandas_potencia(dataset_path):
    print(f"Cargando dataset desde {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    
    # Definir columnas de metadatos (no son señales)
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    # El resto son datos de señal
    cols_senal = [c for c in df.columns if c not in cols_meta]
    
    # Extraer nombres de canales únicos
    # Asumimos formato "NombreCanal_NumeroMuestra"
    # Ejemplo: Cz_1, Cz_2...
    canales_set = set()
    for col in cols_senal:
        if '_' in col:
            nombre_canal = col.rsplit('_', 1)[0]
            canales_set.add(nombre_canal)
    
    # Ordenar canales para consistencia
    # (Opcional: podrías usar el orden del archivo .locs si es crítico, 
    # pero alfabético o por aparición está bien para features, siempre que sea consistente)
    lista_canales = sorted(list(canales_set))
    print(f"Canales identificados ({len(lista_canales)}): {lista_canales}")
    
    # Definir bandas de frecuencia (en Hz)
    bandas = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45) # Nyquist es 64Hz
    }
    
    fs = 128 # Frecuencia de muestreo
    
    # Crear un DataFrame para los resultados de características
    df_features = df[cols_meta].copy()
    
    print("Calculando potencia espectral (PSD) por canal...")
    
    for canal in lista_canales:
        # Filtrar columnas de este canal específico
        # Usamos filter con regex tal vez, o comprensión de lista
        # La lista ordenada de columnas para este canal:
        cols_este_canal = [f"{canal}_{i+1}" for i in range(128)]
        
        # Verificar que existan (por seguridad)
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
            
    print(f"Extracción completada. Dimensiones finales: {df_features.shape}")
    
    # Guardar
    output_parquet = 'datos_bandas.parquet'
    output_csv = 'datos_bandas.csv'
    
    print(f"Guardando en {output_parquet}...")
    df_features.to_parquet(output_parquet, index=False)
    
    print(f"Guardando en {output_csv}...")
    df_features.to_csv(output_csv, index=False)
    
    return df_features

if __name__ == "__main__":
    archivo_entrada = 'datos_completo_epocas.parquet'
    if os.path.exists(archivo_entrada):
        df_final = calcular_bandas_potencia(archivo_entrada)

        print("\nVista previa de las características:")
        print(df_final.head())
    else:
        print(f"Error: No se encuentra el archivo {archivo_entrada}")
