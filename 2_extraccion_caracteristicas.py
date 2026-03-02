import pandas as pd
import numpy as np
from scipy.signal import welch
import os

def calcular_bandas_potencia(dataset_path):
    df = pd.read_parquet(dataset_path)
    
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    cols_senal = [c for c in df.columns if c not in cols_meta]
    
    canales_set = set()
    for col in cols_senal:
        if '_' in col:
            nombre_canal = col.rsplit('_', 1)[0]
            canales_set.add(nombre_canal)

    lista_canales = sorted(list(canales_set))
    
    # Bandas de frecuencia (Hz)
    bandas = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45) 
    }
    
    fs = 128 
    
    df_features = df[cols_meta].copy()
    
    for canal in lista_canales:

        cols_este_canal = [f"{canal}_{i+1}" for i in range(128)]
        
        cols_validas = [c for c in cols_este_canal if c in df.columns]
        
        if len(cols_validas) < 128:
            continue
            
        datos_canal = df[cols_validas].values
        
        # Calcular PSD
        freqs, psd = welch(datos_canal, fs=fs, nperseg=128, axis=1)
        
        # Potencia Total (0.5-45Hz)
        idx_total = np.logical_and(freqs >= 0.5, freqs <= 45)
        psd_total = psd[:, idx_total].sum(axis=1) 
        
        for nombre_banda, (baja, alta) in bandas.items():
            idx_banda = np.logical_and(freqs >= baja, freqs <= alta)
            
            potencia_banda_abs = psd[:, idx_banda].sum(axis=1) 
            
            # Potencia Relativa
            potencia_relativa = potencia_banda_abs / (psd_total + 1e-10)
            
            col_feature = f"{canal}_{nombre_banda}"
            df_features[col_feature] = potencia_relativa
            
    is_baseline = df_features['Tarea'].str.startswith('Baseline_')
    
    df_activity = df_features[~is_baseline].copy()
    
    df_features = df_activity.copy()
    
    output_dir = 'Resultados'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_parquet = os.path.join(output_dir, 'datos_bandas.parquet')
    output_csv = os.path.join(output_dir, 'datos_bandas.csv')
    
    df_features.to_parquet(output_parquet, index=False)
    df_features.to_csv(output_csv, index=False)
    
    return df_features

if __name__ == "__main__":
    archivo_entrada = os.path.join('Resultados', 'datos_completo_epocas.parquet')
    if os.path.exists(archivo_entrada):
        calcular_bandas_potencia(archivo_entrada)
    else:
        pass
