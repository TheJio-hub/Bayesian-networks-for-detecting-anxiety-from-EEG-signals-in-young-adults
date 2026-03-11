import pandas as pd
import numpy as np
from scipy.signal import welch
import os

def normalizar_z_score_log_relajacion(df):
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    columnas_caracteristicas = [c for c in df.columns if c not in columnas_metadatos and np.issubdtype(df[c].dtype, np.number)]
    
    epsilon = 1e-25
    df[columnas_caracteristicas] = np.log10(df[columnas_caracteristicas] + epsilon)
    
    sujetos_unicos = df['Sujeto'].unique()
    
    for sujeto in sujetos_unicos:
        mascara_sujeto = df['Sujeto'] == sujeto
        datos_sujeto = df.loc[mascara_sujeto, columnas_caracteristicas]
        
        mascara_relajacion = (df['Sujeto'] == sujeto) & (df['Tarea'] == 'Relajacion')
        datos_relajacion = df.loc[mascara_relajacion, columnas_caracteristicas]
        
        if datos_relajacion.empty:
            continue
            
        mu_relajacion = datos_relajacion.mean()
        sigma_relajacion = datos_relajacion.std()
        sigma_relajacion = sigma_relajacion.replace(0, 1.0)
        
        df.loc[mascara_sujeto, columnas_caracteristicas] = (datos_sujeto - mu_relajacion) / sigma_relajacion

    return df

def calcular_bandas_potencia(ruta_dataset):
    df = pd.read_parquet(ruta_dataset)
    
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    columnas_todas = df.columns.tolist()
    columnas_senal = [c for c in columnas_todas if c not in columnas_metadatos]
    
    conjunto_canales = set()
    for col in columnas_senal:
        if '_' in col:
            nombre_canal = col.rsplit('_', 1)[0]
            conjunto_canales.add(nombre_canal)

    lista_canales = sorted(list(conjunto_canales))
    
    bandas_frecuencia = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45) 
    }
    
    frecuencia_muestreo = 128 
    duracion_epoca = 5 
    muestras_por_epoca = int(frecuencia_muestreo * duracion_epoca)
    
    nuevas_columnas = {}
    
    for canal in lista_canales:
        columnas_tiempo = [f"{canal}_{i+1}" for i in range(muestras_por_epoca)]
        cols_existentes = [c for c in columnas_tiempo if c in df.columns]
        
        if len(cols_existentes) < muestras_por_epoca:
            continue
            
        matriz_senal = df[cols_existentes].values
        frecuencias, psd = welch(matriz_senal, fs=frecuencia_muestreo, nperseg=128, axis=1)
        
        for nombre_banda, (f_min, f_max) in bandas_frecuencia.items():
            mask_banda = np.logical_and(frecuencias >= f_min, frecuencias <= f_max)
            potencia_absoluta = psd[:, mask_banda].sum(axis=1)
            
            nombre_columna = f"{canal}_{nombre_banda}"
            nuevas_columnas[nombre_columna] = potencia_absoluta
            
    df_bands = pd.DataFrame(nuevas_columnas)
    df_caracteristicas = pd.concat([df[columnas_metadatos].reset_index(drop=True), df_bands], axis=1)

    df_final = normalizar_z_score_log_relajacion(df_caracteristicas)
    
    directorio_salida = 'Resultados'
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)

    archivo_parquet = os.path.join(directorio_salida, 'datos_bandas_normalizados.parquet')
    archivo_csv = os.path.join(directorio_salida, 'datos_bandas_normalizados.csv')
    
    df_final.to_parquet(archivo_parquet, index=False)
    df_final.to_csv(archivo_csv, index=False)
    
    return df_final

if __name__ == "__main__":
    ruta_entrada = os.path.join('Resultados', 'datos_completo_epocas.parquet')
    if os.path.exists(ruta_entrada):
        calcular_bandas_potencia(ruta_entrada)
