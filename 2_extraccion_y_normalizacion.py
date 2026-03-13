import pandas as pd
import numpy as np
from scipy.signal import welch
import os

def aplicar_log10(df):
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    columnas_caracteristicas = [c for c in df.columns if c not in columnas_metadatos and np.issubdtype(df[c].dtype, np.number)]
    
    df_log = df.copy()
    epsilon = 1e-25
    df_log[columnas_caracteristicas] = np.log10(df_log[columnas_caracteristicas] + epsilon)
    return df_log

def normalizar_z_score_relajacion(df):
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    columnas_caracteristicas = [c for c in df.columns if c not in columnas_metadatos and np.issubdtype(df[c].dtype, np.number)]
    
    df_norm = df.copy()
    sujetos_unicos = df_norm['Sujeto'].unique()
    
    for sujeto in sujetos_unicos:
        mascara_sujeto = df_norm['Sujeto'] == sujeto
        datos_sujeto = df_norm.loc[mascara_sujeto, columnas_caracteristicas]
        
        mascara_relajacion = (df_norm['Sujeto'] == sujeto) & (df_norm['Tarea'] == 'Relajacion')
        datos_relajacion = df_norm.loc[mascara_relajacion, columnas_caracteristicas]
        
        if datos_relajacion.empty:
            continue
            
        mu_relajacion = datos_relajacion.mean()
        sigma_relajacion = datos_relajacion.std()
        sigma_relajacion = sigma_relajacion.replace(0, 1.0)
        
        df_norm.loc[mascara_sujeto, columnas_caracteristicas] = (datos_sujeto - mu_relajacion) / sigma_relajacion

    return df_norm

def calcular_asimetria(df_log):
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    
    pares = [
        ('Fp1', 'Fp2'),
        ('F3', 'F4'),
        ('F7', 'F8'),
        ('FC5', 'FC6'),
        ('T7', 'T8'), 
        ('C3', 'C4'),
        ('P3', 'P4'),
        ('P7', 'P8'), 
        ('O1', 'O2'),
        ('PO9', 'PO10')
    ]
    
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    features_asimetria = {}
    
    cols = df_log.columns
    
    for (izq, der) in pares:
        for banda in bandas:
            col_izq = f"{izq}_{banda}"
            col_der = f"{der}_{banda}"
            
            if col_izq in cols and col_der in cols:
                nombre_feat = f"Asym_{der}_{izq}_{banda}" 
                features_asimetria[nombre_feat] = df_log[col_der] - df_log[col_izq]
                
    df_asym = pd.concat([df_log[columnas_metadatos], pd.DataFrame(features_asimetria)], axis=1)
    return df_asym

def calcular_ratios(df_log):
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    
    canales = set([c.split('_')[0] for c in df_log.columns if '_' in c])
    canales = sorted(list(canales))
    
    features_ratios = {}
    
    for canal in canales:
        theta = f"{canal}_Theta"
        beta = f"{canal}_Beta"
        alpha = f"{canal}_Alpha"
        delta = f"{canal}_Delta"
        
        if theta in df_log.columns and beta in df_log.columns:
             features_ratios[f"Ratio_ThetaBeta_{canal}"] = df_log[theta] - df_log[beta]
             
        if theta in df_log.columns and alpha in df_log.columns:
             features_ratios[f"Ratio_ThetaAlpha_{canal}"] = df_log[theta] - df_log[alpha]

        if alpha in df_log.columns and beta in df_log.columns:
             features_ratios[f"Ratio_AlphaBeta_{canal}"] = df_log[alpha] - df_log[beta]
             
    df_ratios = pd.concat([df_log[columnas_metadatos], pd.DataFrame(features_ratios)], axis=1)
    return df_ratios

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

    directorio_salida = 'Resultados'
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)

    df_log = aplicar_log10(df_caracteristicas)
    
    archivo_log_parquet = os.path.join(directorio_salida, 'potencias_log10.parquet')
    archivo_log_csv = os.path.join(directorio_salida, 'potencias_log10.csv')
    df_log.to_parquet(archivo_log_parquet, index=False)
    df_log.to_csv(archivo_log_csv, index=False)

    df_final = normalizar_z_score_relajacion(df_log)
    archivo_parquet = os.path.join(directorio_salida, 'datos_bandas_normalizados.parquet')
    archivo_csv = os.path.join(directorio_salida, 'datos_bandas_normalizados.csv')
    df_final.to_parquet(archivo_parquet, index=False)
    df_final.to_csv(archivo_csv, index=False)
    
    df_asym = calcular_asimetria(df_log)
    df_asym_norm = normalizar_z_score_relajacion(df_asym)
    
    archivo_asym_parquet = os.path.join(directorio_salida, 'datos_asimetria_normalizados.parquet')
    archivo_asym_csv = os.path.join(directorio_salida, 'datos_asimetria_normalizados.csv')
    df_asym_norm.to_parquet(archivo_asym_parquet, index=False)
    df_asym_norm.to_csv(archivo_asym_csv, index=False)

    df_ratios = calcular_ratios(df_log)
    df_ratios_norm = normalizar_z_score_relajacion(df_ratios)
    
    archivo_ratios_parquet = os.path.join(directorio_salida, 'datos_ratios_normalizados.parquet')
    archivo_ratios_csv = os.path.join(directorio_salida, 'datos_ratios_normalizados.csv')
    df_ratios_norm.to_parquet(archivo_ratios_parquet, index=False)
    df_ratios_norm.to_csv(archivo_ratios_csv, index=False)
    
    columnas_join = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    
    df_merged = pd.merge(df_final, df_asym_norm, on=columnas_join, how='inner')
    
    df_merged = pd.merge(df_merged, df_ratios_norm, on=columnas_join, how='inner')
    
    archivo_merged_parquet = os.path.join(directorio_salida, 'datos_completos_normalizados.parquet')
    archivo_merged_csv = os.path.join(directorio_salida, 'datos_completos_normalizados.csv')
    
    df_merged.to_parquet(archivo_merged_parquet, index=False)
    df_merged.to_csv(archivo_merged_csv, index=False)
    
    return df_merged

if __name__ == "__main__":
    ruta_entrada = os.path.join('Resultados', 'datos_completo_epocas.parquet')
    if os.path.exists(ruta_entrada):
        calcular_bandas_potencia(ruta_entrada)
