import os

import numpy as np
import pandas as pd
from scipy.signal import welch
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)


def aplicar_log10(df):
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    columnas_caracteristicas = [c for c in df.columns if c not in columnas_metadatos and np.issubdtype(df[c].dtype, np.number)]
    
    df_log = df.copy()
    epsilon = 1e-25
    df_log[columnas_caracteristicas] = np.log10(df_log[columnas_caracteristicas] + epsilon)
    return df_log

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
    
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Delta_Rel', 'Theta_Rel', 'Alpha_Rel', 'Beta_Rel', 'Gamma_Rel']
    
    features_asimetria = {}
    
    cols = df_log.columns
    
    for (izq, der) in tqdm(pares, desc='Asimetria por pares', unit='par', leave=False):
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
    
    for canal in tqdm(canales, desc='Ratios por canal', unit='canal', leave=False):
        delta = f"{canal}_Delta"
        theta = f"{canal}_Theta"
        alpha = f"{canal}_Alpha"
        beta = f"{canal}_Beta"
        gamma = f"{canal}_Gamma"
        
        if theta in df_log.columns and beta in df_log.columns:
             features_ratios[f"Ratio_ThetaBeta_{canal}"] = df_log[theta] - df_log[beta]
             
        if theta in df_log.columns and alpha in df_log.columns:
             features_ratios[f"Ratio_ThetaAlpha_{canal}"] = df_log[theta] - df_log[alpha]

        if alpha in df_log.columns and beta in df_log.columns:
             features_ratios[f"Ratio_AlphaBeta_{canal}"] = df_log[alpha] - df_log[beta]
             
        if theta in df_log.columns and delta in df_log.columns:
             features_ratios[f"Ratio_ThetaDelta_{canal}"] = df_log[theta] - df_log[delta]

        if alpha in df_log.columns and delta in df_log.columns:
             features_ratios[f"Ratio_AlphaDelta_{canal}"] = df_log[alpha] - df_log[delta]

        if theta in df_log.columns and gamma in df_log.columns:
             features_ratios[f"Ratio_ThetaGamma_{canal}"] = df_log[theta] - df_log[gamma]

        if beta in df_log.columns and gamma in df_log.columns:
             features_ratios[f"Ratio_BetaGamma_{canal}"] = df_log[beta] - df_log[gamma]

        if alpha in df_log.columns and gamma in df_log.columns:
             features_ratios[f"Ratio_AlphaGamma_{canal}"] = df_log[alpha] - df_log[gamma]
             
    df_ratios = pd.concat([df_log[columnas_metadatos], pd.DataFrame(features_ratios)], axis=1)
    return df_ratios

def calcular_asimetria_ratios(df_ratios):
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
    ratios_nombres = [
        'Ratio_ThetaBeta',
        'Ratio_ThetaAlpha',
        'Ratio_AlphaBeta',
        'Ratio_ThetaDelta',
        'Ratio_AlphaDelta',
        'Ratio_ThetaGamma',
        'Ratio_BetaGamma',
        'Ratio_AlphaGamma'
    ]
    features_asym_ratios = {}
    cols = df_ratios.columns
    for (izq, der) in tqdm(pares, desc='Asimetria de ratios', unit='par', leave=False):
        for ratio in ratios_nombres:
            col_izq = f"{ratio}_{izq}"
            col_der = f"{ratio}_{der}"
            if col_izq in cols and col_der in cols:
                nombre_feat = f"Asym_{ratio}_{der}_{izq}"
                features_asym_ratios[nombre_feat] = df_ratios[col_der] - df_ratios[col_izq]
                
    df_asym_ratios = pd.concat([df_ratios[columnas_metadatos], pd.DataFrame(features_asym_ratios)], axis=1)
    return df_asym_ratios


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
    
    for canal in tqdm(lista_canales, desc='Bandas por canal', unit='canal'):
        columnas_tiempo = [f"{canal}_{i+1}" for i in range(muestras_por_epoca)]
        cols_existentes = [c for c in columnas_tiempo if c in df.columns]
        
        if len(cols_existentes) < muestras_por_epoca:
            continue
            
        matriz_senal = df[cols_existentes].values
        frecuencias, psd = welch(matriz_senal, fs=frecuencia_muestreo, nperseg=128, axis=1)
        
        bandas_canal = {}
        total_canal = np.zeros(matriz_senal.shape[0])
        
        for nombre_banda, (f_min, f_max) in bandas_frecuencia.items():
            mask_banda = np.logical_and(frecuencias >= f_min, frecuencias <= f_max)
            potencia_absoluta = psd[:, mask_banda].sum(axis=1)
            bandas_canal[nombre_banda] = potencia_absoluta
            total_canal += potencia_absoluta
            
        for nombre_banda, potencia_absoluta in bandas_canal.items():
            nuevas_columnas[f"{canal}_{nombre_banda}"] = potencia_absoluta
            potencia_relativa = potencia_absoluta / (total_canal + 1e-15)
            nuevas_columnas[f"{canal}_{nombre_banda}_Rel"] = potencia_relativa
            
    df_bands = pd.DataFrame(nuevas_columnas)
    df_caracteristicas = pd.concat([df[columnas_metadatos].reset_index(drop=True), df_bands], axis=1)

    directorio_salida = os.path.join('Resultados', 'Exploratorio')
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)

    df_log = aplicar_log10(df_caracteristicas)
    
    archivo_log_parquet = os.path.join(directorio_salida, 'potencias_log10.parquet')
    archivo_log_csv = os.path.join(directorio_salida, 'potencias_log10.csv')
    df_log.to_parquet(archivo_log_parquet, index=False)
    df_log.to_csv(archivo_log_csv, index=False)

    archivo_bandas_log_parquet = os.path.join(directorio_salida, 'datos_bandas_log10.parquet')
    archivo_bandas_log_csv = os.path.join(directorio_salida, 'datos_bandas_log10.csv')
    df_log.to_parquet(archivo_bandas_log_parquet, index=False)
    df_log.to_csv(archivo_bandas_log_csv, index=False)

    df_asym = calcular_asimetria(df_log)
    archivo_asym_log_parquet = os.path.join(directorio_salida, 'datos_asimetria_log10.parquet')
    archivo_asym_log_csv = os.path.join(directorio_salida, 'datos_asimetria_log10.csv')
    df_asym.to_parquet(archivo_asym_log_parquet, index=False)
    df_asym.to_csv(archivo_asym_log_csv, index=False)

    df_ratios = calcular_ratios(df_log)
    archivo_ratios_log_parquet = os.path.join(directorio_salida, 'datos_ratios_log10.parquet')
    archivo_ratios_log_csv = os.path.join(directorio_salida, 'datos_ratios_log10.csv')
    df_ratios.to_parquet(archivo_ratios_log_parquet, index=False)
    df_ratios.to_csv(archivo_ratios_log_csv, index=False)
    
    df_asym_ratios = calcular_asimetria_ratios(df_ratios)
    archivo_asym_ratios_log_parquet = os.path.join(directorio_salida, 'datos_asimetria_ratios_log10.parquet')
    archivo_asym_ratios_log_csv = os.path.join(directorio_salida, 'datos_asimetria_ratios_log10.csv')
    df_asym_ratios.to_parquet(archivo_asym_ratios_log_parquet, index=False)
    df_asym_ratios.to_csv(archivo_asym_ratios_log_csv, index=False)
    
    columnas_join = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    
    df_merged = pd.merge(df_log, df_asym, on=columnas_join, how='inner')
    df_merged = pd.merge(df_merged, df_ratios, on=columnas_join, how='inner')
    df_merged = pd.merge(df_merged, df_asym_ratios, on=columnas_join, how='inner')
    
    archivo_merged_parquet = os.path.join(directorio_salida, 'datos_completos_log10.parquet')
    archivo_merged_csv = os.path.join(directorio_salida, 'datos_completos_log10.csv')
    
    df_merged.to_parquet(archivo_merged_parquet, index=False)
    df_merged.to_csv(archivo_merged_csv, index=False)
    
    return df_merged

if __name__ == "__main__":
    ruta_entrada = os.path.join('Resultados', 'Exploratorio', 'datos_completo_epocas.parquet')
    if os.path.exists(ruta_entrada):
        calcular_bandas_potencia(ruta_entrada)
