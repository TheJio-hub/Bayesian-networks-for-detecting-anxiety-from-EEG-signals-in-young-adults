import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import linregress
import os

def corregir_linea_base_regresion(df):    
    es_relajacion = df['Tarea'] == 'Relajacion'
    df_relajacion = df[es_relajacion].copy()
    
    if df_relajacion.empty:
        print("Advertencia: No se encontraron datos de Relajación.")
        return df

    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    columnas_caracteristicas = [c for c in df.columns if c not in columnas_metadatos and np.issubdtype(df[c].dtype, np.number)]
    
    # Calcular promedio de Relajación por Sujeto
    df_base_sujeto = df_relajacion.groupby('Sujeto')[columnas_caracteristicas].mean().reset_index()
    
    # Unir vector base
    df_unido = pd.merge(
        df, 
        df_base_sujeto, 
        on='Sujeto', 
        suffixes=('', '_base'), 
        how='left'
    )
    
    # Aplicar corrección de línea base (Log-Ratio / Decibelios)
    # dB = 10 * log10(Actividad / Baseline)
    epsilon = 1e-10
    
    for caracteristica in columnas_caracteristicas:
        columna_base = f"{caracteristica}_base"
        if columna_base in df_unido.columns:
            # evitar división por cero
            ratio = (df_unido[caracteristica] + epsilon) / (df_unido[columna_base] + epsilon)
            # Transformar a decibelios
            df_unido[caracteristica] = 10 * np.log10(ratio)
    
    return df_unido[df.columns]

def calcular_bandas_potencia(ruta_dataset):
    df = pd.read_parquet(ruta_dataset)
    
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    columnas_senal = [c for c in df.columns if c not in columnas_metadatos]
    
    conjunto_canales = set()
    for col in columnas_senal:
        if '_' in col:
            nombre_canal = col.rsplit('_', 1)[0]
            conjunto_canales.add(nombre_canal)

    lista_canales = sorted(list(conjunto_canales))
    
    # Definición de Bandas de frecuencia (Hz)
    bandas_frecuencia = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45) 
    }
    
    frecuencia_muestreo = 128 
    duracion_epoca = 5 # Segundos
    muestras_por_epoca = frecuencia_muestreo * duracion_epoca
    
    # Inicializar DataFrame de características solo con metadatos
    df_caracteristicas = df[columnas_metadatos].copy()
    
    for canal in lista_canales:
        columnas_este_canal = [f"{canal}_{i+1}" for i in range(muestras_por_epoca)]
        
        columnas_validas = [c for c in columnas_este_canal if c in df.columns]
        
        # Si faltan muestras, saltar este canal
        if len(columnas_validas) < muestras_por_epoca:
            continue
            
        datos_canal = df[columnas_validas].values
        
        # Calcular Densidad Espectral de Potencia (PSD)
        # welch retorna (N_epocas, N_frecuencias). Al integrar sobre frecuencias, obtenemos (N_epocas,)
        frecuencias, densidad_espectral = welch(datos_canal, fs=frecuencia_muestreo, nperseg=128, axis=1)
        
        # Potencia Total (0.5 - 45 Hz)
        indice_total = np.logical_and(frecuencias >= 0.5, frecuencias <= 45)
        densidad_total = densidad_espectral[:, indice_total].sum(axis=1) 
        
        # Calcular potencia relativa para cada banda
        for nombre_banda, (freq_baja, freq_alta) in bandas_frecuencia.items():
            indice_banda = np.logical_and(frecuencias >= freq_baja, frecuencias <= freq_alta)
            
            potencia_banda_absoluta = densidad_espectral[:, indice_banda].sum(axis=1) 
            
            potencia_relativa = potencia_banda_absoluta / (densidad_total + 1e-10)
            
            nombre_columna_salida = f"{canal}_{nombre_banda}"
            df_caracteristicas[nombre_columna_salida] = potencia_relativa
    
    df_caracteristicas = corregir_linea_base_regresion(df_caracteristicas)
    
    # Definir rutas de salida
    directorio_salida = 'Resultados'
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)

    archivo_salida_parquet = os.path.join(directorio_salida, 'datos_bandas.parquet')
    archivo_salida_csv = os.path.join(directorio_salida, 'datos_bandas.csv')
    
    # Guardar resultados
    df_caracteristicas.to_parquet(archivo_salida_parquet, index=False)
    df_caracteristicas.to_csv(archivo_salida_csv, index=False)
    
    return df_caracteristicas

if __name__ == "__main__":
    ruta_entrada = os.path.join('Resultados', 'datos_completo_epocas.parquet')
    if os.path.exists(ruta_entrada):
        calcular_bandas_potencia(ruta_entrada)
    else:
        print(f"No se encontró el archivo de entrada: {ruta_entrada}")

