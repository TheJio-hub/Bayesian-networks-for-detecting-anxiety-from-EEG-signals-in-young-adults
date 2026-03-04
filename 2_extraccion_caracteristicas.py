import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import linregress
import os

def corregir_linea_base_regresion(df):
    """
    Aplica corrección de línea base usando regresión lineal por sujeto.
    Modelo: Actividad ~ Pendiente * Base + Intercepto
    Resultado: Residuo = Actividad - (Pendiente * Base + Intercepto)
    """
    print("Aplicando corrección de línea base por regresión...")
    
    es_linea_base = df['Tarea'].str.startswith('Baseline_')
    df_base = df[es_linea_base].copy()
    df_actividad = df[~es_linea_base].copy()
    
    if df_base.empty:
        print("Advertencia: No se encontraron datos de línea base. Se omite la corrección.")
        return df_actividad

    df_base['Tarea_Coincidencia'] = df_base['Tarea'].str.replace('Baseline_', '', regex=False)
    
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Tarea_Coincidencia']
    columnas_caracteristicas = [c for c in df.columns if c not in columnas_metadatos and np.issubdtype(df[c].dtype, np.number)]
    
    # Calcular promedio de la línea base por (Sujeto, Trial, Tarea)
    columnas_agrupacion = ['Sujeto', 'Trial', 'Tarea_Coincidencia']
    columnas_existentes = [c for c in columnas_agrupacion if c in df_base.columns]
    
    df_base_promedio = df_base.groupby(columnas_existentes)[columnas_caracteristicas].mean().reset_index()
    
    # Unir los promedios de línea base a la tabla de actividad
    df_unido = pd.merge(
        df_actividad, 
        df_base_promedio, 
        left_on=['Sujeto', 'Trial', 'Tarea'], 
        right_on=['Sujeto', 'Trial', 'Tarea_Coincidencia'], 
        suffixes=('', '_base'), 
        how='left'
    )
    
    # Aplicar regresión por sujeto y por característica
    sujetos_unicos = df_unido['Sujeto'].unique()
    
    for caracteristica in columnas_caracteristicas:
        columna_base = f"{caracteristica}_base"
        
        # Verificar si la columna base existe en el dataframe unido
        if columna_base not in df_unido.columns:
            continue
            
        for sujeto in sujetos_unicos:
            mascara_sujeto = df_unido['Sujeto'] == sujeto
            subconjunto = df_unido.loc[mascara_sujeto]
            
            # Obtener índices válidos (sin NaNs en ninguna de las dos columnas)
            indices_validos = subconjunto[[caracteristica, columna_base]].dropna().index
            
            # Se requieren al menos 3 puntos para una regresión mínimamente estable
            if len(indices_validos) > 2:
                Y = df_unido.loc[indices_validos, caracteristica].values
                X = df_unido.loc[indices_validos, columna_base].values
                
                # Regresión lineal: Y = pendiente * X + intercepto
                pendiente, intercepto, _, _, _ = linregress(X, Y)
                
                # Calcular predicción y residuo
                prediccion = pendiente * X + intercepto
                df_unido.loc[indices_validos, caracteristica] = Y - prediccion
                
            elif len(indices_validos) > 0:
                # Si hay muy pocos puntos, recurrir a resta simple
                 df_unido.loc[indices_validos, caracteristica] = (
                     df_unido.loc[indices_validos, caracteristica] - 
                     df_unido.loc[indices_validos, columna_base]
                 )

    return df_unido[df_actividad.columns].copy()

def calcular_bandas_potencia(ruta_dataset):
    print(f"Procesando archivo: {ruta_dataset}")
    df = pd.read_parquet(ruta_dataset)
    
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    # Las columnas de señal son todas las que no están en metadatos
    columnas_senal = [c for c in df.columns if c not in columnas_metadatos]
    
    conjunto_canales = set()
    for col in columnas_senal:
        if '_' in col:
            # Asume formato "Canal_Indice"
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
        
        # Calcular Densidad Espectral de Potencia (PSD) con método de Welch
        # nperseg=128 equivale a ventanas de 1 segundo para suavizado
        frecuencias, densidad_espectral = welch(datos_canal, fs=frecuencia_muestreo, nperseg=128, axis=1)
        
        # Potencia Total (0.5 - 45 Hz)
        indice_total = np.logical_and(frecuencias >= 0.5, frecuencias <= 45)
        densidad_total = densidad_espectral[:, indice_total].sum(axis=1) 
        
        # Calcular potencia relativa para cada banda
        for nombre_banda, (freq_baja, freq_alta) in bandas_frecuencia.items():
            indice_banda = np.logical_and(frecuencias >= freq_baja, frecuencias <= freq_alta)
            
            potencia_banda_absoluta = densidad_espectral[:, indice_banda].sum(axis=1) 
            
            # Potencia Relativa = Potencia Banda / Potencia Total
            # Se añade un epsilon pequeño (1e-10) para evitar división por cero
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
