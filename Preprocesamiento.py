import os
import re
import numpy as np
import pandas as pd
import scipy.io
import configuracion as config

def cargar_dataset(tipo_datos="filtrados_ica", tipo_prueba="Arithmetic"):
    '''
    Carga y estructura los datos del dataset SAM 40.
    
    Args:
        tipo_datos (str): Tipo de preprocesamiento origen (ej. "filtrados_ica").
        tipo_prueba (str): Tarea experimental (ej. "Arithmetic").
    
    Returns:
        tuple: (matriz_datos_eeg, lista_archivos)
    '''
    # Validaciones
    if tipo_prueba not in config.TIPOS_DE_PRUEBA:
        raise ValueError("Tipo de prueba no válido")

    # Selección de directorio
    dir_origen = config.DIR_DATOS_FILTRADOS 
    clave_mat = 'Clean_data'
    
    if tipo_datos == "crudos":
        dir_origen = config.DIR_DATOS_CRUDOS
        clave_mat = 'Data'
    elif tipo_datos == "filtrados_wt":
        dir_origen = config.DIR_DATOS_FILTRADOS
        clave_mat = 'Clean_data'
        
    lista_datos = []
    lista_nombres = []
    
    # Búsqueda de archivos
    archivos_mat = []
    if os.path.exists(dir_origen):
        archivos_mat = [f for f in os.listdir(dir_origen) if f.endswith('.mat')]
        
        # Ordenamiento natural (1, 2, ... 10) en lugar de ASCII (1, 10, 2...)
        def llave_orden(texto):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', texto)]
            
        archivos_mat.sort(key=llave_orden)
    else:
        print(f"Directorio inaccesible: {dir_origen}")
        return np.array([]), []

    # Carga iterativa
    for nombre_archivo in archivos_mat:
        if tipo_prueba not in nombre_archivo:
            continue

        ruta_completa = os.path.join(dir_origen, nombre_archivo)
        try:
            contenido_mat = scipy.io.loadmat(ruta_completa)
            datos_trial = None
            
            # Búsqueda de la variable correcta dentro del .mat
            if clave_mat in contenido_mat:
                datos_trial = contenido_mat[clave_mat]
            else:
                llaves_validas = [k for k in contenido_mat.keys() if not k.startswith('__')]
                if llaves_validas:
                    datos_trial = contenido_mat[llaves_validas[0]]
            
            if datos_trial is not None:
                lista_datos.append(datos_trial)
                lista_nombres.append(nombre_archivo)
                
        except Exception as e:
            print(f"Fallo al leer {nombre_archivo}: {e}")

    if not lista_datos:
        return np.array([]), []

    # Estructuración Numpy
    conjunto_datos = np.array(lista_datos)
    
    # Ajuste de dimensiones [Trials, Canales, Segundos, Muestras]
    if conjunto_datos.ndim == 3:
         n_trials, n_canales, n_total_muestras = conjunto_datos.shape
         frec_muestreo = config.FRECUENCIA_MUESTREO
         
         n_segs = n_total_muestras // frec_muestreo
         n_muestras_seg = frec_muestreo
         
         if n_total_muestras % frec_muestreo == 0:
             # Redimensionar
             conjunto_datos = conjunto_datos.reshape(n_trials, n_canales, n_segs, n_muestras_seg)
             # Reordenar ejes a (Trials, Segundos, Canales, Muestras)
             conjunto_datos = conjunto_datos.transpose(0, 2, 1, 3)

    return conjunto_datos, lista_nombres

def cargar_etiquetas():
    '''Carga y procesa el archivo de etiquetas Excel.'''
    if not os.path.exists(config.RUTA_ETIQUETAS):
        return None

    df_etiquetas = pd.read_excel(config.RUTA_ETIQUETAS)
    df_etiquetas = df_etiquetas.rename(columns=config.COLUMNAS_A_RENOMBRAR)
    
    # Limpieza de cabecera extra si existe
    if df_etiquetas.shape[0] > 0 and pd.to_numeric(df_etiquetas.iloc[0,0], errors='coerce') is np.nan:
         df_etiquetas = df_etiquetas[1:]
    
    # Conversión a entero
    df_etiquetas = df_etiquetas.astype("int")
    
    # Binarización (Umbral > 5 = Ansiedad)
    return df_etiquetas > 5

def obtener_etiquetas(tipo_prueba="Arithmetic"):
    '''Genera vector de etiquetas alineado a los trials.'''
    df_etiquetas_bin = cargar_etiquetas()
    if df_etiquetas_bin is None:
        return np.array([])
        
    cols_interes = config.COLUMNAS_TIPO_PRUEBA.get(tipo_prueba, [])
    if not cols_interes:
        return np.array([])
        
    subset = df_etiquetas_bin[cols_interes]
    # Aplanar matriz a vector 1D
    return subset.values.ravel().astype(int)
