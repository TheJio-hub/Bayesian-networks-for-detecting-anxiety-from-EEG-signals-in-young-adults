import os
import re
import numpy as np
import pandas as pd
import scipy.io
import configuracion as config

def cargar_dataset(tipo_datos="filtrados_ica", tipo_prueba="Arithmetic"):
    '''
    Carga datos del dataset SAM 40.
    
    Args:
        tipo_datos (string): El tipo de datos a cargar. Por defecto "filtrados_ica".
        tipo_prueba (string): El tipo de prueba a cargar. Por defecto "Arithmetic".
    
    Returns:
        tuple: (dataset, archivos_cargados)
            dataset (ndarray): El dataset especificado con forma (n_trials, n_secs, n_channels, n_samples).
            archivos_cargados (list): Lista de nombres de archivos correspondientes a cada trial.
    '''
    assert (tipo_prueba in config.TIPOS_DE_PRUEBA)
    assert (tipo_datos in config.TIPOS_DE_DATOS)

    if tipo_datos == "crudos":
        dir_datos = config.DIR_DATOS_CRUDOS
        clave_datos = 'Data'
    elif tipo_datos == "filtrados_wt":
        dir_datos = config.DIR_DATOS_FILTRADOS
        clave_datos = 'Clean_data'
    else:
        # Por defecto usamos un directorio de filtrados
        dir_datos = config.DIR_DATOS_FILTRADOS 
        clave_datos = 'Clean_data'
        
    # Inicializamos lista
    datos_cargados = []
    archivos_cargados = []
    
    # Obtenemos lista de archivos
    archivos = []
    if os.path.exists(dir_datos):
        archivos = os.listdir(dir_datos)
        archivos = [a for a in archivos if a.endswith('.mat')]
        
        # Orden natural para alinear con etiquetas (sub_1, sub_2, ..., sub_10)
        def natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
            
        archivos.sort(key=natural_key)
    else:
        print(f"Directorio no encontrado: {dir_datos}")
        return np.array([]), []

    for nombre_archivo in archivos:
        if tipo_prueba not in nombre_archivo:
            continue

        ruta = os.path.join(dir_datos, nombre_archivo)
        try:
            mat = scipy.io.loadmat(ruta)
            datos_trial = None
            if clave_datos in mat:
                datos_trial = mat[clave_datos]
            else:
                # Intenta buscar otras claves si la default falla
                keys = [k for k in mat.keys() if not k.startswith('__')]
                if keys:
                    datos_trial = mat[keys[0]]
            
            if datos_trial is not None:
                datos_cargados.append(datos_trial)
                archivos_cargados.append(nombre_archivo)
                
        except Exception as e:
            print(f"Error leyendo {nombre_archivo}: {e}")

    if not datos_cargados:
        return np.array([]), []

    # Convertir a numpy array
    # Forma esperada original: (120, 32, 3200)
    dataset = np.array(datos_cargados)
    
    # Validar forma
    if dataset.ndim == 3:
         # Reshape para Extraccion.py: (n_trials, n_secs, n_canales, n_samples)
         # Asumiendo 3200 samples = 25 segundos * 128 Hz
         n_trials, n_channels, n_total_samples = dataset.shape
         sfreq = config.FRECUENCIA_MUESTREO
         n_secs = n_total_samples // sfreq
         n_samples = sfreq
         
         if n_total_samples % sfreq == 0:
             # (120, 32, 25, 128)
             dataset = dataset.reshape(n_trials, n_channels, n_secs, n_samples)
             # Transponer a (120, 25, 32, 128)
             dataset = dataset.transpose(0, 2, 1, 3)
         else:
             print(f"Advertencia: Los datos no son divisibles exactamente en segundos a {sfreq}Hz.")

    return dataset, archivos_cargados

def cargar_etiquetas():
    '''
    Carga etiquetas del dataset y las transforma a binarias.

    Returns:
        ndarray: Las etiquetas.
    '''
    if not os.path.exists(config.RUTA_ETIQUETAS):
        print(f"Archivo de etiquetas no encontrado: {config.RUTA_ETIQUETAS}")
        return None

    etiquetas = pd.read_excel(config.RUTA_ETIQUETAS)
    etiquetas = etiquetas.rename(columns=config.COLUMNAS_A_RENOMBRAR)
    # Ajuste según formato original, eliminar primera fila si es metadata
    etiquetas = etiquetas[1:]
    
    # Asegurar tipos numéricos
    etiquetas = etiquetas.astype("int")
    
    # Binarizar: Valores mayores a 5 son 1 (Ansiedad/Estrés), otros 0
    etiquetas = etiquetas > 5
    return etiquetas

def obtener_etiquetas(tipo_prueba="Arithmetic"):
    '''
    Obtiene las etiquetas alineadas con el dataset para un tipo de prueba específico.
    Asume que el dataset está ordenado por sujeto (1..40) y luego por trial (1..3).
    
    Args:
        tipo_prueba (string): "Arithmetic", "Mirror", o "Stroop".
        
    Returns:
        ndarray: Array 1D de etiquetas (booleans o ints).
    '''
    etiquetas_df = cargar_etiquetas()
    if etiquetas_df is None:
        return np.array([])
        
    if tipo_prueba not in config.COLUMNAS_TIPO_PRUEBA:
        print(f"Tipo de prueba {tipo_prueba} no reconocido en configuración.")
        return np.array([])
        
    cols = config.COLUMNAS_TIPO_PRUEBA[tipo_prueba]
    # Seleccionamos las columnas relevantes para la prueba
    # El dataframe tiene índice de sujetos 0..39 (si se cargó correctamente sin la primera fila)
    subset = etiquetas_df[cols]
    
    # Aplanamos la matriz (40, 3) a (120,). 
    # Pandas/Numpy ravel lo hace por filas por defecto (C-style): S1T1, S1T2, S1T3, S2T1...
    # Esto coincide con el orden de archivos si usamos orden natural (sub_1_t1, sub_1_t2...)
    return subset.values.ravel().astype(int)


def formatear_etiquetas(etiquetas, tipo_prueba="Arithmetic", epocas=1):
    '''
    Filtra las etiquetas y las repite por la cantidad especificada de épocas.

    Args:
        etiquetas (ndarray): Las etiquetas.
        tipo_prueba (string): El tipo de prueba para filtrar.
        epocas (int): La cantidad de épocas.

    Returns:
        ndarray: Las etiquetas formateadas.
    '''
    assert (tipo_prueba in config.TIPOS_DE_PRUEBA)

    etiquetas_formateadas = []
    if tipo_prueba in config.COLUMNAS_TIPO_PRUEBA:
        columnas = config.COLUMNAS_TIPO_PRUEBA[tipo_prueba]
        for trial in columnas:
            if trial in etiquetas.columns:
                etiquetas_formateadas.append(etiquetas[trial])
            else:
                print(f"Advertencia: Columna {trial} no encontrada en etiquetas.")
    
    if etiquetas_formateadas:
        etiquetas_formateadas = pd.concat(etiquetas_formateadas).to_numpy()
        etiquetas_formateadas = etiquetas_formateadas.repeat(epocas)
    else:
        return np.array([])

    return etiquetas_formateadas


def dividir_datos(datos, frecuencia_muestreo):
    '''
    Divide los datos EEG en épocas de 1 segundo.

    Args:
        datos (ndarray): Datos EEG.
        frecuencia_muestreo (int): Frecuencia de muestreo.
    
    Returns:
        ndarray: Datos divididos en épocas.
    '''
    if datos.ndim != 3:
        print("Los datos deben tener 3 dimensiones (trials, canales, muestras)")
        return datos

    n_trials, n_canales, n_muestras = datos.shape
    
    segundos = n_muestras // frecuencia_muestreo
    
    if segundos == 0:
        return np.empty((n_trials, 0, n_canales, frecuencia_muestreo))

    datos_epocados = np.empty((n_trials, segundos, n_canales, frecuencia_muestreo))
    
    for i in range(n_trials):
        for j in range(segundos):
            inicio = j * frecuencia_muestreo
            fin = (j + 1) * frecuencia_muestreo
            datos_epocados[i, j] = datos[i, :, inicio:fin]
            
    return datos_epocados
