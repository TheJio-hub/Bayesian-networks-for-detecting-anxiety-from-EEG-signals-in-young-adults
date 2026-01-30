import os
import pandas as pd
import numpy as np
import scipy.io

def cargar_escalas(ruta_escalas):
    """
    Carga y procesa el archivo scales.xls para crear un diccionario de puntajes.
    Retorna: dict {(Sujeto, Tarea, Trial): Puntaje}
    """
    # Leer el excel sin cabecera para manejar la estructura de doble fila
    df = pd.read_excel(ruta_escalas, header=None)
    
    # Mapeo de nombres de tareas en Excel a nombres de archivo
    mapa_tareas = {
        'Maths': 'Aritmetica',
        'Symmetry': 'Espejo',
        'Stroop': 'Stroop'
    }
    
    puntajes = {}
    
    # Iterar sobre las filas de sujetos (empiezan en la fila 2, índice 2)
    for index, fila in df.iterrows():
        if index < 2: continue # Saltar cabeceras
        
        id_sujeto = int(fila[0])
        
        # Estructura del Excel:
        # Trial 1: Cols 1, 2, 3 (Maths, Symmetry, Stroop)
        # Trial 2: Cols 4, 5, 6
        # Trial 3: Cols 7, 8, 9
        
        # Trial 1
        puntajes[(id_sujeto, 'Aritmetica', 1)] = fila[1]
        puntajes[(id_sujeto, 'Espejo', 1)] = fila[2]
        puntajes[(id_sujeto, 'Stroop', 1)] = fila[3]
        
        # Trial 2
        puntajes[(id_sujeto, 'Aritmetica', 2)] = fila[4]
        puntajes[(id_sujeto, 'Espejo', 2)] = fila[5]
        puntajes[(id_sujeto, 'Stroop', 2)] = fila[6]
        
        # Trial 3
        puntajes[(id_sujeto, 'Aritmetica', 3)] = fila[7]
        puntajes[(id_sujeto, 'Espejo', 3)] = fila[8]
        puntajes[(id_sujeto, 'Stroop', 3)] = fila[9]
            
    return puntajes

def cargar_nombres_canales(ruta_locs):
    """
    Lee el archivo Coordinates.locs y extrae los nombres de los canales.
    Asume que el nombre del canal está en la última columna.
    """
    canales = []
    try:
        with open(ruta_locs, 'r') as f:
            for linea in f:
                partes = linea.split()
                if len(partes) > 0:
                    # El nombre del canal es el último elemento
                    canales.append(partes[-1])
    except Exception as e:
        print(f"Error leyendo nombres de canales: {e}")
        # Retorna nombres genéricos si falla
        return [f'Canal{i+1}' for i in range(32)]
    
    if len(canales) != 32:
        print(f"Advertencia: Se esperaban 32 canales, se encontraron {len(canales)}")
        return [f'Canal{i+1}' for i in range(32)]
        
    return canales

def generar_conjunto_datos():
    ruta_base = 'Conjunto de datos/Data'
    ruta_filtrados = os.path.join(ruta_base, 'filtered_data')
    ruta_escalas = os.path.join(ruta_base, 'scales.xls')
    ruta_locs = os.path.join(ruta_base, 'Coordinates.locs')
    
    print("Cargando etiquetas de estrés...")
    busqueda_puntajes = cargar_escalas(ruta_escalas)
    
    print("Cargando nombres de canales...")
    nombres_canales = cargar_nombres_canales(ruta_locs)
    print(f"Canales detectados: {nombres_canales}")
    
    lista_metadatos = []
    lista_datos_eeg = []
    archivos_fallidos = []
    
    # Definir parámetros
    frecuencia_muestreo = 128
    segundos_por_archivo = 25
    muestras_por_epoca = frecuencia_muestreo * 1 # 1 segundo
    
    print("Procesando archivos .mat...")
    
    archivos = sorted([f for f in os.listdir(ruta_filtrados) if f.endswith('.mat')])
    total_archivos = len(archivos)
    
    for idx, nombre_archivo in enumerate(archivos):
        if idx % 50 == 0:
            print(f"Procesando archivo {idx}/{total_archivos}: {nombre_archivo}")
            
        try:
            # Parse filename: e.g., Arithmetic_sub_1_trial1.mat
            partes = nombre_archivo.replace('.mat', '').split('_')
            
            # Manejo de nombres de tareas y traducción al español
            tarea_archivo = ''
            if nombre_archivo.startswith('Mirror_image'):
                tarea_archivo = 'Espejo' # Nombre traducido
                # Mirror_image_sub_1_trial1 -> partes: ['Mirror', 'image', 'sub', '1', 'trial1']
                idx_sujeto = 3
                idx_ensayo = 4
            elif nombre_archivo.startswith('Arithmetic'):
                 tarea_archivo = 'Aritmetica'
                 idx_sujeto = 2
                 idx_ensayo = 3
            elif nombre_archivo.startswith('Relax'):
                 tarea_archivo = 'Relajacion'
                 idx_sujeto = 2
                 idx_ensayo = 3
            else: # Stroop
                 tarea_archivo = 'Stroop'
                 idx_sujeto = 2
                 idx_ensayo = 3
            
            id_sujeto = int(partes[idx_sujeto])
            num_ensayo = int(partes[idx_ensayo].replace('trial', ''))
            
            # Obtener Puntaje
            if tarea_archivo == 'Relajacion':
                puntaje = 0
            else:
                puntaje = busqueda_puntajes.get((id_sujeto, tarea_archivo, num_ensayo), np.nan)
            
            # Cargar .mat
            mat = scipy.io.loadmat(os.path.join(ruta_filtrados, nombre_archivo))
            # Buscar la llave de datos (usualmente 'Clean_data')
            llave_datos = [k for k in mat.keys() if not k.startswith('__')][0]
            datos_senal = mat[llave_datos] # Forma (32, 3200)
            
            # Validar forma
            if datos_senal.shape[1] != 3200:
                print(f"ADVERTENCIA: {nombre_archivo} tiene longitud inusual: {datos_senal.shape}")
                # Podríamos recortar o rellenar, pero por ahora saltamos si es muy diferente
                if datos_senal.shape[1] < 3200:
                    continue
                else:
                    datos_senal = datos_senal[:, :3200]
            
            # Dividir en épocas de 1 segundo (128 muestras)
            num_epocas = datos_senal.shape[1] // muestras_por_epoca
            
            for i_epoca in range(num_epocas):
                inicio = i_epoca * muestras_por_epoca
                fin = inicio + muestras_por_epoca
                datos_epoca = datos_senal[:, inicio:fin] 
                
                # Crear registro de metadatos
                registro = {
                    'Sujeto': id_sujeto,
                    'Tarea': tarea_archivo,
                    'Ensayo': num_ensayo,
                    'Epoca': i_epoca + 1,
                    'Puntaje': puntaje
                }
                lista_metadatos.append(registro)
                
                # Aplanar datos (Canal 1 [1..128], Canal 2 [1..128], ...)
                lista_datos_eeg.append(datos_epoca.flatten())
                
        except Exception as e:
            print(f"Error procesando {nombre_archivo}: {e}")
            archivos_fallidos.append(nombre_archivo)

    print(f"Generación completa. Total épocas: {len(lista_metadatos)}")
    
    # Crear DataFrames
    print("Construyendo DataFrame...")
    df_meta = pd.DataFrame(lista_metadatos)
    
    # Nombres de columnas para los datos EEG con nombres reales de canales
    # Formato: Fp1_1, Fp1_2 ... o Channel_Sample
    # El usuario pidió "etiquetas con el nombre del canal".
    # Dado que hay 128 muestras por canal por segundo, repetiré el nombre del canal con sufijo de muestra.
    nombres_cols_eeg = [f'{canal}_{m+1}' for canal in nombres_canales for m in range(muestras_por_epoca)]
    
    if len(nombres_cols_eeg) != len(lista_datos_eeg[0]):
         print(f"Error dimensional: Cols={len(nombres_cols_eeg)}, Datos={len(lista_datos_eeg[0])}")
    
    df_eeg = pd.DataFrame(lista_datos_eeg, columns=nombres_cols_eeg)
    
    # Concatenar
    df_resultado = pd.concat([df_meta, df_eeg], axis=1)
    
    # Guardar en formato Parquet (mucho más ligero y rápido)
    archivo_parquet = 'datos_completo_epocas.parquet'
    print(f"Guardando archivo Parquet: {archivo_parquet} ...")
    df_resultado.to_parquet(archivo_parquet, index=False)
    print("¡Archivo Parquet guardado exitosamente!")
    
    # Opcional: CSV (comentado para evitar archivos pesados)
    # archivo_csv = 'datos_completo_epocas.csv'
    # df_resultado.to_csv(archivo_csv, index=False)
    
    # Verificación
    print("\nResumen del Dataset:")
    print(df_resultado.groupby(['Tarea'])['Puntaje'].describe())

if __name__ == "__main__":
    generar_conjunto_datos()
