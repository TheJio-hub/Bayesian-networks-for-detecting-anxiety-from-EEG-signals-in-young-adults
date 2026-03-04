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
    
    busqueda_puntajes = cargar_escalas(ruta_escalas)
    
    nombres_canales = cargar_nombres_canales(ruta_locs)
    
    # Lista de metadatos y datos EEG
    lista_metadatos = []
    lista_datos_eeg = []
    
    # Parámetros EEG
    frecuencia_muestreo = 128
    segundos_descarte = 5 
    muestras_descarte = segundos_descarte * frecuencia_muestreo
    
    # Épocas de 5 segundos con solapamiento del 50%
    duracion_epoca = 5 # segundos
    solapamiento = 0.5 # 50%
    muestras_por_epoca = int(frecuencia_muestreo * duracion_epoca)
    paso_muestras = int(muestras_por_epoca * (1 - solapamiento))
        
    archivos = sorted([f for f in os.listdir(ruta_filtrados) if f.endswith('.mat')])
    
    for idx, nombre_archivo in enumerate(archivos):
            
        try:
            # Parse filename
            partes = nombre_archivo.replace('.mat', '').split('_')
            
            # Identificar tarea por nombre de archivo
            tarea_archivo = ''
            if nombre_archivo.startswith('Mirror_image'):
                tarea_archivo = 'Espejo' 
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
            
            if tarea_archivo == 'Relajacion':
                puntaje = 0
            else:
                puntaje = busqueda_puntajes.get((id_sujeto, tarea_archivo, num_ensayo), np.nan)
            
            mat = scipy.io.loadmat(os.path.join(ruta_filtrados, nombre_archivo))
            llave_datos = [k for k in mat.keys() if not k.startswith('__')][0]
            datos_senal = mat[llave_datos] 
            
            if datos_senal.shape[1] < 3200:
                 continue
            
            # Recortar a 25 segundos
            datos_senal = datos_senal[:, :3200]

            if tarea_archivo == 'Relajacion':
                pass
            else:
                # Extraer baseline (primeros 5s)
                datos_base = datos_senal[:, :muestras_descarte]
                n_samples_base = datos_base.shape[1]
                
                # Calcular número de épocas con solapamiento
                if n_samples_base >= muestras_por_epoca:
                    num_epocas_base = (n_samples_base - muestras_por_epoca) // paso_muestras + 1
                else:
                    num_epocas_base = 0
                
                for i_epoca in range(num_epocas_base):
                    inicio = i_epoca * paso_muestras
                    fin = inicio + muestras_por_epoca
                    datos_epoca = datos_base[:, inicio:fin]
                    
                    registro = {
                        'Sujeto': id_sujeto,
                        'Tarea': f"Baseline_{tarea_archivo}", 
                        'Trial': num_ensayo,
                        'Epoca': i_epoca + 1,
                        'Puntaje': puntaje
                    }
                    lista_metadatos.append(registro)
                    lista_datos_eeg.append(datos_epoca.flatten())

                # Datos activos (segundos 5 al 25)
                datos_senal = datos_senal[:, muestras_descarte:]
            
            # Dividir en épocas con solapamiento
            n_samples_active = datos_senal.shape[1]
            if n_samples_active >= muestras_por_epoca:
                num_epocas = (n_samples_active - muestras_por_epoca) // paso_muestras + 1
            else:
                num_epocas = 0
            
            for i_epoca in range(num_epocas):
                inicio = i_epoca * paso_muestras
                fin = inicio + muestras_por_epoca
                datos_epoca = datos_senal[:, inicio:fin] 
                
                registro = {
                    'Sujeto': id_sujeto,
                    'Tarea': tarea_archivo,
                    'Trial': num_ensayo,
                    'Epoca': i_epoca + 1,
                    'Puntaje': puntaje
                }
                lista_metadatos.append(registro)
                
                lista_datos_eeg.append(datos_epoca.flatten())
                
        except Exception:
            pass

    # Crear DataFrames
    df_meta = pd.DataFrame(lista_metadatos)
    
    nombres_cols_eeg = [f'{canal}_{m+1}' for canal in nombres_canales for m in range(muestras_por_epoca)]
    
    if len(nombres_cols_eeg) != len(lista_datos_eeg[0]):
         pass
    
    df_eeg = pd.DataFrame(lista_datos_eeg, columns=nombres_cols_eeg)
    
    df_resultado = pd.concat([df_meta, df_eeg], axis=1)
    
    output_dir = 'Resultados'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    archivo_parquet = os.path.join(output_dir, 'datos_completo_epocas.parquet')
    df_resultado.to_parquet(archivo_parquet, index=False)
    
    # Generar CSV para visualización
    archivo_csv = os.path.join(output_dir, 'datos_completo_epocas.csv')
    df_resultado.to_csv(archivo_csv, index=False)
    
    # Exportar archivo específico con metadatos
    archivo_meta = os.path.join(output_dir, 'metadatos_epocas_dataset.csv')
    df_meta.to_csv(archivo_meta, index=False)


if __name__ == "__main__":
    generar_conjunto_datos()
