import pandas as pd
import os
import glob
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)


def principal():
    dir_base = "Resultados/Análisis global/Modelos generados"
    dir_resultados = os.path.join(dir_base, "Top 30")  # Ruta actualizada al moverse los archivos
    
    # Comprobar si el directorio existe
    if not os.path.exists(dir_resultados):
        print(f"Directorio no encontrado: {dir_resultados}")
        return

    todos_los_archivos = glob.glob(os.path.join(dir_resultados, "*.csv"))
    # Filtrar solo archivos de Top 30 
    archivos_top30 = [f for f in todos_los_archivos if "_top30" in f and "Mejor_Resultado" not in f and "Resumen_Mejores" not in f]

    clasificadores = ["DT", "RF", "KNN", "SVM"]
    nombres_clasificadores = {
        "DT": "Árbol de Decisión",
        "RF": "Random Forest",
        "KNN": "K-Nearest Neighbors",
        "SVM": "Support Vector Machine"
    }

    mejores_filas = []

    print("Buscando los mejores modelos para Top 30 características (Apilado)...")

    for codigo_clf in tqdm(clasificadores, desc='Consolidando Top 30', unit='clasificador'):
        mejor_exactitud = -1.0
        mejores_filas_para_clasificador = [] # Almacenar filas para el mejor clasificador actual
        mejor_nombre_ranking = None

        # Filtrar archivos para este clasificador
        archivos_clasificador = [f for f in archivos_top30 if f"Resultados_{codigo_clf}_" in os.path.basename(f)]

        for ruta_archivo in archivos_clasificador:
            try:
                df = pd.read_csv(ruta_archivo, index_col=0)
                # La fila de exactitud para "Clase 0" es la misma que la exactitud global
                # A veces el CSV guarda en floats o string, forzar float
                try:
                    exactitud = float(df.loc["Clase 0", "Exactitud"])
                except:
                    exactitud = 0.0
                
                if exactitud > mejor_exactitud:
                    mejor_exactitud = exactitud
                    
                    # Extraer el nombre del ranking
                    nombre_archivo = os.path.basename(ruta_archivo)
                    parte_ranking = nombre_archivo.replace(f"Resultados_{codigo_clf}_Ranking_", "").replace("_top30.csv", "")
                    
                    # Crear filas para Clase 0 y Clase 1 siguiendo la estructura solicitada
                    # Fila 1: Clase 0
                    fila_c0 = {
                        "Clasificador": nombres_clasificadores[codigo_clf],
                        "Metodo_Seleccion": parte_ranking,
                        "Clase": "Clase 0 (Relajación)",
                        "Precision": df.loc["Clase 0", "Precision"],
                        "Sensibilidad": df.loc["Clase 0", "Sensibilidad"],
                        "Especificidad": df.loc["Clase 0", "Especificidad"],
                        "Puntaje_F1": df.loc["Clase 0", "Puntaje_F1"],
                        "Exactitud": exactitud
                    }
                    
                    # Fila 2: Clase 1
                    fila_c1 = {
                        "Clasificador": nombres_clasificadores[codigo_clf],
                        "Metodo_Seleccion": parte_ranking,
                        "Clase": "Clase 1 (Ansiedad)",
                        "Precision": df.loc["Clase 1", "Precision"],
                        "Sensibilidad": df.loc["Clase 1", "Sensibilidad"],
                        "Especificidad": df.loc["Clase 1", "Especificidad"],
                        "Puntaje_F1": df.loc["Clase 1", "Puntaje_F1"],
                        "Exactitud": exactitud
                    }
                    
                    # Almacenamos temporalmente; si encontramos un modelo mejor en el bucle, sobrescribimos
                    mejores_filas_para_clasificador = [fila_c0, fila_c1]
                    mejor_nombre_ranking = parte_ranking
            except Exception as e:
                print(f"Error procesando {ruta_archivo}: {e}")

        if mejores_filas_para_clasificador:
            mejores_filas.extend(mejores_filas_para_clasificador)
            print(f"Mejor {codigo_clf}: {mejor_nombre_ranking} (Exactitud: {mejor_exactitud:.4f})")

    # Crear DataFrame final
    df_final = pd.DataFrame(mejores_filas)
    
    if df_final.empty:
        print("No se encontraron resultados.")
        return

    # Ordenar por Exactitud (descendente)
    # Importante: Como queremos agrupar Clase 0 y Clase 1, debemos ser cuidadosos al ordenar
    # Primero ordenamos por clasificador y exactitud, luego aplicamos un sort estable o reestructuramos
    
    # Estrategia: 
    # 1. Encontrar la exactitud de cada 'par' (modelo)
    # 2. Asignar esa exactitud a ambas filas
    # 3. Ordenar por esa exactitud + nombre clasificador + nombre clase
    
    df_final = df_final.sort_values(["Exactitud", "Clasificador", "Clase"], ascending=[False, True, True])

    # Convertir a cadenas para permitir el enmascaramiento con ""
    df_final["Exactitud"] = df_final["Exactitud"].astype(str)

    # Eliminar duplicados de Clasificador, Metodo_Seleccion y Exactitud para mejor lectura
    # Al ordenar, Clase 0 vendrá antes que Clase 1 para el mismo modelo
    
    clave_actual = None
    
    # Restablecer índices para iterar con seguridad
    df_final = df_final.reset_index(drop=True)
    
    # Usaremos una lista para guardar los valores modificados para evitar SettingWithCopy warning si fuera un slice
    for indice in range(len(df_final)):
        # Calcular clave de comparación (Clasificador + Metodo)
        clasif_actual = df_final.at[indice, 'Clasificador']
        metodo_actual = df_final.at[indice, 'Metodo_Seleccion']
        
        # OJO: Si acabamos de procesar Clase 0 de Modelo X, la siguiente fila es Clase 1 de Modelo X.
        # Queremos que la segunda fila (Clase 1) tenga el campo vacio.
        
        clave_fila = (clasif_actual, metodo_actual)
        
        if clave_fila == clave_actual:
            # Enmascarar los valores duplicados (dejar vacio)
            df_final.at[indice, 'Clasificador'] = ""
            df_final.at[indice, 'Metodo_Seleccion'] = ""
            df_final.at[indice, 'Exactitud'] = "" 
        else:
            # Nueva entrada (probablemente Clase 0 de un nuevo modelo)
            clave_actual = clave_fila
            
    # Guardar en un solo CSV
    nombre_archivo_salida = "Resumen_Mejores_Modelos_Top30_Apilado.csv"
    ruta_salida = os.path.join(dir_resultados, nombre_archivo_salida)
    df_final.to_csv(ruta_salida, index=False)
    
    print(f"\nArchivo consolidado guardado en: {ruta_salida}")
    print(df_final.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    principal()
