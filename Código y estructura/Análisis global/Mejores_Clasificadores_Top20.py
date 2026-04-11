import pandas as pd
import os
import glob

def principal():
    dir_base = "Resultados/Análisis global/Modelos (Usando Rankings)"
    dir_resultados = os.path.join(dir_base, "Top 20")
    
    # Comprobar si el directorio existe
    if not os.path.exists(dir_resultados):
        print(f"Directorio no encontrado: {dir_resultados}")
        return

    todos_los_archivos = glob.glob(os.path.join(dir_resultados, "*.csv"))
    # Filtrar solo archivos de Top 20 
    archivos_top20 = [f for f in todos_los_archivos if "_top20" in f and "Mejor_Resultado" not in f and "Resumen_Mejores" not in f]

    clasificadores = ["DT", "RF", "KNN", "SVM"]
    nombres_clasificadores = {
        "DT": "Árbol de Decisión",
        "RF": "Random Forest",
        "KNN": "K-Nearest Neighbors",
        "SVM": "Support Vector Machine"
    }

    mejores_filas = []

    print("Buscando los mejores modelos para Top 20 características (Apilado)...")

    for codigo_clf in clasificadores:
        mejor_exactitud = -1.0
        mejores_filas_para_clasificador = []
        mejor_nombre_ranking = None

        # Filtrar archivos para este clasificador
        archivos_clasificador = [f for f in archivos_top20 if f"Resultados_{codigo_clf}_" in os.path.basename(f)]

        for ruta_archivo in archivos_clasificador:
            try:
                df = pd.read_csv(ruta_archivo, index_col=0)
                try:
                    exactitud = float(df.loc["Clase 0", "Exactitud"])
                except:
                    exactitud = 0.0
                
                if exactitud > mejor_exactitud:
                    mejor_exactitud = exactitud
                    
                    nombre_archivo = os.path.basename(ruta_archivo)
                    parte_ranking = nombre_archivo.replace(f"Resultados_{codigo_clf}_Ranking_", "").replace("_top20.csv", "")
                    
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
                    
                    mejores_filas_para_clasificador = [fila_c0, fila_c1]
                    mejor_nombre_ranking = parte_ranking
            except Exception as e:
                print(f"Error procesando {ruta_archivo}: {e}")

        if mejores_filas_para_clasificador:
            mejores_filas.extend(mejores_filas_para_clasificador)
            print(f"Mejor {codigo_clf}: {mejor_nombre_ranking} (Exactitud: {mejor_exactitud:.4f})")

    df_final = pd.DataFrame(mejores_filas)
    
    if df_final.empty:
        print("No se encontraron resultados.")
        return

    df_final = df_final.sort_values(["Exactitud", "Clasificador", "Clase"], ascending=[False, True, True])
    df_final["Exactitud"] = df_final["Exactitud"].astype(str)

    clave_actual = None
    df_final = df_final.reset_index(drop=True)
    
    for indice in range(len(df_final)):
        clasif_actual = df_final.at[indice, 'Clasificador']
        metodo_actual = df_final.at[indice, 'Metodo_Seleccion']
        clave_fila = (clasif_actual, metodo_actual)
        
        if clave_fila == clave_actual:
            df_final.at[indice, 'Clasificador'] = ""
            df_final.at[indice, 'Metodo_Seleccion'] = ""
            df_final.at[indice, 'Exactitud'] = "" 
        else:
            clave_actual = clave_fila
            
    nombre_archivo_salida = "Resumen_Mejores_Modelos_Top20_Apilado.csv"
    ruta_salida = os.path.join(dir_resultados, nombre_archivo_salida)
    df_final.to_csv(ruta_salida, index=False)
    
    print(f"\nArchivo consolidado guardado en: {ruta_salida}")
    print(df_final.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    principal()
