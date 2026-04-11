import pandas as pd
import os


def generar_matriz_comparativa_visual():
    archivo_global_maestro = os.path.join('Resultados', 'Análisis global', 'Ranking_Multicriterio_Completo.csv')
    dir_modelos = os.path.join('Resultados', 'Análisis por bandas', 'Modelos por banda')
    dir_salida = os.path.join('Resultados', 'Análisis por bandas', 'Comparativa')
    
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)

    if not os.path.exists(archivo_global_maestro):
        return

    df_global_maestro = pd.read_csv(archivo_global_maestro)

    configuracion_columnas = [
        ("Arbol_Global", "Importancia_DT", "global"),
        ("RF_Global",    "Importancia_RF", "global"),
        
        ("Arbol_Delta",  os.path.join(dir_modelos, "Importancia_Arbol_Delta.csv"), "archivo"),
        ("RF_Delta",     os.path.join(dir_modelos, "Importancia_RandomForest_Delta.csv"), "archivo"),
        
        ("Arbol_Theta",  os.path.join(dir_modelos, "Importancia_Arbol_Theta.csv"), "archivo"),
        ("RF_Theta",     os.path.join(dir_modelos, "Importancia_RandomForest_Theta.csv"), "archivo"),
        
        ("Arbol_Alpha",  os.path.join(dir_modelos, "Importancia_Arbol_Alpha.csv"), "archivo"),
        ("RF_Alpha",     os.path.join(dir_modelos, "Importancia_RandomForest_Alpha.csv"), "archivo"),
        
        ("Arbol_Beta",   os.path.join(dir_modelos, "Importancia_Arbol_Beta.csv"), "archivo"),
        ("RF_Beta",      os.path.join(dir_modelos, "Importancia_RandomForest_Beta.csv"), "archivo"),
        
        ("Arbol_Gamma",  os.path.join(dir_modelos, "Importancia_Arbol_Gamma.csv"), "archivo"),
        ("RF_Gamma",     os.path.join(dir_modelos, "Importancia_RandomForest_Gamma.csv"), "archivo")
    ]

    data_final = {}
    
    max_len = 0
    
    listas_features = {}
    
    for nombre_col, fuente, tipo_fuente in configuracion_columnas:
        if tipo_fuente == "global":
            df_temp = df_global_maestro[["Caracteristica", fuente]].copy()
            df_temp = df_temp.sort_values(fuente, ascending=False)
            features = df_temp['Caracteristica'].tolist()
        elif os.path.exists(fuente):
            df_temp = pd.read_csv(fuente)
            features = df_temp['Caracteristica'].tolist()
        else:
            features = []
        listas_features[nombre_col] = features
        if len(features) > max_len:
            max_len = len(features)

    for nombre_col in [cfg[0] for cfg in configuracion_columnas]:
        features = listas_features.get(nombre_col, [])
        padding = ["" for _ in range(max_len - len(features))]
        data_final[nombre_col] = features + padding
        
    df_comparativo = pd.DataFrame(data_final)
    
    df_comparativo.index = range(1, len(df_comparativo) + 1)
    df_comparativo.index.name = 'Ranking'
    
    archivo_salida = os.path.join(dir_salida, 'Matriz_Visual_Rankings.csv')
    df_comparativo.to_csv(archivo_salida)


if __name__ == "__main__":
    generar_matriz_comparativa_visual()
