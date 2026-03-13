import pandas as pd
import os


def generar_matriz_comparativa_visual():
    dir_ranking = os.path.join('Resultados', 'Ranking')
    dir_modelos = os.path.join('Resultados', 'Modelos')
    dir_salida = os.path.join('Resultados', 'Modelos', 'Comparativa')
    
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)

    configuracion_columnas = [
        ("Arbol_Global", os.path.join(dir_ranking, "Importancia_ArbolDecision.csv")),
        ("RF_Global",    os.path.join(dir_ranking, "Importancia_RandomForest.csv")),
        
        ("Arbol_Delta",  os.path.join(dir_modelos, "Importancia_Arbol_Delta.csv")),
        ("RF_Delta",     os.path.join(dir_modelos, "Importancia_RandomForest_Delta.csv")),
        
        ("Arbol_Theta",  os.path.join(dir_modelos, "Importancia_Arbol_Theta.csv")),
        ("RF_Theta",     os.path.join(dir_modelos, "Importancia_RandomForest_Theta.csv")),
        
        ("Arbol_Alpha",  os.path.join(dir_modelos, "Importancia_Arbol_Alpha.csv")),
        ("RF_Alpha",     os.path.join(dir_modelos, "Importancia_RandomForest_Alpha.csv")),
        
        ("Arbol_Beta",   os.path.join(dir_modelos, "Importancia_Arbol_Beta.csv")),
        ("RF_Beta",      os.path.join(dir_modelos, "Importancia_RandomForest_Beta.csv")),
        
        ("Arbol_Gamma",  os.path.join(dir_modelos, "Importancia_Arbol_Gamma.csv")),
        ("RF_Gamma",     os.path.join(dir_modelos, "Importancia_RandomForest_Gamma.csv"))
    ]

    data_final = {}
    
    max_len = 0
    
    listas_features = {}
    
    for nombre_col, ruta_csv in configuracion_columnas:
        if os.path.exists(ruta_csv):
            df_temp = pd.read_csv(ruta_csv)
            features = df_temp['Caracteristica'].tolist()
            listas_features[nombre_col] = features
            if len(features) > max_len:
                max_len = len(features)
        else:
            listas_features[nombre_col] = []

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
