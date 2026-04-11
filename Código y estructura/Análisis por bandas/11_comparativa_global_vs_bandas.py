import pandas as pd
import numpy as np
import os


def comparar_global_vs_bandas():
    archivo_global_maestro = os.path.join('Resultados', 'Análisis global', 'Ranking_Multicriterio_Completo.csv')
    dir_modelos_bandas = os.path.join('Resultados', 'Análisis por bandas', 'Modelos por banda')
    dir_salida = os.path.join('Resultados', 'Análisis por bandas', 'Comparativa')
    
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)

    modelos = [
        ('ArbolDecision', 'Importancia_Arbol'),
        ('RandomForest', 'Importancia_RandomForest')
    ]
    
    comparativa_total = []

    if not os.path.exists(archivo_global_maestro):
        return

    df_global_maestro = pd.read_csv(archivo_global_maestro)

    for nombre_modelo, prefijo_banda in modelos:
        columna_importancia_global = 'Importancia_DT' if nombre_modelo == 'ArbolDecision' else 'Importancia_RF'
        df_global = df_global_maestro[["Caracteristica", columna_importancia_global]].copy()
        df_global = df_global.sort_values(columna_importancia_global, ascending=False).reset_index(drop=True)
        df_global['Ranking'] = df_global.index + 1
        
        cache_bandas = {} 
        
        for idx, row in df_global.iterrows():
            feature = row['Caracteristica']
            importancia_global = row[columna_importancia_global]
            ranking_global = row['Ranking']
            
            try:
                partes = feature.split('_')
                if len(partes) < 2:
                    continue
                banda = partes[-1]
                canal = "_".join(partes[:-1])
            except:
                continue
                
            if banda not in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                continue
                
            if banda not in cache_bandas:
                archivo_banda = f"{prefijo_banda}_{banda}.csv"
                path_banda = os.path.join(dir_modelos_bandas, archivo_banda)
                if os.path.exists(path_banda):
                    df_b = pd.read_csv(path_banda)
                    if 'Ranking' not in df_b.columns:
                        df_b['Ranking'] = df_b.index + 1
                    cache_bandas[banda] = df_b
                else:
                    cache_bandas[banda] = None
            
            df_banda_especifica = cache_bandas[banda]
            
            rank_local = None
            imp_local = None
            
            if df_banda_especifica is not None:
                match = df_banda_especifica[df_banda_especifica['Caracteristica'] == feature]
                if not match.empty:
                    rank_local = match.iloc[0]['Ranking']
                    imp_local = match.iloc[0]['Importancia']
            
            comparativa_total.append({
                'Modelo': nombre_modelo,
                'Caracteristica': feature,
                'Banda': banda,
                'Canal': canal,
                'Ranking_Global': ranking_global,
                'Importancia_Global': importancia_global,
                'Ranking_IntraBanda': rank_local,
                'Importancia_IntraBanda': imp_local
            })

    df_comparativa = pd.DataFrame(comparativa_total)
    
    df_comparativa['Diferencia_Ranking'] = df_comparativa['Ranking_Global'] - df_comparativa['Ranking_IntraBanda']
    
    archivo_final = os.path.join(dir_salida, 'Comparativa_Global_vs_IntraBanda.csv')
    df_comparativa.to_csv(archivo_final, index=False)


if __name__ == "__main__":
    comparar_global_vs_bandas()
