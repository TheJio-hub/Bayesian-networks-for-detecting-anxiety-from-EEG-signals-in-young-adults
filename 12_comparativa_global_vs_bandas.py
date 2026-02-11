import pandas as pd
import numpy as np
import os

def comparar_global_vs_bandas():
    # Rutas Base
    dir_ranking_global = os.path.join('Resultados', 'Ranking')
    dir_modelos_bandas = os.path.join('Resultados', 'Modelos')
    dir_salida = os.path.join('Resultados', 'Modelos', 'Comparativa')
    
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)

    # Modelos a comparar
    modelos = [
        ('ArbolDecision', 'Importancia_ArbolDecision.csv', 'Importancia_Arbol'),
        ('RandomForest', 'Importancia_RandomForest.csv', 'Importancia_RandomForest')
    ]
    
    comparativa_total = []

    for nombre_modelo, archivo_global, prefijo_banda in modelos:
        print(f"\n--- Analizando: {nombre_modelo} ---")
        
        # 1. Cargar Ranking Global
        path_global = os.path.join(dir_ranking_global, archivo_global)
        if not os.path.exists(path_global):
            print(f"Advertencia: No se encontró {path_global}")
            continue
            
        df_global = pd.read_csv(path_global)
        
        # Nos aseguramos de procesar las características
        # Asumimos que el nombre es "Canal_Banda" (ej. F3_Beta)
        
        # Cache para no recargar archivos de bandas mil veces
        cache_bandas = {} 
        
        for idx, row in df_global.iterrows():
            feature = row['Caracteristica']
            importancia_global = row['Importancia']
            ranking_global = row['Ranking'] if 'Ranking' in row else idx + 1
            
            # Extraer Banda del nombre (ej. F3_Beta -> Beta)
            try:
                # Asume formato Canal_Banda. Si hay guiones bajos extra, tomamos el último.
                partes = feature.split('_')
                if len(partes) < 2:
                    continue
                banda = partes[-1]
                canal = "_".join(partes[:-1])
            except:
                continue
                
            # Verificar si es una banda válida
            if banda not in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                continue
                
            # 2. Cargar Ranking Específico de esa Banda
            if banda not in cache_bandas:
                archivo_banda = f"{prefijo_banda}_{banda}.csv"
                path_banda = os.path.join(dir_modelos_bandas, archivo_banda)
                if os.path.exists(path_banda):
                    df_b = pd.read_csv(path_banda)
                    # Añadir ranking si no existe
                    if 'Ranking' not in df_b.columns:
                        df_b['Ranking'] = df_b.index + 1
                    cache_bandas[banda] = df_b
                else:
                    cache_bandas[banda] = None
            
            df_banda_especifica = cache_bandas[banda]
            
            rank_local = None
            imp_local = None
            
            if df_banda_especifica is not None:
                # Buscar la característica en el archivo de la banda
                match = df_banda_especifica[df_banda_especifica['Caracteristica'] == feature]
                if not match.empty:
                    rank_local = match.iloc[0]['Ranking']
                    imp_local = match.iloc[0]['Importancia']
            
            # Guardar comparación
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

    # Crear DataFrame
    df_comparativa = pd.DataFrame(comparativa_total)
    
    # Calcular Delta de Ranking (Opcional: qué tanto cambia su posición)
    # Si Rank Global es 5 y Rank Local es 1, significa que es la reina de su banda pero compite con otras bandas fuera
    df_comparativa['Diferencia_Ranking'] = df_comparativa['Ranking_Global'] - df_comparativa['Ranking_IntraBanda']
    
    # Guardar CSV final
    archivo_final = os.path.join(dir_salida, 'Comparativa_Global_vs_IntraBanda.csv')
    df_comparativa.to_csv(archivo_final, index=False)
    
    print(f"\nComparativa guardada exitosamente en: {archivo_final}")
    print("\nPrimeras 10 filas del análisis comparativo:")
    print(df_comparativa.head(10))

if __name__ == "__main__":
    comparar_global_vs_bandas()
