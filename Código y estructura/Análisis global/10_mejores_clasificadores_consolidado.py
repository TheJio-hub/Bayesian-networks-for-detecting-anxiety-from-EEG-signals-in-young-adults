import os
from glob import glob

import pandas as pd
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)


def generar_csv_mejores_clasificadores():
    """
    Genera un CSV consolidado con los mejores 4 clasificadores para:
    - Detectar Ansiedad (Clase 1)
    - Detectar No Ansiedad (Clase 0)
    
    Formato: Filas separadas para Entrenamiento y Validación
    """
    
    # Recopilar TODOS los datos
    todos_datos = []
    
    base_path = os.path.join('Resultados', 'Análisis global', 'Modelos generados')
    
    for csv_file in tqdm(sorted(glob(os.path.join(base_path, "**/Resultados_*.csv"), recursive=True)), 
                         desc='Leyendo CSVs', unit='archivo'):
        try:
            df = pd.read_csv(csv_file, index_col=0)
            filename = os.path.basename(csv_file)
            parts = filename.replace("Resultados_", "").replace(".csv", "").split("_Ranking_")
            clf = parts[0]
            resto = parts[1]
            
            for metodo in ['Fisher', 'Mutual_Info', 'mRMR', 'DT']:
                if resto.startswith(metodo):
                    top_str = resto.replace(metodo + "_top", "")
                    top = int(top_str)
                    
                    # Clase 1 (Ansiedad)
                    if 'Clase 1 (Validación)' in df.index and 'Clase 1 (Entrenamiento)' in df.index:
                        val_row_1 = df.loc['Clase 1 (Validación)']
                        train_row_1 = df.loc['Clase 1 (Entrenamiento)']
                        
                        # VALIDACIÓN - Clase 1
                        todos_datos.append({
                            'Clase': 'Ansiedad',
                            'Clasificador': clf,
                            'Método_Selección': metodo,
                            'Top': top,
                            'Conjunto': 'Validación',
                            'Precisión': float(val_row_1['Precision']),
                            'Sensibilidad': float(val_row_1['Sensibilidad']),
                            'Especificidad': float(val_row_1['Especificidad']),
                            'F1_Score': float(val_row_1['Puntaje_F1']),
                            'Exactitud': float(val_row_1['Exactitud']),
                        })
                        
                        # ENTRENAMIENTO - Clase 1
                        todos_datos.append({
                            'Clase': 'Ansiedad',
                            'Clasificador': clf,
                            'Método_Selección': metodo,
                            'Top': top,
                            'Conjunto': 'Entrenamiento',
                            'Precisión': float(train_row_1['Precision']),
                            'Sensibilidad': float(train_row_1['Sensibilidad']),
                            'Especificidad': float(train_row_1['Especificidad']),
                            'F1_Score': float(train_row_1['Puntaje_F1']),
                            'Exactitud': float(train_row_1['Exactitud']),
                        })
                    
                    # Clase 0 (No Ansiedad)
                    if 'Clase 0 (Validación)' in df.index and 'Clase 0 (Entrenamiento)' in df.index:
                        val_row_0 = df.loc['Clase 0 (Validación)']
                        train_row_0 = df.loc['Clase 0 (Entrenamiento)']
                        
                        # VALIDACIÓN - Clase 0
                        todos_datos.append({
                            'Clase': 'No_Ansiedad',
                            'Clasificador': clf,
                            'Método_Selección': metodo,
                            'Top': top,
                            'Conjunto': 'Validación',
                            'Precisión': float(val_row_0['Precision']),
                            'Sensibilidad': float(val_row_0['Sensibilidad']),
                            'Especificidad': float(val_row_0['Especificidad']),
                            'F1_Score': float(val_row_0['Puntaje_F1']),
                            'Exactitud': float(val_row_0['Exactitud']),
                        })
                        
                        # ENTRENAMIENTO - Clase 0
                        todos_datos.append({
                            'Clase': 'No_Ansiedad',
                            'Clasificador': clf,
                            'Método_Selección': metodo,
                            'Top': top,
                            'Conjunto': 'Entrenamiento',
                            'Precisión': float(train_row_0['Precision']),
                            'Sensibilidad': float(train_row_0['Sensibilidad']),
                            'Especificidad': float(train_row_0['Especificidad']),
                            'F1_Score': float(train_row_0['Puntaje_F1']),
                            'Exactitud': float(train_row_0['Exactitud']),
                        })
                    
                    break
        except Exception:
            pass
    
    df_todos = pd.DataFrame(todos_datos)
    
    # Separar por clase y tipo de conjunto
    df_ansiedad_val = df_todos[(df_todos['Clase'] == 'Ansiedad') & (df_todos['Conjunto'] == 'Validación')].copy()
    df_no_ansiedad_val = df_todos[(df_todos['Clase'] == 'No_Ansiedad') & (df_todos['Conjunto'] == 'Validación')].copy()
    
    # Top 4 mejores para cada clase (por sensibilidad en validación)
    top_4_ansiedad = df_ansiedad_val.nlargest(4, 'Sensibilidad')
    top_4_no_ansiedad = df_no_ansiedad_val.nlargest(4, 'Sensibilidad')
    
    # Obtener los IDs de estos modelos para recuperar también sus datos de entrenamiento
    indices_ansiedad = []
    for _, row in top_4_ansiedad.iterrows():
        clf_match = df_todos[
            (df_todos['Clasificador'] == row['Clasificador']) &
            (df_todos['Método_Selección'] == row['Método_Selección']) &
            (df_todos['Top'] == row['Top']) &
            (df_todos['Clase'] == 'Ansiedad')
        ]
        indices_ansiedad.extend(clf_match.index.tolist())
    
    indices_no_ansiedad = []
    for _, row in top_4_no_ansiedad.iterrows():
        clf_match = df_todos[
            (df_todos['Clasificador'] == row['Clasificador']) &
            (df_todos['Método_Selección'] == row['Método_Selección']) &
            (df_todos['Top'] == row['Top']) &
            (df_todos['Clase'] == 'No_Ansiedad')
        ]
        indices_no_ansiedad.extend(clf_match.index.tolist())
    
    # Combinar resultados (validación + entrenamiento para cada modelo)
    resultado_ansiedad = df_todos.loc[indices_ansiedad].sort_values(['Sensibilidad', 'Conjunto'], ascending=[False, False])
    resultado_no_ansiedad = df_todos.loc[indices_no_ansiedad].sort_values(['Sensibilidad', 'Conjunto'], ascending=[False, False])
    
    resultado_final = pd.concat([resultado_ansiedad, resultado_no_ansiedad], ignore_index=True)
    
    # Reordenar columnas
    columnas_ordenadas = [
        'Clase',
        'Clasificador',
        'Método_Selección',
        'Top',
        'Conjunto',
        'Sensibilidad',
        'Especificidad',
        'Precisión',
        'F1_Score',
        'Exactitud',
    ]
    
    resultado_final = resultado_final[columnas_ordenadas]
    
    # Guardar
    ruta_salida = os.path.join('Resultados', 'Análisis global', 'Mejores_Clasificadores_Consolidado.csv')
    resultado_final.to_csv(ruta_salida, index=False)
    
    # Mostrar resultados
    print("\n" + "=" * 120)
    print("✅ MEJORES CLASIFICADORES CONSOLIDADOS")
    print("=" * 120)
    print(f"\n📍 Archivo guardado en: {ruta_salida}\n")
    
    print("\n" + "=" * 120)
    print("🔴 TOP 4 PARA DETECTAR ANSIEDAD (Validación + Entrenamiento)")
    print("=" * 120)
    ansiedad_display = resultado_ansiedad[['Clasificador', 'Método_Selección', 'Top', 'Conjunto', 
                                           'Sensibilidad', 'Especificidad', 'Exactitud']].copy()
    print(ansiedad_display.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    print("\n" + "=" * 120)
    print("🟢 TOP 4 PARA DETECTAR NO ANSIEDAD (Validación + Entrenamiento)")
    print("=" * 120)
    no_ansiedad_display = resultado_no_ansiedad[['Clasificador', 'Método_Selección', 'Top', 'Conjunto', 
                                                  'Sensibilidad', 'Especificidad', 'Exactitud']].copy()
    print(no_ansiedad_display.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Estadísticas
    print("\n" + "=" * 120)
    print("📊 ESTADÍSTICAS")
    print("=" * 120)
    
    ans_val = resultado_ansiedad[resultado_ansiedad['Conjunto'] == 'Validación']
    ans_ent = resultado_ansiedad[resultado_ansiedad['Conjunto'] == 'Entrenamiento']
    
    print("\n🔴 ANSIEDAD:")
    print(f"   Validación   - Sensibilidad: {ans_val['Sensibilidad'].mean():.4f}, Exactitud: {ans_val['Exactitud'].mean():.4f}")
    print(f"   Entrenamiento - Sensibilidad: {ans_ent['Sensibilidad'].mean():.4f}, Exactitud: {ans_ent['Exactitud'].mean():.4f}")
    print(f"   Sobreajuste   - Δ Sensibilidad: {(ans_ent['Sensibilidad'].mean() - ans_val['Sensibilidad'].mean()):.4f}")
    
    no_ans_val = resultado_no_ansiedad[resultado_no_ansiedad['Conjunto'] == 'Validación']
    no_ans_ent = resultado_no_ansiedad[resultado_no_ansiedad['Conjunto'] == 'Entrenamiento']
    
    print("\n🟢 NO ANSIEDAD:")
    print(f"   Validación   - Sensibilidad: {no_ans_val['Sensibilidad'].mean():.4f}, Exactitud: {no_ans_val['Exactitud'].mean():.4f}")
    print(f"   Entrenamiento - Sensibilidad: {no_ans_ent['Sensibilidad'].mean():.4f}, Exactitud: {no_ans_ent['Exactitud'].mean():.4f}")
    print(f"   Sobreajuste   - Δ Sensibilidad: {(no_ans_ent['Sensibilidad'].mean() - no_ans_val['Sensibilidad'].mean()):.4f}")
    
    print("\n" + "=" * 120)


if __name__ == "__main__":
    generar_csv_mejores_clasificadores()
