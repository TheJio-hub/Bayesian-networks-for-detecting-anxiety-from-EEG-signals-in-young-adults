import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

def obtener_top_n_por_criterio(df_ranking, criterio, n=30):
    if criterio == 'mRMR':
        sub = df_ranking.dropna(subset=['mRMR_Rank']).sort_values('mRMR_Rank', ascending=True)
        return sub['Caracteristica'].tolist()[:n]
    elif criterio == 'Fisher':
        sub = df_ranking.sort_values('Fisher_Score', ascending=False)
        return sub['Caracteristica'].tolist()[:n]
    elif criterio == 'Mutual_Info':
        sub = df_ranking.sort_values('Mutual_Info', ascending=False)
        return sub['Caracteristica'].tolist()[:n]
    elif criterio == 'Random_Forest':
        sub = df_ranking.sort_values('Importancia_RF', ascending=False)
        return sub['Caracteristica'].tolist()[:n]
    return []

def entrenar_y_evaluar(X, y, grupos, modelo):
    validacion = LeaveOneGroupOut()
    y_real_total = []
    y_predicha_total = []
    
    for indice_entrenamiento, indice_prueba in validacion.split(X, y, grupos):
        X_entrenamiento, X_prueba = X.iloc[indice_entrenamiento], X.iloc[indice_prueba]
        y_entrenamiento, y_prueba = y[indice_entrenamiento], y[indice_prueba]
        
        modelo.fit(X_entrenamiento, y_entrenamiento)
        prediccion = modelo.predict(X_prueba)
        
        y_real_total.extend(y_prueba)
        y_predicha_total.extend(prediccion)
        
    precision, sensibilidad, f1, _ = precision_recall_fscore_support(y_real_total, y_predicha_total, labels=[0, 1], zero_division=0)
    exactitud = accuracy_score(y_real_total, y_predicha_total)
    tn, fp, fn, tp = confusion_matrix(y_real_total, y_predicha_total, labels=[0, 1]).ravel()
    especificidad = tn / (tn + fp) if (tn + fp) else 0.0
    
    return pd.DataFrame([
        {'Precision': precision[0], 'Sensibilidad': sensibilidad[0], 'Especificidad': especificidad, 'Puntaje_F1': f1[0], 'Exactitud': exactitud},
        {'Precision': precision[1], 'Sensibilidad': sensibilidad[1], 'Especificidad': especificidad, 'Puntaje_F1': f1[1], 'Exactitud': exactitud}
    ], index=['Clase 0', 'Clase 1'])

def principal():
    archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Ranking_Multicriterio_Completo.csv')
    archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    dir_salida = os.path.join('Resultados', 'Análisis global', 'Modelos generados')
    
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)
        
    if not os.path.exists(archivo_datos) or not os.path.exists(archivo_ranking):
        return

    df_ranking = pd.read_csv(archivo_ranking)
    df_datos = pd.read_parquet(archivo_datos)
    
    df_datos = df_datos[ (df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 5) ].copy()
    y = df_datos['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
    grupos = df_datos['Sujeto'].values
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    nombre_modelo = 'RF'
    
    fuentes_ranking = ['Fisher', 'Mutual_Info', 'mRMR', 'Random_Forest']
    top_configuraciones = [(15, 'Top 15'), (20, 'Top 20'), (30, 'Top 30')]

    for n_caracteristicas, nombre_carpeta in top_configuraciones:
        directorio_top = os.path.join(dir_salida, nombre_carpeta)
        os.makedirs(directorio_top, exist_ok=True)

        for criterio in fuentes_ranking:
            mejores_caracteristicas = obtener_top_n_por_criterio(df_ranking, criterio, n=n_caracteristicas)
            
            X_subconjunto = df_datos[mejores_caracteristicas]
            df_resultados = entrenar_y_evaluar(X_subconjunto, y, grupos, modelo)
            
            nombre_archivo = f"Resultados_{nombre_modelo}_Ranking_{criterio}_top{n_caracteristicas}.csv"
            archivo_salida = os.path.join(directorio_top, nombre_archivo)
            df_resultados.to_csv(archivo_salida)
            print(f"Guardado: {archivo_salida}")

if __name__ == "__main__":
    principal()
