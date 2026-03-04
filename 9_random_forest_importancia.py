import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def evaluar_importancia_random_forest():
    archivo_entrada = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    directorio_salida = os.path.join('Resultados', 'Ranking')
    
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        
    if not os.path.exists(archivo_entrada):
        archivo_entrada = os.path.join('Resultados', 'datos_bandas.parquet')
        if not os.path.exists(archivo_entrada):
            return

    df = pd.read_parquet(archivo_entrada)
    
    if 'Puntaje' in df.columns:
        df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
        y = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
        grupos = df['Sujeto'].values
    else:
        return

    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    caracteristicas = [c for c in df.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df[c])]
    
    if not caracteristicas:
        return
        
    X = df[caracteristicas]

    bosque = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

    logo = LeaveOneGroupOut()
    scoring = {
        'accuracy': 'accuracy',
        'precision_0': make_scorer(precision_score, pos_label=0, zero_division=0),
        'recall_0':    make_scorer(recall_score, pos_label=0, zero_division=0),
        'f1_0':        make_scorer(f1_score, pos_label=0, zero_division=0),
        'precision_1': make_scorer(precision_score, pos_label=1, zero_division=0),
        'recall_1':    make_scorer(recall_score, pos_label=1, zero_division=0),
        'f1_1':        make_scorer(f1_score, pos_label=1, zero_division=0)
    }
    scores = cross_validate(bosque, X, y, cv=logo, scoring=scoring, groups=grupos)
    
    metricas = {
        'Modelo': ['Random_Forest_Global'],
        'Accuracy_Mean': [np.mean(scores['test_accuracy'])],
        'Accuracy_Std': [np.std(scores['test_accuracy'])],
        'Precision_Relax_0': [np.mean(scores['test_precision_0'])],
        'Recall_Relax_0': [np.mean(scores['test_recall_0'])],
        'F1_Relax_0': [np.mean(scores['test_f1_0'])],
        'Precision_Ansiedad_1': [np.mean(scores['test_precision_1'])],
        'Recall_Ansiedad_1': [np.mean(scores['test_recall_1'])],
        'F1_Ansiedad_1': [np.mean(scores['test_f1_1'])]
    }
    
    df_metricas = pd.DataFrame(metricas)
    
    archivo_metricas = os.path.join(directorio_salida, 'Metricas_Desempeno_RandomForest.csv')
    df_metricas.to_csv(archivo_metricas, index=False)
    
    # Exportar también como Metricas_RF.csv
    df_metricas.to_csv(os.path.join(directorio_salida, 'Metricas_RF.csv'), index=False)

    bosque.fit(X, y)
    
    importancias = bosque.feature_importances_
    
    df_importancia = pd.DataFrame({
        'Caracteristica': caracteristicas,
        'Importancia': importancias
    })
    
    df_importancia = df_importancia.sort_values(by='Importancia', ascending=False).reset_index(drop=True)
    df_importancia['Ranking'] = df_importancia.index + 1
    
    archivo_salida = os.path.join(directorio_salida, 'Importancia_RandomForest.csv')
    df_importancia.to_csv(archivo_salida, index=False)
    
    print("\n--- Top 15 Características más influyentes (Random Forest) ---")
    print(df_importancia.head(15))
    
    print("\n--- Importancia Acumulada por Banda ---")
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    resumen_bandas = []
    
    for banda in bandas:
        filtro = df_importancia['Caracteristica'].str.endswith(f"_{banda}")
        importancia_total = df_importancia.loc[filtro, 'Importancia'].sum()
        resumen_bandas.append({'Banda': banda, 'Importancia_Total': importancia_total})
        
    df_resumen = pd.DataFrame(resumen_bandas).sort_values(by='Importancia_Total', ascending=False)
    
    archivo_resumen = os.path.join(directorio_salida, 'Importancia_RandomForest_ResumenBandas.csv')
    df_resumen.to_csv(archivo_resumen, index=False)

    un_arbol = bosque.estimators_[0]
    
    plt.figure(figsize=(24, 12))
    plot_tree(un_arbol, 
              feature_names=caracteristicas,
              class_names=['Relajación', 'Ansiedad'],
              filled=True, 
              rounded=True, 
              max_depth=3,
              fontsize=11)
    plt.title(f'Estructura del Árbol #0 (Ejemplo aleatorio 1 de {bosque.n_estimators} árboles)')
    
    archivo_estructura = os.path.join(directorio_salida, 'Estructura_RandomForest_Tree0.png')
    plt.savefig(archivo_estructura, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 8))
    colores = plt.cm.Set3(np.linspace(0, 1, len(df_resumen)))
    
    explode = [0.05 if i==0 else 0 for i in range(len(df_resumen))]

    
    plt.pie(df_resumen['Importancia_Total'], 
            labels=df_resumen['Banda'], 
            autopct='%1.1f%%', 
            startangle=140,
            colors=colores,
            explode=explode,
            shadow=True)
            
    plt.title('Importancia Relativa de cada Banda')
    
    archivo_grafica = os.path.join(directorio_salida, 'Distribucion_Importancia_Bandas_random_forest.png')
    plt.savefig(archivo_grafica)
    plt.close() 

