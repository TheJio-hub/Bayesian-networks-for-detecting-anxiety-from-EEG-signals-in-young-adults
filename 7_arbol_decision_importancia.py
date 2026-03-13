import pandas as pd
import numpy as np
import os
import time
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def evaluar_importancia_arbol():
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

    arbol = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    
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
    scores = cross_validate(arbol, X, y, cv=logo, scoring=scoring, groups=grupos)
    
    metricas = {
        'Modelo': ['Arbol_Decision_Global'],
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
    
    archivo_metricas = os.path.join(directorio_salida, 'Metricas_Desempeno_ArbolDecision.csv')
    df_metricas.to_csv(archivo_metricas, index=False)
    
    # Exportar también como Metricas_Arbol.csv
    df_metricas.to_csv(os.path.join(directorio_salida, 'Metricas_Arbol.csv'), index=False)
    
    arbol.fit(X, y)
    
    importancias = arbol.feature_importances_
    
    df_importancia = pd.DataFrame({
        'Caracteristica': caracteristicas,
        'Importancia': importancias
    })
    
    df_importancia = df_importancia.sort_values(by='Importancia', ascending=False).reset_index(drop=True)
    df_importancia['Ranking'] = df_importancia.index + 1
    
    archivo_salida = os.path.join(directorio_salida, 'Importancia_ArbolDecision.csv')
    df_importancia.to_csv(archivo_salida, index=False)
    
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    resumen_bandas = []
    
    for banda in bandas:
        filtro = df_importancia['Caracteristica'].str.endswith(f"_{banda}")
        importancia_total = df_importancia.loc[filtro, 'Importancia'].sum()
        resumen_bandas.append({'Banda': banda, 'Importancia_Total': importancia_total})
        
    df_resumen = pd.DataFrame(resumen_bandas).sort_values(by='Importancia_Total', ascending=False)
    
    archivo_resumen = os.path.join(directorio_salida, 'Importancia_ArbolDecision_ResumenBandas.csv')
    df_resumen.to_csv(archivo_resumen, index=False)

    reglas = export_text(arbol, feature_names=caracteristicas, show_weights=True)
    archivo_reglas = os.path.join(directorio_salida, 'Reglas_Arbol_Completo.txt')
    with open(archivo_reglas, 'w') as f:
        f.write("Reglas de Decisión del Árbol Entrenado (Global):\n")
        f.write("=================================================\n")
        f.write(reglas)

    nodos_detalle = []
    tree = arbol.tree_
    for i in range(tree.node_count):
        es_hoja = (tree.children_left[i] == -1) and (tree.children_right[i] == -1)
        
        info = {
            'Nodo_ID': i,
            'Tipo': 'Hoja' if es_hoja else 'Decisión',
            'Profundidad': 'ND', 
            'Caracteristica_Usada': 'Ninguna' if es_hoja else caracteristicas[tree.feature[i]],
            'Umbral': 'ND' if es_hoja else f"{tree.threshold[i]:.4f}",
            'Impureza (Gini)': f"{tree.impurity[i]:.4f}",
            'Muestras': tree.n_node_samples[i],
            'Valor_Clase0_Relax': tree.value[i][0][0],
            'Valor_Clase1_Ansiedad': tree.value[i][0][1] if len(tree.value[i][0]) > 1 else 0
        }
        nodos_detalle.append(info)
    
    df_nodos = pd.DataFrame(nodos_detalle)
    archivo_nodos = os.path.join(directorio_salida, 'Nodos_Arbol_Detalle.csv')
    df_nodos.to_csv(archivo_nodos, index=False)

    plt.figure(figsize=(24, 12))
    plot_tree(arbol, 
              feature_names=caracteristicas,
              class_names=['Relajación', 'Ansiedad'],
              filled=True, 
              rounded=True, 
              max_depth=5,
              fontsize=10)
    plt.title('Estructura del Árbol de Decisión (Primeros 5 niveles)')
    
    archivo_estructura = os.path.join(directorio_salida, 'Estructura_Arbol.png')
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
            
    plt.title('Importancia Relativa de cada Banda (Árbol de Decisión)')
    
    archivo_grafica = os.path.join(directorio_salida, 'Distribucion_Importancia_Bandas_Arbol.png')
    plt.savefig(archivo_grafica)
    plt.close()

if __name__ == "__main__":
    evaluar_importancia_arbol()
