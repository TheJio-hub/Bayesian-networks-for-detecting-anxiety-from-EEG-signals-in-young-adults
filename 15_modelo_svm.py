import pandas as pd
import numpy as np
import os
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


# Configuración
DATOS_ENTRADA = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
CARACTERISTICAS_ENTRADA = os.path.join('Resultados', 'Ranking', 'Ranking_mRMR.csv')
DIRECTORIO_SALIDA = os.path.join('Resultados', 'Modelos', 'SVM')
TOP_K_CARACTERISTICAS = 10

def main():
    if not os.path.exists(DIRECTORIO_SALIDA):
        os.makedirs(DIRECTORIO_SALIDA)
        
    df = pd.read_parquet(DATOS_ENTRADA)
    
    df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
    y = df['Puntaje'].apply(lambda x: 1 if x >= 5 else 0).values
    grupos = df['Sujeto'].values
    
    try:
        df_caracteristicas = pd.read_csv(CARACTERISTICAS_ENTRADA)
        if 'Caracteristica' in df_caracteristicas.columns:
            caracteristicas_seleccionadas = df_caracteristicas['Caracteristica'].head(TOP_K_CARACTERISTICAS).tolist()
        else:
            caracteristicas_seleccionadas = df_caracteristicas.iloc[:TOP_K_CARACTERISTICAS, 0].tolist()
            
    except Exception as e:
        return

    X = df[caracteristicas_seleccionadas].values
    
    escalador = StandardScaler()
    X = escalador.fit_transform(X)

    logo = LeaveOneGroupOut()
    y_real_todos = []
    y_pred_todos = []
    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    
    pliegue = 1
    for indice_entrenamiento, indice_prueba in logo.split(X, y, groups=grupos):
        X_entrenamiento, X_prueba = X[indice_entrenamiento], X[indice_prueba]
        y_entrenamiento, y_prueba = y[indice_entrenamiento], y[indice_prueba]
        
        svm.fit(X_entrenamiento, y_entrenamiento)
        y_pred = svm.predict(X_prueba)
        
        y_real_todos.extend(y_prueba)
        y_pred_todos.extend(y_pred)
        pliegue += 1

    exactitud = accuracy_score(y_real_todos, y_pred_todos)
    f1 = f1_score(y_real_todos, y_pred_todos)
    
    matriz_confusion = confusion_matrix(y_real_todos, y_pred_todos)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=['Relajación', 'Ansiedad'], 
                yticklabels=['Relajación', 'Ansiedad'])
    plt.title(f'Matriz de Confusión SVM (Top {TOP_K_CARACTERISTICAS} mRMR)\nExactitud: {exactitud:.2f}')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig(os.path.join(DIRECTORIO_SALIDA, 'Matriz_Confusion_SVM.png'))

if __name__ == "__main__":
    main()
