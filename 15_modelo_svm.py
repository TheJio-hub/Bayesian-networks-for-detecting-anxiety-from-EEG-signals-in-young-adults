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
CARACTERISTICAS_ENTRADA = os.path.join('Resultados', 'Ranking', 'mRMR_Global.csv')
DIRECTORIO_SALIDA = os.path.join('Resultados', 'Modelos', 'SVM')
TOP_K_CARACTERISTICAS = 10  # Usaremos las 10 mejores características de mRMR

def main():
    if not os.path.exists(DIRECTORIO_SALIDA):
        os.makedirs(DIRECTORIO_SALIDA)
        
    print(f"[{os.path.basename(__file__)}] Cargando datos...")
    df = pd.read_parquet(DATOS_ENTRADA)
    
    # Filtrar clases extremas (Relajación=0, Ansiedad>=5)
    df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
    y = df['Puntaje'].apply(lambda x: 1 if x >= 5 else 0).values
    grupos = df['Sujeto'].values
    
    # Cargar características seleccionadas por mRMR
    print(f"Cargando mejores características de {CARACTERISTICAS_ENTRADA}...")
    try:
        df_caracteristicas = pd.read_csv(CARACTERISTICAS_ENTRADA)
        # Asumiendo que el CSV tiene una columna 'Caracteristica'
        if 'Caracteristica' in df_caracteristicas.columns:
            caracteristicas_seleccionadas = df_caracteristicas['Caracteristica'].head(TOP_K_CARACTERISTICAS).tolist()
        else:
            # Si el nombre es distinto, tomamos la primera columna
            caracteristicas_seleccionadas = df_caracteristicas.iloc[:TOP_K_CARACTERISTICAS, 0].tolist()
            
        print(f"--- Características Seleccionadas (Top {TOP_K_CARACTERISTICAS} mRMR) ---")
        for i, f in enumerate(caracteristicas_seleccionadas, 1):
            print(f"{i}. {f}")
            
    except Exception as e:
        print(f"Error cargando archivo de características: {e}")
        return

    X = df[caracteristicas_seleccionadas].values
    
    # SVM requiere escalado de características
    escalador = StandardScaler()
    X = escalador.fit_transform(X)

    # Validación Leave-One-Group-Out (LOGO)
    logo = LeaveOneGroupOut()
    y_real_todos = []
    y_pred_todos = []
    
    print(f"\nEntrenando SVM (Kernel RBF, C=1.0) con validación LOGO...")
    
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

    # Métricas Globales
    exactitud = accuracy_score(y_real_todos, y_pred_todos)
    f1 = f1_score(y_real_todos, y_pred_todos)
    
    print("\n" + "="*40)
    print(f"RESULTADOS SVM (Top {TOP_K_CARACTERISTICAS} Características)")
    print("="*40)
    print(f"Exactitud (Accuracy) Global: {exactitud:.4f}")
    print(f"F1-Score Ansiedad: {f1:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_real_todos, y_pred_todos, target_names=['Relajación', 'Ansiedad']))
    
    # Matriz de Confusión
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
    print(f"Gráfica guardada en {DIRECTORIO_SALIDA}")

if __name__ == "__main__":
    main()
