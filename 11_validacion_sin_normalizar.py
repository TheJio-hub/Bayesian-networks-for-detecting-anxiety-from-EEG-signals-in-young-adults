import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

def validacion_robusta():
    # Usamos el archivo SIN normalización global extra
    archivo_entrada = os.path.join('Resultados', 'datos_bandas.parquet')
    
    if not os.path.exists(archivo_entrada):
        print("Error: No se encontró 'datos_bandas.parquet'")
        return

    print(f"Cargando datos CRUDOS: {archivo_entrada}")
    df = pd.read_parquet(archivo_entrada)
    
    # Filtrar Clases
    df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
    y = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
    grupos = df['Sujeto'].values
    
    # Seleccionar características
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    X = df[[c for c in df.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df[c])]]
    
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
    print("Validación: GroupKFold (5 splits) sobre datos NO normalizados globalmente.")
    
    bosque = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    
    cv = GroupKFold(n_splits=5)
    scoring = {'accuracy': 'accuracy', 'f1': 'f1'}
    
    scores = cross_validate(bosque, X, y, cv=cv, scoring=scoring, groups=grupos)
    
    print("\n--- RESULTADOS REALISTAS (Sin Fuga de Datos) ---")
    print(f"Accuracy Promedio: {np.mean(scores['test_accuracy']):.4f} (+/- {np.std(scores['test_accuracy']):.4f})")
    print(f"F1-Score Promedio: {np.mean(scores['test_f1']):.4f}")

if __name__ == "__main__":
    validacion_robusta()
