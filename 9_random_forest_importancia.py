import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def evaluar_importancia_random_forest():
    # Rutas
    archivo_entrada = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    directorio_salida = os.path.join('Resultados', 'Ranking')
    
    # Asegurar que existe el directorio de salida
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        
    # Verificar entrada
    if not os.path.exists(archivo_entrada):
        archivo_entrada = os.path.join('Resultados', 'datos_bandas.parquet')
        if not os.path.exists(archivo_entrada):
            print("Error: No se encontró el archivo de datos.")
            return

    print(f"Cargando datos: {archivo_entrada}")
    df = pd.read_parquet(archivo_entrada)
    
    # Filtrar Clases: Relajación (0) vs Ansiedad (>=5)
    print("Filtrando: Relajación (0) vs Ansiedad (>=5)...")
    if 'Puntaje' in df.columns:
        df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
        y = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
        # Obtener grupos (Sujetos) para validación correcta
        grupos = df['Sujeto'].values
    else:
        print("Advertencia: No se pudo filtrar por puntaje.")
        return

    # Seleccionar características numéricas
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    caracteristicas = [c for c in df.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df[c])]
    
    if not caracteristicas:
        print("Error: No se encontraron características numéricas.")
        return
        
    X = df[caracteristicas]
    print(f"Entrenando Random Forest con {X.shape[1]} características y {X.shape[0]} muestras...")
    print("   - Esto puede tomar unos segundos...")

    # Configurar Random Forest
    bosque = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

    # --- EVALUACIÓN DE DESEMPEÑO (Cross-Validation GroupKFold) ---
    print("   - Calculando métricas de desempeño (Group K-Fold por Sujeto)...")
    # Evitar fuga de información entre épocas del mismo sujeto
    cv = GroupKFold(n_splits=5)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    scores = cross_validate(bosque, X, y, cv=cv, scoring=scoring, groups=grupos)
    
    # Promediar métricas
    metricas = {
        'Modelo': ['Random_Forest_Global'],
        'Accuracy_Mean': [np.mean(scores['test_accuracy'])],
        'Accuracy_Std': [np.std(scores['test_accuracy'])],
        'Precision_Mean': [np.mean(scores['test_precision'])],
        'Recall_Mean': [np.mean(scores['test_recall'])],
        'F1_Mean': [np.mean(scores['test_f1'])]
    }
    
    df_metricas = pd.DataFrame(metricas)
    print("\n--- Métricas de Desempeño (Promedio CV) ---")
    print(df_metricas.T)
    
    # Guardar Métricas
    archivo_metricas = os.path.join(directorio_salida, 'Metricas_Desempeno_RandomForest.csv')
    df_metricas.to_csv(archivo_metricas, index=False)

    # Entrenar feature importance en todos los datos
    bosque.fit(X, y)
    
    # Obtener importancias
    importancias = bosque.feature_importances_
    
    # Crear DataFrame
    df_importancia = pd.DataFrame({
        'Caracteristica': caracteristicas,
        'Importancia': importancias
    })
    
    # Ordenar por importancia descendente
    df_importancia = df_importancia.sort_values(by='Importancia', ascending=False).reset_index(drop=True)
    df_importancia['Ranking'] = df_importancia.index + 1
    
    # Guardar CSV
    archivo_salida = os.path.join(directorio_salida, 'Importancia_RandomForest.csv')
    df_importancia.to_csv(archivo_salida, index=False)
    
    print("\n--- Top 15 Características más influyentes (Random Forest) ---")
    print(df_importancia.head(15))
    
    # --- ANÁLISIS AGREGADO POR BANDA ---
    print("\n--- Importancia Acumulada por Banda ---")
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    resumen_bandas = []
    
    for banda in bandas:
        # Filtrar filas que terminen en _Banda (asegura que capturamos todas)
        filtro = df_importancia['Caracteristica'].str.endswith(f"_{banda}")
        importancia_total = df_importancia.loc[filtro, 'Importancia'].sum()
        resumen_bandas.append({'Banda': banda, 'Importancia_Total': importancia_total})
        
    df_resumen = pd.DataFrame(resumen_bandas).sort_values(by='Importancia_Total', ascending=False)
    print(df_resumen)
    
    # Guardar resumen también
    archivo_resumen = os.path.join(directorio_salida, 'Importancia_RandomForest_ResumenBandas.csv')
    df_resumen.to_csv(archivo_resumen, index=False)

    # --- NUEVO: Visualizar un Árbol del Bosque ---
    print("Generando imagen de la estructura de un árbol del bosque...")
    # Extraer el primer árbol (estimador) del bosque para visualizar
    un_arbol = bosque.estimators_[0]
    
    plt.figure(figsize=(24, 12))
    plot_tree(un_arbol, 
              feature_names=caracteristicas,
              class_names=['Relajación', 'Ansiedad'],
              filled=True, 
              rounded=True, 
              max_depth=3, # Limitamos a 3 niveles
              fontsize=11)
    plt.title(f'Estructura del Árbol #0 (Ejemplo aleatorio 1 de {bosque.n_estimators} árboles)')
    
    archivo_estructura = os.path.join(directorio_salida, 'Estructura_RandomForest_Tree0.png')
    plt.savefig(archivo_estructura, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Imagen de estructura guardada en: {archivo_estructura}")

    # Generar Gráfica de Pastel
    plt.figure(figsize=(8, 8))
    # Usar mapa de colores agradable
    colores = plt.cm.Set3(np.linspace(0, 1, len(df_resumen)))
    
    # Destacar ligeramente la rebanada más grande (explode)
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
    plt.close() # Cerrar figura para liberar memoria
    print(f"Gráfica guardada en: {archivo_grafica}")

    print(f"\nResultados guardados en: {directorio_salida}")

if __name__ == "__main__":
    evaluar_importancia_random_forest()
