import pandas as pd
import numpy as np
import os
import time
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def evaluar_importancia_arbol():
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
        # Crear vector objetivo y
        y = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
    else:
        print("Advertencia: No se pudo filtrar por puntaje.")
        return

    # Seleccionar todas las características numéricas de EEG
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    caracteristicas = [c for c in df.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df[c])]
    
    if not caracteristicas:
        print("Error: No se encontraron características numéricas.")
        return
        
    X = df[caracteristicas]
    print(f"Entrenando Árbol de Decisión con {X.shape[1]} características y {X.shape[0]} muestras...")

    # Configurar y entrenar el Árbol de Decisión
    # random_state=42 para reproducibilidad
    # clase_weight='balanced' es útil si hay desbalance entre Relax y Ansiedad
    arbol = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    arbol.fit(X, y)
    
    # Obtener importancias
    importancias = arbol.feature_importances_
    
    # Crear DataFrame con los resultados
    df_importancia = pd.DataFrame({
        'Caracteristica': caracteristicas,
        'Importancia': importancias
    })
    
    # Ordenar por importancia descendente
    df_importancia = df_importancia.sort_values(by='Importancia', ascending=False).reset_index(drop=True)
    df_importancia['Ranking'] = df_importancia.index + 1
    
    # Guardar CSV
    archivo_salida = os.path.join(directorio_salida, 'Importancia_ArbolDecision.csv')
    df_importancia.to_csv(archivo_salida, index=False)
    
    print("\n--- Top 15 Características más influyentes (Árbol de Decisión) ---")
    print(df_importancia.head(15))
    
    # --- ANÁLISIS AGREGADO POR BANDA ---
    print("\n--- Importancia Acumulada por Banda ---")
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    resumen_bandas = []
    
    for banda in bandas:
        # Filtrar filas que terminen en _Banda
        filtro = df_importancia['Caracteristica'].str.endswith(f"_{banda}")
        importancia_total = df_importancia.loc[filtro, 'Importancia'].sum()
        resumen_bandas.append({'Banda': banda, 'Importancia_Total': importancia_total})
        
    df_resumen = pd.DataFrame(resumen_bandas).sort_values(by='Importancia_Total', ascending=False)
    print(df_resumen)
    
    # Guardar resumen
    archivo_resumen = os.path.join(directorio_salida, 'Importancia_ArbolDecision_ResumenBandas.csv')
    df_resumen.to_csv(archivo_resumen, index=False)

    # --- NUEVO: Visualizar Estructura del Árbol ---
    print("Generando imagen de la estructura del árbol...")
    plt.figure(figsize=(24, 12)) # Tamaño grande para visualizar bien
    plot_tree(arbol, 
              feature_names=caracteristicas,
              class_names=['Relajación', 'Ansiedad'],
              filled=True, 
              rounded=True, 
              max_depth=3, # Limitamos a 3 niveles para que sea legible
              fontsize=11)
    plt.title('Estructura del Árbol de Decisión (Primeros 3 niveles)')
    
    archivo_estructura = os.path.join(directorio_salida, 'Estructura_Arbol.png')
    plt.savefig(archivo_estructura, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Imagen de estructura guardada en: {archivo_estructura}")

    # Generar Gráfica de Pastel
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
    print(f"Gráfica guardada en: {archivo_grafica}")

    print(f"\nResultados guardados en: {directorio_salida}")

if __name__ == "__main__":
    evaluar_importancia_arbol()
