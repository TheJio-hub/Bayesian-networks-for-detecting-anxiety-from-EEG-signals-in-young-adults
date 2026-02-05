import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def normalizar_y_graficar_densidad():
    # Rutas
    input_file = os.path.join('Resultados', 'datos_bandas.parquet')
    output_parquet = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    output_dir = os.path.join('Resultados', 'Análisis espectral (normalizado)')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(input_file):
        print(f"Error: No se encuentra {input_file}")
        return

    print(f"Cargando {input_file}...")
    df = pd.read_parquet(input_file)
    
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo'] 
    cols_meta_presentes = [c for c in cols_meta if c in df.columns]
    cols_features = [c for c in df.columns if c not in cols_meta_presentes and pd.api.types.is_numeric_dtype(df[c])]

    print(f"Calculando Z-score para {len(cols_features)} características...")
    
    # Normalización Z-score por sujeto
    def zscore(x):
        if x.std() == 0: return x - x.mean()
        return (x - x.mean()) / x.std()

    # Copia para normalizar
    df_norm = df.copy()
    
    # Aplicar transformación
    df_norm[cols_features] = df_norm.groupby('Sujeto')[cols_features].transform(zscore)
    
    # Asegurar columna Grupo
    if 'Puntaje' in df_norm.columns:
        # Filtrar datos según criterio SAM (0 vs >=5)
        df_norm = df_norm[ (df_norm['Puntaje'] == 0) | (df_norm['Puntaje'] >= 5) ].copy()
        df_norm['Grupo'] = df_norm['Puntaje'].apply(lambda x: 'Relajacion' if x == 0 else 'Ansiedad')
    
    print(f"Guardando datos normalizados en {output_parquet}...")
    df_norm.to_parquet(output_parquet)

    # --- GRAFICACIÓN ---
    canales_referencia = ['A1', 'A2', 'M1', 'M2', 'REF', 'Ref', 'EXG1', 'EXG2']
    
    feature_cols_graficar = []
    for col in cols_features:
        parts = col.split('_')
        if len(parts) >= 2:
            canal = parts[0]
            if canal not in canales_referencia:
                feature_cols_graficar.append(col)
            
    print(f"Generando gráficos de densidad (Normalizados) para {len(feature_cols_graficar)} características...")
    
    # Configurar estilo
    sns.set_theme(style="whitegrid")
    
    for i, col in enumerate(feature_cols_graficar):
        if i % 20 == 0:
            print(f"Procesando gráfico {i}/{len(feature_cols_graficar)}: {col}")
            
        plt.figure(figsize=(10, 6))
        
        # Plot de densidad (KDE) - Normalizado
        sns.kdeplot(
            data=df_norm, 
            x=col, 
            hue='Grupo', 
            fill=True, 
            common_norm=False, 
            palette={'Relajacion': 'blue', 'Ansiedad': 'red'},
            alpha=0.3, # Transparencia
            linewidth=2,
            clip=(-4, 4) # Clip para centrar la visualización en Z-scores relevantes
        )
        
        parts = col.split('_')
        canal = parts[0]
        banda = parts[1]
        
        plt.title(f'Distribución Normalizada (Z-score): {canal} - {banda}\n(Relajacion vs Ansiedad)')
        plt.xlabel('Potencia Normalizada (Desv. Estándar)')
        plt.ylabel('Densidad')
        
        filename = os.path.join(output_dir, f"Densidad_Norm_{canal}_{banda}.png")
        plt.savefig(filename)
        plt.close() 
        
    print("¡Proceso de normalización y graficación de densidad completado!")

if __name__ == "__main__":
    normalizar_y_graficar_densidad()
