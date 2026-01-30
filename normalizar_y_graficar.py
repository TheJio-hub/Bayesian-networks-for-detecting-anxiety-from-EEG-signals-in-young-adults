import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

def normalizar_y_graficar():
    input_file = 'datos_bandas.parquet'
    output_dir = 'Resultados/Analisis espectral normalizado'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Cargando {input_file}...")
    df = pd.read_parquet(input_file)
    
    # Identificar columnas de características (las que tienen banda de frecuencia)
    cols_meta = ['Sujeto', 'Tarea', 'Ensayo', 'Epoca', 'Puntaje']
    cols_features = [c for c in df.columns if c not in cols_meta]
    
    print("Aplicando Normalización Z-score por Sujeto...")
    # Esta operación puede ser lenta, optimizamos usando transform de pandas
    # Agrupamos por sujeto y normalizamos las características
    
    def zscore(x):
        return (x - x.mean()) / x.std()

    # Aplicamos z-score a todas las features agrupando por sujeto
    # Esto asegura que el "0" sea el promedio del sujeto, eliminando sesgos individuales
    df_norm = df.copy()
    df_norm[cols_features] = df.groupby('Sujeto')[cols_features].transform(zscore)
    
    # Definir grupos nuevamente
    df_norm['Grupo'] = df_norm['Puntaje'].apply(lambda x: 'Relax' if x == 0 else 'Estres')
    
    # Guardar dataset normalizado (útil para la Red Bayesiana después)
    output_parquet = 'datos_bandas_normalizados.parquet'
    output_csv = 'datos_bandas_normalizados.csv'
    
    print(f"Guardando dataset normalizado en {output_parquet}...")
    df_norm.to_parquet(output_parquet)
    
    print(f"Guardando dataset normalizado en {output_csv}...")
    df_norm.to_csv(output_csv, index=False)
    
    # --- Graficar ---
    print("Generando gráficos de densidad normalizados...")
    
    # Filtrar características de interés (Frontales y Temporales, Alpha y Beta)
    bandas_interes = ['Alpha', 'Beta']
    canales_feature = []
    
    for col in cols_features:
        parts = col.split('_')
        if len(parts) < 2: continue
        canal = parts[0]
        banda = parts[1]
        
        if banda in bandas_interes:
            # Filtro simple: Empieza con F (Frontal) o T (Temporal)
            # Incluye Fp1, FC1, FT9, etc.
            if canal.startswith('F') or canal.startswith('T'):
                canales_feature.append(col)
    
    sns.set_theme(style="whitegrid")
    
    for i, col in enumerate(canales_feature):
        if i % 5 == 0:
            print(f"Graficando {i}/{len(canales_feature)}: {col}")
            
        plt.figure(figsize=(10, 6))
        
        # Plot de densidad
        sns.kdeplot(
            data=df_norm, 
            x=col, 
            hue='Grupo', 
            fill=True, 
            common_norm=False, 
            palette={'Relax': 'blue', 'Estres': 'red'},
            alpha=0.3,
            linewidth=2,
            clip=(-3, 3) # Limitar visualización a +/- 3 desviaciones estándar para ver el centro
        )
        
        parts = col.split('_')
        canal = parts[0]
        banda = parts[1]
        
        plt.title(f'Distribución Normalizada (Z-score): {canal} - {banda}')
        plt.xlabel('Desviación Estándar respecto al promedio del sujeto')
        plt.ylabel('Densidad')
        
        filename = f"{output_dir}/Norm_{canal}_{banda}.png"
        plt.savefig(filename)
        plt.close()

    print("¡Proceso finalizado!")

if __name__ == "__main__":
    normalizar_y_graficar()
