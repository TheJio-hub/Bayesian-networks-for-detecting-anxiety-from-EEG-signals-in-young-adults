import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def diagnostico_fuga_datos():
    # CAMBIO: Leemos el archivo "crudo" para ver si el problema viene de origen
    archivo = os.path.join('Resultados', 'datos_bandas.parquet')
    
    if not os.path.exists(archivo):
        print("No se encuentra el archivo.")
        return

    df = pd.read_parquet(archivo)
    
    # Filtrar solo las clases que usamos
    df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
    
    # Vamos a analizar la característica "culpable": F3_Beta
    feature = 'F3_Beta'
    
    if feature not in df.columns:
        print(f"No se encuentra {feature}")
        return

    # Estadísticas por grupo
    grupo_relax = df[df['Puntaje'] == 0][feature]
    grupo_ansiedad = df[df['Puntaje'] >= 5][feature]
    
    print(f"--- Diagnóstico de la variable: {feature} ---")
    print("\nGrupo RELAJACIÓN (Clase 0):")
    print(f"   Media: {grupo_relax.mean():.6f}")
    print(f"   Std:   {grupo_relax.std():.6f}")
    print(f"   Min:   {grupo_relax.min():.6f}")
    print(f"   Max:   {grupo_relax.max():.6f}")

    print("\nGrupo ANSIEDAD (Clase 1):")
    print(f"   Media: {grupo_ansiedad.mean():.6f}")
    print(f"   Std:   {grupo_ansiedad.std():.6f}")
    print(f"   Min:   {grupo_ansiedad.min():.6f}")
    print(f"   Max:   {grupo_ansiedad.max():.6f}")
    
    # Visualización
    plt.figure(figsize=(10, 6))
    sns.histplot(grupo_relax, color='blue', label='Relajación (0)', kde=True, stat="density")
    sns.histplot(grupo_ansiedad, color='red', label='Ansiedad (>=5)', kde=True, stat="density")
    plt.title(f'Distribución de {feature}: ¿Separación artificial?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_img = os.path.join('Resultados', 'Diagnostico_Overfitting.png')
    plt.savefig(out_img)
    print(f"\nGráfica guardada en: {out_img}")

if __name__ == "__main__":
    diagnostico_fuga_datos()
