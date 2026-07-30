import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)

def principal():
    archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')
    
    if not os.path.exists(archivo_datos) or not os.path.exists(archivo_ranking):
        print("Error: No se encontraron los archivos necesarios en Resultados.")
        return
        
    # Cargar datos
    df_datos = pd.read_parquet(archivo_datos)
    
    # Cargar ranking y obtener las mejores características
    df_ranking = pd.read_csv(archivo_ranking)
    
    # Top 5 de Decision Tree (Red Mixta)
    top_dt = df_ranking.sort_values(by='Importancia_DT', ascending=False)['Caracteristica'].head(5).tolist()
    
    # Top 5 de mRMR (Regresión Causal)
    top_mrmr = df_ranking.dropna(subset=['mRMR_40_Rank']).sort_values(by='mRMR_40_Rank', ascending=True)['Caracteristica'].head(5).tolist()
    
    # Combinar para obtener las características clave a graficar
    mejores_features = list(dict.fromkeys(top_dt + top_mrmr))
    print(f"Características clave para graficar densidades por Trials: {mejores_features}")
    
    sns.set_theme(style="whitegrid")
    
    trials = [1, 2, 3]
    
    for trial in trials:
        print(f"\nGenerando gráficos de densidad para el TRIAL: {trial}")
        dir_salida = os.path.join('Resultados', 'Red Bayesiana', 'Trials', f'Trial_{trial}', 'Densidades')
        os.makedirs(dir_salida, exist_ok=True)
        
        # Filtrar datos de este Trial
        df_trial = df_datos[df_datos['Trial'] == trial].copy()
        df_trial = df_trial[(df_trial['Puntaje'] == 0) | (df_trial['Puntaje'] >= 1)].copy()
        
        # Etiquetar grupo (Control vs Ansiedad)
        df_trial['Grupo'] = df_trial['Puntaje'].apply(lambda x: 'Control' if x == 0 else 'Ansiedad')
        
        for feat in mejores_features:
            if feat not in df_trial.columns:
                continue
                
            plt.figure(figsize=(9, 5))
            
            try:
                sns.kdeplot(
                    data=df_trial,
                    x=feat,
                    hue='Grupo',
                    fill=True,
                    common_norm=False,
                    palette={'Control': 'blue', 'Ansiedad': 'red'},
                    alpha=0.3,
                    linewidth=2.0
                )
                
                plt.title(f"Distribución de Densidad: {feat} (Trial {trial})", fontsize=12, fontweight='bold', pad=15)
                plt.xlabel("Valor Normalizado (Z-Score)", fontsize=10)
                plt.ylabel("Densidad", fontsize=10)
                plt.tight_layout()
                
                nombre_grafica = os.path.join(dir_salida, f"Densidad_{feat}.png")
                plt.savefig(nombre_grafica, dpi=300)
                plt.close()
            except Exception as e:
                print(f"Error al graficar {feat} en Trial {trial}: {e}")
                plt.close()

    print("\n¡Gráficos de densidad por Trials generados con éxito!")

if __name__ == "__main__":
    principal()
