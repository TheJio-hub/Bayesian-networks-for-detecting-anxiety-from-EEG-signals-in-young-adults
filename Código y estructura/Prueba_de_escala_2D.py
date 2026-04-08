import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def graficar_scatter_2d():
    archivo_entrada = os.path.join('Resultados', 'datos_bandas_normalizados.csv')
    archivo_ranking = os.path.join('Resultados', 'Ranking', 'Ranking_mRMR.csv')
    output_dir = os.path.join('Resultados', 'Analisis_2D')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(archivo_entrada):
         archivo_entrada = archivo_entrada.replace('.csv', '.parquet')
         if os.path.exists(archivo_entrada):
            df = pd.read_parquet(archivo_entrada)
         else:
            return
    else:
         df = pd.read_csv(archivo_entrada)
         
    if 'Grupo' not in df.columns:
        if 'Puntaje' in df.columns:
             df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
             df['Grupo'] = df['Puntaje'].apply(lambda x: 'Relajacion' if x == 0 else 'Ansiedad')
    else:
         if 'Puntaje' in df.columns:
              df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ]

    if os.path.exists(archivo_ranking):
        df_rank = pd.read_csv(archivo_ranking)
        if 'Caracteristica' in df_rank.columns:
             top_features = df_rank['Caracteristica'].head(4).tolist()
        else:
             top_features = df_rank.iloc[:4, 0].tolist()
    else:
        top_features = ['Fp1_Delta', 'T8_Beta', 'P7_Alpha', 'F3_Theta'] 

    if len(top_features) < 4:
        return

    pares = [(top_features[0], top_features[1]), (top_features[2], top_features[3])]
    
    for x_col, y_col in pares:
        if x_col not in df.columns or y_col not in df.columns:
            continue
            
        plt.figure(figsize=(10, 8))
        
        sns.scatterplot(
            data=df, 
            x=x_col, 
            y=y_col, 
            hue='Grupo', 
            style='Grupo',
            alpha=0.6,
            palette={'Relajacion': '#1f77b4', 'Ansiedad': '#d62728'},
            s=60
        )
        
        medias = df.groupby('Grupo')[[x_col, y_col]].mean()
        
        plt.scatter(
            medias.loc['Relajacion', x_col], medias.loc['Relajacion', y_col], 
            color='blue', s=300, marker='X', edgecolors='black', label='Centroide Relax', zorder=10
        )
        plt.scatter(
            medias.loc['Ansiedad', x_col], medias.loc['Ansiedad', y_col], 
            color='red', s=300, marker='X', edgecolors='black', label='Centroide Ansiedad', zorder=10
        )
        
        plt.title(f'Separación 2D: {x_col} vs {y_col}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = os.path.join(output_dir, f"Scatter_{x_col}_vs_{y_col}.png")
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    graficar_scatter_2d()
