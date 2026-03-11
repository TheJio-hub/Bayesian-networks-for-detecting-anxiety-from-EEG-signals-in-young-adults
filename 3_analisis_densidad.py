import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def generar_graficos_densidad():
    input_file = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    output_dir = os.path.join('Resultados', 'Análisis espectral')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(input_file):
        print(f"Error: No se encontró {input_file}. Ejecute el script 2 primero.")
        return

    df = pd.read_parquet(input_file)
    
    df_filtrado = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
    
    # Asignar etiquetas
    df_filtrado['Grupo'] = df_filtrado['Puntaje'].apply(lambda x: 'Relajacion' if x == 0 else 'Ansiedad')
    
    col_exclude = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    feature_cols = [c for c in df_filtrado.columns if c not in col_exclude and pd.api.types.is_numeric_dtype(df_filtrado[c])]
    
    canales_referencia = ['A1', 'A2', 'M1', 'M2', 'REF', 'Ref', 'EXG1', 'EXG2']
    
    canales_feature = []
    for col in feature_cols:
        parts = col.split('_')
        if len(parts) >= 2: 
            canal = parts[0]
            if canal not in canales_referencia:
                canales_feature.append(col)
                
    sns.set_theme(style="whitegrid")
        
    for i, col in enumerate(canales_feature):
        plt.figure(figsize=(10, 6))
        
        try:
            sns.kdeplot(
                data=df_filtrado, 
                x=col, 
                hue='Grupo', 
                fill=True, 
                common_norm=False, 
                palette={'Relajacion': 'blue', 'Ansiedad': 'red'},
                alpha=0.3,
                linewidth=2
            )
            
            parts = col.split('_')
            canal = parts[0]
            banda = parts[1] if len(parts) > 1 else "Banda"
            
            plt.title(f'Densidad (Z-Score Rel.): {canal} - {banda}')
            plt.xlabel('Z-Score (Log10 Power - Ref. Relajación)')
            plt.ylabel('Densidad')
            
            filename = os.path.join(output_dir, f"Densidad_{canal}_{banda}.png")
            plt.savefig(filename)
            plt.close()
            
        except Exception as e:
            print(f"Error graficando {col}: {e}")
            plt.close()

if __name__ == "__main__":
    generar_graficos_densidad()