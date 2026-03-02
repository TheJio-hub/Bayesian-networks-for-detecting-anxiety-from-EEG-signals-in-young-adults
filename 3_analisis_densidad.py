import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generar_graficos_densidad():
    input_file = os.path.join('Resultados', 'datos_bandas.parquet')
    output_dir = os.path.join('Resultados', 'Análisis espectral')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_parquet(input_file)
    
    # Filtrar datos SAM
    df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
    
    df['Grupo'] = df['Puntaje'].apply(lambda x: 'Relajacion' if x == 0 else 'Ansiedad')
    
    all_columns = df.columns.tolist()
    feature_cols = [c for c in all_columns if '_' in c and c not in ['Sujeto', 'Tarea', 'Ensayo', 'Epoca', 'Puntaje', 'Grupo']]
    
    # Excluir referencias
    canales_referencia = ['A1', 'A2', 'M1', 'M2', 'REF', 'Ref', 'EXG1', 'EXG2']
    
    canales_feature = []
    for col in feature_cols:
        parts = col.split('_')
        if len(parts) < 2: continue
        canal = parts[0]
        
        if canal in canales_referencia:
            continue
            
        canales_feature.append(col)
                
    sns.set_theme(style="whitegrid")
    
    for i, col in enumerate(canales_feature):
            
        plt.figure(figsize=(10, 6))
        
        sns.kdeplot(
            data=df, 
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
        banda = parts[1]
        
        plt.title(f'Distribución de Densidad: Canal {canal} - Banda {banda}\n(Relajacion vs Ansiedad)')
        plt.xlabel('Potencia Espectral (uV^2/Hz)')
        plt.ylabel('Densidad')
        
        filename = os.path.join(output_dir, f"Densidad_{canal}_{banda}.png")
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    generar_graficos_densidad()
