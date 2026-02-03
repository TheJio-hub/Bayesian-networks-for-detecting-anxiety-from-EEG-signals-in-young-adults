import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generar_graficos_boxplot():
    input_file = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    output_dir = os.path.join('Resultados', 'Análisis boxplot (normalizado)')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Cargando {input_file}...")
    df = pd.read_parquet(input_file)
    
    # Definir grupos: Relajacion vs Ansiedad
    print("Creando grupos: Relajacion (Puntaje == 0) vs Ansiedad (Puntaje >= 1)...")
    df['Grupo'] = df['Puntaje'].apply(lambda x: 'Relajacion' if x == 0 else 'Ansiedad')
    
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo']
    cols_features = [c for c in df.columns if c not in cols_meta]
    
    bandas_interes = ['Alpha', 'Beta']
    canales_feature = []
    
    for col in cols_features:
        parts = col.split('_')
        if len(parts) < 2: continue
        canal = parts[0]
        banda = parts[1]
        
        if banda in bandas_interes:
            if canal.startswith('F') or canal.startswith('T'):
                canales_feature.append(col)
                
    
    sns.set_theme(style="whitegrid")
    
    for i, col in enumerate(canales_feature):
        if i % 10 == 0:
            print(f"Procesando {i}/{len(canales_feature)}: {col}")
            
        plt.figure(figsize=(8, 6))
        

        sns.boxplot(
            data=df, 
            x='Grupo', 
            y=col, 
            hue='Grupo',
            palette={'Relajacion': 'lightblue', 'Ansiedad': 'salmon'},
            width=0.5,
            showfliers=False 
        )
        
        parts = col.split('_')
        canal = parts[0]
        banda = parts[1]
        plt.title(f'Distribución Normalizada (Boxplot): {canal} - {banda}')
        plt.ylabel('Z-Score (Desviaciones Estándar)')
        plt.xlabel('Condición')
        
        filename = os.path.join(output_dir, f"Boxplot_Norm_{canal}_{banda}.png")
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    generar_graficos_boxplot()
