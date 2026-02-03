import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generar_graficos_densidad():
    # Configuración de rutas
    input_file = os.path.join('Resultados', 'datos_bandas.parquet')
    output_dir = os.path.join('Resultados', 'Análisis espectral')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Cargando {input_file}...")
    df = pd.read_parquet(input_file)
    
    # Definir grupos: Relajacion vs Ansiedad
    print("Creando grupos: Relajacion (Puntaje == 0) vs Ansiedad (Puntaje >= 1)...")
    df['Grupo'] = df['Puntaje'].apply(lambda x: 'Relajacion' if x == 0 else 'Ansiedad')
    
    all_columns = df.columns.tolist()
    feature_cols = [c for c in all_columns if '_' in c and c not in ['Sujeto', 'Tarea', 'Ensayo', 'Epoca', 'Puntaje', 'Grupo']]
    
    # Bandas de interés
    bandas_interes = ['Alpha', 'Beta']
    
    canales_feature = []
    for col in feature_cols:
        canal, banda = col.split('_')
        if banda in bandas_interes:
            if canal.startswith('F') or canal.startswith('T'):
                canales_feature.append(col)
                
    print(f"Total de características a graficar: {len(canales_feature)}")
    
    # Configurar estilo visual
    sns.set_theme(style="whitegrid")
    
    # Generar gráficos    
    for i, col in enumerate(canales_feature):
        if i % 10 == 0:
            print(f"Generando gráfico {i}/{len(canales_feature)}: {col}")
            
        plt.figure(figsize=(10, 6))
        
        # Plot de densidad (KDE)
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
        
        canal, banda = col.split('_')
        plt.title(f'Distribución de Densidad: Canal {canal} - Banda {banda}\n(Relajacion vs Ansiedad)')
        plt.xlabel('Potencia Espectral (uV^2/Hz)')
        plt.ylabel('Densidad')
        
        # Guardar archivo
        filename = f"{output_dir}/Densidad_{canal}_{banda}.png"
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    generar_graficos_densidad()
