import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generar_graficos_densidad():
    # Configuración de rutas
    input_file = 'datos_bandas.parquet'
    output_dir = 'Resultados/Analisis espectral'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Cargando {input_file}...")
    df = pd.read_parquet(input_file)
    
    # Definir grupos: Relax vs Estrés
    print("Creando grupos: Relax (Puntaje == 0) vs Estrés (Puntaje >= 1)...")
    df['Grupo'] = df['Puntaje'].apply(lambda x: 'Relax' if x == 0 else 'Estres')
    
    # Filtrar canales Frontales (F...) y Temporales (T...)
    # Lista de canales obtenida previamente:
    # ['C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6', 'Cz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'FT10', 'FT9', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'Oz', 'P3', 'P4', 'P7', 'P8', 'PO10', 'PO9', 'Pz', 'T7', 'T8']
    
    all_columns = df.columns.tolist()
    feature_cols = [c for c in all_columns if '_' in c and c not in ['Sujeto', 'Tarea', 'Ensayo', 'Epoca', 'Puntaje', 'Grupo']]
    
    # Bandas de interés
    bandas_interes = ['Alpha', 'Beta']
    
    # Canales de interés: Empiezan con F (Frontal) o T (Temporal)
    # Excluimos los que no sean F o T puros si queremos ser muy estrictos, pero Fp, FC, FT son relevantes.
    # El usuario pidió "Frontales y Temporales".
    
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
    # Para no saturar, guardaremos un gráfico por característica
    
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
            palette={'Relax': 'blue', 'Estres': 'red'},
            alpha=0.3,
            linewidth=2
        )
        
        canal, banda = col.split('_')
        plt.title(f'Distribución de Densidad: Canal {canal} - Banda {banda}\n(Relax vs Estrés)')
        plt.xlabel('Potencia Espectral (uV^2/Hz)')
        plt.ylabel('Densidad')
        
        # Guardar archivo
        filename = f"{output_dir}/Densidad_{canal}_{banda}.png"
        plt.savefig(filename)
        plt.close()

    print("¡Análisis gráfico completado!")

if __name__ == "__main__":
    generar_graficos_densidad()
