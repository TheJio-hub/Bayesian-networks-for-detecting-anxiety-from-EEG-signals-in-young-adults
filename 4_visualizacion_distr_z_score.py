import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def graficar_densidad_normalizada():
    archivo_entrada = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    directorio_salida_graficos = os.path.join('Resultados', 'Análisis espectral (normalizado)')
    
    if not os.path.exists(directorio_salida_graficos):
        os.makedirs(directorio_salida_graficos)
    
    if not os.path.exists(archivo_entrada):
        print(f"No se encontró el archivo de entrada: {archivo_entrada}. Ejecuta el script 2 primero.")
        return

    df = pd.read_parquet(archivo_entrada)
    
    col_exclude = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    columnas_caracteristicas = [c for c in df.columns if c not in col_exclude and pd.api.types.is_numeric_dtype(df[c])]

    df_grafico = df.copy()
    if 'Puntaje' in df_grafico.columns:
        df_grafico = df_grafico[ (df_grafico['Puntaje'] == 0) | (df_grafico['Puntaje'] >= 5) ].copy()
        df_grafico['Grupo'] = df_grafico['Puntaje'].apply(lambda x: 'Relajacion' if x == 0 else 'Ansiedad')
    
    canales_referencia = ['A1', 'A2', 'M1', 'M2', 'REF', 'Ref', 'EXG1', 'EXG2']
    
    columnas_para_graficar = []
    for col in columnas_caracteristicas:
        partes = col.split('_')
        if len(partes) >= 2:
            canal = partes[0]
            if canal not in canales_referencia:
                columnas_para_graficar.append(col)
            
    sns.set_theme(style="whitegrid")
        
    for i, columna in enumerate(columnas_para_graficar):
            
        plt.figure(figsize=(10, 6))
        
        try:
            sns.kdeplot(
                data=df_grafico, 
                x=columna, 
                hue='Grupo', 
                fill=True, 
                common_norm=False, 
                palette={'Relajacion': 'blue', 'Ansiedad': 'red'},
                alpha=0.3, 
                linewidth=2,
                clip=(-5, 5) 
            )
            
            partes = columna.split('_')
            canal = partes[0]
            banda = partes[1] if len(partes) > 1 else "Banda"
            
            plt.title(f'Distribución Normalizada (Z): {canal} - {banda}\n(Referencia: Media de Relajación)')
            plt.xlabel('Z-Score (Log10 Power - Ref. Relajación)')
            plt.ylabel('Densidad')
            
            nombre_archivo = os.path.join(directorio_salida_graficos, f"Densidad_Norm_{canal}_{banda}.png")
            plt.savefig(nombre_archivo)
            plt.close()
            
        except Exception as e:
            print(f"Error {columna}: {e}")
            plt.close()
    
if __name__ == "__main__":
    graficar_densidad_normalizada()