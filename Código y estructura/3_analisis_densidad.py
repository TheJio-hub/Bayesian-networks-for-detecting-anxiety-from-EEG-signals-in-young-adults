import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def graficar_densidades(input_file, output_dir, tipo_analisis):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(input_file):
        print(f"Error: No se encontró {input_file}. Ejecute el script 2 primero.")
        return

    df = pd.read_parquet(input_file)
    
    # Filtrar solo Relajación (0) y Ansiedad (>=5)
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
            
            if tipo_analisis == "Log10":
                titulo = f'Densidad (Log10 Power): {col}'
                ejex = 'Log10 Power (dB)'
            elif tipo_analisis == "Asimetria":
                titulo = f'Densidad (Asimetría Z-Score): {col}'
                ejex = 'Z-Score (Asimetría - Ref. Relajación)'
            elif tipo_analisis == "Ratios":
                titulo = f'Densidad (Ratio Z-Score): {col}'
                ejex = 'Z-Score (Ratio - Ref. Relajación)'
            else:
                titulo = f'Densidad (Power Z-Score): {col}'
                ejex = 'Z-Score (Log10 Power - Ref. Relajación)'

            plt.title(titulo)
            plt.xlabel(ejex)
            plt.ylabel('Densidad')
            
            # Limpiar nombre de archivo de caracteres raros si los hubiera (ej. /)
            safe_filename = col.replace('/', '_')
            filename = os.path.join(output_dir, f"Densidad_{safe_filename}.png")
            plt.savefig(filename)
            plt.close()
            
        except Exception as e:
            print(f"Error graficando {col}: {e}")
            plt.close()

def main():
    archivo_norm = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    dir_norm = os.path.join('Resultados', 'Análisis espectral')
    graficar_densidades(archivo_norm, dir_norm, "Normalizado")

    archivo_log = os.path.join('Resultados', 'potencias_log10.parquet')
    dir_log = os.path.join('Resultados', 'Análisis espectral Log10')
    graficar_densidades(archivo_log, dir_log, "Log10")

    archivo_asim = os.path.join('Resultados', 'datos_asimetria_normalizados.parquet')
    dir_asim = os.path.join('Resultados', 'Análisis Asimetría')
    graficar_densidades(archivo_asim, dir_asim, "Asimetria")

    archivo_ratios = os.path.join('Resultados', 'datos_ratios_normalizados.parquet')
    dir_ratios = os.path.join('Resultados', 'Análisis Ratios')
    graficar_densidades(archivo_ratios, dir_ratios, "Ratios")

if __name__ == "__main__":
    main()