import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def normalizar_y_graficar_densidad():
    """
    Normaliza las características espectrales mediante Z-score por sujeto y genera gráficos de densidad.
    """
    archivo_entrada = os.path.join('Resultados', 'datos_bandas.parquet')
    archivo_salida_parquet = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    directorio_salida_graficos = os.path.join('Resultados', 'Análisis espectral (normalizado)')
    
    if not os.path.exists(directorio_salida_graficos):
        os.makedirs(directorio_salida_graficos)
    
    if not os.path.exists(archivo_entrada):
        print(f"No se encontró el archivo de entrada: {archivo_entrada}")
        return

    print(f"Cargando datos desde: {archivo_entrada}")
    df = pd.read_parquet(archivo_entrada)
    
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo'] 
    columnas_metadatos_presentes = [c for c in columnas_metadatos if c in df.columns]
    columnas_caracteristicas = [c for c in df.columns if c not in columnas_metadatos_presentes and pd.api.types.is_numeric_dtype(df[c])]

    def puntuacion_z(x):
        if x.std() == 0: return x - x.mean()
        return (x - x.mean()) / x.std()

    df_normalizado = df.copy()
    
    print("Normalizando características por sujeto (Z-Score)...")
    # Agrupar por sujeto y aplicar transformación Z-score a las características
    df_normalizado[columnas_caracteristicas] = df_normalizado.groupby('Sujeto')[columnas_caracteristicas].transform(puntuacion_z)
    
    if 'Puntaje' in df_normalizado.columns:
        # Filtrar Grupos: Relajación (0) vs Ansiedad (>= 5)
        # Esto elimina los puntajes intermedios (1-4) para limpiar el análisis
        mascara_filtro = (df_normalizado['Puntaje'] == 0) | (df_normalizado['Puntaje'] >= 5)
        df_normalizado = df_normalizado[mascara_filtro].copy()
        
        # Etiquetar grupos
        df_normalizado['Grupo'] = df_normalizado['Puntaje'].apply(lambda x: 'Relajacion' if x == 0 else 'Ansiedad')
    
    print(f"Guardando datos normalizados en: {archivo_salida_parquet}")
    df_normalizado.to_parquet(archivo_salida_parquet, index=False)
    
    # Exportar CSV para visualización externa si es necesario
    df_normalizado.to_csv(archivo_salida_parquet.replace('.parquet', '.csv'), index=False)

    canales_referencia = ['A1', 'A2', 'M1', 'M2', 'REF', 'Ref', 'EXG1', 'EXG2']
    
    columnas_para_graficar = []
    for col in columnas_caracteristicas:
        partes = col.split('_')
        if len(partes) >= 2:
            canal = partes[0]
            # Excluir canales de referencia si aparecen en las características
            if canal not in canales_referencia:
                columnas_para_graficar.append(col)
            
    sns.set_theme(style="whitegrid")
    
    print(f"Generando gráficos de densidad en {directorio_salida_graficos}...")
    for i, columna in enumerate(columnas_para_graficar):
            
        plt.figure(figsize=(10, 6))
        
        try:
            sns.kdeplot(
                data=df_normalizado, 
                x=columna, 
                hue='Grupo', 
                fill=True, 
                common_norm=False, 
                palette={'Relajacion': 'blue', 'Ansiedad': 'red'},
                alpha=0.3, 
                linewidth=2,
                clip=(-4, 4) # Limitar visualización a +/- 4 desviaciones estándar
            )
            
            partes = columna.split('_')
            canal = partes[0]
            banda = partes[1] if len(partes) > 1 else "Banda"
            
            plt.title(f'Distribución Normalizada (Z-score): {canal} - {banda}\n(Relajacion vs Ansiedad)')
            plt.xlabel('Potencia Normalizada (Desviaciones Estándar)')
            plt.ylabel('Densidad')
            
            nombre_archivo = os.path.join(directorio_salida_graficos, f"Densidad_Norm_{canal}_{banda}.png")
            plt.savefig(nombre_archivo)
            plt.close()
            
        except Exception as e:
            print(f"Error graficando {columna}: {e}")
            plt.close()
    
    print("Proceso de normalización y graficado completado.")

if __name__ == "__main__":
    normalizar_y_graficar_densidad()
