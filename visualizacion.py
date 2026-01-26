import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualizar_asimetrias():
    # Rutas de archivos
    dir_base = "Resultados/Arithmetic"
    ruta_X = os.path.join(dir_base, "X.npy")
    ruta_ids = os.path.join(dir_base, "identifiers.npy")

    if not os.path.exists(ruta_X) or not os.path.exists(ruta_ids):
        print("Archivos de resultados no encontrados.")
        return

    # Carga de datos
    matriz_X = np.load(ruta_X)
    ids_archivos = np.load(ruta_ids)

    # Identificar nombres de los canales desde archivo de coordenadas
    ruta_locs = os.path.join(dir_base, "../../Conjunto de datos/Data/Coordinates.locs")
    if os.path.exists(ruta_locs):
        try:
            df_locs = pd.read_csv(ruta_locs, sep='\t', header=None, names=['idx', 'theta', 'radius', 'etiqueta'])
            nombres_canales = df_locs['etiqueta'].str.strip().tolist()
        except Exception as e:
            nombres_canales = [f"Ch{i+1}" for i in range(32)]
    else:
        nombres_canales = [f"Ch{i+1}" for i in range(32)]

    # Clasificar por intento (Trial)
    lista_intentos = []
    lista_sujetos = []
    
    for id_archivo in ids_archivos:
        texto_id = str(id_archivo)
        
        # Determinar trial
        if "trial1" in texto_id:
            lista_intentos.append("Intento 1")
        elif "trial2" in texto_id:
            lista_intentos.append("Intento 2")
        elif "trial3" in texto_id:
            lista_intentos.append("Intento 3")
        else:
            lista_intentos.append("Desconocido")
            
        # Determinar sujeto
        partes = texto_id.split('_')
        idx_sub = -1
        for k, p in enumerate(partes):
            if p == 'sub':
                idx_sub = k + 1
                break
        
        if idx_sub != -1 and idx_sub < len(partes):
            lista_sujetos.append(partes[idx_sub])
        else:
            lista_sujetos.append('desc')

    # Indices de caracteristicas espectrales
    # 0-95: Tiempo, 96-255: Frecuencia
    inicio_frec = 96
    num_bandas = 5 
    
    # Calculo de Asimetrias (Ratios)
    print("Calculando asimetrías inter-hemisféricas...")
    
    def buscar_indice(nombre, lista):
        return -1
        
    ind_f3 = buscar_indice('F3', nombres_canales)
    ind_f4 = buscar_indice('F4', nombres_canales)
    ind_p3 = buscar_indice('P3', nombres_canales)
    ind_p4 = buscar_indice('P4', nombres_canales)
    
    nuevas_caract = {}
    
    # Asimetria Frontal Alpha (F4/F3)
    if ind_f3 != -1 and ind_f4 != -1:
        col_f3_alpha = inicio_frec + (ind_f3 * num_bandas) + 2
        col_f4_alpha = inicio_frec + (ind_f4 * num_bandas) + 2
        
        # Log ratio: ln(Derecha) - ln(Izquierda)
        valores = np.log(matriz_X[:, col_f4_alpha] + 1e-6) - np.log(matriz_X[:, col_f3_alpha] + 1e-6)
        nuevas_caract["Asimetría Frontal Alpha (F4/F3)"] = valores
        
        # Asimetria Frontal Beta
        col_f3_beta = inicio_frec + (ind_f3 * num_bandas) + 3
        col_f4_beta = inicio_frec + (ind_f4 * num_bandas) + 3
        
        valores = np.log(matriz_X[:, col_f4_beta] + 1e-6) - np.log(matriz_X[:, col_f3_beta] + 1e-6)
        nuevas_caract["Asimetría Frontal Beta (F4/F3)"] = valores

    # Asimetria Parietal Alpha (P4/P3)
    if ind_p3 != -1 and ind_p4 != -1:
        col_p3_alpha = inicio_frec + (ind_p3 * num_bandas) + 2
        col_p4_alpha = inicio_frec + (ind_p4 * num_bandas) + 2
        
        valores = np.log(matriz_X[:, col_p4_alpha] + 1e-6) - np.log(matriz_X[:, col_p3_alpha] + 1e-6)
        nuevas_caract["Asimetría Parietal Alpha (P4/P3)"] = valores

    if not nuevas_caract:
        return

    # DataFrame para visualizacion
    df_derivado = pd.DataFrame(nuevas_caract)
    df_derivado["Intento"] = lista_intentos
    df_derivado["Sujeto"] = lista_sujetos
    
    # Normalizacion Z-score por sujeto
    print("Normalizando datos por sujeto...")
    cols_numericas = list(nuevas_caract.keys())
    def aplicar_zscore(x):
        if x.std() == 0: return x - x.mean()
        return (x - x.mean()) / x.std()

    df_derivado[cols_numericas] = df_derivado.groupby("Sujeto")[cols_numericas].transform(aplicar_zscore)
    
    # Generacion de graficos
    plt.figure(figsize=(15, 6))
    plt.suptitle(f"Comparación de Asimetrías Normalizadas por Intento", fontsize=16)
    
    colores = {"Intento 1": "#2ecc71", "Intento 2": "#3498db", "Intento 3": "#e74c3c"}
    
    for i, columna in enumerate(cols_numericas):
        plt.subplot(1, 3, i+1)
        datos_col = df_derivado[columna]
        # Filtrar valores extremos para mejorar visualización
        datos_col = datos_col[datos_col.between(-4, 4)]
        
        sns.kdeplot(data=df_derivado, x=columna, hue="Intento", fill=True, common_norm=False, palette=colores, alpha=0.3, clip=(-4, 4))
        plt.title(columna)
        plt.xlabel("Asimetría (Z-score)")
        plt.ylabel("Densidad")
        plt.grid(True, alpha=0.3)
        
        if i > 0:
            plt.legend([],[], frameon=False)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    ruta_salida = os.path.join(dir_base, "grafico_asimetrias.png")
    plt.savefig(ruta_salida)
    print(f"Gráfico guardado en: {ruta_salida}")

if __name__ == "__main__":
    visualizar_asimetrias()
