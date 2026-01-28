import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def analizar_todas_caracteristicas_relax_ansiedad():
    '''
    Genera gráficos de densidad comparativa para TODAS las características extraídas.
    Compara Clase 0 (Relax recortado 25s) vs Clase 1 (Ansiedad Aritmética > 5).
    '''
    # 1. Configuración de Rutas y Carga
    dir_base = "Resultados"
    tipos_feat = ['time', 'frequency', 'hjorth', 'fractal']
    
    # Mapeo de nombres legibles para las características
    # Time: 3 feats (Var, RMS, PTP)
    # Freq: 5 bandas (Delta, Theta, Alpha, Beta, Gamma)
    # Hjorth: 2 feats (Movilidad, Complejidad)
    # Fractal: 2 feats (Higuchi, Katz)
    nombres_feats = {
        'time': ['Varianza', 'RMS', 'Pico-a-Pico'],
        'frequency': ['Delta (0.5-4Hz)', 'Theta (4-8Hz)', 'Alpha (8-13Hz)', 'Beta (13-30Hz)', 'Gamma (30-45Hz)'],
        'hjorth': ['Movilidad', 'Complejidad'],
        'fractal': ['Higuchi FD', 'Katz FD']
    }
    
    # Cargar identificadores para normalización
    ruta_relax_ids = os.path.join(dir_base, "Relax", "identifiers.npy")
    ruta_stress_ids = os.path.join(dir_base, "Arithmetic", "identifiers.npy")
    ruta_stress_y = os.path.join(dir_base, "Arithmetic", "y.npy") # Etiquetas subjetivas
    
    if not os.path.exists(ruta_relax_ids) or not os.path.exists(ruta_stress_ids):
        print("Faltan archivos de identificadores.")
        return
        
    ids_relax = np.load(ruta_relax_ids)
    ids_stress = np.load(ruta_stress_ids)
    etiquetas_stress = np.load(ruta_stress_y)
    
    # Función auxiliar para extraer ID sujeto
    def extraer_sujeto(nombre_archivo):
        partes = str(nombre_archivo).split('_')
        try:
            if 'sub' in partes:
                idx = partes.index('sub')
                return partes[idx+1]
        except:
            pass
        return 'unknown'

    sujetos_relax = [extraer_sujeto(f) for f in ids_relax]
    # Filtrar Ansiedad (Clase 1)
    mask_ansiedad = etiquetas_stress == 1 
    ids_ansiedad = ids_stress[mask_ansiedad]
    sujetos_ansiedad = [extraer_sujeto(f) for f in ids_ansiedad]
    
    if len(sujetos_ansiedad) == 0:
        print("Advertencia: No hay sujetos con etiqueta de ansiedad alta. Usando todos los de aritmética.")
        mask_ansiedad = np.ones(len(ids_stress), dtype=bool)
        ids_ansiedad = ids_stress
        sujetos_ansiedad = [extraer_sujeto(f) for f in ids_ansiedad]

    # Identificar canales y regiones
    # Ajuste de ruta: Asumiendo ejecución desde directorio raíz del proyecto
    ruta_locs = "Conjunto de datos/Data/Coordinates.locs"
    nombres_canales = [f"Ch{i+1}" for i in range(32)]
    
    if os.path.exists(ruta_locs):
        try:
            # Usar sep='\s+' para manejar cualquier cantidad de espacios/tabs
            df_locs = pd.read_csv(ruta_locs, sep=r'\s+', header=None, names=['idx', 'theta', 'radius', 'etiqueta'], engine='python')
            nombres_canales = df_locs['etiqueta'].str.strip().tolist()
        except Exception as e:
            print(f"Error leyendo coordenadas: {e}")
    else:
        print(f"Advertencia: No se encontró {ruta_locs}, usando nombres genéricos.")

    # Definición de Regiones (Mapeo estricto basado en Coordinates.locs)
    regiones = {
        'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
        'Parietal': ['P7', 'P3', 'Pz', 'P4', 'P8']
    }
    
    # Obtener índices por región
    indices_regiones = {}
    for nombre_region, lista_canales in regiones.items():
        indices = []
        for ch in lista_canales:
            if ch in nombres_canales:
                indices.append(nombres_canales.index(ch))
        indices_regiones[nombre_region] = indices
        print(f"Región {nombre_region}: {len(indices)} canales encontrados ({lista_canales})")

    # Directorio salida
    dir_salida = os.path.join(dir_base, "Graficos_Regionales")
    os.makedirs(dir_salida, exist_ok=True)

    # 2. Iteración por Tipo de Grupo de Características
    # Filtramos solo frecuencia como se solicitó "sus respectivas 5 bandas", pero mantenemos flexibilidad
    for grupo in ['frequency']: 
        print(f"Procesando grupo: {grupo}...")
        
        # Cargar matrices de datos .npy especificas
        ruta_feat_relax = os.path.join(dir_base, "Relax", "features", f"{grupo}.npy")
        ruta_feat_stress = os.path.join(dir_base, "Arithmetic", "features", f"{grupo}.npy")
        
        if not os.path.exists(ruta_feat_relax): continue
        
        X_relax = np.load(ruta_feat_relax)
        X_stress_full = np.load(ruta_feat_stress)
        
        # Filtrar estrés
        X_ansiedad = X_stress_full[mask_ansiedad]
        
        # Determinar número de canales (32) y features por canal
        n_canales = 32
        n_subfeats = X_relax.shape[1] // n_canales
        lista_nombres_sub = nombres_feats.get(grupo, [f"Feat_{i}" for i in range(n_subfeats)])
        
        # 3. Iteración por Sub-característica (Bandas)
        for i, nombre_sub in enumerate(lista_nombres_sub):
            
            # 4. Iteración por Región (Frontal, Parietal) - CANAL POR CANAL
            for nombre_region, idxs_canales in indices_regiones.items():
                if not idxs_canales: continue
                
                # Iterar sobre cada canal INDIVIDUAL de la región
                for ch_idx in idxs_canales:
                    nombre_canal = nombres_canales[ch_idx]
                    
                    # Indice especifico para este canal y esta banda
                    # La estructura es: [C1_B1, C1_B2... C1_B5, C2_B1...] 
                    # i = indice de la banda (0 a 4)
                    # ch_idx = indice del canal (0 a 31)
                    # n_subfeats = numero de bandas (5)
                    indice_feature = i + (ch_idx * n_subfeats)
                    
                    # Valores crudos del canal específico
                    val_relax = X_relax[:, indice_feature]
                    val_ansiedad = X_ansiedad[:, indice_feature]
                    
                    # Metadatos para gráfica
                    n_sujetos_relax = len(np.unique(sujetos_relax))
                    unique_ansiedad = np.unique(sujetos_ansiedad)
                    n_sujetos_ansiedad = len(unique_ansiedad)
                    
                    ids_str = ",".join(unique_ansiedad)
                    if len(ids_str) > 40: ids_str = ids_str[:37] + "..."

                    # Normalización Z-Score por Sujeto
                    df_comp = pd.DataFrame({
                        'Valor': np.concatenate([val_relax, val_ansiedad]),
                        'Condicion': ['Relajación (25s)'] * len(val_relax) + ['Ansiedad (Aritmética)'] * len(val_ansiedad),
                        'Sujeto': sujetos_relax + sujetos_ansiedad
                    })
                    
                    # Z-Score por sujeto
                    df_comp['Valor_Norm'] = df_comp.groupby('Sujeto')['Valor'].transform(
                        lambda x: (x - x.mean()) / (x.std() + 1e-6)
                    )
                    
                    # Graficar
                    plt.figure(figsize=(10, 7))
                    sns.kdeplot(data=df_comp[df_comp['Condicion'].str.contains('Relajación')], x='Valor_Norm', fill=True, label=f'Relajación (N={n_sujetos_relax})', color='blue', alpha=0.3)
                    sns.kdeplot(data=df_comp[df_comp['Condicion'].str.contains('Ansiedad')], x='Valor_Norm', fill=True, label=f'Ansiedad (N={n_sujetos_ansiedad})', color='red', alpha=0.3)
                    
                    titulo = f'Densidad: {nombre_sub} - Canal {nombre_canal} ({nombre_region})'
                    subtitulo = (f"Sujetos Ansiosos: {ids_str}\n"
                                 f"Normalización: Z-Score intra-sujeto")
                                 
                    plt.suptitle(titulo, fontsize=14, weight='bold')
                    plt.title(subtitulo, fontsize=10, color='gray', pad=15)
                    
                    plt.xlabel('Amplitud Normalizada (Z-Score)', fontsize=11)
                    plt.ylabel('Densidad de Probabilidad', fontsize=11)
                    plt.legend(title="Estado", loc='upper right')
                    plt.grid(True, alpha=0.3, linestyle='--')
                    
                    plt.figtext(0.5, 0.01, "Nota: Z-Score=0 representa la media del sujeto en este canal.", 
                                ha="center", fontsize=8, style='italic')

                    # Guardar archivo individual
                    archivo_png = f"Comparacion_{nombre_region}_{nombre_canal}_{nombre_sub.replace(' ', '_').replace('(', '').replace(')', '')}.png"
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.savefig(os.path.join(dir_salida, archivo_png))
                    plt.close()
            
    print(f"Gráficos generados en {dir_salida}")

def visualizar_proyeccion_dimensionalidad():
    '''
    Genera proyecciones 2D (PCA y t-SNE) para visualizar la separabilidad
    entre el estado de Relajación y Ansiedad utilizando todas las características.
    '''
    print("Generando proyecciones de dimensionalidad (PCA/t-SNE)...")
    dir_base = "Resultados"
    
    # Rutas
    ruta_relax_X = os.path.join(dir_base, "Relax", "X.npy")
    ruta_stress_X = os.path.join(dir_base, "Arithmetic", "X.npy")
    ruta_stress_y = os.path.join(dir_base, "Arithmetic", "y.npy")
    
    if not os.path.exists(ruta_relax_X) or not os.path.exists(ruta_stress_X):
        print("Faltan archivos de datos principales.")
        return

    # Cargar datos
    try:
        X_relax = np.load(ruta_relax_X)
        X_stress = np.load(ruta_stress_X)
        y_stress = np.load(ruta_stress_y)
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return
        
    # Filtrar Ansiedad (Clase 1 > 5)
    X_ansiedad = X_stress[y_stress == 1]
    
    if X_ansiedad.shape[0] == 0:
        print("No hay muestras de ansiedad alta, usando todo Arithmetic.")
        X_ansiedad = X_stress
    
    # Concatenar para análisis
    # Etiqueta 0: Relax, Etiqueta 1: Ansiedad
    X_total = np.concatenate([X_relax, X_ansiedad], axis=0)
    y_total = np.concatenate([np.zeros(len(X_relax)), np.ones(len(X_ansiedad))])
    
    # Estandarización (Crucial para PCA/t-SNE)
    # Se eliminan NaNs o Infinitos si existen
    X_total = np.nan_to_num(X_total)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_total)
    
    # Directorio salida
    dir_salida = os.path.join(dir_base, "Proyecciones_2D")
    os.makedirs(dir_salida, exist_ok=True)
    
    # --- PCA ---
    print("Calculando PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_explicada = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 8))
    # Usamos scatter plot
    plt.scatter(X_pca[y_total==0, 0], X_pca[y_total==0, 1], c='blue', alpha=0.5, s=20, label='Relajación')
    plt.scatter(X_pca[y_total==1, 0], X_pca[y_total==1, 1], c='red', alpha=0.5, s=20, label='Ansiedad')
    
    plt.title(f'PCA - Análisis de Componentes Principales\nVar. Explicada: {sum(var_explicada)*100:.2f}%')
    plt.xlabel(f'PC1 ({var_explicada[0]*100:.2f}%)')
    plt.ylabel(f'PC2 ({var_explicada[1]*100:.2f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(dir_salida, "Proyeccion_PCA.png"))
    plt.close()
    
    # --- t-SNE ---
    print("Calculando t-SNE (esto puede tardar un poco)...")
    # Reducir dimensionalidad previa con PCA si hay muchas features para acelerar t-SNE
    # Si hay > 50 features, se recomienda PCA previo a 50 componentes
    if X_scaled.shape[1] > 50:
        pca_50 = PCA(n_components=50)
        X_pca_50 = pca_50.fit_transform(X_scaled)
        X_para_tsne = X_pca_50
    else:
        X_para_tsne = X_scaled

    # Sampling aleatorio si hay demasiados puntos (>10k) para que no sea eterno
    MAX_MUESTRAS = 5000
    if X_para_tsne.shape[0] > MAX_MUESTRAS:
        indices = np.random.choice(X_para_tsne.shape[0], MAX_MUESTRAS, replace=False)
        X_para_tsne = X_para_tsne[indices]
        y_labels_tsne = y_total[indices]
    else:
        y_labels_tsne = y_total
            
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_para_tsne)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[y_labels_tsne==0, 0], X_tsne[y_labels_tsne==0, 1], c='blue', alpha=0.5, s=20, label='Relajación')
    plt.scatter(X_tsne[y_labels_tsne==1, 0], X_tsne[y_labels_tsne==1, 1], c='red', alpha=0.5, s=20, label='Ansiedad')
    
    plt.title('t-SNE - Visualización de Separabilidad No Lineal')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(dir_salida, "Proyeccion_tSNE.png"))
    plt.close()
    
    print(f"Gráficos de proyección guardados en {dir_salida}")

def visualizar_proyecciones_individuales():
    '''
    Genera PCA y t-SNE separadas por Grupo de Características (Time, Freq...)
    y por Región Cerebral (Frontal, Parietal), para identificar mejor separabilidad.
    Estructura de datos asume Feature-Major (Bloques de 32 canales por feature).
    '''
    print("Generando proyecciones detalladas por Grupo y Región...")
    dir_base = "Resultados"
    
    # Configuración de Grupos
    # Nombre del grupo -> Cantidad de sub-features
    # El orden de concatenación en main.py es: Time, Freq, Hjorth, Fractal
    info_grupos = [
        ('time', 3),       # Var, RMS, PTP
        ('frequency', 5),  # Delta, Theta, Alpha, Beta, Gamma
        ('hjorth', 2),     # Mobility, Complexity
        ('fractal', 2)     # Higuchi, Katz
    ]
    
    # Cargar datos base
    ruta_relax_X = os.path.join(dir_base, "Relax", "X.npy")
    ruta_stress_X = os.path.join(dir_base, "Arithmetic", "X.npy")
    ruta_stress_y = os.path.join(dir_base, "Arithmetic", "y.npy")
    
    if not os.path.exists(ruta_relax_X): return
    
    X_relax_full = np.load(ruta_relax_X)
    X_stress_full = np.load(ruta_stress_X)
    y_stress = np.load(ruta_stress_y)
    
    # Filtrar Ansiedad
    X_ansiedad_full = X_stress_full[y_stress == 1]
    if X_ansiedad_full.shape[0] == 0: X_ansiedad_full = X_stress_full
    
    # Identificar Canales
    ruta_locs = "Conjunto de datos/Data/Coordinates.locs"
    nombres_canales = [f"Ch{i+1}" for i in range(32)]
    if os.path.exists(ruta_locs):
        try:
            df_locs = pd.read_csv(ruta_locs, sep=r'\s+', header=None, names=['idx', 'theta', 'radius', 'etiqueta'], engine='python')
            nombres_canales = df_locs['etiqueta'].str.strip().tolist()
        except: pass
        
    regiones = {
        'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
        'Parietal': ['P7', 'P3', 'Pz', 'P4', 'P8']
    }
    
    indices_regiones = {}
    for r, lista in regiones.items():
        idxs = [nombres_canales.index(ch) for ch in lista if ch in nombres_canales]
        if idxs: indices_regiones[r] = idxs

    dir_salida = os.path.join(dir_base, "Proyecciones_Regionales")
    os.makedirs(dir_salida, exist_ok=True)
    
    # Offset global para saber dónde empieza cada grupo en la matriz concatenada gigante
    offset_global = 0
    n_canales = 32
    
    # Iterar por Grupo
    for nombre_grupo, n_subfeats in info_grupos:
        # Calcular ancho de este grupo en columnas
        ancho_grupo = n_subfeats * n_canales
        
        # Extraer sub-matriz solo para este grupo
        # X_relax_full tiene shape (N, Total_Cols)
        # Slicing: [:, offset : offset+ancho]
        X_relax_grupo = X_relax_full[:, offset_global : offset_global + ancho_grupo]
        X_ansiedad_grupo = X_ansiedad_full[:, offset_global : offset_global + ancho_grupo]
        
        offset_global += ancho_grupo # Actualizar para el siguiente grupo
        
        # Iterar por Región
        for nombre_region, idxs_canales in indices_regiones.items():
            print(f"  Analizando {nombre_grupo} - {nombre_region}...")
            
            # Construir máscara de columnas para esta región dentro del grupo
            # Asumimos Feature-Major: [F1_C1...F1_C32, F2_C1...F2_C32...]
            cols_a_usar = []
            for f in range(n_subfeats):
                offset_feat = f * n_canales
                for ch in idxs_canales:
                    cols_a_usar.append(offset_feat + ch)
            
            X_rel = X_relax_grupo[:, cols_a_usar]
            X_ans = X_ansiedad_grupo[:, cols_a_usar]
            
            # Preparar dataset combinado
            X_total = np.concatenate([X_rel, X_ans], axis=0)
            y_total = np.concatenate([np.zeros(len(X_rel)), np.ones(len(X_ans))])
            
            # Limpieza y Scaling
            X_total = np.nan_to_num(X_total)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_total)
            
            # PCA
            try:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                var = pca.explained_variance_ratio_
                
                plt.figure(figsize=(8, 6))
                plt.scatter(X_pca[y_total==0, 0], X_pca[y_total==0, 1], c='blue', alpha=0.4, s=15, label='Relax')
                plt.scatter(X_pca[y_total==1, 0], X_pca[y_total==1, 1], c='red', alpha=0.4, s=15, label='Ansiedad')
                plt.title(f'PCA: {nombre_grupo} ({nombre_region})\nVar: {sum(var)*100:.1f}%')
                plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(dir_salida, f"PCA_{nombre_grupo}_{nombre_region}.png"))
                plt.close()
                
                # t-SNE (Solo si hay suficientes datos/dims, reduciendo antes con PCA si es necesario)
                # t-SNE es lento, hacemos downsample si > 3000 muestras
                if X_scaled.shape[0] > 3000:
                    idx_rand = np.random.choice(X_scaled.shape[0], 3000, replace=False)
                    X_tsne_input = X_scaled[idx_rand]
                    y_tsne_labels = y_total[idx_rand]
                else:
                    X_tsne_input = X_scaled
                    y_tsne_labels = y_total
                
                # PCA previo a 30 dims si hay muchas columnas (para acelerar t-SNE)
                if X_tsne_input.shape[1] > 30:
                    X_tsne_input = PCA(n_components=30).fit_transform(X_tsne_input)
                    
                tsne = TSNE(n_components=2, perplexity=30, n_iter=600, init='pca', learning_rate='auto')
                X_tsne = tsne.fit_transform(X_tsne_input)
                
                plt.figure(figsize=(8, 6))
                plt.scatter(X_tsne[y_tsne_labels==0, 0], X_tsne[y_tsne_labels==0, 1], c='blue', alpha=0.4, s=15, label='Relax')
                plt.scatter(X_tsne[y_tsne_labels==1, 0], X_tsne[y_tsne_labels==1, 1], c='red', alpha=0.4, s=15, label='Ansiedad')
                plt.title(f't-SNE: {nombre_grupo} ({nombre_region})')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(dir_salida, f"tSNE_{nombre_grupo}_{nombre_region}.png"))
                plt.close()
                
            except Exception as e:
                print(f"Error en proyección {nombre_grupo}-{nombre_region}: {e}")

def visualizar_proyecciones_por_canal_especifico():
    '''
    Genera PCA y t-SNE para cada uno de los 12 canales de interés (Frontales/Parietales)
    separando por Grupo de Características.
    Finalidad: Determinar qué canal específico y qué métrica tiene mayor poder discriminante.
    '''
    print("Generando proyecciones detalladas por CANAL INDIVIDUAL...")
    dir_base = "Resultados"
    
    # 12 Canales de interés
    canales_interes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'P7', 'P3', 'Pz', 'P4', 'P8']
    
    info_grupos = [
        ('time', 3),       # Var, RMS, PTP
        ('frequency', 5),  # Delta, Theta, Alpha, Beta, Gamma
        ('hjorth', 2),     # Mobility, Complexity
        ('fractal', 2)     # Higuchi, Katz
    ]
    
    # Cargar datos base
    ruta_relax_X = os.path.join(dir_base, "Relax", "X.npy")
    ruta_stress_X = os.path.join(dir_base, "Arithmetic", "X.npy")
    ruta_stress_y = os.path.join(dir_base, "Arithmetic", "y.npy")
    
    if not os.path.exists(ruta_relax_X): return
    X_relax_full = np.load(ruta_relax_X)
    X_stress_full = np.load(ruta_stress_X)
    y_stress = np.load(ruta_stress_y)
    X_ansiedad_full = X_stress_full[y_stress == 1]
    
    # Mapear nombres a indices
    nombres_canales = [f"Ch{i+1}" for i in range(32)]
    ruta_locs = "Conjunto de datos/Data/Coordinates.locs"
    if os.path.exists(ruta_locs):
        try:
            df_locs = pd.read_csv(ruta_locs, sep=r'\s+', header=None, names=['idx', 'theta', 'radius', 'etiqueta'], engine='python')
            nombres_canales = df_locs['etiqueta'].str.strip().tolist()
        except: pass
        
    indices_map = {}
    for ch in canales_interes:
        if ch in nombres_canales:
            indices_map[ch] = nombres_canales.index(ch)
            
    if not indices_map:
        print("No se encontraron los canales especificados en .locs")
        return

    dir_salida = os.path.join(dir_base, "Proyecciones_Canales_Detalle")
    os.makedirs(dir_salida, exist_ok=True)
    
    # Datos completos
    offset_global = 0
    n_canales = 32
    
    for nombre_grupo, n_subfeats in info_grupos:
        ancho_grupo = n_subfeats * n_canales
        # Slicing del grupo entero
        X_relax_grupo = X_relax_full[:, offset_global : offset_global + ancho_grupo]
        X_ansiedad_grupo = X_ansiedad_full[:, offset_global : offset_global + ancho_grupo]
        
        offset_global += ancho_grupo
        
        print(f"  Analizando Grupo: {nombre_grupo}...")
        
        for ch_nombre, ch_idx in indices_map.items():
            # Construir columnas para este canal dentro de este grupo
            # Feature-Major: [Feat1_Ch0...Feat1_Ch31, Feat2_Ch0...]
            cols_canal = []
            for f in range(n_subfeats):
                offset_feat = f * n_canales
                cols_canal.append(offset_feat + ch_idx)
                
            X_rel_ch = X_relax_grupo[:, cols_canal]
            X_ans_ch = X_ansiedad_grupo[:, cols_canal]
            
            # Combinar
            X_total = np.concatenate([X_rel_ch, X_ans_ch], axis=0)
            y_total = np.concatenate([np.zeros(len(X_rel_ch)), np.ones(len(X_ans_ch))])
            
            # Limpieza y Scaling
            X_total = np.nan_to_num(X_total)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_total)
            
            try:
                # PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                var = pca.explained_variance_ratio_
                
                plt.figure(figsize=(6, 5))
                plt.scatter(X_pca[y_total==0, 0], X_pca[y_total==0, 1], c='blue', alpha=0.4, s=10, label='Relax')
                plt.scatter(X_pca[y_total==1, 0], X_pca[y_total==1, 1], c='red', alpha=0.4, s=10, label='Ansiedad')
                plt.title(f'PCA: {nombre_grupo} - {ch_nombre}\nVar: {sum(var)*100:.1f}%')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(dir_salida, f"PCA_{nombre_grupo}_{ch_nombre}.png"))
                plt.close()
                
                # t-SNE (con downsample si es grande)
                if X_scaled.shape[0] > 3000:
                    idx = np.random.choice(X_scaled.shape[0], 3000, replace=False)
                    X_tsne_in = X_scaled[idx]
                    y_tsne_lab = y_total[idx]
                else:
                    X_tsne_in = X_scaled
                    y_tsne_lab = y_total
                    
                tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
                X_tsne = tsne.fit_transform(X_tsne_in)
                
                plt.figure(figsize=(6, 5))
                plt.scatter(X_tsne[y_tsne_lab==0, 0], X_tsne[y_tsne_lab==0, 1], c='blue', alpha=0.4, s=10, label='Relax')
                plt.scatter(X_tsne[y_tsne_lab==1, 0], X_tsne[y_tsne_lab==1, 1], c='red', alpha=0.4, s=10, label='Ansiedad')
                plt.title(f't-SNE: {nombre_grupo} - {ch_nombre}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(dir_salida, f"tSNE_{nombre_grupo}_{ch_nombre}.png"))
                plt.close()
                
            except Exception as e:
                print(f"Skip {ch_nombre}: {e}")

if __name__ == "__main__":
    visualizar_proyecciones_por_canal_especifico()
    # visualizar_proyecciones_individuales()
    # visualizar_proyeccion_dimensionalidad()
    # analizar_todas_caracteristicas_relax_ansiedad()
