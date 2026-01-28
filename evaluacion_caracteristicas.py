import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Configuración: Usar ruta absoluta basada en la ubicación del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_BASE = os.path.join(SCRIPT_DIR, "Resultados")

RUTA_RELAX_X = os.path.join(DIR_BASE, "Relax", "X.npy")
RUTA_STRESS_X = os.path.join(DIR_BASE, "Arithmetic", "X.npy")
RUTA_STRESS_Y = os.path.join(DIR_BASE, "Arithmetic", "y.npy")
RUTA_LOCS = os.path.join(SCRIPT_DIR, "Conjunto de datos", "Data", "Coordinates.locs")

# Estructura de características (Debe coincidir con extracción)
N_CANALES = 32
INFO_GRUPOS = [ # Nombre Grupo, Subfeatures, Nombres Subfeatures
    ('Time', 3, ['Var', 'RMS', 'PTP']),
    ('Frequency', 5, ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']),
    ('Hjorth', 2, ['Mobility', 'Complexity']),
    ('Fractal', 2, ['Higuchi', 'Katz'])
]

def cargar_datos_y_etiquetas():
    print("Cargando datos...")
    
    # 1. Cargar Relax (Clase 0)
    # Ya asumimos que el preprocesamiento guardó los primeros 25s
    if not os.path.exists(RUTA_RELAX_X):
        raise FileNotFoundError(f"No se encontró {RUTA_RELAX_X}")
    X_relax = np.load(RUTA_RELAX_X)
    y_relax = np.zeros(X_relax.shape[0])
    
    # 2. Cargar Stress (Clase 1)
    if not os.path.exists(RUTA_STRESS_X):
        raise FileNotFoundError(f"No se encontró {RUTA_STRESS_X}")
    X_stress = np.load(RUTA_STRESS_X)
    y_labels = np.load(RUTA_STRESS_Y)
    
    # Filtrar solo clase Ansioso (y=1) del dataset Arithmetic
    # Según instrucción: "clase 1: ansioso"
    mask_ansioso = (y_labels == 1)
    X_ansioso = X_stress[mask_ansioso]
    y_ansioso = np.ones(X_ansioso.shape[0])
    
    print(f"Muestras Relax (Clase 0): {X_relax.shape[0]}")
    print(f"Muestras Ansioso (Clase 1): {X_ansioso.shape[0]}")
    
    # Concatenar
    X = np.concatenate([X_relax, X_ansioso], axis=0)
    y = np.concatenate([y_relax, y_ansioso], axis=0)
    
    # Limpieza básica de NaNs/Infs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y

def obtener_nombres_canales():
    nombres = [f"Ch{i+1}" for i in range(N_CANALES)]
    if os.path.exists(RUTA_LOCS):
        try:
            df = pd.read_csv(RUTA_LOCS, sep=r'\s+', header=None, names=['idx', 'theta', 'radius', 'etiqueta'], engine='python')
            nombres = df['etiqueta'].str.strip().tolist()
        except:
            pass
    return nombres

def generar_nombres_features(nombres_canales):
    nombres_cols = []
    
    # El orden de features en X es: 
    # [ G1_S1_C1...G1_S1_C32, G1_S2_C1... ]
    
    for grupo, n_sub, sub_names in INFO_GRUPOS:
        for sub_idx in range(n_sub):
            sub_nombre = sub_names[sub_idx]
            for canal in nombres_canales:
                nombre_feat = f"{grupo}_{sub_nombre}_{canal}"
                nombres_cols.append(nombre_feat)
                
    return nombres_cols

def fisher_score(X, y):
    '''
    Calcula Fisher Score para cada feature.
    F = (mu1 - mu2)^2 / (sigma1^2 + sigma2^2)
    '''
    # Separar clases
    X0 = X[y == 0]
    X1 = X[y == 1]
    
    mean0 = np.mean(X0, axis=0)
    mean1 = np.mean(X1, axis=0)
    var0 = np.var(X0, axis=0)
    var1 = np.var(X1, axis=0)
    
    # Evitar división por cero
    epsilon = 1e-10
    fisher = ((mean0 - mean1)**2) / (var0 + var1 + epsilon)
    return fisher

def analizar_caracteristicas():
    X, y = cargar_datos_y_etiquetas()
    canales = obtener_nombres_canales()
    nombres_feats = generar_nombres_features(canales)
    
    if X.shape[1] != len(nombres_feats):
        print(f"ADVERTENCIA: Dimensiones no coinciden. X: {X.shape[1]}, Generado: {len(nombres_feats)}")
        # Ajuste de emergencia si no coinciden
        nombres_feats = [f"Feat_{i}" for i in range(X.shape[1])]

    print("Calculando Fisher Score...")
    scores_fisher = fisher_score(X, y)
    
    print("Calculando Información Mutua (esto puede tardar un poco)...")
    # MMI requiere datos discretos o vecinos cercanos. Para continuo, usa métrica por defecto de sklearn.
    scores_mmi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    
    # Crear DataFrame
    df_res = pd.DataFrame({
        'Feature': nombres_feats,
        'Fisher_Score': scores_fisher,
        'Mutual_Info': scores_mmi
    })
    
    # Ordenar por Fisher
    df_fisher = df_res.sort_values(by='Fisher_Score', ascending=False)
    
    # Guardar
    ruta_csv = os.path.join(DIR_BASE, "Evaluacion_Caracteristicas.csv")
    df_fisher.to_csv(ruta_csv, index=False)
    print(f"Resultados guardados en: {ruta_csv}")
    
    # Mostrar Top 10
    print("\n--- CARACTERÍSTICAS (FISHER) ---")
    print(df_fisher[['Feature', 'Fisher_Score', 'Mutual_Info']].head(10))
    
    # Opcional: Graficar Top 20 Fisher
    import matplotlib.pyplot as plt
    try:
        top_20 = df_fisher.head(20)
        plt.figure(figsize=(12, 8))
        plt.barh(top_20['Feature'][::-1], top_20['Fisher_Score'][::-1], color='skyblue')
        plt.title('Características Discriminantes (Criterio Fisher)')
        plt.xlabel('Fisher Score')
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_BASE, "Features_Fisher.png"))
        print("Gráfico generado: Features_Fisher.png")
    except Exception as e:
        print(f"No se pudo graficar: {e}")

if __name__ == "__main__":
    analizar_caracteristicas()
