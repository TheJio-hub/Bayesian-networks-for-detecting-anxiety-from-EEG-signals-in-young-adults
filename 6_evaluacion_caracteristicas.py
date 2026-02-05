import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif

def fisher_score_func(X, y):
    """
    Calcula el Fisher Score para cada característica.
    F = (mu1 - mu2)^2 / (sigma1^2 + sigma2^2)
    """
    # Asegurar que y sea numérico para indexación
    classes = np.unique(y)
    # Suponemos binario
    c0 = classes[0]
    c1 = classes[1]
    
    X0 = X[y == c0]
    X1 = X[y == c1]
    
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    
    var0 = X0.var(axis=0)
    var1 = X1.var(axis=0)
    
    # Evitar división por cero
    denom = var0 + var1
    denom[denom == 0] = 1e-10
    
    fisher = ((mu0 - mu1)**2) / denom
    return fisher.fillna(0)

def evaluar_caracteristicas_por_banda():
    # 1. Configuración de Rutas
    input_file = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    output_dir = os.path.join('Resultados', 'Ranking')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(input_file):
        # Fallback si no existe el normalizado, buscar el crudo
        input_file = os.path.join('Resultados', 'datos_bandas.parquet')
        if not os.path.exists(input_file):
            print(f"Error: No se encuentra {input_file}")
            return

    print(f"Cargando dataset: {input_file}")
    df = pd.read_parquet(input_file)

    # 2. Filtrado de Clases (Relajación (0) vs Ansiedad (>=5))
    print("Filtrando Clases: Relajación (0) vs Ansiedad (>=5)...")
    # Aseguramos que solo filtramos si hay datos mixtos, aunque deberia venir limpio del script 4
    if 'Puntaje' in df.columns:
        df_filtered = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
    else:
        print("Advertencia: No se encontró columna 'Puntaje'. Usando dataset completo.")
        df_filtered = df.copy()
    
    
    # Crear vector de etiquetas (0 -> Clase 0, 1 -> Clase 1)
    # Nota: Si ya filtramos, el grupo "Ansiedad" son los >=5.
    y = df_filtered['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
    
    print(f"Muestras después del filtrado: {len(df_filtered)}")
    print(f"Distribución: Relajación (0)={sum(y==0)}, Ansiedad (1)={sum(y==1)}")

    # 3. Definición de Bandas y Features
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    # Identificar columnas de características (excluir meta)
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    all_features = [c for c in df_filtered.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df_filtered[c])]

    for banda in bandas:
        print(f"\n--- Procesando Banda: {banda} ---")
        
        # Filtrar columnas que pertenecen a esta banda (sufijo _Banda)
        # Asumimos formato "Canal_Banda"
        cols_banda = [c for c in all_features if c.endswith(f"_{banda}")]
        
        if not cols_banda:
            print(f"Advertencia: No se encontraron columnas para la banda {banda}")
            continue
            
        X_banda = df_filtered[cols_banda]
        
        # --- A. Fisher Score ---
        print(f"Calculando Fisher Score ({len(cols_banda)} canales)...")
        try:
            f_scores = fisher_score_func(X_banda, y)
            
            df_fisher = pd.DataFrame({
                'Canal': [c.replace(f"_{banda}", "") for c in cols_banda],
                'Fisher_Score': f_scores.values
            }).sort_values(by='Fisher_Score', ascending=False)
            
            # Guardar Fisher
            file_fisher = os.path.join(output_dir, f'Fisher_{banda}.csv')
            df_fisher.to_csv(file_fisher, index=False)
            print(f"Guardado: {file_fisher}")
        except Exception as e:
            print(f"Error calculando Fisher para {banda}: {e}")
        
        # --- B. Mutual Information (MMI) ---
        print(f"Calculando MMI ({len(cols_banda)} canales)...")
        try:
            m_scores = mutual_info_classif(X_banda, y, discrete_features=False, random_state=42)
            
            df_mmi = pd.DataFrame({
                'Canal': [c.replace(f"_{banda}", "") for c in cols_banda],
                'MMI_Score': m_scores
            }).sort_values(by='MMI_Score', ascending=False)
            
            # Guardar MMI
            file_mmi = os.path.join(output_dir, f'MMI_{banda}.csv')
            df_mmi.to_csv(file_mmi, index=False)
            print(f"Guardado: {file_mmi}")
        except Exception as e:
            print(f"Error calculando MMI para {banda}: {e}")

    print("\nProceso completado. Se han generado los archivos en Resultados/Ranking.")

if __name__ == "__main__":
    evaluar_caracteristicas_por_banda()
