import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def fisher_score(X, y):
    """
    Calcula el Fisher Score para cada característica en un problema de clasificación binaria.
    F = (mu1 - mu2)^2 / (sigma1^2 + sigma2^2)
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("Fisher Score implementado para clasificación binaria.")
    
    c0 = classes[0]
    c1 = classes[1]
    
    X0 = X[y == c0]
    X1 = X[y == c1]
    
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    
    var0 = X0.var(axis=0)
    var1 = X1.var(axis=0)
    
    fisher = ((mu0 - mu1)**2) / (var0 + var1)
    
    # Manejar posibles divisiones por cero o NaNs
    return fisher.fillna(0)

def evaluar_caracteristicas():
    input_file = os.path.join('Resultados', 'datos_bandas.parquet')
    output_csv = os.path.join('Resultados', 'ranking_caracteristicas.csv')
    
    if not os.path.exists(input_file):
        print(f"Error: No se encuentra {input_file}")
        return

    print(f"Cargando dataset {input_file}...")
    df = pd.read_parquet(input_file)

    print("Generando etiquetas de clase (0: Relajacion, 1: Ansiedad)...")
    y = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1)
    
    # Separar características
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    # Filtrar solo columnas numéricas que no sean meta
    X = df.drop(columns=[c for c in cols_meta if c in df.columns])
    
    feature_names = X.columns.tolist()
    print(f"Evaluando {len(feature_names)} características...")
    
    # 1. Calcular Fisher Score
    print("Calculando Fisher Score...")
    scores_fisher = fisher_score(X, y)
    
    # 2. Calcular Información Mutua (MMI)
    print("Calculando Información Mutua ...")
    # discrete_features=False porque las bandas son continuas
    scores_mmi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    
    # Crear DataFrame de resultados
    df_ranking = pd.DataFrame({
        'Caracteristica': feature_names,
        'Fisher_Score': scores_fisher.values,
        'Mutual_Information': scores_mmi
    })
    
    # Ordenar por Información Mutua descendente
    df_ranking = df_ranking.sort_values(by='Mutual_Information', ascending=False)
    
    print("\nInformación Mutua:")
    print(df_ranking.head(10))
    
    print("\nFisher Score:")
    print(df_ranking.sort_values(by='Fisher_Score', ascending=False).head(10))
    
    df_ranking.to_csv(output_csv, index=False)

if __name__ == "__main__":
    evaluar_caracteristicas()
