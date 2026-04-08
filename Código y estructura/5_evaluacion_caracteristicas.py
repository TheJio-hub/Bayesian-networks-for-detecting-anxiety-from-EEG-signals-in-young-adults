import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif

def fisher_score_func(X, y):
    classes = np.unique(y)
    c0 = classes[0]
    c1 = classes[1]
    
    X0 = X[y == c0]
    X1 = X[y == c1]
    
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    
    var0 = X0.var(axis=0)
    var1 = X1.var(axis=0)
    
    denom = var0 + var1
    denom[denom == 0] = 1e-10
    
    fisher = ((mu0 - mu1)**2) / denom
    return fisher.fillna(0)

def evaluar_caracteristicas_por_banda():
    input_file = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    output_dir = os.path.join('Resultados', 'Ranking')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(input_file):
        input_file = os.path.join('Resultados', 'datos_bandas.parquet')
        if not os.path.exists(input_file):
            return

    df = pd.read_parquet(input_file)

    if 'Puntaje' in df.columns:
        df_filtered = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
    else:
        df_filtered = df.copy()
    
    y = df_filtered['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
    
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    all_features = [c for c in df_filtered.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df_filtered[c])]

    for banda in bandas:
        
        cols_banda = [c for c in all_features if c.endswith(f"_{banda}")]
        
        if not cols_banda:
            continue
            
        X_banda = df_filtered[cols_banda]
        
        # Fisher Score
        try:
            f_scores = fisher_score_func(X_banda, y)
            
            df_fisher = pd.DataFrame({
                'Canal': [c.replace(f"_{banda}", "") for c in cols_banda],
                'Fisher_Score': f_scores.values
            }).sort_values(by='Fisher_Score', ascending=False)
            
            file_fisher = os.path.join(output_dir, f'Fisher_{banda}.csv')
            df_fisher.to_csv(file_fisher, index=False)
        except Exception as e:
            print(f"Error Fisher {banda}: {e}")
        
        # Mutual Information (MMI)
        try:
            m_scores = mutual_info_classif(X_banda, y, discrete_features=False, random_state=42)
            
            df_mmi = pd.DataFrame({
                'Canal': [c.replace(f"_{banda}", "") for c in cols_banda],
                'MMI_Score': m_scores
            }).sort_values(by='MMI_Score', ascending=False)
            
            file_mmi = os.path.join(output_dir, f'MMI_{banda}.csv')
            df_mmi.to_csv(file_mmi, index=False)
            
        except Exception as e:
            print(f"Error MMI {banda}: {e}")
        except Exception as e:
            print(f"Error calculando MMI para {banda}: {e}")

if __name__ == "__main__":
    evaluar_caracteristicas_por_banda()
