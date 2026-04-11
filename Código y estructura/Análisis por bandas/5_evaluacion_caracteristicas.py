import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

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


def columnas_ratio_por_banda(columnas, banda):
    seleccionadas = []
    for col in columnas:
        if not col.startswith('Ratio_'):
            continue
        partes = col.split('_')
        if len(partes) < 3:
            continue
        tipo_ratio = partes[1]
        if banda in tipo_ratio:
            seleccionadas.append(col)
    return seleccionadas


def guardar_ranking_fisher_mi(X, y, ruta_fisher, ruta_mi):
    if X.shape[1] == 0:
        pd.DataFrame(columns=['Caracteristica', 'Fisher_Score']).to_csv(ruta_fisher, index=False)
        pd.DataFrame(columns=['Caracteristica', 'MI_Score']).to_csv(ruta_mi, index=False)
        return

    f_scores = fisher_score_func(X, y)
    df_fisher = pd.DataFrame({
        'Caracteristica': X.columns,
        'Fisher_Score': f_scores.values
    }).sort_values(by='Fisher_Score', ascending=False)
    df_fisher.to_csv(ruta_fisher, index=False)

    m_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    df_mi = pd.DataFrame({
        'Caracteristica': X.columns,
        'MI_Score': m_scores
    }).sort_values(by='MI_Score', ascending=False)
    df_mi.to_csv(ruta_mi, index=False)


def guardar_ranking_dt(X, y, ruta_dt):
    if X.shape[1] == 0:
        pd.DataFrame(columns=['Caracteristica', 'Importancia_DT']).to_csv(ruta_dt, index=False)
        return

    X_num = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    dt = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5)
    dt.fit(X_num, y)
    df_dt = pd.DataFrame({
        'Caracteristica': X.columns,
        'Importancia_DT': dt.feature_importances_
    }).sort_values(by='Importancia_DT', ascending=False)
    df_dt.to_csv(ruta_dt, index=False)

def evaluar_caracteristicas_por_banda():
    input_file = os.path.join('Resultados', 'Exploratorio', 'datos_bandas_normalizados.parquet')
    input_asimetria = os.path.join('Resultados', 'Exploratorio', 'datos_asimetria_normalizados.parquet')
    input_ratios = os.path.join('Resultados', 'Exploratorio', 'datos_ratios_normalizados.parquet')
    output_dir = os.path.join('Resultados', 'Análisis por bandas', 'Ranking de características')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(input_file):
        input_file = os.path.join('Resultados', 'datos_bandas.parquet')
        if not os.path.exists(input_file):
            return

    df = pd.read_parquet(input_file)

    if 'Puntaje' in df.columns:
        df_filtered = df[(df['Puntaje'] == 0) | (df['Puntaje'] >= 5)].copy()
    else:
        df_filtered = df.copy()

    y = df_filtered['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values

    df_asim = pd.read_parquet(input_asimetria) if os.path.exists(input_asimetria) else pd.DataFrame()
    if not df_asim.empty and 'Puntaje' in df_asim.columns:
        df_asim = df_asim[(df_asim['Puntaje'] == 0) | (df_asim['Puntaje'] >= 5)].copy()

    df_ratio = pd.read_parquet(input_ratios) if os.path.exists(input_ratios) else pd.DataFrame()
    if not df_ratio.empty and 'Puntaje' in df_ratio.columns:
        df_ratio = df_ratio[(df_ratio['Puntaje'] == 0) | (df_ratio['Puntaje'] >= 5)].copy()
    
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    all_features = [c for c in df_filtered.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df_filtered[c])]

    for banda in bandas:
        
        cols_banda = [c for c in all_features if c.endswith(f"_{banda}")]
        
        if not cols_banda:
            continue
            
        X_banda = df_filtered[cols_banda]
        os.makedirs(os.path.join(output_dir, banda), exist_ok=True)

        try:
            guardar_ranking_fisher_mi(
                X_banda,
                y,
                os.path.join(output_dir, banda, f'Fisher_{banda}.csv'),
                os.path.join(output_dir, banda, f'MI_{banda}.csv')
            )
        except Exception as e:
            print(f"Error Fisher/MI {banda}: {e}")

        try:
            guardar_ranking_dt(
                X_banda,
                y,
                os.path.join(output_dir, banda, f'DT_{banda}.csv')
            )
        except Exception as e:
            print(f"Error DT {banda}: {e}")

        # Asimetria por banda
        dir_asim = os.path.join(output_dir, banda, 'Asimetria')
        os.makedirs(dir_asim, exist_ok=True)
        cols_asim = [c for c in df_asim.columns if c.startswith('Asym_') and c.endswith(f'_{banda}')] if not df_asim.empty else []
        X_asim = df_asim[cols_asim] if cols_asim else pd.DataFrame(index=df_filtered.index)

        try:
            guardar_ranking_fisher_mi(
                X_asim,
                y,
                os.path.join(dir_asim, f'Fisher_Asimetria_{banda}.csv'),
                os.path.join(dir_asim, f'MI_Asimetria_{banda}.csv')
            )
        except Exception as e:
            print(f"Error Fisher/MI Asimetria {banda}: {e}")

        try:
            guardar_ranking_dt(
                X_asim,
                y,
                os.path.join(dir_asim, f'DT_Asimetria_{banda}.csv')
            )
        except Exception as e:
            print(f"Error DT Asimetria {banda}: {e}")

        # Ratios por banda
        dir_ratios = os.path.join(output_dir, banda, 'Ratios')
        os.makedirs(dir_ratios, exist_ok=True)
        cols_ratio = columnas_ratio_por_banda(df_ratio.columns.tolist(), banda) if not df_ratio.empty else []
        X_ratio = df_ratio[cols_ratio] if cols_ratio else pd.DataFrame(index=df_filtered.index)

        try:
            guardar_ranking_fisher_mi(
                X_ratio,
                y,
                os.path.join(dir_ratios, f'Fisher_Ratios_{banda}.csv'),
                os.path.join(dir_ratios, f'MI_Ratios_{banda}.csv')
            )
        except Exception as e:
            print(f"Error Fisher/MI Ratios {banda}: {e}")

        try:
            guardar_ranking_dt(
                X_ratio,
                y,
                os.path.join(dir_ratios, f'DT_Ratios_{banda}.csv')
            )
        except Exception as e:
            print(f"Error DT Ratios {banda}: {e}")

if __name__ == "__main__":
    evaluar_caracteristicas_por_banda()
