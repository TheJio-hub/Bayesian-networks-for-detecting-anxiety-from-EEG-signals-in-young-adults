import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calcular_fisher_score(X, y):
    clases = np.unique(y)
    if len(clases) != 2:
        raise ValueError("Fisher Score requiere clasificación binaria.")
    
    c0 = X[y == clases[0]]
    c1 = X[y == clases[1]]
    
    mu0 = c0.mean()
    mu1 = c1.mean()
    var0 = c0.var()
    var1 = c1.var()
    
    fisher = ((mu0 - mu1)**2) / (var0 + var1)
    return fisher

def calcular_mutual_information(X, y):
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    return pd.Series(mi, index=X.columns)

def seleccion_mrmr_rapido(X, y, n_seleccion):
    n_caracteristicas = X.shape[1]
    nombres_caracteristicas = list(X.columns)
    
    indices_seleccionados = []
    indices_candidatos = list(range(n_caracteristicas))
    
    relevancia = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    
    redundancia_acumulada = np.zeros(n_caracteristicas)
    
    primera = np.argmax(relevancia)
    indices_seleccionados.append(primera)
    indices_candidatos.remove(primera)
    
    for i in range(1, n_seleccion):
        idx_ultimo = indices_seleccionados[-1]
        dat_ultimo = X.iloc[:, idx_ultimo].values.reshape(-1, 1)
        
        for idx_cand in indices_candidatos:
            dat_cand = X.iloc[:, idx_cand].values.reshape(-1, 1)
            mi_feat = mutual_info_regression(dat_cand, dat_ultimo.ravel(), discrete_features=False, random_state=42)[0]
            redundancia_acumulada[idx_cand] += mi_feat
            
        mejor_score = -np.inf
        mejor_idx = -1
        
        for idx_cand in indices_candidatos:
            red_promedio = redundancia_acumulada[idx_cand] / len(indices_seleccionados)
            score = relevancia[idx_cand] - red_promedio
            if score > mejor_score:
                mejor_score = score
                mejor_idx = idx_cand
        
        indices_seleccionados.append(mejor_idx)
        indices_candidatos.remove(mejor_idx)
            
    return [nombres_caracteristicas[i] for i in indices_seleccionados]

def analizar_modelos_y_ramas(X, y, grupos, output_dir):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X, y)
    imp_rf = pd.DataFrame({'Caracteristica': X.columns, 'Importancia_RF': rf.feature_importances_})
    imp_rf = imp_rf.sort_values('Importancia_RF', ascending=False)
    
    dt = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5)
    dt.fit(X, y)
    imp_dt = pd.DataFrame({'Caracteristica': X.columns, 'Importancia_DT': dt.feature_importances_})
    imp_dt = imp_dt.sort_values('Importancia_DT', ascending=False)
    
    reglas = export_text(dt, feature_names=list(X.columns))
    
    with open(os.path.join(output_dir, "Arbol_Decision_Reglas.txt"), "w") as f:
        f.write(reglas)
        
    return imp_rf, imp_dt

def main():
    archivo_entrada = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    directorio_salida = os.path.join('Resultados', 'Análisis global')
    
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        
    if not os.path.exists(archivo_entrada):
        return

    df = pd.read_parquet(archivo_entrada)
    
    df = df[(df['Puntaje'] == 0) | (df['Puntaje'] >= 5)].copy()
    y = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
    grupos = df['Sujeto'].values
    
    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    feats = [c for c in df.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats]
    
    fisher = calcular_fisher_score(X, y)
    fisher_df = pd.DataFrame({'Caracteristica': fisher.index, 'Fisher_Score': fisher.values})
    
    mi = calcular_mutual_information(X, y)
    mi_df = pd.DataFrame({'Caracteristica': mi.index, 'Mutual_Info': mi.values})
    
    top_n_mrmr = 30
    ranking_mrmr = seleccion_mrmr_rapido(X, y, top_n_mrmr)
    mrmr_df = pd.DataFrame({'Caracteristica': ranking_mrmr, 'mRMR_Rank': range(1, top_n_mrmr + 1)})
    
    imp_rf, imp_dt = analizar_modelos_y_ramas(X, y, grupos, directorio_salida)
    
    maestra = fisher_df.merge(mi_df, on='Caracteristica', how='outer')
    maestra = maestra.merge(mrmr_df, on='Caracteristica', how='outer')
    maestra = maestra.merge(imp_rf, on='Caracteristica', how='outer')
    maestra = maestra.merge(imp_dt, on='Caracteristica', how='outer')
    
    maestra = maestra.sort_values('Fisher_Score', ascending=False)
    
    archivo_maestro = os.path.join(directorio_salida, 'Ranking_Multicriterio_Completo.csv')
    maestra.to_csv(archivo_maestro, index=False)

if __name__ == "__main__":
    main()
