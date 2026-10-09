import os
import re
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier, export_text


BANDAS_ARTICULO = ('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
RAZONES_ARTICULO = ('ThetaBeta', 'ThetaAlpha', 'AlphaBeta')
NUM_CARACTERISTICAS_ARTICULO = 306
SEED_PARTICION = 42
N_SUJETOS_DESARROLLO = 32


def obtener_caracteristicas_articulo(columnas):
    """Devuelve exactamente las 306 variables especificadas en el artículo."""
    columnas = list(columnas)
    potencias = [
        c for c in columnas
        if not c.startswith(('Asym_', 'Ratio_'))
        and any(re.fullmatch(rf'.+_{banda}', c) for banda in BANDAS_ARTICULO)
    ]
    asimetrias = [
        c for c in columnas
        if c.startswith('Asym_')
        and any(c.endswith(f'_{banda}') for banda in BANDAS_ARTICULO)
    ]
    razones = [
        c for c in columnas
        if c.startswith('Ratio_')
        and any(c.startswith(f'Ratio_{razon}_') for razon in RAZONES_ARTICULO)
    ]

    caracteristicas = potencias + asimetrias + razones
    conteos = (len(potencias), len(asimetrias), len(razones))
    if conteos != (160, 50, 96) or len(caracteristicas) != NUM_CARACTERISTICAS_ARTICULO:
        raise ValueError(
            'El conjunto de características no coincide con el artículo: '
            f'potencias={conteos[0]}, asimetrías={conteos[1]}, '
            f'razones={conteos[2]}, total={len(caracteristicas)}; '
            'se esperan 160, 50, 96 y 306, respectivamente.'
        )
    return caracteristicas


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

def seleccion_mrmr_rapido(X, y, n_seleccion, desc='mRMR'):
    n_caracteristicas = X.shape[1]
    nombres_caracteristicas = list(X.columns)

    if n_caracteristicas == 0:
        return []

    n_seleccion = min(n_seleccion, n_caracteristicas)
    
    indices_seleccionados = []
    indices_candidatos = list(range(n_caracteristicas))
    
    relevancia = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    
    redundancia_acumulada = np.zeros(n_caracteristicas, dtype=float)
    
    primera = np.argmax(relevancia)
    indices_seleccionados.append(primera)
    indices_candidatos.remove(primera)
    
    for i in tqdm(range(1, n_seleccion), desc=f"{desc} ({n_caracteristicas})", unit='iter'):
        idx_ultimo = indices_seleccionados[-1]
        dat_ultimo = X.iloc[:, idx_ultimo].values.reshape(-1, 1)
        
        # Una llamada estima la redundancia entre la última característica
        # seleccionada y todas las candidatas; no restringe el conjunto a un
        # prefiltrado por información mutua.
        mi_redundancia = mutual_info_regression(
            X, dat_ultimo.ravel(), discrete_features=False, random_state=42
        )
        redundancia_acumulada += mi_redundancia
            
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

def seleccionar_top_mi(X, y, n_top):
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    ranking_mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return ranking_mi.index[: min(n_top, len(ranking_mi))].tolist()

def construir_ranking_mrmr(X, y, nombre_columna_ranking):
    ranking = seleccion_mrmr_rapido(X, y, X.shape[1], desc=nombre_columna_ranking)
    return pd.DataFrame({
        'Caracteristica': ranking,
        nombre_columna_ranking: range(1, len(ranking) + 1)
    })

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
    directorio_seleccion = os.path.join(directorio_salida, 'Selección de características')
    
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
    if not os.path.exists(directorio_seleccion):
        os.makedirs(directorio_seleccion)
        
    if not os.path.exists(archivo_entrada):
        return

    progreso = tqdm(total=7, desc='Pipeline selección global', unit='paso')

    df = pd.read_parquet(archivo_entrada)
    progreso.update(1)

    df = df[(df['Puntaje'] == 0) | (df['Puntaje'] >= 1)].copy()
    sujetos = np.sort(df['Sujeto'].dropna().unique())
    if len(sujetos) != 40:
        raise ValueError(f'Se esperaban 40 sujetos SAM-40; se encontraron {len(sujetos)}.')
    sujetos_barajados = np.random.RandomState(SEED_PARTICION).permutation(sujetos)
    sujetos_desarrollo = sujetos_barajados[:N_SUJETOS_DESARROLLO]
    sujetos_prueba = sujetos_barajados[N_SUJETOS_DESARROLLO:]
    df = df[df['Sujeto'].isin(sujetos_desarrollo)].copy()
    if df['Sujeto'].nunique() != N_SUJETOS_DESARROLLO:
        raise ValueError('El conjunto de desarrollo no contiene exactamente 32 sujetos.')
    y = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
    grupos = df['Sujeto'].values
    with open(os.path.join(directorio_seleccion, 'Particion_sujetos_32_8.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'semilla': SEED_PARTICION,
            'criterio': 'SAM=0 control; SAM>=1 caso',
            'sujetos_desarrollo_32': list(map(str, sujetos_desarrollo)),
            'sujetos_prueba_externa_8': list(map(str, sujetos_prueba)),
            'uso_del_conjunto': 'El ranking se calcula solo con los sujetos de desarrollo; los ocho sujetos externos no participan.'
        }, f, ensure_ascii=False, indent=2)
    
    feats = obtener_caracteristicas_articulo(df.columns)
    X = df[feats]
    progreso.update(1)
    
    fisher = calcular_fisher_score(X, y)
    fisher_df = pd.DataFrame({'Caracteristica': fisher.index, 'Fisher_Score': fisher.values})
    progreso.update(1)
    
    mi = calcular_mutual_information(X, y)
    mi_df = pd.DataFrame({'Caracteristica': mi.index, 'Mutual_Info': mi.values})
    progreso.update(1)

    mrmr_df = construir_ranking_mrmr(X, y, 'mRMR_Rank')
    # Un único ranking completo de 306 candidatas; Top 10/15/20/30/40 son prefijos.
    # Compatibilidad con los scripts consumidores existentes: estas columnas
    # contienen el mismo ranking generado sobre las 306 candidatas.
    for n in (40, 30, 20, 15, 10):
        mrmr_df[f'mRMR_{n}_Rank'] = mrmr_df['mRMR_Rank']
    progreso.update(1)
    
    imp_rf, imp_dt = analizar_modelos_y_ramas(X, y, grupos, directorio_salida)
    progreso.update(1)
    
    maestra = fisher_df.merge(mi_df, on='Caracteristica', how='outer')
    maestra = maestra.merge(mrmr_df, on='Caracteristica', how='outer')
    maestra = maestra.merge(imp_rf, on='Caracteristica', how='outer')
    maestra = maestra.merge(imp_dt, on='Caracteristica', how='outer')
    
    maestra = maestra.sort_values('Fisher_Score', ascending=False)
    
    archivo_maestro = os.path.join(directorio_seleccion, 'Ranking_Multicriterio_Completo.csv')
    maestra.to_csv(archivo_maestro, index=False)
    progreso.update(1)
    progreso.close()

if __name__ == "__main__":
    main()
