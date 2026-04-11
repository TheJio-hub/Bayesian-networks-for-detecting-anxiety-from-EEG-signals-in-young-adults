import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def seleccion_mrmr(X, y, n_seleccion):
    n_caracteristicas = X.shape[1]
    nombres_caracteristicas = list(X.columns)

    indices_seleccionados = []
    indices_candidatos = list(range(n_caracteristicas))

    relevancia_inicial = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    redundancia_acumulada = np.zeros(n_caracteristicas)

    primera_mejor = np.argmax(relevancia_inicial)
    indices_seleccionados.append(primera_mejor)
    indices_candidatos.remove(primera_mejor)

    for i in range(1, n_seleccion):
        idx_ultimo = indices_seleccionados[-1]
        datos_ultimo = X.iloc[:, idx_ultimo].values.reshape(-1, 1)

        for idx_candidato in indices_candidatos:
            datos_candidato = X.iloc[:, idx_candidato].values.reshape(-1, 1)
            mi_redundancia = mutual_info_regression(
                datos_candidato,
                datos_ultimo.ravel(),
                discrete_features=False,
                random_state=42,
            )[0]
            redundancia_acumulada[idx_candidato] += mi_redundancia

        puntajes_mrmr = -np.inf * np.ones(n_caracteristicas)
        for idx_candidato in indices_candidatos:
            promedio_redundancia = redundancia_acumulada[idx_candidato] / len(indices_seleccionados)
            puntajes_mrmr[idx_candidato] = relevancia_inicial[idx_candidato] - promedio_redundancia

        idx_mejor = np.argmax(puntajes_mrmr)
        indices_seleccionados.append(idx_mejor)
        indices_candidatos.remove(idx_mejor)

    nombres_seleccionados = [nombres_caracteristicas[i] for i in indices_seleccionados]
    relevancia_seleccionada = [relevancia_inicial[i] for i in indices_seleccionados]

    return pd.DataFrame({
        'Orden_Seleccion': range(1, n_seleccion + 1),
        'Caracteristica': nombres_seleccionados,
        'Relevancia_Original': relevancia_seleccionada,
    })


def ejecutar_mrmr_global_completo():
    archivo_completo = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    if not os.path.exists(archivo_completo):
        return None

    print('Iniciando mRMR Global Completo (Todas las caracteristicas)...')
    df_completo = pd.read_parquet(archivo_completo)

    df_completo = df_completo[(df_completo['Puntaje'] == 0) | (df_completo['Puntaje'] >= 5)].copy()
    y_completo = df_completo['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values

    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    feats_completo = [
        c for c in df_completo.columns
        if c not in cols_meta and pd.api.types.is_numeric_dtype(df_completo[c])
    ]

    n_total = len(feats_completo)
    n_seleccion = int(n_total * 0.25)
    if n_seleccion < 5:
        n_seleccion = 5

    X_completo = df_completo[feats_completo]
    df_mrmr_completo = seleccion_mrmr(X_completo, y_completo, n_seleccion)

    return df_mrmr_completo


if __name__ == '__main__':
    resultado = ejecutar_mrmr_global_completo()
    if resultado is not None:
        print(resultado.head(10).to_string(index=False))
