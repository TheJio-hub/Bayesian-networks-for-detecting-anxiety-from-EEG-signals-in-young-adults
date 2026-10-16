"""Red bayesiana TAN discreta para detección de ansiedad (SAM >= 1).

La clase es raíz y padre de todos los predictores. El árbol EEG maximiza
la suma de información mutua condicionada por la clase; se orienta desde
la primera característica del ranking DT del entrenamiento. Las CPD se
estiman con K2 y el posterior se evalúa exactamente de forma vectorizada.
LOSO recalcula imputación, DT, cuantiles y árbol con 31 sujetos; el ajuste
final usa 32 y evalúa los ocho externos. Las flechas no prueban causalidad.
"""

import argparse
import json
import os
import threading
import time
import warnings
from contextlib import contextmanager
from hashlib import blake2b
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm as _tqdm

_CACHE_RANKING_DT = {}
VERSION_MODELO = 'tan_discreta_k2_v1'


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)


def graficar_dag(dag, titulo, nombre_archivo):
    plt.figure(figsize=(12, 10))
    G = nx.DiGraph(dag.edges())
    G.add_nodes_from(dag.nodes())
    generations = list(nx.topological_generations(G))
    pos = {}
    num_gens = len(generations)
    for gen_idx, nodes in enumerate(generations):
        y = 1.0 - (gen_idx / (num_gens - 1) if num_gens > 1 else 0.0)
        for node_idx, node in enumerate(nodes):
            x = 0.1 + 0.8 * node_idx / (len(nodes) - 1) if len(nodes) > 1 else 0.5
            pos[node] = (x, y)

    nx.draw_networkx_nodes(
        G, pos,
        node_size=1500,
        node_color='lightgreen',
        edgecolors='darkgreen',
        linewidths=1.5
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_family='sans-serif',
        font_weight='bold'
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=list(G.edges()),
        node_size=1500,
        edge_color='darkgreen',
        arrowstyle='-|>',
        arrowsize=20,
        width=1.5,
        alpha=0.8
    )

    plt.title(titulo, fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()


def seleccionar_top_dt(df_entrenamiento, candidatas, columna_objetivo, n_top):
    """Ranking DT del entrenamiento; reutiliza solo entradas exactamente iguales."""
    candidatas = list(candidatas)
    X = df_entrenamiento[candidatas].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df_entrenamiento[columna_objetivo].astype(int).to_numpy()
    if np.unique(y).size != 2:
        raise ValueError("El entrenamiento debe incluir ambas clases para seleccionar con DT.")
    # El contenido, el orden de filas, los índices, las etiquetas y las variables
    # forman la clave: nunca se comparte un ranking entre particiones distintas.
    firma = blake2b(digest_size=32)
    firma.update(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    firma.update(y.tobytes())
    clave = (tuple(candidatas), columna_objetivo, firma.digest())
    importancia = _CACHE_RANKING_DT.get(clave)
    if importancia is None:
        selector = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=5)
        selector.fit(X, y)
        importancia = selector.feature_importances_.copy()
        if len(_CACHE_RANKING_DT) >= 256:
            del _CACHE_RANKING_DT[next(iter(_CACHE_RANKING_DT))]
        _CACHE_RANKING_DT[clave] = importancia
    if np.count_nonzero(importancia > 0) < n_top:
        raise ValueError(f"DT asignó importancia positiva a menos de {n_top} variables en este entrenamiento.")
    orden = sorted(range(len(candidatas)), key=lambda j: (-importancia[j], j))
    return [candidatas[j] for j in orden[:n_top]]


def bootstrap_ci_por_sujeto(y_real, y_pred, sujetos, prefijo, n_replicas=500, semilla=42):
    """Bootstrap de sujetos completos mediante conteos suficientes por sujeto.

    Produce el mismo remuestreo y las mismas métricas que concatenar épocas
    en cada réplica, sin reconstruir cientos de veces los índices de filas.
    """
    y_real = np.asarray(y_real).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sujetos = np.asarray(sujetos)
    if not (len(y_real) == len(y_pred) == len(sujetos)):
        raise ValueError("Etiquetas, predicciones y sujetos deben estar alineados.")
    if pd.isna(sujetos).any():
        raise ValueError("La columna de sujetos contiene valores faltantes.")
    ids, grupos = np.unique(sujetos, return_inverse=True)
    if len(ids) < 2:
        raise ValueError("Se requieren al menos dos sujetos para el bootstrap.")
    estadisticas = np.column_stack([
        np.bincount(grupos, minlength=len(ids)),
        np.bincount(grupos, weights=(y_real == y_pred), minlength=len(ids)),
        np.bincount(grupos, weights=(y_real == 1), minlength=len(ids)),
        np.bincount(grupos, weights=((y_real == 1) & (y_pred == 1)), minlength=len(ids)),
        np.bincount(grupos, weights=(y_real == 0), minlength=len(ids)),
        np.bincount(grupos, weights=((y_real == 0) & (y_pred == 0)), minlength=len(ids)),
    ])
    rng = np.random.RandomState(semilla)
    muestras = rng.choice(len(ids), size=(n_replicas, len(ids)), replace=True)
    totales = estadisticas[muestras].sum(axis=1)
    exactitudes = totales[:, 1] / totales[:, 0]
    validas = (totales[:, 2] > 0) & (totales[:, 4] > 0)
    balanceadas = (totales[validas, 3] / totales[validas, 2] + totales[validas, 5] / totales[validas, 4]) / 2
    acc_ci = np.percentile(exactitudes, [2.5, 97.5])
    bal_ci = np.percentile(balanceadas, [2.5, 97.5]) if len(balanceadas) else (np.nan, np.nan)
    return {
        f"{prefijo}_IC95_exactitud_inf": float(acc_ci[0]),
        f"{prefijo}_IC95_exactitud_sup": float(acc_ci[1]),
        f"{prefijo}_IC95_balanced_accuracy_inf": float(bal_ci[0]),
        f"{prefijo}_IC95_balanced_accuracy_sup": float(bal_ci[1]),
        f"{prefijo}_n_sujetos": int(len(ids)),
    }


def metricas_agregadas_por_sujeto(y_real, y_pred, sujetos, prefijo):
    """Calcula métricas por registro con peso total igual para cada sujeto.

    Se conservan las etiquetas por época: un mismo sujeto puede aportar
    observaciones de ambas clases.
    """
    yt = np.asarray(y_real).astype(int)
    yp = np.asarray(y_pred).astype(int)
    grupos = np.asarray(sujetos)
    if not (len(yt) == len(yp) == len(grupos)) or len(grupos) == 0:
        raise ValueError("Etiquetas, predicciones y sujetos deben tener la misma longitud no vacía.")
    if pd.isna(grupos).any():
        raise ValueError("La columna de sujetos contiene valores faltantes.")
    conteos = pd.Series(grupos).value_counts()
    pesos = np.asarray([1.0 / conteos[sujeto] for sujeto in grupos], dtype=float)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1], sample_weight=pesos).ravel()
    sensibilidad = tp / (tp + fn) if (tp + fn) else np.nan
    especificidad = tn / (tn + fp) if (tn + fp) else np.nan
    balanced_accuracy = (sensibilidad + especificidad) / 2 if np.isfinite(sensibilidad) and np.isfinite(especificidad) else np.nan
    return {
        f"{prefijo}_sujeto_exactitud": float(accuracy_score(yt, yp, sample_weight=pesos)),
        f"{prefijo}_sujeto_balanced_accuracy": float(balanced_accuracy),
        f"{prefijo}_sujeto_sensibilidad": float(sensibilidad),
        f"{prefijo}_sujeto_especificidad": float(especificidad),
        f"{prefijo}_sujeto_precision": float(precision_score(yt, yp, zero_division=0, sample_weight=pesos)),
        f"{prefijo}_sujeto_f1": float(f1_score(yt, yp, zero_division=0, sample_weight=pesos)),
        f"{prefijo}_n_sujetos": int(pd.Series(grupos).nunique()),
    }


def obtener_version(paquete):
    try:
        return version(paquete)
    except PackageNotFoundError:
        return 'no instalado'



@contextmanager
def mostrar_actividad(etiqueta, intervalo=30):
    inicio = time.perf_counter()
    fin = threading.Event()

    def avisar():
        while not fin.wait(intervalo):
            _tqdm.write(f'{etiqueta}: búsqueda activa, {time.perf_counter() - inicio:.1f} s transcurridos.')

    hilo = threading.Thread(target=avisar, daemon=True)
    _tqdm.write(f'INICIO {etiqueta}')
    hilo.start()
    try:
        yield
    finally:
        fin.set()
        hilo.join(timeout=1)
        _tqdm.write(f'FIN {etiqueta}: {time.perf_counter() - inicio:.2f} s')


def guardar_csv(tabla, ruta):
    ruta = Path(ruta)
    temporal = ruta.with_suffix(ruta.suffix + '.tmp')
    tabla.to_csv(temporal, index=False)
    os.replace(temporal, ruta)


def guardar_evidencia(contenido, ruta):
    """Guarda JSON en Evidencia, con subcarpetas por tipo de información.

    Actualiza cada archivo por reemplazo atómico. Conserva evidencia parcial
    si una corrida falla, pero no implementa reanudación automática.
    """
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    for nombre, datos in contenido.items():
        destino = ruta / nombre
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = destino.with_suffix(destino.suffix + '.tmp')
        temporal.write_text(
            json.dumps(datos, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        os.replace(temporal, destino)


def metricas_clasificacion(y_real, y_pred, y_prob):
    tn, fp, _, _ = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    sensibilidad = recall_score(y_real, y_pred, zero_division=0)
    especificidad = tn / (tn + fp) if (tn + fp) else np.nan
    return {'Exact.': accuracy_score(y_real, y_pred),
            'Prec.': precision_score(y_real, y_pred, zero_division=0),
            'Sens.': sensibilidad, 'Esp.': especificidad,
            'F1': f1_score(y_real, y_pred, zero_division=0),
            'Balanced_accuracy': (sensibilidad + especificidad) / 2,
            'ROC_AUC': roc_auc_score(y_real, y_prob),
            'Brier': brier_score_loss(y_real, y_prob),
            'Log_loss': log_loss(y_real, np.column_stack([1 - y_prob, y_prob]), labels=[0, 1])}


def validar_dag(dag, columnas):
    faltantes = set(columnas) - set(dag.nodes())
    extras = set(dag.nodes()) - set(columnas)
    if faltantes or extras:
        raise ValueError(f'Nodos incorrectos: faltantes={sorted(faltantes)}, extras={sorted(extras)}.')
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError(f'Ciclo dirigido: {nx.find_cycle(dag)}.')


def preparar_particion(train, test, candidatas, objetivo, top, bins):
    """Imputación por mediana, DT y cuantiles: todo se ajusta con entrenamiento."""
    X = train[candidatas].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    Z = test[candidatas].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    medianas = X.median()
    if medianas.isna().any():
        raise ValueError(f'Variables sin datos finitos en entrenamiento: {medianas[medianas.isna()].index.tolist()}')
    if set(train[objetivo].unique()) != {0, 1}:
        raise ValueError('El entrenamiento debe contener ambas clases.')
    # Construir las columnas juntas y consolidar bloques de memoria.
    base = pd.concat([X.fillna(medianas), train[[objetivo]].astype(int)], axis=1).copy()
    seleccionadas = seleccionar_top_dt(base, candidatas, objetivo, top)
    discretizador = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile',
                                     subsample=None, random_state=42)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        a = discretizador.fit_transform(base[seleccionadas])
    b = discretizador.transform(Z[seleccionadas].fillna(medianas[seleccionadas]))
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('La discretización produjo valores no finitos.')
    a = pd.DataFrame(a.astype(int), columns=seleccionadas, index=train.index)
    b = pd.DataFrame(b.astype(int), columns=seleccionadas, index=test.index)
    a[objetivo] = train[objetivo].to_numpy(dtype=int)
    estados = {c: list(range(int(n))) for c, n in zip(seleccionadas, discretizador.n_bins_)}
    estados[objetivo] = [0, 1]
    for c in seleccionadas:
        if not a[c].isin(estados[c]).all() or not b[c].isin(estados[c]).all():
            raise ValueError(f'Estado fuera de los intervalos efectivos de {c}.')
    meta = {'Caracteristicas_DT': seleccionadas, 'n_bins_solicitados': bins,
            'n_bins_efectivos': {c: len(estados[c]) for c in seleccionadas},
            'Limites_intervalos': {c: v.tolist() for c, v in zip(seleccionadas, discretizador.bin_edges_)},
            'Medianas_imputacion': medianas[seleccionadas].to_dict(),
            'N_valores_imputados_train': int(X[seleccionadas].isna().sum().sum()),
            'N_valores_imputados_evaluacion': int(Z[seleccionadas].isna().sum().sum()),
            'Advertencias_discretizacion': [str(aviso.message) for aviso in avisos],
            'Estrategia': 'Cuantiles de entrenamiento; sin submuestreo; mediana de entrenamiento'}
    return a, b, estados, meta


def ajustar_cpds(dag, datos, estados, max_celdas):
    """Estimación K2 exacta por conteos vectorizados, incluida la red sin aristas."""
    validar_dag(dag, datos.columns)
    modelo = DiscreteBayesianNetwork()
    modelo.add_nodes_from(datos.columns)
    modelo.add_edges_from(dag.edges())
    diagnostico = []
    for nodo in datos.columns:
        padres = sorted(dag.predecessors(nodo))
        tarjetas = [len(estados[p]) for p in padres]
        ri = len(estados[nodo]); qi = int(np.prod(tarjetas, dtype=np.int64))
        if ri * qi > max_celdas:
            raise ValueError(f'CPD de {nodo} excede el límite de celdas antes de asignar memoria.')
        codigo = np.zeros(len(datos), dtype=np.int64)
        for p in padres:
            codigo = codigo * len(estados[p]) + datos[p].to_numpy(dtype=np.int64)
        counts = np.bincount(datos[nodo].to_numpy(dtype=np.int64) * qi + codigo,
                             minlength=ri * qi).reshape(ri, qi)
        valores = (counts + 1.0) / (counts.sum(axis=0, keepdims=True) + ri)
        kwargs = {'variable': nodo, 'variable_card': ri, 'values': valores,
                  'state_names': {c: estados[c] for c in [nodo] + padres}}
        if padres:
            kwargs.update(evidence=padres, evidence_card=tarjetas)
        modelo.add_cpds(TabularCPD(**kwargs))
        totales = counts.sum(axis=0)
        diagnostico.append({'nodo': nodo, 'padres': padres, 'cardinalidad': ri,
                            'configuraciones_padres': qi, 'celdas_CPD': ri * qi,
                            'parametros_libres': (ri - 1) * qi,
                            'configuraciones_sin_datos': int(sum(totales == 0)),
                            'configuraciones_con_menos_de_5': int(sum(totales < 5)),
                            'min_observaciones_por_configuracion': int(totales.min()),
                            'pseudoconteo_K2_por_celda': 1.0})
    modelo.check_model()
    return modelo, diagnostico


def inferir_posterior(modelo, datos, objetivo):
    """Posterior exacto de la red generativa con clase raíz.

    Cada CPD EEG contiene la clase como padre. Las log-razones por factor
    evitan restar dos sumas conjuntas grandes. No se redondean las
    probabilidades ni las métricas.
    """
    esperado = set(modelo.nodes()) - {objetivo}
    if set(datos.columns) != esperado or not datos.index.is_unique:
        raise ValueError('Predictores o índices de inferencia incorrectos.')
    if modelo.in_degree(objetivo) != 0 or set(modelo.successors(objetivo)) != esperado:
        raise ValueError('La red debe tener la clase como raíz y padre de todos los predictores.')
    cpd_clase = modelo.get_cpds(objetivo)
    if cpd_clase.state_names[objetivo] != [0, 1]:
        raise ValueError('La clase no está codificada como [0, 1].')
    for c in esperado:
        estados = modelo.get_cpds(c).state_names[c]
        if not datos[c].isin(estados).all():
            raise ValueError(f'Estado faltante o no definido para {c}.')
    factores = sorted(modelo.get_cpds(), key=lambda cpd: str(cpd.variable))
    if any(objetivo not in cpd.variables for cpd in factores):
        raise ValueError('Una CPD no cumple la arquitectura generativa declarada.')

    def evaluar(cpd, clase):
        indices = []
        for variable in cpd.variables:
            valores = np.full(len(datos), clase) if variable == objetivo else datos[variable].to_numpy()
            mapa = {estado: j for j, estado in enumerate(cpd.state_names[variable])}
            codigos = pd.Series(valores).map(mapa)
            if codigos.isna().any():
                raise ValueError(f'Estado no definido en la CPD de {variable}.')
            indices.append(codigos.to_numpy(dtype=np.int64))
        p = np.asarray(cpd.values)[tuple(indices)]
        if not np.isfinite(p).all() or np.any(p <= 0) or np.any(p > 1):
            raise ValueError('CPD inválida; K2 debe producir probabilidades positivas y finitas.')
        return p

    log_razon = np.zeros(len(datos))
    for cpd in factores:
        log_razon += np.log(evaluar(cpd, 1)) - np.log(evaluar(cpd, 0))
    if not np.isfinite(log_razon).all():
        raise ValueError('Log-razones no finitas durante la inferencia.')
    p = np.exp(-np.logaddexp(0.0, -log_razon))
    if len(p) != len(datos) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Inferencia incompleta o probabilidades inválidas.')
    return pd.Series(p, index=datos.index, name='Probabilidad_ansiedad')


def serializar_modelo(modelo):
    return {'nodos': list(modelo.nodes()), 'arcos': list(modelo.edges()), 'prior_parametros': 'K2',
            'Version_modelo': VERSION_MODELO,
            'Factorizacion': 'P(Y) * P(X_raiz | Y) * producto_no_raiz P(X_j | Y, padre_EEG(X_j))',
            'cpds': [{'variables': list(cpd.variables), 'cardinality': cpd.cardinality.tolist(),
                      'state_names': cpd.state_names, 'values': np.asarray(cpd.values).tolist()}
                     for cpd in modelo.get_cpds()]}


def validar_tan(dag, columnas, raiz_eeg, objetivo='Ansiedad'):
    """Comprueba la arquitectura completa, no solo la existencia de aristas."""
    validar_dag(dag, columnas)
    eeg = [c for c in columnas if c != objetivo]
    d = len(eeg)
    if d < 2 or raiz_eeg not in eeg:
        raise ValueError('TAN requiere al menos dos predictores y una raíz EEG válida.')
    if dag.in_degree(objetivo) != 0 or set(dag.successors(objetivo)) != set(eeg):
        raise ValueError('La clase debe ser raíz y padre de todos los predictores EEG.')
    arbol = nx.DiGraph(dag.subgraph(eeg))
    if arbol.number_of_edges() != d - 1 or not nx.is_tree(arbol.to_undirected()):
        raise ValueError(f'El árbol EEG debe ser conexo y contener {d-1} aristas.')
    if arbol.in_degree(raiz_eeg) != 0 or any(arbol.in_degree(c) != 1 for c in eeg if c != raiz_eeg):
        raise ValueError('El árbol EEG no está orientado desde la raíz declarada.')
    if dag.number_of_edges() != 2*d - 1:
        raise ValueError(f'La TAN debe contener {2*d-1} aristas totales.')
    if any(dag.in_degree(c) > 2 for c in eeg):
        raise ValueError('Un predictor TAN tiene más de un padre EEG y la clase.')


def aprender_tan(datos, estados, objetivo='Ansiedad'):
    """TAN estándar: árbol de expansión máxima de I(Xi; Xj | Y).

    Se incluyen explícitamente las parejas con peso cero. Los empates se
    resuelven de forma reproducible según el orden DT de entrenamiento;
    no implican evidencia de una dependencia o dirección causal.
    """
    eeg = [c for c in datos.columns if c != objetivo]
    y = datos[objetivo].to_numpy(dtype=np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('El entrenamiento TAN debe contener ambas clases.')
    valores = datos[eeg].to_numpy(dtype=np.int64)
    grafo = nx.Graph()
    grafo.add_nodes_from(eeg)
    pesos = []
    for i, j in combinations(range(len(eeg)), 2):
        ri, rj = len(estados[eeg[i]]), len(estados[eeg[j]])
        peso = 0.0
        for clase in (0, 1):
            mascara = y == clase
            n = int(mascara.sum())
            conteos = np.bincount(valores[mascara, i]*rj + valores[mascara, j],
                                  minlength=ri*rj).reshape(ri, rj).astype(float)
            marginales = conteos.sum(axis=1, keepdims=True) * conteos.sum(axis=0, keepdims=True)
            presentes = conteos > 0
            mi = np.sum((conteos[presentes]/n) * np.log(conteos[presentes]*n / marginales[presentes]))
            if not np.isfinite(mi) or mi < -1e-10:
                raise ValueError(f'Información mutua inválida entre {eeg[i]} y {eeg[j]}.')
            peso += (n/len(datos))*max(0.0, float(mi))
        grafo.add_edge(eeg[i], eeg[j], weight=peso)
        pesos.append({'Variable_1': eeg[i], 'Variable_2': eeg[j], 'CMI_nats': peso})
    # Kruskal mantiene el orden de inserción para pesos iguales (orden DT).
    arbol = nx.maximum_spanning_tree(grafo, algorithm='kruskal', weight='weight')
    raiz = eeg[0]
    orientado = nx.bfs_tree(arbol, raiz)
    dag = nx.DiGraph()
    dag.add_nodes_from(datos.columns)
    dag.add_edges_from(orientado.edges())
    dag.add_edges_from((objetivo, c) for c in eeg)
    validar_tan(dag, datos.columns, raiz, objetivo)
    pesos_arbol = [{'Padre_EEG': a, 'Hijo_EEG': b, 'CMI_nats': grafo[a][b]['weight']}
                   for a, b in orientado.edges()]
    meta = {'Motor': 'CMI empírica NumPy + árbol máximo NetworkX; CPD pgmpy',
            'Criterio': 'Máxima suma de I(Xi;Xj|Ansiedad), ponderada por prevalencia de entrenamiento',
            'Version_modelo': VERSION_MODELO, 'Clase_raiz': True,
            'N_arcos_clase_impuestos': len(eeg), 'Raiz_EEG': raiz,
            'Orientacion': 'Desde primera característica DT; clase raíz impuesta; sin interpretación causal',
            'Desempate': 'Orden de inserción de parejas según ranking DT; Kruskal estable',
            'Pesos_CMI_parejas': pesos, 'Pesos_arbol_EEG': pesos_arbol,
            'N_arcos_EEG_peso_cero': sum(p['CMI_nats'] == 0.0 for p in pesos_arbol),
            'Arbol_EEG_completo': True}
    return dag, meta


def ajustar_y_predecir(train, test, candidatas, top, bins, algoritmo, args, etiqueta):
    if algoritmo != 'TAN':
        raise ValueError('Este script implementa únicamente TAN.')
    inicio = time.perf_counter()
    a, b, estados, prepro = preparar_particion(train, test, candidatas, 'Ansiedad', top, bins)
    tiempo_prepro = time.perf_counter() - inicio
    inicio = time.perf_counter()
    with mostrar_actividad(etiqueta):
        dag, meta = aprender_tan(a, estados)
    tiempo_busqueda = time.perf_counter() - inicio
    inicio = time.perf_counter()
    # Una TAN con hasta tres intervalos tiene como máximo 3*3*2=18 celdas por CPD.
    modelo, diagnostico = ajustar_cpds(dag, a, estados, max_celdas=2*bins*bins)
    validar_tan(modelo, a.columns, meta['Raiz_EEG'])
    tiempo_parametros = time.perf_counter() - inicio
    inicio = time.perf_counter()
    probabilidades = inferir_posterior(modelo, b, 'Ansiedad')
    if not probabilidades.index.equals(test.index):
        raise ValueError('Las predicciones no conservan el orden de las épocas.')
    meta.update({'Tiempo_preprocesamiento_s': tiempo_prepro,
                 'Tiempo_estructura_s': tiempo_busqueda,
                 'Tiempo_parametros_s': tiempo_parametros,
                 'Tiempo_inferencia_s': time.perf_counter() - inicio,
                 'N_arcos': dag.number_of_edges(),
                 'N_parametros': sum(d['parametros_libres'] for d in diagnostico),
                 'Max_padres_efectivos': max(len(d['padres']) for d in diagnostico),
                 'Max_celdas_CPD_efectivas': max(d['celdas_CPD'] for d in diagnostico),
                 'N_configuraciones_CPD_sin_datos': sum(d['configuraciones_sin_datos'] for d in diagnostico),
                 'N_caracteristicas_hijas_clase': dag.out_degree('Ansiedad'),
                 'N_arcos_EEG_final': top-1,
                 'Inferencia': 'Posterior exacto por suma vectorizada de log-razones de CPD; sin redondeo',
                 'Umbral_fijo': 0.5})
    evidencia = {'Preprocesamiento': prepro, 'Detalles': meta, 'Diagnostico_CPD': diagnostico,
                 'Modelo': serializar_modelo(modelo)}
    if args.diagnosticos_estructura:
        from pgmpy.metrics import CorrelationScore, FisherC
        diag = {'Tipo': 'Diagnósticos estructurales; no son métricas predictivas ni evidencia causal',
                'Advertencia': 'Chi cuadrado asintótico y épocas dependientes dentro de sujeto; interpretar con cautela'}
        for nombre, estimador in [('CS', CorrelationScore(ci_test='chi_square', significance_level=.05)),
                                  ('FisherC', FisherC(ci_test='chi_square', compute_rmsea=True, show_progress=False))]:
            try:
                diag[nombre] = np.asarray(estimador.evaluate(X=a, causal_graph=dag)).tolist()
            except Exception as exc:
                diag[nombre] = {'Estado': 'NO_DISPONIBLE', 'Error': f'{type(exc).__name__}: {exc}'}
        evidencia['Diagnosticos_estructura'] = diag
    return probabilidades.to_numpy(), modelo, meta, evidencia


def tabla_predicciones(datos, prob):
    prob = np.asarray(prob, dtype=float)
    if (prob.ndim != 1 or len(datos) != len(prob) or not np.isfinite(prob).all()
            or np.any((prob < 0) | (prob > 1))):
        raise ValueError('No hay una probabilidad finita por época.')
    return pd.DataFrame({'Indice_fila': datos.index.to_numpy(), 'Sujeto': datos['Sujeto'].to_numpy(),
                         'Ansiedad_real': datos['Ansiedad'].to_numpy(dtype=int),
                         'Probabilidad_ansiedad': prob, 'Prediccion': (prob >= 0.5).astype(int)})


def ejecutar_loso(datos, candidatas, top, bins, algoritmo, args, carpeta_evidencia):
    sujetos = datos['Sujeto'].to_numpy()
    logo = LeaveOneGroupOut()
    prob = np.full(len(datos), np.nan)
    tiempos = []
    detalle = {'Estado': 'EN_CURSO', 'Pliegues': [], 'Predicciones': []}
    clave = f'LOSO/{algoritmo}_top{top}_bins{bins}.json'
    guardar_evidencia({clave: detalle}, carpeta_evidencia)
    particiones = logo.split(datos, datos['Ansiedad'], sujetos)
    barra = tqdm(particiones, total=32, desc=f'LOSO {algoritmo} Top {top}, bins {bins}', unit='fold')
    for fold, (it, iv) in enumerate(barra, start=1):
        sujeto = str(sujetos[iv[0]])
        contexto = f'{algoritmo}, Top {top}, bins {bins}, fold {fold}/32, sujeto {sujeto}'
        registro = {'Fold': fold, 'Sujeto_validacion': sujeto,
                    'N_entrenamiento': len(it), 'N_validacion': len(iv),
                    'Sujetos_entrenamiento': sorted(set(map(str, sujetos[it]))), 'Estado': 'INICIO'}
        inicio = time.perf_counter()
        try:
            if (set(sujetos[it]) & set(sujetos[iv]) or len(set(sujetos[it])) != 31
                    or len(set(sujetos[iv])) != 1):
                raise ValueError('La partición LOSO no contiene 31/1 sujetos disjuntos.')
            p, _, meta, evidencia = ajustar_y_predecir(
                datos.iloc[it], datos.iloc[iv], candidatas, top, bins, algoritmo, args, contexto)
            prob[iv] = p
            registro.update({k: v for k, v in meta.items() if k not in {'Pesos_CMI_parejas', 'Pesos_arbol_EEG'}},
                            Estado='OK', N_predicciones_validas=len(p),
                            Tiempo_fold_s=time.perf_counter() - inicio)
            tiempos.append(registro)
            detalle['Pliegues'].append({'Registro': registro, **evidencia})
        except Exception as exc:
            registro.update(Estado='ERROR', Error=f'{type(exc).__name__}: {exc}',
                            Tiempo_fold_s=time.perf_counter() - inicio)
            detalle.update(Estado='ERROR', Error=registro)
            guardar_evidencia({clave: detalle}, carpeta_evidencia)
            raise RuntimeError(f'Falló {contexto}. {type(exc).__name__}: {exc}') from exc
        listos = np.flatnonzero(np.isfinite(prob))
        detalle['Predicciones'] = tabla_predicciones(datos.iloc[listos], prob[listos]).to_dict(orient='records')
        guardar_evidencia({clave: detalle}, carpeta_evidencia)
        barra.set_postfix(sujeto=sujeto, segundos=f'{registro["Tiempo_fold_s"]:.1f}')
    if not np.isfinite(prob).all():
        raise RuntimeError(f'LOSO incompleto para {algoritmo}, Top {top}, bins {bins}.')
    detalle['Estado'] = 'COMPLETO'
    guardar_evidencia({clave: detalle}, carpeta_evidencia)
    return prob, pd.DataFrame(tiempos)


def cargar_datos(raiz):
    archivo = raiz / 'Resultados' / 'Exploratorio' / 'datos_completos_normalizados.parquet'
    ranking = raiz / 'Resultados' / 'Análisis global' / 'Selección de características' / 'Ranking_Multicriterio_Completo.csv'
    if not archivo.exists() or not ranking.exists():
        raise FileNotFoundError(f'Faltan datos o lista de candidatas: {archivo}; {ranking}')
    datos = pd.read_parquet(archivo)
    requeridas = {'Sujeto', 'Puntaje'}
    if not requeridas.issubset(datos.columns):
        raise ValueError(f'Faltan columnas: {sorted(requeridas - set(datos.columns))}')
    if not datos.index.is_unique or not datos.columns.is_unique or datos['Sujeto'].isna().any():
        raise ValueError('Hay índices/columnas duplicados o sujetos sin identificar.')
    sam = pd.to_numeric(datos['Puntaje'], errors='coerce')
    if not np.isfinite(sam.to_numpy()).all():
        raise ValueError('Puntaje contiene valores faltantes, no numéricos o infinitos.')
    conservar = (sam == 0) | (sam >= 1)
    excluidas = int((~conservar).sum())
    datos = datos.loc[conservar].copy()
    datos['Ansiedad'] = (sam.loc[conservar] >= 1).astype(int)
    sujetos = np.sort(datos['Sujeto'].unique())
    if len(sujetos) != 40:
        raise ValueError(f'Se requieren exactamente 40 sujetos, se encontraron {len(sujetos)}.')
    tabla = pd.read_csv(ranking)
    if 'Caracteristica' not in tabla:
        raise ValueError('La lista de candidatas no contiene Caracteristica.')
    candidatas = sorted(tabla['Caracteristica'].dropna().tolist())
    if len(candidatas) != 306 or len(set(candidatas)) != 306:
        raise ValueError('Se requieren 306 características candidatas distintas.')
    if {'Ansiedad', 'Sujeto', 'Puntaje'} & set(candidatas):
        raise ValueError('La lista EEG incluye una etiqueta o un identificador.')
    faltantes = set(candidatas) - set(datos.columns)
    if faltantes:
        raise ValueError(f'Faltan características en los datos: {sorted(faltantes)}')
    mezclados = np.random.RandomState(42).permutation(sujetos)
    desarrollo = datos[datos['Sujeto'].isin(mezclados[:32])].copy()
    externa = datos[datos['Sujeto'].isin(mezclados[32:])].copy()
    if set(desarrollo.Sujeto) & set(externa.Sujeto):
        raise ValueError('Los sujetos externos se solapan con desarrollo.')
    for nombre, parte in [('desarrollo', desarrollo), ('externa', externa)]:
        if set(parte['Ansiedad'].unique()) != {0, 1}:
            raise ValueError(f'Falta una clase en la partición {nombre}.')
    return desarrollo, externa, candidatas, mezclados, excluidas


def principal(argv=None):
    parser = argparse.ArgumentParser(description='Red bayesiana TAN: DT y cuantiles por entrenamiento; CPD K2 e inferencia exacta.')
    parser.add_argument('--tops', nargs='+', type=int, choices=[10, 15, 20], default=[10, 15, 20])
    parser.add_argument('--bins', nargs='+', type=int, choices=[2, 3], default=[2, 3])
    parser.add_argument('--diagnosticos-estructura', action='store_true',
                        help='Añade CS y FisherC a Evidencia, separados de las métricas predictivas; aumenta el tiempo.')
    args = parser.parse_args(argv)
    raiz = Path(__file__).resolve().parents[3]
    salida = raiz / 'Resultados' / 'Red Bayesiana' / 'Global' / 'TAN'
    evidencia_dir = salida / 'Evidencia'
    salida.mkdir(parents=True, exist_ok=True)
    desarrollo, externa, candidatas, sujetos, excluidas = cargar_datos(raiz)
    configuracion = {**vars(args), 'Estado_ejecucion': 'EN_CURSO',
                     'Inicio_UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     'Semilla': 42, 'Criterio': 'SAM=0 control; SAM>=1 caso',
                     'Version_modelo': VERSION_MODELO,
                     'Arquitectura': 'Ansiedad raíz y padre de todos los predictores; árbol EEG completo de d-1 aristas',
                     'Limite_padres': 'Dos por predictor: clase y a lo sumo un padre EEG',
                     'Sujetos_desarrollo_32': list(map(str, sujetos[:32])),
                     'Sujetos_externos_8': list(map(str, sujetos[32:])),
                     'Epocas_excluidas_por_criterio_SAM': excluidas,
                     'Candidatas': candidatas, 'N_candidatas': len(candidatas),
                     'Version_pgmpy': obtener_version('pgmpy'), 'Version_sklearn': obtener_version('scikit-learn'),
                     'Version_numpy': obtener_version('numpy'), 'Version_pandas': obtener_version('pandas'),
                     'Version_networkx': obtener_version('networkx'),
                     'Implementacion_estructura': 'CMI empírica y Kruskal sobre grafo completo, incluidos pesos cero',
                     'Umbral_fijo': 0.5, 'Prior_parametros': 'K2, un pseudoconteo por celda',
                     'Imputacion': 'Mediana ajustada dentro de cada entrenamiento, antes de DT; infinitos tratados como faltantes',
                     'Interpretacion': 'TAN de máxima información mutua condicionada; raíz EEG primera DT; conserva pesos cero con desempate reproducible. No identifica causalidad.',
                     'Nota_validacion': 'Los intervalos se remuestrean por sujeto; las épocas no son sujetos independientes. Los ocho externos ya se observaron en análisis previos.'}
    guardar_evidencia({'Configuracion.json': configuracion}, evidencia_dir)
    resumen = salida / 'Resultados_Red_TAN.csv'
    resultados = []
    guardar_csv(pd.DataFrame(columns=['Estructura', 'n_bins', 'Top', 'Estado']), resumen)
    print(f'Resultados: {salida}', flush=True)
    print('Red bayesiana TAN: árbol EEG completo, clase raíz y CPD K2.', flush=True)
    print(f'Configuraciones: {len(set(args.tops)) * len(set(args.bins))}; '
          f'Tops {list(dict.fromkeys(args.tops))}, intervalos {list(dict.fromkeys(args.bins))}; umbral fijo 0.5.', flush=True)
    inicio_total = time.perf_counter()
    contexto = ''
    try:
        for bins in dict.fromkeys(args.bins):
            for top in dict.fromkeys(args.tops):
                for algoritmo in ('TAN',):
                    contexto = f'{algoritmo}, Top {top}, bins {bins}'
                    inicio = time.perf_counter()
                    pval, tiempos = ejecutar_loso(desarrollo, candidatas, top, bins, algoritmo, args, evidencia_dir)
                    ptest, modelo, meta, evidencia = ajustar_y_predecir(
                        desarrollo, externa, candidatas, top, bins, algoritmo, args,
                        f'{contexto}, ajuste final con 32 sujetos')
                    key = f'{algoritmo}_top{top}_bins{bins}'
                    guardar_evidencia({f'Modelos_Finales/{key}.json': evidencia,
                                       f'Predicciones_Externas/{key}.json': tabla_predicciones(externa, ptest).to_dict(orient='records')},
                                      evidencia_dir)
                    yval = desarrollo['Ansiedad'].to_numpy(dtype=int)
                    ytest = externa['Ansiedad'].to_numpy(dtype=int)
                    predval = (pval >= 0.5).astype(int); predtest = (ptest >= 0.5).astype(int)
                    efectivos = evidencia['Preprocesamiento']['n_bins_efectivos']
                    fila = {'Estructura': algoritmo, 'Motor': meta['Motor'], 'Criterio': meta['Criterio'],
                            'Top': top, 'n_bins': bins, 'Estado': 'COMPLETO', 'Umbral_fijo': 0.5,
                            'Version_modelo': meta['Version_modelo'],
                            'Clase_raiz': meta['Clase_raiz'],
                            'N_arcos_clase_impuestos': meta['N_arcos_clase_impuestos'],
                            'N_arcos_EEG_final': meta['N_arcos_EEG_final'],
                            'N_caracteristicas_hijas_clase': meta['N_caracteristicas_hijas_clase'],
                            'Orientacion': meta['Orientacion'],
                            'Raiz_EEG_final': meta['Raiz_EEG'],
                            'Arbol_EEG_completo_final': meta['Arbol_EEG_completo'],
                            'N_arcos_EEG_peso_cero_final': meta['N_arcos_EEG_peso_cero'],
                            'Caracteristicas_DT_final': json.dumps(evidencia['Preprocesamiento']['Caracteristicas_DT'], ensure_ascii=False),
                            'Bins_efectivos_final': json.dumps(efectivos, ensure_ascii=False),
                            'N_features_bins_reducidos_final': sum(n < bins for n in efectivos.values()),
                            'Max_padres_configurado': 2, 'Max_celdas_CPD_configurado': 2*bins*bins,
                            'Max_padres_efectivos_final': meta['Max_padres_efectivos'],
                            'Max_celdas_CPD_final': meta['Max_celdas_CPD_efectivas'],
                            'N_arcos_final': meta['N_arcos'], 'N_parametros_final': meta['N_parametros'],
                            'N_configuraciones_CPD_sin_datos_final': meta['N_configuraciones_CPD_sin_datos'],
                            'N_pliegues_LOSO_completos': len(tiempos),
                            'N_predicciones_LOSO': len(pval), 'N_predicciones_externas': len(ptest),
                            'Tiempo_preprocesamiento_LOSO_s': float(tiempos['Tiempo_preprocesamiento_s'].sum()),
                            'Tiempo_parametros_LOSO_s': float(tiempos['Tiempo_parametros_s'].sum()),
                            'Tiempo_inferencia_LOSO_s': float(tiempos['Tiempo_inferencia_s'].sum()),
                            'Tiempo_preprocesamiento_final_s': meta['Tiempo_preprocesamiento_s'],
                            'Tiempo_parametros_final_s': meta['Tiempo_parametros_s'],
                            'Tiempo_LOSO_s': float(tiempos['Tiempo_fold_s'].sum()),
                            'Tiempo_estructura_LOSO_s': float(tiempos['Tiempo_estructura_s'].sum()),
                            'Tiempo_estructura_final_s': meta['Tiempo_estructura_s'],
                            'Tiempo_inferencia_final_s': meta['Tiempo_inferencia_s'],
                            'Interpretacion_grafo': 'Estructura predictiva observacional; no demuestra causalidad'}
                    for prefijo, y, pred, prob, grupos in [('Val', yval, predval, pval, desarrollo.Sujeto.to_numpy()),
                                                          ('Test', ytest, predtest, ptest, externa.Sujeto.to_numpy())]:
                        fila.update({f'{k} ({prefijo})': v for k, v in metricas_clasificacion(y, pred, prob).items()})
                        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
                        fila.update({f'{k} ({prefijo})': int(v) for k, v in zip(['TN', 'FP', 'FN', 'TP'], [tn, fp, fn, tp])})
                        fila.update(bootstrap_ci_por_sujeto(y, pred, grupos, prefijo))
                        fila.update(metricas_agregadas_por_sujeto(y, pred, grupos, prefijo))
                    fila['Tiempo_configuracion_s'] = time.perf_counter() - inicio
                    resultados.append(fila)
                    guardar_csv(pd.DataFrame(resultados), resumen)
                    graficar_dag(modelo, f'Red bayesiana TAN: Top {top}, {bins} intervalos',
                                 salida / f'Estructura_Global_{algoritmo}_top{top}_bins{bins}.png')
                    print(f'COMPLETO {contexto}: {time.perf_counter() - inicio:.1f} s', flush=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        configuracion.update(Estado_ejecucion='ERROR', Configuracion_fallida=contexto, Error=error)
        guardar_evidencia({'Configuracion.json': configuracion}, evidencia_dir)
        resultados.append({'Estructura': algoritmo, 'n_bins': bins, 'Top': top,
                           'Estado': 'ERROR', 'Error': error})
        guardar_csv(pd.DataFrame(resultados), resumen)
        raise RuntimeError(f'Falló {contexto}. La evidencia parcial y el resumen quedaron guardados. {error}') from exc
    configuracion.update(Estado_ejecucion='COMPLETO', Tiempo_total_s=time.perf_counter() - inicio_total)
    guardar_evidencia({'Configuracion.json': configuracion}, evidencia_dir)
    print(f'Resumen: {resumen}', flush=True)
    print(f'Evidencia: {evidencia_dir}', flush=True)
    print(pd.DataFrame(resultados).to_string(index=False, float_format=lambda x: f'{x:.4f}'))



if __name__ == '__main__':
    principal()
