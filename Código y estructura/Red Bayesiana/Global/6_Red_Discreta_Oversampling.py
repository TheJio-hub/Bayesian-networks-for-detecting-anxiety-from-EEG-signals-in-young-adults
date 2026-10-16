"""Red bayesiana discreta con controles sintéticos generados por otra red.

SAM >= 1; 32 sujetos de desarrollo y ocho externos; LOSO agrupado por sujeto.
Imputación, DT y cuantiles se ajustan con entrenamiento real. La estructura
del clasificador y sus límites son los mismos de la discreta sin sobremuestreo:
PC condicionado por clase o GES con BIC discreto condicional, clase raíz.
El generador aprende solo de controles reales del entrenamiento. Los sintéticos
intervienen exclusivamente en las CPD EEG; P(Y) conserva el K2 original.
Cada ajuste compara CPD originales y aumentadas sobre idéntica estructura.
Los sintéticos no son sujetos nuevos ni evidencia independiente para tests PC,
BIC, validación, bootstrap o diagnósticos. Sin afirmaciones causales.
"""

import argparse
import json
import os
import threading
import time
import warnings
from uuid import uuid4
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
from pgmpy.causal_discovery import GES, PC
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.structure_score import BaseStructureScore
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm as _tqdm

_CACHE_RANKING_DT = {}
VERSION_MODELO = 'discreta_sinteticos_cpd_prior_original_v1'

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
    X = df_entrenamiento[candidatas].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("DT requiere predictores imputados y finitos.")
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


class BICDiscreto(BaseStructureScore):
    """BIC multinomial estándar; cuenta solo configuraciones observadas.

    El término de penalización incluye todas las configuraciones posibles.
    Evita reservar tablas enormes durante la búsqueda de estructura.
    """

    _tags = {'name': 'bic-discreto-conteos', 'supported_datatype': 'discrete',
             'default_for': None, 'is_parameteric': False}

    def __init__(self, data, state_names):
        super().__init__(data, state_names=state_names)
        self.codigos = {c: data[c].to_numpy(dtype=np.int64) for c in data.columns}
        self.card = {c: len(state_names[c]) for c in data.columns}
        self.n = len(data)

    def _local_score(self, variable, parents):
        ri = self.card[variable]
        qi = int(np.prod([self.card[p] for p in parents], dtype=np.int64))
        codigos = np.zeros(self.n, dtype=np.int64)
        for padre in parents:
            codigos = codigos * self.card[padre] + self.codigos[padre]
        _, inversos, totales = np.unique(codigos, return_inverse=True, return_counts=True)
        conteos = np.bincount(inversos * ri + self.codigos[variable],
                              minlength=len(totales) * ri).reshape(-1, ri)
        filas, columnas = np.nonzero(conteos)
        positivos = conteos[filas, columnas]
        ll = float(np.sum(positivos * (np.log(positivos) - np.log(totales[filas]))))
        valor = ll - 0.5 * np.log(self.n) * qi * (ri - 1)
        if not np.isfinite(valor):
            raise ValueError(f'BIC discreto no finito: {variable}, padres={parents}.')
        return float(valor)


class BICDiscretoCondicionado(BICDiscreto):
    """BIC de las CPD EEG con la clase como padre obligatorio.

    El buscador solo recorre nodos EEG. La contribución P(Y) es constante
    durante esa búsqueda. El score incluye los estados de Y y el número
    completo de parámetros de cada CPD condicionada por clase.
    """

    def _local_score(self, variable, parents):
        if variable == 'Ansiedad':
            if parents:
                raise ValueError('La clase raíz no admite padres.')
            return super()._local_score(variable, ())
        padres = tuple(sorted(set(parents) | {'Ansiedad'}))
        return super()._local_score(variable, padres)


def validar_dag(dag, columnas):
    faltantes = set(columnas) - set(dag.nodes())
    extras = set(dag.nodes()) - set(columnas)
    if faltantes or extras:
        raise ValueError(f'Nodos incorrectos: faltantes={sorted(faltantes)}, extras={sorted(extras)}.')
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError(f'Ciclo dirigido: {nx.find_cycle(dag)}.')


def orientar_pc(esqueleto, columnas, score):
    """Mantiene el esqueleto EEG de PC y orienta por BIC discreto condicional.

    Orden inicial: características EEG seleccionadas por DT. Los intercambios adyacentes
    aceptan únicamente mejoras estrictas. Es una heurística predictiva.
    """
    orden = list(columnas)
    esqueleto = nx.Graph(esqueleto)
    if set(esqueleto.nodes()) != set(orden) or nx.number_of_selfloops(esqueleto):
        raise ValueError('El esqueleto PC no coincide con las variables discretas.')
    posicion = {c: j for j, c in enumerate(orden)}
    padres = {c: {p for p in esqueleto.neighbors(c) if posicion[p] < posicion[c]} for c in orden}

    def local(c, ps):
        return score.local_score(c, tuple(sorted(ps)))

    inicial = sum(local(c, padres[c]) for c in orden)
    cambios = 0
    convergio = False
    for pasada in range(len(orden) ** 2):
        mejoro = False
        for j in range(len(orden) - 1):
            u, v = orden[j:j + 2]
            if not esqueleto.has_edge(u, v):
                continue
            nuevos_u, nuevos_v = padres[u] | {v}, padres[v] - {u}
            delta = local(u, nuevos_u) + local(v, nuevos_v) - local(u, padres[u]) - local(v, padres[v])
            if delta > 1e-8:
                orden[j], orden[j + 1] = v, u
                padres[u], padres[v] = nuevos_u, nuevos_v
                cambios += 1
                mejoro = True
        if not mejoro:
            convergio = True
            break
    dag = nx.DiGraph()
    dag.add_nodes_from(columnas)
    dag.add_edges_from((p, c) for c, ps in padres.items() for p in sorted(ps))
    validar_dag(dag, columnas)
    if {frozenset(e) for e in dag.edges()} != {frozenset(e) for e in esqueleto.edges()}:
        raise RuntimeError('La orientación predictiva alteró el esqueleto PC.')
    return dag, {'Orientacion': 'PC-esqueleto EEG condicionado por clase; orientación predictiva BIC discreto condicional',
                 'Orden_orientacion': orden, 'BIC_orientacion_inicial': inicial,
                 'BIC_orientacion_final': sum(local(c, padres[c]) for c in orden),
                 'Intercambios_orientacion': cambios, 'Pasadas_orientacion': pasada + 1,
                 'Orientacion_convergio': convergio}


def aprender_estructura(datos, estados, algoritmo):
    objetivo = 'Ansiedad'
    columnas_eeg = [c for c in datos.columns if c != objetivo]
    eeg = datos[columnas_eeg]
    score = BICDiscretoCondicionado(datos, estados)
    if algoritmo == 'PC':
        from pgmpy.ci_tests import get_ci_test
        test_chi2 = get_ci_test(test='chi_square', data=datos)

        def independencia_dada_clase(X, Y, Z=(), significance_level=0.05):
            # El test se instancia con los datos completos del entrenamiento;
            # PC solo ve nodos EEG. La clase se añade a todo conjunto separador.
            condicion = sorted(set(Z) | {objetivo})
            return test_chi2(X, Y, condicion, significance_level=significance_level)

        estimador = PC(variant='stable', ci_test=independencia_dada_clase,
                       significance_level=0.05, return_type='pdag', show_progress=False)
        estimador.fit(eeg)
        pdag = estimador.causal_graph_
        dirigido = nx.DiGraph()
        dirigido.add_nodes_from(columnas_eeg)
        dirigido.add_edges_from(pdag.directed_edges)
        dag_eeg, meta = orientar_pc(estimador.skeleton_, columnas_eeg, score)
        meta.update({'Ciclo_PC_detectado': not nx.is_directed_acyclic_graph(dirigido),
                     'PDAG_EEG_original': {'dirigidas': sorted(pdag.directed_edges),
                                           'no_dirigidas': sorted(pdag.undirected_edges)},
                     'Criterio': 'PC stable chi_square dado Ansiedad; orientación BIC discreto condicional'})
    elif algoritmo == 'GES':
        # El score añade Ansiedad a cada conjunto de padres EEG. GES no
        # puede orientar aristas hacia la clase porque esta no es un nodo
        # de su espacio de búsqueda; se incorpora como raíz al terminar.
        estimador = GES(scoring_method=score, return_type='pdag')
        estimador.fit(eeg)
        pdag = estimador.causal_graph_
        dag_eeg = pdag.to_dag()
        validar_dag(dag_eeg, columnas_eeg)
        esqueleto = {frozenset(e) for e in pdag.edges()}
        if {frozenset(e) for e in dag_eeg.edges()} != esqueleto:
            raise ValueError('La conversión GES alteró su esqueleto EEG.')
        if not set(pdag.directed_edges).issubset(set(dag_eeg.edges())):
            raise ValueError('La conversión GES invirtió una arista dirigida EEG.')

        def colliders(arcos):
            g = nx.DiGraph(); g.add_nodes_from(columnas_eeg); g.add_edges_from(arcos)
            return {(c, frozenset((a, b))) for c in g.nodes()
                    for a, b in combinations(g.predecessors(c), 2)
                    if frozenset((a, b)) not in esqueleto}

        if colliders(pdag.directed_edges) != colliders(dag_eeg.edges()):
            raise ValueError('GES no produjo una extensión consistente de su PDAG EEG.')
        meta = {'Orientacion': 'Extensión consistente del PDAG EEG de GES; clase raíz impuesta',
                'Criterio': 'GES pgmpy con BIC discreto condicional a Ansiedad',
                'PDAG_EEG_original': {'dirigidas': sorted(pdag.directed_edges),
                                      'no_dirigidas': sorted(pdag.undirected_edges)}}
    else:
        raise ValueError(f'Algoritmo no reconocido: {algoritmo}.')
    validar_dag(dag_eeg, columnas_eeg)
    dag = nx.DiGraph()
    dag.add_nodes_from(datos.columns)
    dag.add_edges_from(dag_eeg.edges())
    dag.add_edges_from((objetivo, c) for c in columnas_eeg)
    validar_dag(dag, datos.columns)
    meta.update({'Motor': 'pgmpy', 'Version_modelo': VERSION_MODELO,
                 'Clase_raiz': True, 'N_arcos_clase_impuestos': len(columnas_eeg),
                 'N_arcos_EEG_aprendidos': dag_eeg.number_of_edges(),
                 'Arcos_clase_protegidos': [(objetivo, c) for c in columnas_eeg],
                 'N_arcos_antes_limites': dag.number_of_edges(),
                 'Arcos_antes_limites': list(dag.edges())})
    return dag, score, meta


def limitar_complejidad(dag, score, estados, max_padres, max_celdas):
    """Refinamiento posterior explícito; no cambia los tests PC ni la búsqueda GES.

    Las aristas Ansiedad -> EEG están protegidas. Si un nodo excede un
    límite, se elimina el padre EEG cuya retirada deja el
    mejor BIC local del entrenamiento. Se registra cada retirada; no se
    interpreta la estructura refinada como una extensión causal del PDAG.
    """
    dag = nx.DiGraph(dag)
    if dag.in_degree('Ansiedad') != 0:
        raise ValueError('La clase debe ser raíz antes de aplicar límites.')
    for nodo in set(dag.nodes()) - {'Ansiedad'}:
        if not dag.has_edge('Ansiedad', nodo):
            raise ValueError(f'Falta la arista de clase protegida hacia {nodo}.')
    retirados = []
    for nodo in list(dag.nodes()):
        while True:
            padres = sorted(dag.predecessors(nodo))
            celdas = len(estados[nodo]) * int(np.prod([len(estados[p]) for p in padres], dtype=np.int64))
            if len(padres) <= max_padres and celdas <= max_celdas:
                break
            candidatos_eeg = [p for p in padres if p != 'Ansiedad']
            if not candidatos_eeg:
                raise ValueError(f'Los límites no permiten la CPD mínima de {nodo} dada la clase; aumenta los límites.')
            antes = score.local_score(nodo, tuple(padres))
            candidatos = [(score.local_score(nodo, tuple(p for p in padres if p != quitar)), quitar)
                          for quitar in candidatos_eeg]
            mejor = max(v for v, _ in candidatos)
            quitar = min(p for v, p in candidatos if v == mejor)
            dag.remove_edge(quitar, nodo)
            retirados.append({'padre': quitar, 'hijo': nodo, 'BIC_antes': antes,
                              'BIC_despues': mejor, 'celdas_antes': celdas,
                              'motivo': 'Límite declarado de padres o celdas de CPD'})
    validar_dag(dag, estados)
    if dag.in_degree('Ansiedad') != 0 or set(dag.successors('Ansiedad')) != set(estados) - {'Ansiedad'}:
        raise RuntimeError('El refinamiento alteró las aristas de clase protegidas.')
    return dag, retirados


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
            'Factorizacion': 'P(Y) * producto_j P(X_j | Y, padres_EEG(X_j))',
            'cpds': [{'variables': list(cpd.variables), 'cardinality': cpd.cardinality.tolist(),
                      'state_names': cpd.state_names, 'values': np.asarray(cpd.values).tolist()}
                     for cpd in modelo.get_cpds()]}


def tabla_predicciones(datos, prob):
    prob = np.asarray(prob, dtype=float)
    if (prob.ndim != 1 or len(datos) != len(prob) or not np.isfinite(prob).all()
            or np.any((prob < 0) | (prob > 1))):
        raise ValueError('No hay una probabilidad válida y finita por época.')
    return pd.DataFrame({'Indice_fila': datos.index.to_numpy(), 'Sujeto': datos['Sujeto'].to_numpy(),
                         'Ansiedad_real': datos['Ansiedad'].to_numpy(dtype=int),
                         'Probabilidad_ansiedad': prob, 'Prediccion': (prob >= 0.5).astype(int)})



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





def validar_clasificador(modelo, estados):
    validar_dag(modelo, estados)
    eeg = set(estados) - {'Ansiedad'}
    if modelo.in_degree('Ansiedad') != 0 or set(modelo.successors('Ansiedad')) != eeg:
        raise ValueError('La clase debe ser raíz y padre de todos los predictores.')
    modelo.check_model()
    if modelo.get_cpds('Ansiedad').state_names['Ansiedad'] != [0, 1]:
        raise ValueError('La CPD de clase debe contener estados [0, 1].')
    for nodo in estados:
        cpd = modelo.get_cpds(nodo)
        if cpd is None or cpd.state_names[nodo] != estados[nodo]:
            raise ValueError(f'La CPD de {nodo} no conserva todos los estados efectivos.')
        if set(cpd.variables[1:]) != set(modelo.predecessors(nodo)):
            raise ValueError(f'Padres y CPD incompatibles para {nodo}.')
        p = np.asarray(cpd.values)
        if not np.isfinite(p).all() or np.any(p <= 0) or np.any(p > 1):
            raise ValueError(f'CPD no positiva o no finita para {nodo}.')


def extension_consistente(pdag, nombres):
    dag = pdag.to_dag()
    validar_dag(dag, nombres)
    esqueleto = {frozenset(e) for e in pdag.edges()}
    if {frozenset(e) for e in dag.edges()} != esqueleto:
        raise ValueError('La extensión del generador GES cambió su esqueleto.')
    if not set(pdag.directed_edges).issubset(set(dag.edges())):
        raise ValueError('La extensión del generador invirtió una flecha de GES.')
    def colliders(arcos):
        g = nx.DiGraph(); g.add_nodes_from(nombres); g.add_edges_from(arcos)
        return {(c, frozenset((a, b))) for c in g.nodes()
                for a, b in combinations(g.predecessors(c), 2)
                if frozenset((a, b)) not in esqueleto}
    if colliders(pdag.directed_edges) != colliders(dag.edges()):
        raise ValueError('La extensión del generador cambió estructuras en V.')
    return dag


def aprender_generador(controles, estados, algoritmo, args):
    """Aprende y estima un generador con controles reales exclusivamente."""
    score = BICDiscreto(controles, estados)
    if algoritmo == 'PC':
        estimador = PC(variant='stable', ci_test='chi_square', significance_level=.05,
                       return_type='pdag', show_progress=False)
        estimador.fit(controles)
        dag, orientacion = orientar_pc(estimador.skeleton_, controles.columns, score)
        orientacion['Orientacion'] = 'Esqueleto PC de controles; orientación predictiva por BIC discreto de controles'
        orientacion['PDAG_original'] = {'Dirigidas': sorted(estimador.causal_graph_.directed_edges),
                                      'No_dirigidas': sorted(estimador.causal_graph_.undirected_edges)}
    else:
        estimador = GES(scoring_method=score, return_type='pdag')
        estimador.fit(controles)
        dag = extension_consistente(estimador.causal_graph_, controles.columns)
        orientacion = {'Orientacion': 'Extensión consistente del PDAG GES de controles'}
    dag = nx.DiGraph(dag)
    # Igual máximo de padres EEG que el clasificador; el generador no tiene nodo clase.
    max_padres = args.max_padres - 1
    retirados = []
    for nodo in controles.columns:
        while True:
            padres = sorted(dag.predecessors(nodo))
            celdas = len(estados[nodo])*int(np.prod([len(estados[p]) for p in padres], dtype=np.int64))
            if len(padres) <= max_padres and celdas <= args.max_celdas_cpd:
                break
            if not padres:
                raise ValueError(f'El límite no permite la CPD generadora de {nodo}.')
            opciones = [(score.local_score(nodo, tuple(p for p in padres if p != quitar)), quitar)
                        for quitar in padres]
            mejor = max(v for v, _ in opciones)
            quitar = min(p for v, p in opciones if v == mejor)
            retirados.append({'Padre': quitar, 'Hijo': nodo, 'Celdas_antes': celdas,
                              'Motivo': 'Límite de complejidad del generador'})
            dag.remove_edge(quitar, nodo)
    validar_dag(dag, estados)
    modelo, diagnostico = ajustar_cpds(dag, controles, estados, args.max_celdas_cpd)
    for nodo in estados:
        if modelo.get_cpds(nodo).state_names[nodo] != estados[nodo]:
            raise ValueError(f'El generador perdió estados de {nodo}.')
    return modelo, {'N_controles_reales': len(controles), 'Algoritmo': algoritmo,
                    'Motor': 'pgmpy', 'Fuente': 'Solo controles del entrenamiento real',
                    'N_arcos': dag.number_of_edges(), 'Max_padres_EEG': max_padres,
                    'Arcos_retirados': retirados, 'Diagnostico_CPD': diagnostico,
                    **orientacion, 'Modelo': {**serializar_modelo(modelo),
                    'Tipo': 'Generador discreto de controles, sin nodo clase',
                    'Version_modelo': 'generador_controles_k2_v1',
                    'Factorizacion': 'producto_j P(X_j | padres_EEG(X_j))'}}


def obtener_semilla(datos, top, bins, algoritmo, semilla_base):
    firma = blake2b(digest_size=8)
    firma.update(pd.util.hash_pandas_object(datos, index=True).to_numpy().tobytes())
    firma.update(json.dumps([int(semilla_base), int(top), int(bins), algoritmo,
                            list(datos.columns)], ensure_ascii=False).encode('utf-8'))
    return int.from_bytes(firma.digest(), byteorder='little')


def muestrear_generador(modelo, n, estados, semilla):
    """Muestreo ancestral vectorizado; permite nodos aislados y grafos sin aristas."""
    if n < 0:
        raise ValueError('El número de sintéticos debe ser no negativo.')
    columnas = list(estados)
    salida = pd.DataFrame(index=pd.RangeIndex(n), columns=columnas, dtype=np.int64)
    if n == 0:
        return salida
    rng = np.random.default_rng(semilla)
    posicion = {c: i for i, c in enumerate(columnas)}
    for nodo in nx.lexicographical_topological_sort(modelo, key=posicion.get):
        cpd = modelo.get_cpds(nodo)
        padres = list(cpd.variables[1:])
        codigo = np.zeros(n, dtype=np.int64)
        for padre in padres:
            codigo = codigo*len(estados[padre]) + salida[padre].to_numpy(dtype=np.int64)
        p = np.asarray(cpd.values).reshape(len(estados[nodo]), -1)[:, codigo].T
        if (not np.isfinite(p).all() or np.any(p < 0)
                or not np.allclose(p.sum(axis=1), 1.0)):
            raise ValueError(f'CPD inválida al generar {nodo}.')
        acumulada = np.cumsum(p, axis=1)
        acumulada[:, -1] = 1.0
        salida[nodo] = (rng.random(n)[:, None] >= acumulada).sum(axis=1).astype(np.int64)
        if not salida[nodo].isin(estados[nodo]).all():
            raise ValueError(f'El generador produjo un estado inválido para {nodo}.')
    return salida[columnas].astype(np.int64)


def diagnostico_sinteticos(reales, sinteticos, estados):
    n = len(sinteticos)
    por_variable = {}
    for c, categorias in estados.items():
        p = reales[c].value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
        q = (sinteticos[c].value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
             if n else np.full(len(categorias), np.nan))
        por_variable[c] = {'Estados': categorias, 'Frecuencia_real': p.tolist(),
                           'Frecuencia_sintetica': q.tolist() if n else None,
                           'Distancia_variacion_total': float(.5*np.abs(p-q).sum()) if n else None}
    filas_reales = set(reales.itertuples(index=False, name=None))
    coincidentes = sum(t in filas_reales for t in sinteticos.itertuples(index=False, name=None))
    return {'N_sinteticos': n, 'N_sinteticos_unicos': int(len(sinteticos.drop_duplicates())),
            'N_coincidentes_con_controles_reales': coincidentes,
            'Distribuciones_marginales': por_variable,
            'Nota': 'Diagnósticos descriptivos. Coincidencias son posibles al discretizar; no demuestran fuga. '
                    'Las marginales no validan por sí solas todas las dependencias conjuntas.'}


def ajustar_y_predecir(train, test, candidatas, top, bins, algoritmo, args, etiqueta):
    inicio = time.perf_counter()
    a, b, estados, prepro = preparar_particion(train, test, candidatas, 'Ansiedad', top, bins)
    tpre = time.perf_counter()-inicio
    inicio = time.perf_counter()
    with mostrar_actividad(f'{etiqueta}: estructura del clasificador, solo datos reales'):
        dag, score, meta = aprender_estructura(a, estados, algoritmo)
        dag, retirados = limitar_complejidad(dag, score, estados, args.max_padres, args.max_celdas_cpd)
    testructura = time.perf_counter()-inicio
    inicio = time.perf_counter()
    base, diagnostico_base = ajustar_cpds(dag, a, estados, args.max_celdas_cpd)
    validar_clasificador(base, estados)
    tbase = time.perf_counter()-inicio
    columnas = [c for c in a if c != 'Ansiedad']
    controles = a.loc[a.Ansiedad == 0, columnas].copy()
    n0, n1 = len(controles), int((a.Ansiedad == 1).sum())
    if n0 == 0 or n1 == 0:
        raise ValueError('Se requieren ambas clases en el entrenamiento real.')
    n_sinteticos = max(0, n1-n0)
    semilla = obtener_semilla(a, top, bins, algoritmo, args.semilla_muestreo)
    inicio = time.perf_counter()
    with mostrar_actividad(f'{etiqueta}: generador, solo controles reales'):
        generador, gen = aprender_generador(controles, {c: estados[c] for c in columnas}, algoritmo, args)
    tgenerador = time.perf_counter()-inicio
    inicio = time.perf_counter()
    sinteticas = muestrear_generador(generador, n_sinteticos, {c: estados[c] for c in columnas}, semilla)
    diag_sinteticos = diagnostico_sinteticos(controles, sinteticas, {c: estados[c] for c in columnas})
    sinteticas['Ansiedad'] = np.zeros(n_sinteticos, dtype=np.int64)
    sinteticas = sinteticas[a.columns]
    aumentado = pd.concat([a, sinteticas], ignore_index=True)
    tmuestreo = time.perf_counter()-inicio
    inicio = time.perf_counter()
    modelo, diagnostico = ajustar_cpds(dag, aumentado, estados, args.max_celdas_cpd)
    # El prior K2 se copia del entrenamiento real, no de las clases balanceadas.
    modelo.remove_cpds(modelo.get_cpds('Ansiedad'))
    modelo.add_cpds(base.get_cpds('Ansiedad').copy())
    validar_clasificador(modelo, estados)
    if not np.array_equal(modelo.get_cpds('Ansiedad').values, base.get_cpds('Ansiedad').values):
        raise ValueError('No se conservó el prior del entrenamiento original.')
    if set(modelo.edges()) != set(base.edges()):
        raise ValueError('El sobremuestreo alteró la estructura del clasificador.')
    for c in columnas:
        # Solo se añaden controles: P(X|padres,Y=1) debe permanecer idéntica.
        cpd, cpd_base = modelo.get_cpds(c), base.get_cpds(c)
        eje = list(cpd.variables).index('Ansiedad')
        if not np.array_equal(np.take(cpd.values, 1, axis=eje), np.take(cpd_base.values, 1, axis=eje)):
            raise ValueError(f'El aumento de controles alteró inesperadamente la CPD positiva de {c}.')
    diagnostico = [next(d for d in diagnostico_base if d['nodo']=='Ansiedad') if d['nodo']=='Ansiedad' else d
                  for d in diagnostico]
    tparam = time.perf_counter()-inicio
    inicio = time.perf_counter()
    prob = inferir_posterior(modelo, b, 'Ansiedad')
    prob_base = inferir_posterior(base, b, 'Ansiedad')
    if not prob.index.equals(test.index) or not prob_base.index.equals(test.index):
        raise ValueError('Las predicciones no conservan las filas esperadas.')
    tinferencia = time.perf_counter()-inicio
    meta.update({'ID_ejecucion': args.ejecucion_id, 'Version_modelo': VERSION_MODELO,
                 'Tiempo_preprocesamiento_s': tpre, 'Tiempo_estructura_s': testructura,
                 'Tiempo_parametros_base_s': tbase, 'Tiempo_generador_s': tgenerador,
                 'Tiempo_muestreo_s': tmuestreo, 'Tiempo_parametros_s': tparam,
                 'Tiempo_inferencia_s': tinferencia,
                 'N_reales': len(a), 'N_controles_reales': n0, 'N_positivos_reales': n1,
                 'N_sinteticos': n_sinteticos, 'N_aumentado': len(aumentado),
                 'Prevalencia_original': n1/len(a), 'Prevalencia_aumentada': n1/len(aumentado),
                 'Prior_clase_usado': np.asarray(base.get_cpds('Ansiedad').values).tolist(),
                 'Prior_clase_fuente': 'K2 de entrenamiento real, (N_c+1)/(N+2)',
                 'Semilla_muestreo': str(semilla),
                 'Arcos_retirados_por_limites': retirados,
                 'N_arcos': dag.number_of_edges(), 'N_arcos_EEG_final': dag.number_of_edges()-len(columnas),
                 'N_caracteristicas_hijas_clase': len(columnas),
                 'N_parametros': sum(d['parametros_libres'] for d in diagnostico),
                 'Max_padres_efectivos': max(len(d['padres']) for d in diagnostico),
                 'Max_celdas_CPD_efectivas': max(d['celdas_CPD'] for d in diagnostico),
                 'N_configuraciones_CPD_sin_datos': sum(d['configuraciones_sin_datos'] for d in diagnostico),
                 'Umbral_fijo': .5, 'Estructura_solo_reales': True,
                 'Metodo': 'Aumento generativo de controles solo para CPD; estructura y prior originales'})
    evidencia = {'Preprocesamiento': prepro, 'Detalles': meta,
                 'Generador': gen, 'Diagnostico_sinteticos': diag_sinteticos,
                 'Diagnostico_CPD_aumentadas': diagnostico, 'Diagnostico_CPD_originales': diagnostico_base,
                 'Modelo_aumentado': serializar_modelo(modelo), 'Modelo_sin_sobremuestreo': serializar_modelo(base)}
    if args.diagnosticos_estructura:
        from pgmpy.metrics import CorrelationScore, FisherC
        diag = {'Fuente': 'Entrenamiento real, nunca sintéticos',
                'Tipo': 'Diagnósticos estructurales, no métricas predictivas ni evidencia causal',
                'Advertencia': 'Épocas dependientes dentro de sujeto y aproximaciones chi-cuadrado asintóticas'}
        for nombre, estimador in [('CS', CorrelationScore(ci_test='chi_square', significance_level=.05)),
                                  ('FisherC', FisherC(ci_test='chi_square', compute_rmsea=True, show_progress=False))]:
            try:
                diag[nombre] = np.asarray(estimador.evaluate(X=a, causal_graph=dag)).tolist()
            except Exception as exc:
                diag[nombre] = {'Estado': 'NO_DISPONIBLE', 'Error': f'{type(exc).__name__}: {exc}'}
        evidencia['Diagnosticos_estructura'] = diag
    return prob.to_numpy(), modelo, meta, evidencia, prob_base.to_numpy(), sinteticas


def guardar_ajuste(evidencia, sinteticas, carpeta, clave):
    guardar_evidencia({f'Ajustes/{clave}.json': evidencia}, carpeta)
    destino = Path(carpeta)/'Sinteticos'/f'{clave}.csv'
    destino.parent.mkdir(parents=True, exist_ok=True)
    guardar_csv(sinteticas, destino)


def ejecutar_loso(datos, candidatas, top, bins, algoritmo, args, carpeta):
    key = f'{algoritmo}_top{top}_bins{bins}'
    grupos = datos.Sujeto.to_numpy()
    prob, pbase = np.full(len(datos), np.nan), np.full(len(datos), np.nan)
    registros = []
    estado = {'ID_ejecucion': args.ejecucion_id, 'Estado': 'EN_CURSO', 'Pliegues': []}
    ruta = f'LOSO/{key}/Resumen.json'
    guardar_evidencia({ruta: estado}, carpeta)
    guardar_csv(pd.DataFrame(columns=['ID_ejecucion','Indice_fila','Sujeto','Ansiedad_real',
                                      'Probabilidad_ansiedad','Prediccion','Probabilidad_sin_sobremuestreo',
                                      'Prediccion_sin_sobremuestreo']),Path(carpeta)/f'LOSO/{key}/Predicciones.csv')
    barra = tqdm(LeaveOneGroupOut().split(datos, datos.Ansiedad, grupos), total=32,
                 desc=f'LOSO Discreta sintéticos {algoritmo} Top {top}, bins {bins}', unit='fold')
    for fold, (it, iv) in enumerate(barra, 1):
        sujeto = str(grupos[iv[0]])
        contexto = f'{algoritmo}, Top {top}, bins {bins}, fold {fold}/32, sujeto {sujeto}'
        registro = {'ID_ejecucion': args.ejecucion_id, 'Fold': fold, 'Sujeto_validacion': sujeto,
                    'Sujetos_entrenamiento': sorted(set(map(str, grupos[it]))),
                    'N_entrenamiento': len(it), 'N_validacion': len(iv)}
        inicio = time.perf_counter()
        try:
            if len(set(grupos[it])) != 31 or len(set(grupos[iv])) != 1 or set(grupos[it]) & set(grupos[iv]):
                raise ValueError('LOSO requiere 31/1 sujetos disjuntos.')
            p, _, meta, evidencia, pb, sinteticas = ajustar_y_predecir(
                datos.iloc[it], datos.iloc[iv], candidatas, top, bins, algoritmo, args, contexto)
            guardar_ajuste(evidencia, sinteticas, carpeta, f'{key}/Fold_{fold:02d}')
            registro.update({k:v for k,v in meta.items() if k.startswith('Tiempo_') or k in
                             {'N_sinteticos','N_controles_reales','N_positivos_reales','Semilla_muestreo',
                              'N_arcos','N_arcos_EEG_final'}},
                            Estado='OK', Tiempo_fold_s=time.perf_counter()-inicio)
            prob[iv], pbase[iv] = p, pb
            registros.append(registro)
            estado['Pliegues'].append(registro)
            listos = np.flatnonzero(np.isfinite(prob))
            tabla = tabla_predicciones(datos.iloc[listos], prob[listos])
            tabla.insert(0, 'ID_ejecucion', args.ejecucion_id)
            tabla['Probabilidad_sin_sobremuestreo'] = pbase[listos]
            tabla['Prediccion_sin_sobremuestreo'] = (pbase[listos]>=.5).astype(int)
            guardar_csv(tabla, Path(carpeta)/f'LOSO/{key}/Predicciones.csv')
            estado.update(N_pliegues_completos=fold, N_predicciones=len(listos))
            guardar_evidencia({ruta: estado}, carpeta)
            barra.set_postfix(sujeto=sujeto, segundos=f'{registro["Tiempo_fold_s"]:.1f}')
        except Exception as exc:
            registro.update(Estado='ERROR', Error=f'{type(exc).__name__}: {exc}')
            estado.update(Estado='ERROR', Error=registro)
            guardar_evidencia({ruta: estado}, carpeta)
            raise RuntimeError(f'Falló {contexto}. {type(exc).__name__}: {exc}') from exc
    if not np.isfinite(prob).all() or not np.isfinite(pbase).all():
        raise RuntimeError(f'Predicciones LOSO incompletas para {key}.')
    estado['Estado'] = 'COMPLETO'
    guardar_evidencia({ruta: estado}, carpeta)
    return prob, pd.DataFrame(registros), pbase


def principal(argv=None):
    parser = argparse.ArgumentParser(description='Discreta con controles sintéticos: estructura y prior originales; comparación pareada de CPD.')
    parser.add_argument('--algoritmos', nargs='+', choices=['GES','PC'], default=['GES','PC'])
    parser.add_argument('--tops', nargs='+', type=int, choices=[10,15,20], default=[10,15,20])
    parser.add_argument('--bins', nargs='+', type=int, choices=[2,3], default=[2,3])
    parser.add_argument('--max-padres', type=int, default=3, help='Padres totales del clasificador, incluida la clase; generador usa como máximo uno menos.')
    parser.add_argument('--max-celdas-cpd', type=int, default=256)
    parser.add_argument('--semilla-muestreo', type=int, default=42, help='Base de semillas derivadas de cada entrenamiento, Top, intervalos y algoritmo.')
    parser.add_argument('--diagnosticos-estructura', action='store_true', help='CS/FisherC opcionales, solo con registros reales; aumenta el tiempo.')
    args = parser.parse_args(argv)
    args.ejecucion_id = uuid4().hex
    if args.max_padres < 1 or args.max_celdas_cpd < 2*max(args.bins) or args.semilla_muestreo < 0:
        parser.error('max-padres debe ser >=1; max-celdas-cpd >=2*max(bins); semilla-muestreo >=0.')
    raiz = Path(__file__).resolve().parents[3]
    salida = raiz/'Resultados'/'Red Bayesiana'/'Global'/'Discreta_Oversampling'
    evidencia_dir = salida/'Evidencia'
    salida.mkdir(parents=True, exist_ok=True)
    desarrollo, externa, candidatas, sujetos, excluidas = cargar_datos(raiz)
    config = {**vars(args), 'ID_ejecucion': args.ejecucion_id, 'Estado_ejecucion': 'EN_CURSO',
              'Version_modelo': VERSION_MODELO,
              'Inicio_UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'Criterio': 'SAM=0 control; SAM>=1 caso', 'Semilla_particion': 42,
              'Sujetos_desarrollo_32': list(map(str,sujetos[:32])),
              'Sujetos_externos_8': list(map(str,sujetos[32:])),
              'Epocas_excluidas_por_criterio_SAM': excluidas,
              'N_candidatas': len(candidatas), 'Candidatas': candidatas,
              'Estructura_clasificador': 'Solo entrenamiento real: EEG condicionado por clase; clase raíz; límites como discreta sin aumento',
              'Generador': 'Red discreta PC/GES ajustada solo a controles reales del entrenamiento, con todos los estados del discretizador',
              'Aumento': 'Añade controles hasta igualar positivos; mantiene todos los registros reales',
              'Uso_sinteticos': 'Solo CPD EEG; nunca DT, cuantiles, estructura del clasificador, prior, validación, bootstrap ni diagnósticos de estructura',
              'Prior_clase': 'K2 de entrenamiento original: (N_c+1)/(N+2); se fija directamente en la CPD raíz',
              'Comparador_pareado': 'Sin sobremuestreo, idénticos datos reales, estructura, límites y preprocesamiento en la misma corrida',
              'Umbral_fijo': .5, 'Prior_parametros': 'K2: un pseudoconteo por celda',
              'Version_pgmpy': obtener_version('pgmpy'), 'Version_numpy': obtener_version('numpy'),
              'Version_pandas': obtener_version('pandas'), 'Version_sklearn': obtener_version('scikit-learn'),
              'Interpretacion': 'Aumento predictivo experimental, sin afirmaciones causales. Los sintéticos no son sujetos ni evidencia independiente.',
              'Nota_validacion': 'Épocas agrupadas por sujeto. Ocho sujetos externos ya observados en experimentos anteriores.',
              'Lectura_deltas': 'Delta = aumentado - sin sobremuestreo. Positivo favorece BA/AUC; negativo favorece Brier/log_loss.',
              'Evidencia_vigente': 'Usar solo artefactos del ID_ejecucion actual; no hay reanudación automática.'}
    guardar_evidencia({'Configuracion.json':config}, evidencia_dir)
    resumen = salida/'Resultados_Red_Discreta_Oversampling.csv'
    resultados = []
    guardar_csv(pd.DataFrame(columns=['Estructura','Top','n_bins','Estado']),resumen)
    print(f'Resultados: {salida}',flush=True)
    print('Estructura y prior solo con registros reales; controles sintéticos únicamente para CPD EEG.',flush=True)
    print(f'Configuraciones: {len(set(args.algoritmos))*len(set(args.tops))*len(set(args.bins))}; '
          'cada una guarda también el comparador sin sobremuestreo.',flush=True)
    inicio_total = time.perf_counter()
    algoritmo, top, bins, contexto = 'INICIO', None, None, 'Inicio'
    try:
        for bins in dict.fromkeys(args.bins):
            for top in dict.fromkeys(args.tops):
                for algoritmo in dict.fromkeys(args.algoritmos):
                    inicio = time.perf_counter()
                    key = f'{algoritmo}_top{top}_bins{bins}'
                    contexto = f'{algoritmo}, Top {top}, bins {bins}, LOSO'
                    pval, tiempos, pval_base = ejecutar_loso(desarrollo,candidatas,top,bins,algoritmo,args,evidencia_dir)
                    contexto = f'{algoritmo}, Top {top}, bins {bins}, ajuste final 32 sujetos'
                    ptest, modelo, meta, evidencia, ptest_base, sinteticas = ajustar_y_predecir(
                        desarrollo,externa,candidatas,top,bins,algoritmo,args,contexto)
                    guardar_ajuste(evidencia,sinteticas,evidencia_dir,f'{key}/Final')
                    guardar_evidencia({f'Modelos_Finales/{key}.json':evidencia},evidencia_dir)
                    tabla = tabla_predicciones(externa,ptest)
                    tabla.insert(0,'ID_ejecucion',args.ejecucion_id)
                    tabla['Probabilidad_sin_sobremuestreo'] = ptest_base
                    tabla['Prediccion_sin_sobremuestreo'] = (ptest_base>=.5).astype(int)
                    carpeta_pred = evidencia_dir/'Predicciones_Externas'
                    carpeta_pred.mkdir(parents=True,exist_ok=True)
                    guardar_csv(tabla,carpeta_pred/f'{key}.csv')
                    efectivos = evidencia['Preprocesamiento']['n_bins_efectivos']
                    fila = {'ID_ejecucion':args.ejecucion_id,'Estructura':algoritmo,'Motor':meta['Motor'],
                            'Top':top,'n_bins':bins,'Estado':'COMPLETO','Version_modelo':VERSION_MODELO,
                            'Umbral_fijo':.5,'Metodo':meta['Metodo'],'Estructura_solo_reales':True,
                            'Clase_raiz':True,'N_arcos_clase_impuestos':top,
                            'N_arcos_EEG_final':meta['N_arcos_EEG_final'],'N_arcos_final':meta['N_arcos'],
                            'N_parametros_final':meta['N_parametros'],
                            'Max_padres_configurado':args.max_padres,'Max_celdas_CPD_configurado':args.max_celdas_cpd,
                            'Max_padres_efectivos_final':meta['Max_padres_efectivos'],
                            'Max_celdas_CPD_final':meta['Max_celdas_CPD_efectivas'],
                            'N_reales_final':meta['N_reales'],'N_controles_reales_final':meta['N_controles_reales'],
                            'N_positivos_reales_final':meta['N_positivos_reales'],
                            'N_sinteticos_final':meta['N_sinteticos'],'N_aumentado_final':meta['N_aumentado'],
                            'N_sinteticos_unicos_final':evidencia['Diagnostico_sinteticos']['N_sinteticos_unicos'],
                            'N_sinteticos_coincidentes_final':evidencia['Diagnostico_sinteticos']['N_coincidentes_con_controles_reales'],
                            'Prevalencia_original_final':meta['Prevalencia_original'],
                            'Prevalencia_aumentada_final':meta['Prevalencia_aumentada'],
                            'Prior_clase_usado_final':json.dumps(meta['Prior_clase_usado']),
                            'Semilla_muestreo_final':meta['Semilla_muestreo'],
                            'Caracteristicas_DT_final':json.dumps(evidencia['Preprocesamiento']['Caracteristicas_DT'],ensure_ascii=False),
                            'Bins_efectivos_final':json.dumps(efectivos,ensure_ascii=False),
                            'N_features_bins_reducidos_final':sum(n<bins for n in efectivos.values()),
                            'N_pliegues_LOSO_completos':len(tiempos),'N_predicciones_LOSO':len(pval),
                            'N_predicciones_externas':len(ptest),'Tiempo_LOSO_s':float(tiempos.Tiempo_fold_s.sum()),
                            'N_sinteticos_LOSO':int(tiempos.N_sinteticos.sum()),
                            'Interpretacion_grafo':'Dependencias predictivas; sintéticos solo para CPD; sin evidencia causal'}
                    for etapa in ['preprocesamiento','estructura','parametros_base','generador','muestreo','parametros','inferencia']:
                        fila[f'Tiempo_{etapa}_LOSO_s']=float(tiempos[f'Tiempo_{etapa}_s'].sum())
                        fila[f'Tiempo_{etapa}_final_s']=meta[f'Tiempo_{etapa}_s']
                    for prefijo,datos,prob,pbase in [('Val',desarrollo,pval,pval_base),('Test',externa,ptest,ptest_base)]:
                        y=datos.Ansiedad.to_numpy(dtype=int);grupos=datos.Sujeto.to_numpy()
                        for sufijo,p in [('',prob),('_sin_sobremuestreo',pbase)]:
                            etiqueta=prefijo+sufijo;pred=(p>=.5).astype(int)
                            fila.update({f'{k} ({etiqueta})':v for k,v in metricas_clasificacion(y,pred,p).items()})
                            tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
                            fila.update({f'{k} ({etiqueta})':int(v) for k,v in zip(['TN','FP','FN','TP'],[tn,fp,fn,tp])})
                            fila.update(bootstrap_ci_por_sujeto(y,pred,grupos,etiqueta))
                            fila.update(metricas_agregadas_por_sujeto(y,pred,grupos,etiqueta))
                        for metrica in ['Balanced_accuracy','ROC_AUC','Brier','Log_loss']:
                            fila[f'Delta_{metrica} ({prefijo})']=fila[f'{metrica} ({prefijo})']-fila[f'{metrica} ({prefijo}_sin_sobremuestreo)']
                    fila['Tiempo_configuracion_s']=time.perf_counter()-inicio
                    resultados.append(fila)
                    guardar_csv(pd.DataFrame(resultados),resumen)
                    contexto=f'{algoritmo}, Top {top}, bins {bins}, figura final'
                    graficar_dag(modelo,f'Discreta con controles sintéticos solo en CPD: {algoritmo}, Top {top}, {bins} intervalos',
                                 salida/f'Estructura_Global_{algoritmo}_top{top}_bins{bins}.png')
                    print(f'COMPLETO {key}: {time.perf_counter()-inicio:.1f} s',flush=True)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        config.update(Estado_ejecucion='ERROR',Contexto_fallido=contexto,Error=error)
        guardar_evidencia({'Configuracion.json':config},evidencia_dir)
        resultados.append({'ID_ejecucion':args.ejecucion_id,'Estructura':algoritmo,'Top':top,'n_bins':bins,'Estado':'ERROR','Error':error})
        guardar_csv(pd.DataFrame(resultados),resumen)
        raise RuntimeError(f'Falló {contexto}. Se conservan resumen y evidencia parcial. {error}') from exc
    config.update(Estado_ejecucion='COMPLETO',Tiempo_total_s=time.perf_counter()-inicio_total)
    guardar_evidencia({'Configuracion.json':config},evidencia_dir)
    print(f'Resumen: {resumen}',flush=True)
    print(f'Evidencia: {evidencia_dir}',flush=True)
    columnas=['Estructura','Top','n_bins','Estado','Balanced_accuracy (Val)','Balanced_accuracy (Test)',
              'Balanced_accuracy (Test_sin_sobremuestreo)','Delta_Balanced_accuracy (Test)',
              'ROC_AUC (Test)','Brier (Test)','Tiempo_configuracion_s']
    print(pd.DataFrame(resultados)[columnas].to_string(index=False,float_format=lambda x:f'{x:.4f}'))


if __name__ == '__main__':
    principal()
