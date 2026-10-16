"""Clasificador bayesiano con EEG continuo y clase binaria (SAM >= 1).

La estructura compartida entre características se busca con GES/PC sobre
EEG centrado por clase dentro del entrenamiento. Es una heurística de
estructura, no una búsqueda exacta del BIC condicional gaussiano completo.
PC conserva su esqueleto y usa una orientación predictiva acíclica por BIC.
La clase es raíz por diseño predictivo, sin interpretación causal.
Las CPD gaussianas específicas por clase y su prior empírico determinan
P(Ansiedad=1 | EEG). No se utiliza regresión logística.
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
from pgmpy.base import PDAG
from pgmpy.structure_score import BICGauss
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm as _tqdm

VERSION_CAUSAL_LEARN = '0.1.4.8'
_CACHE_RANKING_DT = {}

class RobustBICGauss(BICGauss):
    def _local_score(self, variable, parents):
        try:
            score = super()._local_score(variable, parents)
            if np.isnan(score) or np.isinf(score):
                raise ValueError(f"BIC no finito para {variable} con padres {parents}.")
            return score
        except Exception as exc:
            raise RuntimeError(f"Falló el BIC gaussiano para {variable}.") from exc

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
        node_color='lightblue',
        edgecolors='darkblue',
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
        edge_color='navy',
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


def cargar_ges_causal_learn():
    try:
        from causallearn.search.ScoreBased.GES import ges
    except ImportError as exc:
        raise RuntimeError(
            'No está disponible causal-learn o alguno de sus auxiliares. '
            f'En bn_anxiety_env ejecuta: python -m pip install causal-learn=={VERSION_CAUSAL_LEARN}. '
            'No se sustituirá automáticamente por otro motor.'
        ) from exc
    instalada = obtener_version('causal-learn')
    if instalada != VERSION_CAUSAL_LEARN:
        warnings.warn(f'Esta prueba se verificó con causal-learn {VERSION_CAUSAL_LEARN}; se detectó {instalada}.')
    return ges


def convertir_cpdag_a_dag(matriz, nombres):
    """Conserva el esqueleto, las flechas identificadas y las estructuras en V."""
    matriz = np.asarray(matriz)
    if matriz.shape != (len(nombres), len(nombres)) or np.any(np.diag(matriz) != 0):
        raise ValueError('La matriz del CPDAG tiene dimensiones o diagonal inválidas.')
    dirigidas, no_dirigidas = [], []
    for i, j in combinations(range(len(nombres)), 2):
        extremos = (int(matriz[i, j]), int(matriz[j, i]))
        if extremos == (0, 0):
            continue
        if extremos == (-1, 1):
            dirigidas.append((nombres[i], nombres[j]))
        elif extremos == (1, -1):
            dirigidas.append((nombres[j], nombres[i]))
        elif extremos == (-1, -1):
            no_dirigidas.append((nombres[i], nombres[j]))
        else:
            raise ValueError(f'Extremos no admitidos por GES: {extremos} en {nombres[i]}, {nombres[j]}.')
    pdag = PDAG(directed_ebunch=dirigidas, undirected_ebunch=no_dirigidas)
    pdag.add_nodes_from(nombres)
    dag = pdag.to_dag()
    esqueleto = {frozenset(arco) for arco in dirigidas + no_dirigidas}
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError('La conversión del CPDAG produjo un ciclo.')
    if set(dag.nodes()) != set(nombres) or {frozenset(arco) for arco in dag.edges()} != esqueleto:
        raise ValueError('La conversión alteró nodos o conexiones del CPDAG.')
    if not set(dirigidas).issubset(set(dag.edges())):
        raise ValueError('La conversión invirtió una flecha del CPDAG.')

    def estructuras_v(aristas):
        padres = {nodo: [] for nodo in nombres}
        for padre, hijo in aristas:
            padres[hijo].append(padre)
        return {(hijo, frozenset((a, b))) for hijo in nombres
                for a, b in combinations(padres[hijo], 2)
                if frozenset((a, b)) not in esqueleto}

    if estructuras_v(dirigidas) != estructuras_v(dag.edges()):
        raise ValueError('No se obtuvo una extensión consistente: cambiaron las estructuras en V.')
    return dag, {
        'Arcos_dirigidos_CPDAG': len(dirigidas),
        'Conexiones_no_dirigidas_CPDAG': len(no_dirigidas),
        'Orientacion': 'Extensión consistente del CPDAG; no identifica direcciones causales.',
    }


def orientar_esqueleto_pc(esqueleto, datos):
    """Orientación predictiva acíclica del esqueleto PC, sin quitar conexiones.

    Inicio: orden de las características seleccionadas por DT. Se prueban
    intercambios adyacentes que mejoren el BIC gaussiano del EEG centrado.
    Toda orientación sigue un orden total y por ello siempre es acíclica.
    Es una búsqueda local, no una extensión causal del PDAG de PC ni una
    optimización exacta del BIC de la red híbrida completa.
    """
    nombres = list(datos.columns)
    esqueleto = nx.Graph(esqueleto)
    if set(esqueleto.nodes()) != set(nombres) or nx.number_of_selfloops(esqueleto):
        raise ValueError('El esqueleto PC no coincide con las características EEG.')
    orden = nombres.copy()
    score = RobustBICGauss(datos)
    cache = {}

    def padres_actuales(secuencia):
        posicion = {nodo: i for i, nodo in enumerate(secuencia)}
        return {nodo: {vecino for vecino in esqueleto.neighbors(nodo)
                       if posicion[vecino] < posicion[nodo]} for nodo in nombres}

    def local(nodo, padres):
        clave = (nodo, tuple(sorted(padres)))
        if clave not in cache:
            cache[clave] = float(score.local_score(*clave))
        return cache[clave]

    padres = padres_actuales(orden)
    inicio = sum(local(nodo, padres[nodo]) for nodo in nombres)
    cambios = 0
    convergio = False
    # Un límite fijo mantiene acotado el coste de esta heurística. Incluso
    # si se alcanza, el orden obtenido sigue definiendo un DAG válido.
    for pasada in range(max(1, len(nombres) ** 2)):
        mejoro = False
        for i in range(len(orden) - 1):
            u, v = orden[i:i + 2]
            if not esqueleto.has_edge(u, v):
                continue
            nuevos_u, nuevos_v = padres[u] | {v}, padres[v] - {u}
            incremento = (local(u, nuevos_u) + local(v, nuevos_v)
                          - local(u, padres[u]) - local(v, padres[v]))
            if incremento > 1e-8:
                orden[i], orden[i + 1] = v, u
                padres[u], padres[v] = nuevos_u, nuevos_v
                mejoro = True
                cambios += 1
        if not mejoro:
            convergio = True
            break
    posicion = {nodo: i for i, nodo in enumerate(orden)}
    dag = nx.DiGraph()
    dag.add_nodes_from(nombres)
    dag.add_edges_from((u, v) if posicion[u] < posicion[v] else (v, u)
                       for u, v in esqueleto.edges())
    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError('La orientación predictiva PC produjo un ciclo.')
    if {frozenset(e) for e in dag.edges()} != {frozenset(e) for e in esqueleto.edges()}:
        raise RuntimeError('La orientación predictiva alteró conexiones del esqueleto PC.')
    return dag, {
        'Orientacion': 'PC-esqueleto + orientación predictiva por orden y BIC gaussiano; sin interpretación causal',
        'Orden_orientacion': orden,
        'BIC_orientacion_inicial': inicio,
        'BIC_orientacion_final': sum(local(nodo, padres[nodo]) for nodo in nombres),
        'Intercambios_orientacion': cambios,
        'Pasadas_orientacion': pasada + 1,
        'Orientacion_convergio': convergio,
    }


def aprender_estructura(df_train, algoritmo, motor_ges='causal-learn', max_padres=None):
    if not np.isfinite(df_train.to_numpy(dtype=float)).all():
        raise ValueError('El entrenamiento contiene valores no finitos.')
    if algoritmo == 'PC':
        from pgmpy.causal_discovery import PC
        # No solicitar DAG: la orientación del PDAG puede contener ciclos
        # con independencias incompatibles en una muestra finita.
        estimador = PC(variant='stable', ci_test='fisher_z', significance_level=0.05,
                       return_type='pdag', show_progress=False)
        estimador.fit(df_train)
        pdag = estimador.causal_graph_
        dirigido = nx.DiGraph()
        dirigido.add_nodes_from(df_train.columns)
        dirigido.add_edges_from(pdag.directed_edges)
        dag, detalles = orientar_esqueleto_pc(estimador.skeleton_, df_train)
        detalles.update({
            'Motor': 'pgmpy',
            'Puntuacion': 'PC stable: Fisher Z, alpha=0.05; orientación predictiva BIC gaussiano',
            'Ciclo_PC_detectado': not nx.is_directed_acyclic_graph(dirigido),
            'PDAG_PC_original': {
                'arcos_dirigidos': sorted(pdag.directed_edges),
                'conexiones_no_dirigidas': sorted(pdag.undirected_edges),
            },
        })
        return dag, detalles
    if algoritmo != 'GES':
        raise ValueError(f'Algoritmo no reconocido: {algoritmo}')
    if motor_ges == 'pgmpy':
        if max_padres is not None:
            raise ValueError('El límite de padres de esta prueba solo se implementa para causal-learn.')
        from pgmpy.causal_discovery import GES
        estimador = GES(scoring_method=RobustBICGauss(df_train), return_type='dag')
        estimador.fit(df_train)
        return estimador.causal_graph_, {'Motor': 'pgmpy', 'Puntuacion': 'BICGauss'}
    if motor_ges != 'causal-learn':
        raise ValueError(f'Motor GES no reconocido: {motor_ges}')
    ges = cargar_ges_causal_learn()
    nombres = list(df_train.columns)
    limite = len(nombres) - 1 if max_padres is None else min(max_padres, len(nombres) - 1)
    resultado = ges(df_train.to_numpy(dtype=float), score_func='local_score_BIC',
                    maxP=limite, parameters={'lambda_value': 0.5}, node_names=nombres)
    dag, detalles = convertir_cpdag_a_dag(resultado['G'].graph, nombres)
    puntuacion = float(np.asarray(resultado['score']).reshape(-1)[0])
    if not np.isfinite(puntuacion):
        raise ValueError('GES de causal-learn devolvió una puntuación no finita.')
    detalles.update({'Motor': 'causal-learn', 'Puntuacion': 'BIC gaussiano lineal; lambda=0.5',
                     'Score_GES': puntuacion, 'Max_padres_busqueda': limite})
    return dag, detalles


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


class ClasificadorBayesianoCondicionalGaussiano:
    """Red con clase binaria raíz y CPD gaussianas lineales por clase.

    P(Y, X) = P(Y) * producto_j P(X_j | Y, padres_EEG(X_j)).
    La estructura EEG es común, pero cada clase tiene interceptos,
    coeficientes y varianzas propios. Las probabilidades finales proceden
    exclusivamente de esta factorización, sin un clasificador de respaldo.
    """

    def __init__(self, columna_objetivo, ridge=1.0, piso_varianza=1e-6):
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError('La regularización ridge debe ser finita y no negativa.')
        if not np.isfinite(piso_varianza) or piso_varianza <= 0:
            raise ValueError('El piso de varianza debe ser finito y positivo.')
        self.objetivo = columna_objetivo
        self.ridge = float(ridge)
        self.piso_varianza = float(piso_varianza)

    def preparar_entrenamiento(self, datos):
        self.columnas = [c for c in datos.columns if c != self.objetivo]
        if not self.columnas:
            raise ValueError('No hay características EEG para entrenar la red.')
        X = datos[self.columnas].to_numpy(dtype=float)
        y = datos[self.objetivo].to_numpy(dtype=float)
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('El entrenamiento contiene valores no finitos.')
        if set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError('La clase debe ser binaria y ambas clases deben estar presentes.')
        self.media = X.mean(axis=0)
        self.escala = X.std(axis=0, ddof=0)
        if np.any(self.escala <= 0):
            constantes = [c for c, sd in zip(self.columnas, self.escala) if sd <= 0]
            raise ValueError(f'Características constantes en entrenamiento: {constantes}')
        self.Z = (X - self.media) / self.escala
        self.y = y.astype(int)
        # Heurística de estructura: GES/PC trabaja solo con EEG centrado
        # dentro de cada clase del entrenamiento. No es optimización exacta
        # del BIC de la red condicional gaussiana completa ni evidencia causal.
        centradas = self.Z.copy()
        for clase in (0, 1):
            mascara = self.y == clase
            if mascara.sum() < 3:
                raise ValueError(f'Menos de tres observaciones de la clase {clase}.')
            centradas[mascara] -= self.Z[mascara].mean(axis=0)
        if np.any(centradas.std(axis=0) <= 0):
            raise ValueError('Hay una característica sin variación dentro de las clases.')
        return pd.DataFrame(centradas, columns=self.columnas, index=datos.index)

    def ajustar_parametros(self, dag_eeg):
        faltantes = set(self.columnas) - set(dag_eeg.nodes())
        sobrantes = set(dag_eeg.nodes()) - set(self.columnas)
        if faltantes or sobrantes:
            raise ValueError(f'Nodos EEG incorrectos. Faltantes: {sorted(faltantes)}; sobrantes: {sorted(sobrantes)}.')
        if not nx.is_directed_acyclic_graph(dag_eeg):
            raise ValueError(f'La estructura EEG tiene un ciclo dirigido: {nx.find_cycle(dag_eeg)}.')
        self.dag = nx.DiGraph(dag_eeg.edges())
        self.dag.add_nodes_from(self.columnas + [self.objetivo])
        self.dag.add_edges_from((self.objetivo, c) for c in self.columnas)
        self.priores = np.asarray([(self.y == c).mean() for c in (0, 1)])
        indices = {c: j for j, c in enumerate(self.columnas)}
        self.cpds = []
        pisos = 0
        n_parametros = 1  # P(Y=1).
        for nodo in self.columnas:
            padres = sorted(dag_eeg.predecessors(nodo), key=indices.get)
            idx_padres = [indices[c] for c in padres]
            parametros = []
            for clase in (0, 1):
                Z = self.Z[self.y == clase]
                respuesta = Z[:, indices[nodo]]
                diseno = np.column_stack([np.ones(len(Z)), Z[:, idx_padres]])
                if len(Z) <= diseno.shape[1]:
                    raise ValueError(f'Datos insuficientes para {nodo}, clase {clase}.')
                # Penaliza las pendientes; el intercepto queda libre.
                penalizacion = np.diag([0.0] + [self.ridge] * len(padres))
                if self.ridge == 0:
                    coef, _, rango, _ = np.linalg.lstsq(diseno, respuesta, rcond=None)
                    if rango < diseno.shape[1]:
                        raise ValueError(f'Diseño singular en {nodo}, clase {clase}; usa --ridge positivo.')
                else:
                    coef = np.linalg.solve(diseno.T @ diseno + penalizacion, diseno.T @ respuesta)
                residuo = respuesta - diseno @ coef
                varianza_original = float(np.mean(residuo ** 2))
                varianza = max(varianza_original, self.piso_varianza)
                pisos += int(varianza_original < self.piso_varianza)
                if not np.isfinite(coef).all() or not np.isfinite(varianza):
                    raise ValueError(f'CPD no finita en {nodo}, clase {clase}.')
                parametros.append({'clase': clase, 'n': int(len(Z)),
                                   'intercepto': float(coef[0]), 'coeficientes': coef[1:].tolist(),
                                   'varianza': varianza, 'varianza_sin_piso': varianza_original})
                n_parametros += len(coef) + 1
            self.cpds.append({'nodo': nodo, 'padres_EEG': padres, 'parametros_por_clase': parametros})
        self.dag.graph['modelo_probabilistico'] = {
            'tipo': 'Condicional gaussiano lineal; clase discreta raíz',
            'factorizacion': 'P(Y) * producto_j P(X_j | Y, padres_EEG(X_j))',
            'columnas': self.columnas, 'objetivo': self.objetivo,
            'media_entrenamiento': self.media.tolist(), 'escala_entrenamiento': self.escala.tolist(),
            'priores_clase': self.priores.tolist(), 'ridge': self.ridge,
            'piso_varianza': self.piso_varianza, 'cpds': self.cpds,
            'n_parametros': n_parametros, 'n_varianzas_con_piso': pisos,
        }
        # No conservar los datos de entrenamiento en el modelo ajustado.
        del self.Z, self.y
        return self

    def predecir_probabilidades(self, datos):
        X = datos[self.columnas].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError('Los predictores de evaluación contienen valores no finitos.')
        Z = (X - self.media) / self.escala
        indices = {c: j for j, c in enumerate(self.columnas)}
        log_conjunta = np.tile(np.log(self.priores), (len(Z), 1))
        for cpd in self.cpds:
            padres = [indices[c] for c in cpd['padres_EEG']]
            for par in cpd['parametros_por_clase']:
                media = par['intercepto'] + Z[:, padres] @ np.asarray(par['coeficientes'])
                residuo = Z[:, indices[cpd['nodo']]] - media
                log_conjunta[:, par['clase']] += -0.5 * (
                    np.log(2 * np.pi * par['varianza']) + residuo ** 2 / par['varianza'])
        if not np.isfinite(log_conjunta).all():
            raise RuntimeError('Densidades no finitas: revisa las CPD y los predictores.')
        # Normalización en logaritmos: evita el subdesbordamiento del producto
        # de densidades. El jacobiano de estandarización es común a las clases
        # y se cancela en la probabilidad posterior.
        normalizador = np.logaddexp(log_conjunta[:, 0], log_conjunta[:, 1])
        prob = np.exp(log_conjunta[:, 1] - normalizador)
        if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
            raise RuntimeError('La inferencia bayesiana produjo probabilidades inválidas.')
        return prob


def ajustar_y_predecir(df_train, df_test, columna_objetivo, algoritmo, motor_ges,
                       max_padres, etiqueta, ridge=1.0, piso_varianza=1e-6):
    inicio = time.perf_counter()
    modelo = ClasificadorBayesianoCondicionalGaussiano(columna_objetivo, ridge, piso_varianza)
    eeg_centrado = modelo.preparar_entrenamiento(df_train)
    with mostrar_actividad(etiqueta):
        dag_eeg, detalles = aprender_estructura(eeg_centrado, algoritmo, motor_ges, max_padres)
    tiempo_estructura = time.perf_counter() - inicio
    inicio = time.perf_counter()
    modelo.ajustar_parametros(dag_eeg)
    tiempo_parametros = time.perf_counter() - inicio
    inicio = time.perf_counter()
    # Se accede exclusivamente a los predictores: las etiquetas externas no
    # participan en la estructura, los parámetros ni la inferencia.
    probabilidades = modelo.predecir_probabilidades(df_test)
    if len(probabilidades) != len(df_test):
        raise RuntimeError('La red no produjo todas las probabilidades requeridas.')
    detalles.update({'Tiempo_estructura_s': tiempo_estructura,
                     'Tiempo_parametros_s': tiempo_parametros,
                     'Tiempo_inferencia_s': time.perf_counter() - inicio,
                     'N_arcos': int(modelo.dag.number_of_edges()),
                     'N_arcos_EEG': int(dag_eeg.number_of_edges()),
                     'N_caracteristicas': len(modelo.columnas),
                     'Caracteristicas_modelo': json.dumps(modelo.columnas, ensure_ascii=False),
                     'Modelo_probabilidades': 'Inferencia condicional gaussiana por factorización de CPD',
                     'Aprendizaje_estructura': 'GES o PC-esqueleto con orientación predictiva; EEG centrado por clase, solo entrenamiento',
                     'Restriccion_clase': 'Ansiedad raíz; padre de todas las características EEG',
                     'Ridge': modelo.ridge, 'Piso_varianza': modelo.piso_varianza,
                     'Priores_clase': json.dumps(modelo.priores.tolist()),
                     'N_parametros': modelo.dag.graph['modelo_probabilistico']['n_parametros'],
                     'N_varianzas_con_piso': modelo.dag.graph['modelo_probabilistico']['n_varianzas_con_piso']})
    return probabilidades, modelo.dag, detalles


def guardar_csv(tabla, ruta):
    ruta = Path(ruta)
    temporal = ruta.with_suffix(ruta.suffix + '.tmp')
    tabla.to_csv(temporal, index=False)
    os.replace(temporal, ruta)


def serializar_grafo(dag, detalles):
    return {'nodos': list(dag.nodes()), 'arcos': list(dag.edges()), 'detalles': detalles,
            'modelo_probabilistico': dag.graph.get('modelo_probabilistico')}


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


def ejecutar_loso(df_datos, caracteristicas_seleccionadas, columna_objetivo, grupos, algoritmo,
                  n_top=10, motor_ges='causal-learn', max_padres=None,
                  ridge=1.0, piso_varianza=1e-6, evidencia=None, ruta_evidencia=None):
    candidatas = list(caracteristicas_seleccionadas)
    y_real = df_datos[columna_objetivo].to_numpy().astype(int)
    grupos = np.asarray(grupos)
    if len(grupos) != len(df_datos):
        raise ValueError('Los sujetos no están alineados con las épocas.')
    datos = df_datos[candidatas + [columna_objetivo]].apply(pd.to_numeric, errors='coerce').fillna(0)
    logo = LeaveOneGroupOut()
    total = logo.get_n_splits(datos, y_real, grupos)
    y_prob = np.full(len(datos), np.nan)
    tiempos = []
    motor = motor_ges if algoritmo == 'GES' else 'pgmpy'
    barra = tqdm(logo.split(datos, y_real, grupos), total=total,
                 desc=f'LOSO {algoritmo}/{motor} Top {n_top}', unit='fold')
    clave = f'LOSO/{algoritmo}_top{n_top}.json'
    detalle_loso = {'estado': 'EN_CURSO', 'pliegues': [], 'predicciones': []}
    if evidencia is not None:
        evidencia[clave] = detalle_loso
    for fold, (idx_train, idx_test) in enumerate(barra, start=1):
        sujeto = str(grupos[idx_test[0]])
        inicio_fold = time.perf_counter()
        registro = {'Algoritmo': algoritmo, 'Motor': motor, 'Top': n_top, 'Fold': fold,
                    'Sujeto_validacion': sujeto, 'N_entrenamiento': len(idx_train),
                    'N_validacion': len(idx_test), 'Estado': 'INICIO'}
        try:
            if set(grupos[idx_train]) & set(grupos[idx_test]):
                raise ValueError('Se detectó un sujeto compartido entre entrenamiento y validación.')
            inicio_dt = time.perf_counter()
            columnas = seleccionar_top_dt(datos.iloc[idx_train], candidatas, columna_objetivo, n_top)
            registro['Tiempo_DT_s'] = time.perf_counter() - inicio_dt
            registro['Caracteristicas_DT'] = json.dumps(columnas, ensure_ascii=False)
            train = datos.iloc[idx_train][columnas + [columna_objetivo]]
            test = datos.iloc[idx_test][columnas + [columna_objetivo]]
            etiqueta = f'{algoritmo}/{motor} Top {n_top}, fold {fold}/{total}, sujeto {sujeto}'
            probabilidades, dag, detalles = ajustar_y_predecir(
                train, test, columna_objetivo, algoritmo, motor_ges, max_padres, etiqueta, ridge, piso_varianza)
            y_prob[idx_test] = probabilidades
            registro.update(detalles, Estado='OK', Tiempo_fold_s=time.perf_counter() - inicio_fold)
            detalle_loso['pliegues'].append(serializar_grafo(dag, registro))
        except Exception as exc:
            registro.update(Estado='ERROR', Error=f'{type(exc).__name__}: {exc}',
                            Tiempo_fold_s=time.perf_counter() - inicio_fold)
            detalle_loso.update(estado='ERROR', error=registro)
            if evidencia is not None and ruta_evidencia is not None:
                evidencia['Configuracion.json']['Estado_ejecucion'] = 'ERROR'
                guardar_evidencia(evidencia, ruta_evidencia)
            raise RuntimeError(f'Falló {algoritmo}/{motor}, Top {n_top}, fold {fold}/{total}, sujeto {sujeto}.') from exc
        tiempos.append(registro)
        if evidencia is not None and ruta_evidencia is not None:
            listos = np.flatnonzero(np.isfinite(y_prob))
            tabla = pd.DataFrame({'Indice_fila': df_datos.index.to_numpy()[listos],
                                  'Sujeto': grupos[listos], 'Ansiedad_real': y_real[listos],
                                  'Probabilidad_ansiedad': y_prob[listos],
                                  'Prediccion': (y_prob[listos] >= 0.5).astype(int)})
            detalle_loso['predicciones'] = tabla.to_dict(orient='records')
            guardar_evidencia(evidencia, ruta_evidencia)
        barra.set_postfix(sujeto=sujeto, segundos=f'{registro["Tiempo_fold_s"]:.1f}')
    if not np.isfinite(y_prob).all():
        raise RuntimeError('LOSO quedó incompleto; no se calcularán métricas finales.')
    detalle_loso['estado'] = 'COMPLETO'
    return y_real, (y_prob >= 0.5).astype(int), y_prob, 0.5, pd.DataFrame(tiempos)


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


def principal(argv=None):
    parser = argparse.ArgumentParser(description='Clasificador bayesiano condicional gaussiano: EEG continuo y clase discreta, LOSO por sujeto.')
    parser.add_argument('--motor-ges', choices=['causal-learn', 'pgmpy'], default='causal-learn')
    parser.add_argument('--algoritmos', nargs='+', choices=['GES', 'PC'], default=['GES', 'PC'],
                        help='Por defecto evalúa GES y PC. Usa --algoritmos GES o --algoritmos PC para ejecutar solo uno.')
    parser.add_argument('--tops', nargs='+', type=int, choices=[10, 15, 20], default=[10, 15, 20])
    parser.add_argument('--max-padres', type=int, default=None,
                        help='Restricción experimental opcional de GES causal-learn; por defecto no se restringe.')
    parser.add_argument('--ridge', type=float, default=1.0,
                        help='Regularización de pendientes de las CPD gaussianas, sin penalizar interceptos.')
    parser.add_argument('--piso-varianza', type=float, default=1e-6,
                        help='Varianza residual mínima en unidades de EEG estandarizado.')
    args = parser.parse_args(argv)
    if not np.isfinite(args.ridge) or args.ridge < 0 or not np.isfinite(args.piso_varianza) or args.piso_varianza <= 0:
        parser.error('--ridge debe ser no negativo y --piso-varianza positivo; ambos finitos.')
    if args.max_padres is not None and (args.max_padres < 1 or args.motor_ges != 'causal-learn' or 'GES' not in args.algoritmos):
        parser.error('--max-padres debe ser positivo y solo puede usarse con GES causal-learn.')
    if 'GES' in args.algoritmos and args.motor_ges == 'causal-learn':
        cargar_ges_causal_learn()
    raiz = Path(__file__).resolve().parents[3]
    salida = raiz / 'Resultados' / 'Red Bayesiana' / 'Global' / 'Condicional Gaussiana'
    salida.mkdir(parents=True, exist_ok=True)
    archivo_datos = raiz / 'Resultados' / 'Exploratorio' / 'datos_completos_normalizados.parquet'
    archivo_ranking = raiz / 'Resultados' / 'Análisis global' / 'Selección de características' / 'Ranking_Multicriterio_Completo.csv'
    if not archivo_datos.exists() or not archivo_ranking.exists():
        raise FileNotFoundError(f'Faltan datos o ranking: {archivo_datos}; {archivo_ranking}')
    datos = pd.read_parquet(archivo_datos)
    datos = datos[(datos['Puntaje'] == 0) | (datos['Puntaje'] >= 1)].copy()
    datos['Ansiedad'] = (datos['Puntaje'] >= 1).astype(float)
    sujetos = np.sort(datos['Sujeto'].dropna().unique())
    if len(sujetos) != 40 or datos['Sujeto'].isna().any():
        raise ValueError(f'Se requieren los 40 sujetos identificados; se encontraron {len(sujetos)}.')
    mezclados = np.random.RandomState(42).permutation(sujetos)
    desarrollo = datos[datos['Sujeto'].isin(mezclados[:32])].copy()
    externa = datos[datos['Sujeto'].isin(mezclados[32:])].copy()
    candidatas = sorted(pd.read_csv(archivo_ranking)['Caracteristica'].dropna().tolist())
    if len(candidatas) != 306 or len(set(candidatas)) != 306:
        raise ValueError('El ranking debe enumerar 306 características distintas.')
    faltantes = set(candidatas) - set(datos.columns)
    if faltantes:
        raise ValueError(f'Faltan variables de EEG en los datos: {sorted(faltantes)}')
    configuracion = vars(args).copy()
    configuracion.update({'Criterio': 'SAM=0 control; SAM>=1 caso', 'Semilla': 42,
                          'Sujetos_desarrollo_32': list(map(str, mezclados[:32])),
                          'Sujetos_externos_8': list(map(str, mezclados[32:])),
                          'Version_pgmpy': obtener_version('pgmpy'),
                          'Version_causal_learn': obtener_version('causal-learn'),
                          'Clasificador': 'Inferencia bayesiana con CPD gaussianas específicas por clase',
                          'Script': Path(__file__).name,
                          'Inicio_UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          'Estado_ejecucion': 'EN_CURSO',
                          'Preprocesamiento': 'Conversión numérica; faltantes a cero como en la prueba previa; estandarización solo con entrenamiento',
                          'Umbral': 0.5, 'Prior': 'Prevalencia empírica del entrenamiento sin balance artificial',
                          'Nota_metodologica': 'Clase binaria raíz. Estructura EEG compartida aprendida sobre variables centradas por clase. BIC gaussiano de esa representación es una heurística, no BIC exacto del modelo híbrido. PC usa su esqueleto y orientación predictiva por orden y BIC, no una extensión causal de su PDAG. Parámetros gaussianos propios de cada clase. Las flechas no identifican causalidad.'})
    evidencia = {'Configuracion.json': configuracion}
    ruta_evidencia = salida / 'Evidencia'
    guardar_evidencia(evidencia, ruta_evidencia)
    print(f'Resultados: {salida}', flush=True)
    print('Red condicional gaussiana: clase discreta y predicción por inferencia bayesiana. Umbral fijo 0.5.', flush=True)
    print('Estructura EEG compartida: búsqueda sobre entrenamiento centrado por clase; sin afirmaciones causales.', flush=True)
    inicio_total = time.perf_counter()
    resultados = []
    ruta_resultados = salida / 'Resultados_Red_Bayesiana_Condicional_Gaussiana.csv'
    # El resumen corresponde exclusivamente a esta ejecución, incluso si falla.
    guardar_csv(pd.DataFrame(columns=['Estructura', 'Top', 'Estado']), ruta_resultados)
    for n_top in dict.fromkeys(args.tops):
        for algoritmo in dict.fromkeys(args.algoritmos):
            inicio_config = time.perf_counter()
            y_val, pred_val, prob_val, umbral, tiempos = ejecutar_loso(
                desarrollo, candidatas, 'Ansiedad', desarrollo['Sujeto'].to_numpy(), algoritmo,
                n_top=n_top, motor_ges=args.motor_ges, max_padres=args.max_padres,
                ridge=args.ridge, piso_varianza=args.piso_varianza,
                evidencia=evidencia, ruta_evidencia=ruta_evidencia)
            columnas = seleccionar_top_dt(desarrollo, candidatas, 'Ansiedad', n_top)
            train = desarrollo[columnas + ['Ansiedad']].apply(pd.to_numeric, errors='coerce').fillna(0)
            test = externa[columnas + ['Ansiedad']].apply(pd.to_numeric, errors='coerce').fillna(0)
            prob_test, dag, meta = ajustar_y_predecir(
                train, test, 'Ansiedad', algoritmo, args.motor_ges, args.max_padres,
                f'{algoritmo} Top {n_top}, ajuste final con 32 sujetos', args.ridge, args.piso_varianza)
            evidencia[f'Modelos_Finales/{algoritmo}_top{n_top}.json'] = serializar_grafo(dag, meta)
            y_test, pred_test = test['Ansiedad'].to_numpy().astype(int), (prob_test >= umbral).astype(int)
            fila = {'Estructura': algoritmo, 'Motor': meta['Motor'], 'Puntuacion': meta['Puntuacion'],
                    'Top': n_top, 'Umbral': umbral, 'Estado': 'COMPLETO',
                    'Modelo_probabilidades': meta['Modelo_probabilidades'],
                    'Aprendizaje_estructura': meta['Aprendizaje_estructura'],
                    'Orientacion_EEG': meta.get('Orientacion', 'DAG del estimador'),
                    'Ciclo_PC_detectado': meta.get('Ciclo_PC_detectado', False),
                    'Ridge': args.ridge, 'Piso_varianza': args.piso_varianza,
                    'N_arcos_EEG_final': meta['N_arcos_EEG'],
                    'N_parametros_final': meta['N_parametros'],
                    'N_varianzas_con_piso_final': meta['N_varianzas_con_piso'],
                    'Caracteristicas_DT_final': json.dumps(columnas, ensure_ascii=False),
                    'Interpretacion_grafo': 'Clase raíz impuesta para clasificación; arcos EEG exploratorios. No identifica causalidad.',
                    'Tiempo_LOSO_s': float(tiempos['Tiempo_fold_s'].sum()),
                    'Tiempo_estructura_LOSO_s': float(tiempos['Tiempo_estructura_s'].sum()),
                    'Tiempo_estructura_final_s': meta['Tiempo_estructura_s'],
                    'N_arcos_final': meta['N_arcos'], 'N_caracteristicas_final': meta['N_caracteristicas']}
            fila.update({f'{k} (Val)': v for k, v in metricas_clasificacion(y_val, pred_val, prob_val).items()})
            fila.update({f'{k} (Test)': v for k, v in metricas_clasificacion(y_test, pred_test, prob_test).items()})
            fila.update(bootstrap_ci_por_sujeto(y_val, pred_val, desarrollo['Sujeto'].to_numpy(), 'Val'))
            fila.update(metricas_agregadas_por_sujeto(y_val, pred_val, desarrollo['Sujeto'].to_numpy(), 'Val'))
            fila.update(bootstrap_ci_por_sujeto(y_test, pred_test, externa['Sujeto'].to_numpy(), 'Test'))
            fila.update(metricas_agregadas_por_sujeto(y_test, pred_test, externa['Sujeto'].to_numpy(), 'Test'))
            fila['Tiempo_configuracion_s'] = time.perf_counter() - inicio_config
            resultados.append(fila)
            guardar_csv(pd.DataFrame(resultados), ruta_resultados)
            evidencia[f'Predicciones_Externas/{algoritmo}_top{n_top}.json'] = pd.DataFrame({
                'Indice_fila': externa.index, 'Sujeto': externa['Sujeto'].to_numpy(),
                'Ansiedad_real': y_test, 'Probabilidad_ansiedad': prob_test,
                'Prediccion': pred_test}).to_dict(orient='records')
            guardar_evidencia(evidencia, ruta_evidencia)
            graficar_dag(dag, f'Red condicional gaussiana: {algoritmo}, {meta["Motor"]}, Top {n_top}',
                         salida / f'Estructura_Global_{algoritmo}_top{n_top}.png')
            print(f'COMPLETO {algoritmo} Top {n_top}: {fila["Tiempo_configuracion_s"]:.1f} s', flush=True)
    configuracion['Estado_ejecucion'] = 'COMPLETO'
    configuracion['Tiempo_total_s'] = time.perf_counter() - inicio_total
    guardar_evidencia(evidencia, ruta_evidencia)
    print(f'Tiempo total de la corrida: {configuracion["Tiempo_total_s"]:.1f} s', flush=True)
    print(f'Resumen: {ruta_resultados}', flush=True)
    print(f'Evidencia para figuras, análisis por sujeto y estabilidad: {ruta_evidencia}', flush=True)
    print(pd.DataFrame(resultados).to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    principal()
