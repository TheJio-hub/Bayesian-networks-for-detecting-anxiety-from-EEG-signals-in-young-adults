"""Red bayesiana condicional gaussiana regional (SAM >= 1).

25 predictores fijos: media de canales por región y banda, sobre valores
estandarizados respecto a relajación. Imputación de canales exclusivamente
con entrenamiento, antes de promediar. Sin selección DT ni discretización.
GES de pgmpy optimiza BIC condicional gaussiano con parámetros propios de cada clase.
PC de causal-learn obtiene esqueletos EEG por clase, los une y orienta de forma predictiva
con el mismo BIC; esta variante no es PC mixto ni una extensión causal.
CPD gaussianas lineales por clase y prior empírico producen el posterior.
LOSO 31/1 sobre 32 sujetos; ajuste final con 32 y evaluación de ocho externos.
"""

import argparse
import json
import os
import threading
import time
from uuid import uuid4
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.causal_discovery import GES
from pgmpy.structure_score import BaseStructureScore
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from tqdm.auto import tqdm as _tqdm

VERSION_MODELO = 'regional_condicional_gaussiana_v2_pc_causallearn'
# Agrupación operacional de electrodos del cuero cabelludo, no fuentes corticales.
# Se conserva el mapa original para que su justificación pueda revisarse en la tesis.
REGIONES = {
    'Frontal': ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8'],
    'Central': ['Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'C3', 'C4'],
    'Temporal': ['FT9', 'FT10', 'T7', 'T8', 'CP5', 'CP6'],
    'Parietal': ['CP1', 'CP2', 'P3', 'P4', 'P7', 'P8', 'Pz'],
    'Occipital': ['O1', 'O2', 'Oz', 'PO9', 'PO10'],
}
BANDAS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
PREDICTORES = [f'Reg_{region}_{banda}' for region in REGIONES for banda in BANDAS]
COLUMNAS_ORIGEN = [f'{canal}_{banda}' for canales in REGIONES.values() for canal in canales for banda in BANDAS]


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)


def graficar_dag(dag, titulo, nombre_archivo):
    """Diagrama con cinco columnas regionales; todos los nodos son visibles."""
    validar_dag(dag, PREDICTORES + ['Ansiedad'])
    fig, ax = plt.subplots(figsize=(16, 11))
    posiciones = {'Ansiedad': (2.0, 6.0)}
    etiquetas = {'Ansiedad': 'Ansiedad'}
    colores = ['#dceaf7', '#dff0df', '#ffe6cc', '#e9dff4', '#f8dce5']
    for i, region in enumerate(REGIONES):
        ax.text(i, 5.15, region, ha='center', fontsize=12, fontweight='bold')
        for j, banda in enumerate(BANDAS):
            nombre = f'Reg_{region}_{banda}'
            posiciones[nombre] = (float(i), float(4-j))
            etiquetas[nombre] = banda
        nx.draw_networkx_nodes(dag, posiciones,
                               nodelist=[f'Reg_{region}_{b}' for b in BANDAS],
                               node_color=colores[i], edgecolors='#46566b',
                               node_shape='s', node_size=2600, ax=ax)
    nx.draw_networkx_nodes(dag, posiciones, nodelist=['Ansiedad'], node_size=3000,
                           node_color='#c5e6c9', edgecolors='#306137', ax=ax)
    nx.draw_networkx_labels(dag, posiciones, labels=etiquetas, font_size=10, ax=ax)
    nx.draw_networkx_edges(dag, posiciones,
                           edgelist=[(a,b) for a,b in dag.edges() if a=='Ansiedad'],
                           edge_color='#777777', style='dashed', width=.8, alpha=.25,
                           node_size=2800, arrowsize=12, ax=ax)
    nx.draw_networkx_edges(dag, posiciones,
                           edgelist=[(a,b) for a,b in dag.edges() if a!='Ansiedad'],
                           edge_color='#244a70', width=1.3, alpha=.65,
                           connectionstyle='arc3,rad=0.13', node_size=2800, arrowsize=16, ax=ax)
    ax.set_title(titulo, fontsize=14, pad=18)
    fig.text(.5, .035, 'Línea discontinua: clase → predictor (impuesta). Línea continua: dependencia EEG aprendida.\n'
             'Cada nodo representa la media de canales de una región y banda sobre valores estandarizados. Flechas sin interpretación causal.',
             ha='center', fontsize=10)
    ax.set_xlim(-.55, 4.55); ax.set_ylim(-.6, 6.6)
    ax.axis('off')
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)


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


def tabla_predicciones(datos, prob, ejecucion_id=None):
    prob = np.asarray(prob, dtype=float)
    if (prob.ndim != 1 or len(datos) != len(prob) or not np.isfinite(prob).all()
            or np.any((prob < 0) | (prob > 1))):
        raise ValueError('No hay una probabilidad finita por época.')
    tabla = pd.DataFrame({'Indice_fila': datos.index.to_numpy(), 'Sujeto': datos['Sujeto'].to_numpy(),
                          'Ansiedad_real': datos['Ansiedad'].to_numpy(dtype=int),
                          'Probabilidad_ansiedad': prob, 'Prediccion': (prob >= 0.5).astype(int)})
    if ejecucion_id is not None:
        tabla.insert(0, 'ID_ejecucion', ejecucion_id)
    return tabla



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
        if not np.isfinite(self.media).all() or not np.isfinite(self.escala).all():
            raise ValueError('Media o escala de entrenamiento no finita.')
        if np.any(self.escala <= 0):
            constantes = [c for c, sd in zip(self.columnas, self.escala) if sd <= 0]
            raise ValueError(f'Características constantes en entrenamiento: {constantes}')
        self.Z = (X - self.media) / self.escala
        self.y = y.astype(int)
        for clase in (0, 1):
            if np.count_nonzero(self.y == clase) <= len(self.columnas) + 2:
                raise ValueError(f'Datos insuficientes en la clase {clase} para 25 predictores.')
        return pd.DataFrame(self.Z, columns=self.columnas, index=datos.index)

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
        log_razon = np.full(len(Z), np.log(self.priores[1]) - np.log(self.priores[0]))
        for cpd in self.cpds:
            padres = [indices[c] for c in cpd['padres_EEG']]
            densidades = []
            for par in cpd['parametros_por_clase']:
                media = par['intercepto'] + Z[:, padres] @ np.asarray(par['coeficientes'])
                residuo = Z[:, indices[cpd['nodo']]] - media
                densidades.append(-0.5*(np.log(2*np.pi*par['varianza']) + residuo**2/par['varianza']))
            log_razon += densidades[1] - densidades[0]
        if not np.isfinite(log_razon).all():
            raise RuntimeError('Log-razones gaussianas no finitas.')
        prob = np.exp(-np.logaddexp(0.0, -log_razon))
        if (prob.ndim != 1 or len(prob) != len(datos) or not np.isfinite(prob).all()
                or np.any((prob < 0) | (prob > 1))):
            raise RuntimeError('La inferencia bayesiana produjo probabilidades inválidas o incompletas.')
        return prob



class BICCondicionalRegional(BaseStructureScore):
    """BIC de P(EEG|Y), con intercepto, pendientes y varianza por clase.

    P(Y) es constante entre estructuras y se omite solo en la búsqueda.
    La clase se pasa por separado y nunca se infiere su tipo desde un entero.
    Estadísticos suficientes por clase y caché evitan reajustar sobre filas
    para cada estructura. El BIC usa MLE sin ridge; las CPD predictivas usan
    la regularización declarada. El tamaño muestral BIC cuenta épocas;
    sus diferencias no se interpretan como evidencia causal entre sujetos.
    """
    def __init__(self, data, etiquetas):
        super().__init__(data)
        self.etiquetas = etiquetas
        self.n = len(data)
        self.indices = {c: i for i, c in enumerate(data.columns)}
        etiquetas = np.asarray(etiquetas)
        if set(np.unique(etiquetas)) != {0, 1} or len(etiquetas) != self.n:
            raise ValueError('BIC condicional requiere etiquetas binarias alineadas.')
        self.estadisticos = []
        X = data.to_numpy(dtype=float)
        for clase in (0, 1):
            a = X[etiquetas == clase]
            centrada = a - a.mean(axis=0)
            cov = centrada.T @ centrada / len(a)
            if not np.isfinite(cov).all() or np.any(np.diag(cov) <= 0):
                raise ValueError(f'Covarianza inválida o variable constante en clase {clase}.')
            self.estadisticos.append((len(a), cov))

    def _local_score(self, variable, parents):
        j = self.indices[variable]
        padres = [self.indices[p] for p in sorted(parents)]
        ll = 0.0
        for n, cov in self.estadisticos:
            varianza = float(cov[j, j])
            if padres:
                cruzada = cov[np.ix_(padres, [j])].ravel()
                matriz = cov[np.ix_(padres, padres)]
                try:
                    coeficientes = np.linalg.solve(matriz, cruzada)
                except np.linalg.LinAlgError as exc:
                    raise ValueError(f'Diseño singular al puntuar {variable} con {parents}.') from exc
                varianza -= float(cruzada @ coeficientes)
            if not np.isfinite(varianza) or varianza <= np.finfo(float).eps * cov[j, j]:
                raise ValueError(f'Varianza residual no positiva al puntuar {variable} con {parents}.')
            ll -= 0.5*n*(np.log(2*np.pi) + 1 + np.log(varianza))
        k = 2*(len(padres) + 2)
        valor = ll - 0.5*k*np.log(self.n)
        if not np.isfinite(valor):
            raise ValueError(f'BIC condicional no finito para {variable}.')
        return float(valor)


def orientar_esqueleto_pc(esqueleto, datos, score):
    """Orienta la unión de esqueletos por orden total y BIC condicional.

    Inicio en el orden regional fijo. Intercambios adyacentes de mejora
    estricta mantienen un DAG y todas las conexiones de la unión.
    Es una orientación predictiva, sin afirmaciones causales.
    """
    nombres = list(datos.columns)
    esqueleto = nx.Graph(esqueleto)
    if set(esqueleto.nodes()) != set(nombres) or nx.number_of_selfloops(esqueleto):
        raise ValueError('El esqueleto PC no coincide con las características EEG.')
    orden = nombres.copy()
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
        'Orientacion': 'PC-esqueleto + orientación predictiva por orden y BIC condicional gaussiano; sin interpretación causal',
        'Orden_orientacion': orden,
        'BIC_orientacion_inicial': inicio,
        'BIC_orientacion_final': sum(local(nodo, padres[nodo]) for nodo in nombres),
        'Intercambios_orientacion': cambios,
        'Pasadas_orientacion': pasada + 1,
        'Orientacion_convergio': convergio,
    }


def validar_modelo(modelo):
    dag = modelo.dag
    validar_dag(dag, PREDICTORES + ['Ansiedad'])
    if dag.in_degree('Ansiedad') != 0 or set(dag.successors('Ansiedad')) != set(PREDICTORES):
        raise ValueError('La clase debe ser raíz y padre de los 25 predictores.')
    cpds = {c['nodo']: c for c in modelo.cpds}
    if len(cpds) != 25 or set(cpds) != set(PREDICTORES):
        raise ValueError('Faltan CPD regionales o hay nodos duplicados.')
    if not np.isfinite(modelo.priores).all() or np.any(modelo.priores <= 0) or not np.isclose(modelo.priores.sum(), 1):
        raise ValueError('Prior de clase inválido.')
    for nodo, cpd in cpds.items():
        padres = set(dag.predecessors(nodo)) - {'Ansiedad'}
        if padres != set(cpd['padres_EEG']):
            raise ValueError(f'CPD y estructura no coinciden en {nodo}.')
        if [p['clase'] for p in cpd['parametros_por_clase']] != [0, 1]:
            raise ValueError(f'CPD sin ambas clases en {nodo}.')
        for p in cpd['parametros_por_clase']:
            valores = [p['intercepto'], p['varianza']] + p['coeficientes']
            if len(p['coeficientes']) != len(padres) or p['varianza'] <= 0 or not np.isfinite(valores).all():
                raise ValueError(f'Parámetros inválidos en {nodo}.')


def cargar_pc_causal_learn():
    """Carga explícita sin sustituir motores ni cambiar el límite PC."""
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError as exc:
        raise RuntimeError(
            'PC regional requiere causal-learn en el entorno activo. '
            'En bn_anxiety_env instala: python -m pip install causal-learn==0.1.4.8. '
            'No se sustituirá automáticamente por PC de pgmpy.'
        ) from exc
    from inspect import signature
    if 'max_k' not in signature(pc).parameters:
        raise RuntimeError(
            f'causal-learn {obtener_version("causal-learn")} no expone max_k en PC. '
            'Se requiere una versión compatible para conservar --pc-max-cond-vars; '
            'la API instalada revisada es 0.1.4.8.'
        )
    return pc


def esqueleto_causal_learn(grafo, nombres):
    """Extrae solo adyacencias; la orientación predictiva se realiza después."""
    nombres = list(nombres)
    matriz = np.asarray(grafo.G.graph)
    nodos = [n.get_name() for n in grafo.G.get_nodes()]
    if nodos != nombres or matriz.shape != (len(nombres), len(nombres)):
        raise ValueError('Los nodos o su orden en PC/causal-learn no coinciden con los predictores.')
    if (not np.isfinite(matriz).all() or np.any(np.diag(matriz) != 0)
            or not np.isin(matriz, [-1, 0, 1]).all()):
        raise ValueError('La matriz PC contiene extremos o diagonal inválidos.')
    presencia = matriz != 0
    if not np.array_equal(presencia, presencia.T):
        raise ValueError('La matriz PC contiene una conexión con un solo extremo.')
    esqueleto = nx.Graph()
    esqueleto.add_nodes_from(nombres)
    esqueleto.add_edges_from((nombres[i], nombres[j]) for i, j in combinations(range(len(nombres)), 2)
                             if presencia[i, j])
    return esqueleto, matriz.tolist()


def aprender_estructura(X, y, algoritmo, args):
    score = BICCondicionalRegional(X, y)
    if algoritmo == 'GES':
        estimador = GES(scoring_method=score, return_type='pdag')
        estimador.fit(X)
        pdag = estimador.causal_graph_
        dag = pdag.to_dag()
        validar_dag(dag, X.columns)
        esqueleto = {frozenset(e) for e in pdag.edges()}
        if {frozenset(e) for e in dag.edges()} != esqueleto:
            raise ValueError('La conversión GES cambió el esqueleto.')
        if not set(pdag.directed_edges).issubset(set(dag.edges())):
            raise ValueError('La conversión GES invirtió una arista dirigida.')
        def colliders(arcos):
            g = nx.DiGraph(); g.add_nodes_from(X.columns); g.add_edges_from(arcos)
            return {(c, frozenset((a, b))) for c in g.nodes()
                    for a, b in combinations(g.predecessors(c), 2)
                    if frozenset((a, b)) not in esqueleto}
        if colliders(pdag.directed_edges) != colliders(dag.edges()):
            raise ValueError('La extensión del PDAG GES alteró estructuras en V.')
        meta = {'Motor': 'pgmpy',
                'Motor_orientacion': 'pgmpy, extensión consistente del PDAG',
                'Orientacion': 'Extensión consistente de PDAG GES; sin interpretación causal',
                'Procedimiento': 'GES con BIC condicional gaussiano compartido entre clases',
                'PDAG_original': {'Dirigidas': sorted(pdag.directed_edges),
                                  'No_dirigidas': sorted(pdag.undirected_edges)}}
    elif algoritmo == 'PC':
        pc_causal_learn = cargar_pc_causal_learn()
        union = nx.Graph(); union.add_nodes_from(X.columns)
        originales = []
        for clase in (0, 1):
            parte = X.loc[np.asarray(y) == clase]
            # La clase no entra en Fisher-Z. Dos búsquedas estratificadas;
            # alpha/2 por clase, sin afirmar corrección global de todos los tests PC.
            inicio_pc = time.perf_counter()
            grafo_pc = pc_causal_learn(
                parte.to_numpy(dtype=float), alpha=args.pc_alpha/2,
                indep_test='fisherz', stable=True, show_progress=False, verbose=False,
                node_names=list(X.columns), max_k=args.pc_max_cond_vars)
            esqueleto, matriz = esqueleto_causal_learn(grafo_pc, X.columns)
            union.add_edges_from(esqueleto.edges())
            originales.append({'Clase': clase, 'N_epocas': len(parte),
                               'Motor': 'causal-learn', 'Arcos_esqueleto': sorted(esqueleto.edges()),
                               'Orden_nodos': list(X.columns), 'Matriz_extremos_PC_original': matriz,
                               'Alpha_por_clase': args.pc_alpha/2,
                               'Max_condicionantes': args.pc_max_cond_vars,
                               'Tiempo_PC_clase_s': time.perf_counter()-inicio_pc})
        dag, meta = orientar_esqueleto_pc(union, X, score)
        meta.update({'Motor': 'causal-learn',
                     'Motor_orientacion': 'Orden total y BIC condicional NumPy; orientación predictiva',
                     'Version_causal_learn': obtener_version('causal-learn'),
                     'Procedimiento': 'PC causal-learn por clase; unión de esqueletos; orientación predictiva BIC condicional',
                     'Esqueletos_por_clase': originales,
                     'Max_condicionantes_PC': args.pc_max_cond_vars,
                     'Nota_PC': 'Fisher-Z asume relaciones lineales gaussianas dentro de clase; épocas dependientes. '
                                'La unión es conservadora en conexiones; no es PC mixto ni una extensión causal.'})
    else:
        raise ValueError(f'Algoritmo desconocido: {algoritmo}.')
    validar_dag(dag, X.columns)
    bic = sum(score.local_score(n, tuple(sorted(dag.predecessors(n)))) for n in X.columns)
    meta.update({'Implementacion_BIC': 'BIC condicional NumPy con estadísticos por clase y caché',
                 'BIC_condicional_EEG': float(bic),
                 'Criterio': 'Suma log-verosimilitud gaussiana MLE por clase - k*log(N_epocas)/2',
                 'Tipos': {'Ansiedad': 'discreta binaria explícita; fuera de búsquedas EEG',
                           'EEG': '25 predictores continuos'},
                 'N_arcos_EEG': dag.number_of_edges()})
    return dag, meta


def preparar_regionales(train, test):
    """Imputa 160 entradas con entrenamiento y promedia canales por banda."""
    if len(COLUMNAS_ORIGEN) != 160 or len(set(COLUMNAS_ORIGEN)) != 160 or len(PREDICTORES) != 25:
        raise ValueError('El mapa regional no define exactamente 32 canales y cinco bandas.')
    a = train[COLUMNAS_ORIGEN].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    b = test[COLUMNAS_ORIGEN].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    medianas = a.median()
    if medianas.isna().any():
        raise ValueError(f'Canales/bandas sin valores finitos en entrenamiento: {medianas[medianas.isna()].index.tolist()}')
    meta = {'Medianas_canales_entrenamiento': medianas.to_dict(),
            'Valores_imputados_entrenamiento': int(a.isna().sum().sum()),
            'Valores_imputados_evaluacion': int(b.isna().sum().sum()),
            'Imputacion': 'Medianas de entrenamiento por canal/banda, antes de promediar; infinitos como faltantes',
            'Definicion_regional': 'Media aritmética de valores estandarizados respecto a relajación por sujeto, por banda',
            'Caracteristicas_fijas': PREDICTORES, 'Seleccion_DT': False}
    a, b = a.fillna(medianas), b.fillna(medianas)
    def agregar(tabla):
        columnas = {}
        for region, canales in REGIONES.items():
            for banda in BANDAS:
                columnas[f'Reg_{region}_{banda}'] = tabla[[f'{c}_{banda}' for c in canales]].mean(axis=1)
        return pd.DataFrame(columnas, index=tabla.index)
    X, Z = agregar(a), agregar(b)
    if not X.index.equals(train.index) or not Z.index.equals(test.index):
        raise ValueError('La agregación regional alteró el orden de filas.')
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(Z.to_numpy()).all():
        raise ValueError('La agregación regional produjo valores no finitos.')
    X['Ansiedad'] = train['Ansiedad'].to_numpy(dtype=int)
    return X, Z, meta


def ajustar_y_predecir(train, test, algoritmo, args, contexto):
    inicio = time.perf_counter()
    a, b, prepro = preparar_regionales(train, test)
    modelo = ClasificadorBayesianoCondicionalGaussiano('Ansiedad', args.ridge, args.piso_varianza)
    X = modelo.preparar_entrenamiento(a)
    meta = {'Tiempo_preprocesamiento_s': time.perf_counter() - inicio}
    inicio = time.perf_counter()
    with mostrar_actividad(contexto):
        dag, detalles = aprender_estructura(X, a['Ansiedad'].to_numpy(), algoritmo, args)
    meta.update(detalles, Tiempo_estructura_s=time.perf_counter() - inicio)
    inicio = time.perf_counter()
    modelo.ajustar_parametros(dag)
    validar_modelo(modelo)
    meta['Tiempo_parametros_s'] = time.perf_counter() - inicio
    inicio = time.perf_counter()
    prob = modelo.predecir_probabilidades(b)
    if len(prob) != len(test) or not b.index.equals(test.index):
        raise ValueError('Las predicciones no corresponden a las épocas esperadas.')
    meta.update(ID_ejecucion=args.ejecucion_id, Tiempo_inferencia_s=time.perf_counter() - inicio,
                N_arcos=modelo.dag.number_of_edges(), N_arcos_clase_impuestos=25,
                Clase_raiz=True, Version_modelo=VERSION_MODELO, Umbral_fijo=0.5,
                N_parametros=modelo.dag.graph['modelo_probabilistico']['n_parametros'],
                N_varianzas_con_piso=modelo.dag.graph['modelo_probabilistico']['n_varianzas_con_piso'],
                Max_padres_EEG=max(dict(dag.in_degree()).values()))
    evidencia = {'Preprocesamiento': prepro, 'Detalles': meta,
                 'Modelo': {'Nodos': list(modelo.dag.nodes()), 'Arcos': list(modelo.dag.edges()),
                            **modelo.dag.graph['modelo_probabilistico']}}
    return prob, modelo.dag, meta, evidencia


def cargar_datos(raiz):
    archivo = raiz/'Resultados'/'Exploratorio'/'datos_completos_normalizados.parquet'
    if not archivo.exists():
        raise FileNotFoundError(f'Falta el archivo de datos: {archivo}')
    datos = pd.read_parquet(archivo)
    if not datos.columns.is_unique or not datos.index.is_unique:
        raise ValueError('Hay nombres de columnas o índices de épocas duplicados.')
    faltantes = set(COLUMNAS_ORIGEN + ['Sujeto', 'Puntaje']) - set(datos.columns)
    if faltantes:
        raise ValueError(f'Faltan columnas para construir regiones: {sorted(faltantes)}')
    if datos['Sujeto'].isna().any():
        raise ValueError('Hay sujetos sin identificar.')
    sam = pd.to_numeric(datos['Puntaje'], errors='coerce')
    if not np.isfinite(sam.to_numpy()).all():
        raise ValueError('Puntaje contiene valores faltantes o no finitos.')
    conservar = (sam == 0) | (sam >= 1)
    excluidas = int((~conservar).sum())
    datos = datos.loc[conservar, COLUMNAS_ORIGEN + ['Sujeto', 'Puntaje']].copy()
    datos['Ansiedad'] = (sam.loc[conservar] >= 1).astype(int)
    sujetos = np.sort(datos['Sujeto'].unique())
    if len(sujetos) != 40:
        raise ValueError(f'Se requieren 40 sujetos, se encontraron {len(sujetos)}.')
    mezclados = np.random.RandomState(42).permutation(sujetos)
    desarrollo = datos[datos.Sujeto.isin(mezclados[:32])].copy()
    externa = datos[datos.Sujeto.isin(mezclados[32:])].copy()
    if (len(set(desarrollo.Sujeto)) != 32 or len(set(externa.Sujeto)) != 8
            or set(desarrollo.Sujeto) & set(externa.Sujeto)):
        raise ValueError('La partición no contiene 32/8 sujetos disjuntos.')
    for nombre, parte in [('desarrollo', desarrollo), ('externa', externa)]:
        if set(parte.Ansiedad.unique()) != {0, 1}:
            raise ValueError(f'Falta una clase en {nombre}.')
    return desarrollo, externa, mezclados, excluidas


def resumen_estabilidad(arcos_por_fold, arcos_finales):
    """Frecuencia empírica de adyacencia y dirección, sin umbral de consenso."""
    n = len(arcos_por_fold)
    if n == 0:
        return {'N_pliegues': 0, 'Conexiones_EEG': []}
    finales = set(map(tuple, arcos_finales))
    registros = []
    for a, b in combinations(PREDICTORES, 2):
        ab = sum((a, b) in arcos for arcos in arcos_por_fold)
        ba = sum((b, a) in arcos for arcos in arcos_por_fold)
        registros.append({'Nodo_A': a, 'Nodo_B': b, 'Pliegues_A_a_B': ab,
                          'Pliegues_B_a_A': ba, 'Frecuencia_adyacencia': (ab+ba)/n,
                          'Frecuencia_A_a_B': ab/n, 'Frecuencia_B_a_A': ba/n,
                          'Direccion_A_a_B_dada_adyacencia': ab/(ab+ba) if ab+ba else None,
                          'Arco_final_A_a_B': (a, b) in finales, 'Arco_final_B_a_A': (b, a) in finales})
    return {'N_pliegues': n, 'Conexiones_EEG': registros,
            'Nota': 'Frecuencias descriptivas entre entrenamientos LOSO solapados; no son probabilidades causales ni intervalos independientes. '
                    'Las 25 aristas de clase son impuestas y se excluyen de la estabilidad aprendida.'}


def ejecutar_loso(datos, algoritmo, args, evidencia_dir):
    prob = np.full(len(datos), np.nan)
    grupos = datos.Sujeto.to_numpy()
    tiempos, arcos = [], []
    estado = {'ID_ejecucion': args.ejecucion_id, 'Estado': 'EN_CURSO', 'Algoritmo': algoritmo, 'Pliegues_completos': 0}
    clave = f'LOSO/{algoritmo}/Resumen.json'
    guardar_evidencia({clave: estado}, evidencia_dir)
    guardar_csv(pd.DataFrame(columns=['ID_ejecucion', 'Indice_fila', 'Sujeto', 'Ansiedad_real',
                                      'Probabilidad_ansiedad', 'Prediccion']),
                evidencia_dir/f'LOSO/{algoritmo}/Predicciones.csv')
    barra = tqdm(LeaveOneGroupOut().split(datos, datos.Ansiedad, grupos), total=32,
                 desc=f'LOSO Regional {algoritmo}/{"causal-learn" if algoritmo == "PC" else "pgmpy"}, 25 predictores', unit='fold')
    for fold, (it, iv) in enumerate(barra, 1):
        sujeto = str(grupos[iv[0]])
        motor = 'causal-learn' if algoritmo == 'PC' else 'pgmpy'
        contexto = f'Regional {algoritmo}/{motor}, 25 predictores, fold {fold}/32, sujeto {sujeto}'
        registro = {'ID_ejecucion': args.ejecucion_id, 'Fold': fold, 'Sujeto_validacion': sujeto,
                    'Sujetos_entrenamiento': sorted(set(map(str, grupos[it]))),
                    'N_entrenamiento': len(it), 'N_validacion': len(iv)}
        inicio = time.perf_counter()
        try:
            if set(grupos[it]) & set(grupos[iv]) or len(set(grupos[it])) != 31 or len(set(grupos[iv])) != 1:
                raise ValueError('El pliegue no contiene 31/1 sujetos disjuntos.')
            p, dag, meta, evidencia = ajustar_y_predecir(datos.iloc[it], datos.iloc[iv], algoritmo, args, contexto)
            registro.update(Estado='OK', N_predicciones_validas=len(p), Tiempo_fold_s=time.perf_counter()-inicio)
            guardar_evidencia({f'LOSO/{algoritmo}/Fold_{fold:02d}.json': {'Registro': registro, **evidencia}}, evidencia_dir)
            prob[iv] = p
            tiempos.append({**registro, **{k: v for k, v in meta.items() if k.startswith('Tiempo_')},
                            'N_arcos_EEG': meta['N_arcos_EEG']})
            arcos.append({(a, b) for a, b in dag.edges() if a != 'Ansiedad'})
            listos = np.flatnonzero(np.isfinite(prob))
            guardar_csv(tabla_predicciones(datos.iloc[listos], prob[listos], args.ejecucion_id), evidencia_dir/f'LOSO/{algoritmo}/Predicciones.csv')
            estado.update(Pliegues_completos=fold, N_predicciones=len(listos))
            guardar_evidencia({clave: estado,
                               f'Estabilidad/{algoritmo}.json': {'ID_ejecucion': args.ejecucion_id, **resumen_estabilidad(arcos, [])}}, evidencia_dir)
            barra.set_postfix(sujeto=sujeto, segundos=f'{registro["Tiempo_fold_s"]:.1f}')
        except Exception as exc:
            registro.update(Estado='ERROR', Error=f'{type(exc).__name__}: {exc}', Tiempo_fold_s=time.perf_counter()-inicio)
            estado.update(Estado='ERROR', Error=registro)
            guardar_evidencia({clave: estado, f'LOSO/{algoritmo}/Fold_{fold:02d}_ERROR.json': registro}, evidencia_dir)
            raise RuntimeError(f'Falló {contexto}. {type(exc).__name__}: {exc}') from exc
    if not np.isfinite(prob).all():
        raise RuntimeError(f'LOSO regional {algoritmo} no produjo una probabilidad por época.')
    estado.update(Estado='COMPLETO')
    guardar_evidencia({clave: estado}, evidencia_dir)
    return prob, pd.DataFrame(tiempos), arcos


def diagnosticos_estructurales(evidencia):
    """Separación explícita: CS/FisherC clásicos no se aplican al modelo mixto."""
    return {'CS': None, 'FisherC_p': None, 'FisherC_RMSEA': None,
            'Estado': 'NO_APLICADOS',
            'Motivo': 'No se aplica Pearson/Fisher-Z a un grafo que contiene clase binaria y EEG continuo. '
                      'Se conservan BIC condicional de entrenamiento, parámetros, varianzas y estabilidad de arcos.',
            'BIC_condicional_EEG': evidencia['Detalles']['BIC_condicional_EEG'],
            'Nota': 'Diagnósticos de estructura separados de métricas predictivas; sin validación causal.'}


def principal(argv=None):
    parser = argparse.ArgumentParser(description='Red bayesiana condicional gaussiana regional: 25 predictores fijos, LOSO y prueba 32/8.')
    parser.add_argument('--algoritmos', nargs='+', choices=['GES', 'PC'], default=['GES', 'PC'])
    parser.add_argument('--ridge', type=float, default=1.0, help='Regularización de pendientes en las CPD predictivas; no del BIC MLE.')
    parser.add_argument('--piso-varianza', type=float, default=1e-6)
    parser.add_argument('--pc-alpha', type=float, default=.05, help='Se usa alpha/2 en cada una de las dos búsquedas PC por clase.')
    parser.add_argument('--pc-max-cond-vars', type=int, default=4, help='Máximo de condicionantes por test PC; búsqueda acotada, registrada en evidencia.')
    args = parser.parse_args(argv)
    args.ejecucion_id = uuid4().hex
    if not np.isfinite(args.ridge) or args.ridge < 0 or not np.isfinite(args.piso_varianza) or args.piso_varianza <= 0:
        parser.error('ridge debe ser finito y no negativo; piso-varianza debe ser finito y positivo.')
    if not 0 < args.pc_alpha < 1 or not 0 <= args.pc_max_cond_vars <= 23:
        parser.error('pc-alpha debe estar entre 0 y 1 y pc-max-cond-vars entre 0 y 23.')
    if 'PC' in args.algoritmos:
        cargar_pc_causal_learn()
    raiz = Path(__file__).resolve().parents[3]
    salida = raiz/'Resultados'/'Red Bayesiana'/'Global'/'Regional'
    evidencia_dir = salida/'Evidencia'
    salida.mkdir(parents=True, exist_ok=True)
    desarrollo, externa, sujetos, excluidas = cargar_datos(raiz)
    configuracion = {**vars(args), 'Version_modelo': VERSION_MODELO, 'Estado_ejecucion': 'EN_CURSO',
                     'Inicio_UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     'Semilla': 42, 'Criterio': 'SAM=0 control; SAM>=1 caso',
                     'ID_ejecucion': args.ejecucion_id, 'Regiones_electrodos': REGIONES, 'Bandas': BANDAS, 'Predictores': PREDICTORES,
                     'N_predictores': 25, 'Seleccion_DT': False, 'Umbral_fijo': .5,
                     'Sujetos_desarrollo_32': list(map(str, sujetos[:32])),
                     'Sujetos_externos_8': list(map(str, sujetos[32:])),
                     'Epocas_excluidas_por_criterio_SAM': excluidas,
                     'Motores_estructura': {'GES': 'pgmpy', 'PC': 'causal-learn'},
                     'Version_causal_learn': obtener_version('causal-learn'),
                     'Version_pgmpy': obtener_version('pgmpy'), 'Version_numpy': obtener_version('numpy'),
                     'Version_pandas': obtener_version('pandas'), 'Version_sklearn': obtener_version('scikit-learn'),
                     'Version_networkx': obtener_version('networkx'),
                     'Tipos': {'Ansiedad': 'discreta binaria', 'Predictores': 'continuos'},
                     'Arquitectura': 'Clase raíz y padre de los 25 predictores; estructura EEG común y CPD por clase',
                     'BIC': 'Condicional gaussiano MLE por clase, sin ridge; épocas como tamaño muestral',
                     'Regiones': 'Agrupación operacional conservada del script original; requiere justificación en tesis. No localiza fuentes corticales.',
                     'Entrada': 'Valores estandarizados respecto a relajación por sujeto; se requiere ese baseline para predecir en este esquema.',
                     'Evidencia_vigente': 'Usar solo artefactos con ID_ejecucion de esta configuración y pliegues declarados completos; no hay reanudación automática.',
                     'Nota_validacion': 'Etiquetas por época; métricas y bootstrap agrupados por sujeto. Los ocho externos ya se observaron en experimentos anteriores.',
                     'Interpretacion': 'Estructura predictiva observacional; PC estratificado con unión, no PC mixto; sin afirmaciones causales.'}
    guardar_evidencia({'Configuracion.json': configuracion}, evidencia_dir)
    resumen = salida/'Resultados_Red_Regional.csv'
    resultados = []
    guardar_csv(pd.DataFrame(columns=['Estructura', 'Estado', 'Version_modelo']), resumen)
    print(f'Resultados: {salida}', flush=True)
    print('Regional condicional gaussiana: 25 predictores fijos; CPD por clase; SAM>=1; 32/8; umbral 0.5.', flush=True)
    print('GES/pgmpy: BIC condicional. PC/causal-learn: esqueletos por clase, unión y orientación predictiva; no causal.', flush=True)
    inicio_total = time.perf_counter()
    algoritmo = 'INICIO'
    contexto = 'Inicio'
    try:
        for algoritmo in dict.fromkeys(args.algoritmos):
            inicio = time.perf_counter()
            contexto = f'Regional {algoritmo}, LOSO con 25 predictores'
            pval, tiempos, arcos = ejecutar_loso(desarrollo, algoritmo, args, evidencia_dir)
            contexto = f'Regional {algoritmo}, ajuste final 32 sujetos, 25 predictores'
            ptest, dag, meta, evidencia = ajustar_y_predecir(desarrollo, externa, algoritmo, args, contexto)
            evidencia['Diagnosticos_estructura'] = diagnosticos_estructurales(evidencia)
            guardar_evidencia({f'Modelos_Finales/{algoritmo}.json': evidencia,
                               f'Estabilidad/{algoritmo}.json': {'ID_ejecucion': args.ejecucion_id, **resumen_estabilidad(arcos, dag.edges())}}, evidencia_dir)
            guardar_csv(tabla_predicciones(externa, ptest, args.ejecucion_id), evidencia_dir/f'Predicciones_Externas_{algoritmo}.csv')
            fila = {'ID_ejecucion': args.ejecucion_id, 'Estructura': algoritmo, 'Modelo': 'Condicional gaussiana regional', 'Estado': 'COMPLETO',
                    'Version_modelo': VERSION_MODELO, 'Motor': meta['Motor'],
                    'Motor_orientacion': meta['Motor_orientacion'],
                    'Version_pgmpy': obtener_version('pgmpy'),
                    'Version_causal_learn': obtener_version('causal-learn'),
                    'Procedimiento': meta['Procedimiento'],
                    'N_caracteristicas': 25, 'Caracteristicas_modelo': json.dumps(PREDICTORES), 'Seleccion_DT': False,
                    'Clase_raiz': True, 'N_arcos_clase_impuestos': 25, 'N_arcos_EEG_final': meta['N_arcos_EEG'],
                    'N_arcos_final': meta['N_arcos'], 'N_parametros_final': meta['N_parametros'],
                    'Max_padres_EEG_final': meta['Max_padres_EEG'], 'N_varianzas_con_piso_final': meta['N_varianzas_con_piso'],
                    'Ridge': args.ridge, 'Piso_varianza': args.piso_varianza, 'Umbral_fijo': .5,
                    'N_pliegues_LOSO_completos': len(tiempos), 'N_predicciones_LOSO': len(pval),
                    'N_predicciones_externas': len(ptest),
                    'N_arcos_EEG_promedio_LOSO': float(tiempos.N_arcos_EEG.mean()),
                    'Tiempo_LOSO_s': float(tiempos.Tiempo_fold_s.sum()),
                    'Orientacion': meta['Orientacion'], 'Interpretacion_grafo': 'Estructura predictiva observacional; no demuestra causalidad'}
            for etapa in ['preprocesamiento', 'estructura', 'parametros', 'inferencia']:
                fila[f'Tiempo_{etapa}_LOSO_s'] = float(tiempos[f'Tiempo_{etapa}_s'].sum())
                fila[f'Tiempo_{etapa}_final_s'] = meta[f'Tiempo_{etapa}_s']
            for prefijo, datos, prob in [('Val', desarrollo, pval), ('Test', externa, ptest)]:
                y = datos.Ansiedad.to_numpy(dtype=int)
                pred = (prob >= .5).astype(int)
                fila.update({f'{k} ({prefijo})': v for k, v in metricas_clasificacion(y, pred, prob).items()})
                tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
                fila.update({f'{k} ({prefijo})': int(v) for k, v in zip(['TN', 'FP', 'FN', 'TP'], [tn, fp, fn, tp])})
                fila.update(bootstrap_ci_por_sujeto(y, pred, datos.Sujeto.to_numpy(), prefijo))
                fila.update(metricas_agregadas_por_sujeto(y, pred, datos.Sujeto.to_numpy(), prefijo))
            fila['Tiempo_configuracion_s'] = time.perf_counter()-inicio
            resultados.append(fila)
            guardar_csv(pd.DataFrame(resultados), resumen)
            contexto = f'Regional {algoritmo}, figura final'
            graficar_dag(dag, f'Red bayesiana condicional gaussiana regional — {algoritmo}, 25 predictores',
                         salida/f'Red_Mixta_Regional_{algoritmo}.png')
            print(f'COMPLETO {algoritmo}: {time.perf_counter()-inicio:.1f} s', flush=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        configuracion.update(Estado_ejecucion='ERROR', Contexto_fallido=contexto, Error=error)
        guardar_evidencia({'Configuracion.json': configuracion}, evidencia_dir)
        resultados.append({'Estructura': algoritmo, 'Estado': 'ERROR', 'Error': error, 'Version_modelo': VERSION_MODELO, 'ID_ejecucion': args.ejecucion_id})
        guardar_csv(pd.DataFrame(resultados), resumen)
        raise RuntimeError(f'Falló {contexto}. Se conservan resumen y evidencia parcial. {error}') from exc
    configuracion.update(Estado_ejecucion='COMPLETO', Tiempo_total_s=time.perf_counter()-inicio_total)
    guardar_evidencia({'Configuracion.json': configuracion}, evidencia_dir)
    print(f'Resumen: {resumen}', flush=True)
    print(f'Evidencia: {evidencia_dir}', flush=True)
    columnas = ['Estructura', 'Motor', 'Estado', 'Exact. (Val)', 'Balanced_accuracy (Val)', 'ROC_AUC (Val)',
                'Exact. (Test)', 'Balanced_accuracy (Test)', 'ROC_AUC (Test)', 'Tiempo_configuracion_s']
    print(pd.DataFrame(resultados)[columnas].to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    principal()

