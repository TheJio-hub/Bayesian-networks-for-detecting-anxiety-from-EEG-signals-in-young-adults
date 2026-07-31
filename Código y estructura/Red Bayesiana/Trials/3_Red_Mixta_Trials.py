import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.causal_discovery import GES, PC
from pgmpy.metrics import CorrelationScore, FisherC
from pgmpy.structure_score import BICCondGauss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)

import pgmpy.base.DAG
import pgmpy.base.PDAG

original_to_pdag = pgmpy.base.DAG.to_pdag
original_to_cpdag = pgmpy.base.PDAG.to_cpdag

CALL_COUNTER = 0

def patched_to_pdag(self, *args, **kwargs):
    global CALL_COUNTER
    CALL_COUNTER += 1
    if CALL_COUNTER > 150:
        raise RuntimeError("Bucle infinito detectado en la conversión causal del grafo")
    return original_to_pdag(self, *args, **kwargs)

def patched_to_cpdag(self, *args, **kwargs):
    global CALL_COUNTER
    CALL_COUNTER += 1
    if CALL_COUNTER > 150:
        raise RuntimeError("Bucle infinito detectado en la conversión causal del grafo")
    return original_to_cpdag(self, *args, **kwargs)

pgmpy.base.DAG.to_pdag = patched_to_pdag
pgmpy.base.PDAG.to_cpdag = patched_to_cpdag

class RobustBICCondGauss(BICCondGauss):
    def _local_score(self, variable, parents):
        try:
            score = super()._local_score(variable, parents)
            if np.isnan(score) or np.isinf(score):
                return -1e9
            return score
        except Exception:
            return -1e9

class MDLPDiscretizer:
    """Discretizador supervisado por entropía basado en el Principio de Longitud Mínima de Descripción (MDLP - Fayyad & Irani)."""
    def __init__(self, min_samples_split=5):
        self.min_samples_split = min_samples_split
        self.cut_points_ = {}

    def _entropy(self, y):
        if len(y) == 0:
            return 0.0
        p1 = np.mean(y)
        p0 = 1.0 - p1
        if p0 <= 0 or p1 <= 0:
            return 0.0
        return - (p0 * np.log2(p0) + p1 * np.log2(p1))

    def _mdlp_cut(self, x, y):
        n = len(x)
        if n < self.min_samples_split or len(np.unique(y)) <= 1:
            return []

        sort_idx = np.argsort(x)
        x_sorted, y_sorted = x[sort_idx], y[sort_idx]
        unique_x = np.unique(x_sorted)
        if len(unique_x) <= 1:
            return []

        candidates = (unique_x[:-1] + unique_x[1:]) / 2.0
        best_gain = -1.0
        best_cut = None

        ent_s = self._entropy(y_sorted)
        k_s = len(np.unique(y_sorted))

        for cut in candidates:
            left_mask = x_sorted <= cut
            y_l, y_r = y_sorted[left_mask], y_sorted[~left_mask]
            if len(y_l) == 0 or len(y_r) == 0:
                continue

            ent_l = self._entropy(y_l)
            ent_r = self._entropy(y_r)
            e_cond = (len(y_l) / n) * ent_l + (len(y_r) / n) * ent_r
            gain = ent_s - e_cond

            if gain > best_gain:
                best_gain = gain
                best_cut = cut

        if best_cut is None or best_gain <= 0:
            return []

        left_mask = x_sorted <= best_cut
        y_l, y_r = y_sorted[left_mask], y_sorted[~left_mask]
        k1, k2 = len(np.unique(y_l)), len(np.unique(y_r))
        ent_l, ent_r = self._entropy(y_l), self._entropy(y_r)
        delta = np.log2(3**k_s - 2) - (k_s * ent_s - k1 * ent_l - k2 * ent_r)
        threshold = (np.log2(n - 1) + delta) / n

        if best_gain > threshold:
            left_cuts = self._mdlp_cut(x_sorted[left_mask], y_l)
            right_cuts = self._mdlp_cut(x_sorted[~left_mask], y_r)
            return sorted(list(set(left_cuts + [best_cut] + right_cuts)))
        else:
            return []

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        y_arr = np.asarray(y)

        for col in X_df.columns:
            cuts = self._mdlp_cut(X_df[col].values, y_arr)
            if len(cuts) == 0:
                cuts = [np.median(X_df[col].values)]
            self.cut_points_[col] = cuts
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_disc = pd.DataFrame(index=X_df.index)

        for col in X_df.columns:
            cuts = self.cut_points_.get(col, [])
            bins = [-np.inf] + cuts + [np.inf]
            X_disc[col] = pd.cut(X_df[col], bins=bins, labels=False).fillna(0).astype(int)
        return X_disc

def graficar_dag(dag, titulo, nombre_archivo):
    plt.figure(figsize=(12, 10))
    G = nx.DiGraph(dag.edges())
    G.add_nodes_from(dag.nodes())
    try:
        generations = list(nx.topological_generations(G))
        pos = {}
        num_gens = len(generations)
        for gen_idx, nodes in enumerate(generations):
            y = 1.0 - (gen_idx / (num_gens - 1) if num_gens > 1 else 0.0)
            num_nodes = len(nodes)
            for node_idx, node in enumerate(nodes):
                if num_nodes > 1:
                    x = 0.1 + 0.8 * (node_idx / (num_nodes - 1))
                else:
                    x = 0.5
                pos[node] = (x, y)
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=1.5)
    
    colores_nodos = []
    for nodo in G.nodes():
        if nodo == 'Ansiedad':
            colores_nodos.append('lightgreen')
        else:
            colores_nodos.append('lightblue')
            
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=1500, 
        node_color=colores_nodos, 
        edgecolors='navy', 
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

def split_ansiedad_into_three(df_ansiedad, n_control, seed=42):
    df_ansiedad_shuffled = df_ansiedad.sample(frac=1.0, random_state=seed)
    
    if len(df_ansiedad) < n_control:
        sub_A = df_ansiedad.sample(n=n_control, replace=True, random_state=seed)
        sub_B = df_ansiedad.sample(n=n_control, replace=True, random_state=seed+1)
        sub_C = df_ansiedad.sample(n=n_control, replace=True, random_state=seed+2)
        return sub_A, sub_B, sub_C
        
    sub_A = df_ansiedad_shuffled.iloc[0:n_control]
    sub_B = df_ansiedad_shuffled.iloc[n_control:2*n_control]
    n_remaining = len(df_ansiedad_shuffled) - 2 * n_control
    remaining = df_ansiedad_shuffled.iloc[2*n_control:]
    
    if n_remaining < n_control:
        n_to_fill = n_control - n_remaining
        poblacion = df_ansiedad_shuffled.iloc[0:2*n_control]
        replace_val = True if len(poblacion) < n_to_fill else False
        fill = poblacion.sample(n=n_to_fill, replace=replace_val, random_state=seed+1)
        sub_C = pd.concat([remaining, fill])
    else:
        sub_C = remaining.iloc[0:n_control]
        
    return sub_A, sub_B, sub_C

def ejecutar_loso(df_datos, caracteristicas_seleccionadas, columna_objetivo, grupos, algoritmo):
    particion_grupos = LeaveOneGroupOut()
    y_real = df_datos[columna_objetivo].values
    y_pred = np.zeros(len(df_datos))
    
    columnas_subconjunto = caracteristicas_seleccionadas + [columna_objetivo]
    datos_mixed = df_datos[columnas_subconjunto].copy()
    datos_mixed[columna_objetivo] = datos_mixed[columna_objetivo].astype(int)
    for col in caracteristicas_seleccionadas:
         datos_mixed[col] = pd.to_numeric(datos_mixed[col], errors='coerce').fillna(0).astype(float)
    
    total_pliegues = particion_grupos.get_n_splits(datos_mixed, y_real, grupos)
    barra_progreso = tqdm(
        particion_grupos.split(datos_mixed, y_real, grupos), 
        total=total_pliegues, 
        desc=f"LOSO Mixta {algoritmo}", 
        unit='fold', 
        leave=False
    )
    
    for indices_entrenamiento, indices_prueba in barra_progreso:
        df_entrenamiento = datos_mixed.iloc[indices_entrenamiento]
        df_prueba = datos_mixed.iloc[indices_prueba]
        
        df_control = df_entrenamiento[df_entrenamiento[columna_objetivo] == 0]
        df_ansiedad = df_entrenamiento[df_entrenamiento[columna_objetivo] == 1]
        
        n_control = len(df_control)
        df_ansiedad_A, df_ansiedad_B, df_ansiedad_C = split_ansiedad_into_three(df_ansiedad, n_control, seed=42)
        
        df_train_A = pd.concat([df_control, df_ansiedad_A])
        df_train_B = pd.concat([df_control, df_ansiedad_B])
        df_train_C = pd.concat([df_control, df_ansiedad_C])
        
        # --- Model A ---
        manta_A = []
        try:
            global CALL_COUNTER
            CALL_COUNTER = 0
            if algoritmo == "PC":
                est = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
                est.fit(df_train_A)
                dag_A = est.causal_graph_
            else:
                scoring_fn = RobustBICCondGauss(df_train_A)
                est = GES(scoring_method=scoring_fn, return_type='dag')
                est.fit(df_train_A)
                dag_A = est.causal_graph_
            
            if columna_objetivo in dag_A.nodes():
                manta_A = dag_A.get_markov_blanket(columna_objetivo)
        except Exception:
            pass
        if len(manta_A) == 0:
            manta_A = caracteristicas_seleccionadas
            
        clf_A = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        clf_A.fit(df_train_A[manta_A], df_train_A[columna_objetivo])
        prob_A = clf_A.predict_proba(df_prueba[manta_A])[:, 1]
        
        # --- Model B ---
        manta_B = []
        try:
            CALL_COUNTER = 0
            if algoritmo == "PC":
                est = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
                est.fit(df_train_B)
                dag_B = est.causal_graph_
            else:
                scoring_fn = RobustBICCondGauss(df_train_B)
                est = GES(scoring_method=scoring_fn, return_type='dag')
                est.fit(df_train_B)
                dag_B = est.causal_graph_
            
            if columna_objetivo in dag_B.nodes():
                manta_B = dag_B.get_markov_blanket(columna_objetivo)
        except Exception:
            pass
        if len(manta_B) == 0:
            manta_B = caracteristicas_seleccionadas
            
        clf_B = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        clf_B.fit(df_train_B[manta_B], df_train_B[columna_objetivo])
        prob_B = clf_B.predict_proba(df_prueba[manta_B])[:, 1]
        
        # --- Model C ---
        manta_C = []
        try:
            CALL_COUNTER = 0
            if algoritmo == "PC":
                est = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
                est.fit(df_train_C)
                dag_C = est.causal_graph_
            else:
                scoring_fn = RobustBICCondGauss(df_train_C)
                est = GES(scoring_method=scoring_fn, return_type='dag')
                est.fit(df_train_C)
                dag_C = est.causal_graph_
            
            if columna_objetivo in dag_C.nodes():
                manta_C = dag_C.get_markov_blanket(columna_objetivo)
        except Exception:
            pass
        if len(manta_C) == 0:
            manta_C = caracteristicas_seleccionadas
            
        clf_C = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        clf_C.fit(df_train_C[manta_C], df_train_C[columna_objetivo])
        prob_C = clf_C.predict_proba(df_prueba[manta_C])[:, 1]
        
        y_pred[indices_prueba] = (prob_A + prob_B + prob_C) / 3.0
            
    umbral = 0.5
    y_pred_bin = (y_pred >= umbral).astype(int)
    y_real_bin = y_real.astype(int)
    
    return y_real_bin, y_pred_bin

def principal(trial, tarea=None):
    # Determinar rutas y nombres
    if tarea:
        dir_resultados = os.path.join('Resultados', 'Red Bayesiana', 'Trials', f'Trial_{trial}', tarea, 'Mixta')
        label_ejecucion = f"TRIAL {trial} ({tarea.upper()})"
    else:
        dir_resultados = os.path.join('Resultados', 'Red Bayesiana', 'Trials', f'Trial_{trial}', 'Mixta')
        label_ejecucion = f"TRIAL {trial} (GLOBAL)"
        
    os.makedirs(dir_resultados, exist_ok=True)
    
    archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')
    
    if not os.path.exists(archivo_datos) or not os.path.exists(archivo_ranking):
        print("Error: No se encontraron los archivos necesarios en Resultados.")
        return
        
    df_datos = pd.read_parquet(archivo_datos)
    
    # 1. Filtrar por Trial
    df_datos = df_datos[df_datos['Trial'] == trial].copy()
    
    # 2. Filtrar por Tarea (si se especifica)
    if tarea:
        df_datos = df_datos[df_datos['Tarea'].isin(['Relajacion', tarea])].copy()
        
    df_datos = df_datos[(df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1)].copy()
    df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0 if x == 0 else 1)
    
    # Si tras el filtrado no quedan suficientes muestras
    if len(df_datos) == 0:
        print(f"Advertencia: No hay datos para la combinación {label_ejecucion}.")
        return
        
    sujetos_unicos = np.sort(df_datos['Sujeto'].unique())
    np.random.seed(42)
    sujetos_mezclados = np.random.permutation(sujetos_unicos)
    
    # Determinar tamaño de desarrollo (LOGO)
    n_dev = min(32, len(sujetos_mezclados) - 1)
    if n_dev <= 1:
        print(f"Error: Muy pocos sujetos ({len(sujetos_mezclados)}) para validación LOGO.")
        return
        
    sujetos_desarrollo = sujetos_mezclados[:n_dev]
    sujetos_prueba_externa = sujetos_mezclados[n_dev:]
    
    df_desarrollo = df_datos[df_datos['Sujeto'].isin(sujetos_desarrollo)].copy()
    df_prueba_externa = df_datos[df_datos['Sujeto'].isin(sujetos_prueba_externa)].copy()
    
    grupos_desarrollo = df_desarrollo['Sujeto'].values
    
    df_ranking = pd.read_csv(archivo_ranking)
    df_ranking = df_ranking.sort_values(by='Importancia_DT', ascending=False)
    mejores_caracteristicas = df_ranking['Caracteristica'].tolist()
    
    tamanos_top = [10, 15]
    resultados_loso = []
    
    for n_caracteristicas in tamanos_top:
        caracteristicas = mejores_caracteristicas[:n_caracteristicas]
        
        for algoritmo in ["PC", "GES"]:
            y_real_val, y_pred_val = ejecutar_loso(df_desarrollo, caracteristicas, 'Ansiedad', grupos_desarrollo, algoritmo)
            
            val_acc = accuracy_score(y_real_val, y_pred_val)
            val_prec = precision_score(y_real_val, y_pred_val, zero_division=0)
            val_sens = recall_score(y_real_val, y_pred_val, zero_division=0)
            val_spec = recall_score(y_real_val, y_pred_val, pos_label=0, zero_division=0)
            val_f1 = f1_score(y_real_val, y_pred_val, zero_division=0)
            
            columnas_subconjunto = caracteristicas + ['Ansiedad']
            df_desarrollo_sub = df_desarrollo[columnas_subconjunto].copy()
            df_desarrollo_sub['Ansiedad'] = df_desarrollo_sub['Ansiedad'].astype(int)
            for col in caracteristicas:
                 df_desarrollo_sub[col] = df_desarrollo_sub[col].astype(float)
                 
            df_prueba_externa_sub = df_prueba_externa[columnas_subconjunto].copy()
            df_prueba_externa_sub['Ansiedad'] = df_prueba_externa_sub['Ansiedad'].astype(int)
            for col in caracteristicas:
                 df_prueba_externa_sub[col] = df_prueba_externa_sub[col].astype(float)
            
            global CALL_COUNTER
            CALL_COUNTER = 0
            
            df_control_final = df_desarrollo_sub[df_desarrollo_sub['Ansiedad'] == 0]
            df_ansiedad_final = df_desarrollo_sub[df_desarrollo_sub['Ansiedad'] == 1]
            
            n_control_final = len(df_control_final)
            df_ansiedad_final_A, df_ansiedad_final_B, df_ansiedad_final_C = split_ansiedad_into_three(df_ansiedad_final, n_control_final, seed=42)
            
            df_train_final_A = pd.concat([df_control_final, df_ansiedad_final_A])
            df_train_final_B = pd.concat([df_control_final, df_ansiedad_final_B])
            df_train_final_C = pd.concat([df_control_final, df_ansiedad_final_C])
            
            # --- Model A ---
            manta_final_A = []
            dag_final_A = None
            try:
                CALL_COUNTER = 0
                if algoritmo == "PC":
                    est_f_A = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
                    est_f_A.fit(df_train_final_A)
                    dag_final_A = est_f_A.causal_graph_
                else:
                    scoring_fn = RobustBICCondGauss(df_train_final_A)
                    est_f_A = GES(scoring_method=scoring_fn, return_type='dag')
                    est_f_A.fit(df_train_final_A)
                    dag_final_A = est_f_A.causal_graph_
                if 'Ansiedad' in dag_final_A.nodes():
                    manta_final_A = dag_final_A.get_markov_blanket('Ansiedad')
            except Exception:
                pass
            if len(manta_final_A) == 0:
                manta_final_A = caracteristicas
            
            clf_final_A = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            clf_final_A.fit(df_train_final_A[manta_final_A], df_train_final_A['Ansiedad'])
            
            # Predicción en test (si hay datos de test disponibles)
            if len(df_prueba_externa_sub) > 0:
                prob_test_A = clf_final_A.predict_proba(df_prueba_externa_sub[manta_final_A])[:, 1]
            else:
                prob_test_A = np.zeros(0)
            
            # --- Model B ---
            manta_final_B = []
            dag_final_B = None
            try:
                CALL_COUNTER = 0
                if algoritmo == "PC":
                    est_f_B = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
                    est_f_B.fit(df_train_final_B)
                    dag_final_B = est_f_B.causal_graph_
                else:
                    scoring_fn = RobustBICCondGauss(df_train_final_B)
                    est_f_B = GES(scoring_method=scoring_fn, return_type='dag')
                    est_f_B.fit(df_train_final_B)
                    dag_final_B = est_f_B.causal_graph_
                if 'Ansiedad' in dag_final_B.nodes():
                    manta_final_B = dag_final_B.get_markov_blanket('Ansiedad')
            except Exception:
                pass
            if len(manta_final_B) == 0:
                manta_final_B = caracteristicas
            
            clf_final_B = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            clf_final_B.fit(df_train_final_B[manta_final_B], df_train_final_B['Ansiedad'])
            
            if len(df_prueba_externa_sub) > 0:
                prob_test_B = clf_final_B.predict_proba(df_prueba_externa_sub[manta_final_B])[:, 1]
            else:
                prob_test_B = np.zeros(0)
            
            # --- Model C ---
            manta_final_C = []
            dag_final_C = None
            try:
                CALL_COUNTER = 0
                if algoritmo == "PC":
                    est_f_C = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
                    est_f_C.fit(df_train_final_C)
                    dag_final_C = est_f_C.causal_graph_
                else:
                    scoring_fn = RobustBICCondGauss(df_train_final_C)
                    est_f_C = GES(scoring_method=scoring_fn, return_type='dag')
                    est_f_C.fit(df_train_final_C)
                    dag_final_C = est_f_C.causal_graph_
                if 'Ansiedad' in dag_final_C.nodes():
                    manta_final_C = dag_final_C.get_markov_blanket('Ansiedad')
            except Exception:
                pass
            if len(manta_final_C) == 0:
                manta_final_C = caracteristicas
            
            clf_final_C = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            clf_final_C.fit(df_train_final_C[manta_final_C], df_train_final_C['Ansiedad'])
            
            if len(df_prueba_externa_sub) > 0:
                prob_test_C = clf_final_C.predict_proba(df_prueba_externa_sub[manta_final_C])[:, 1]
            else:
                prob_test_C = np.zeros(0)
            
            # Métricas en test
            if len(df_prueba_externa_sub) > 0:
                y_prueba_externa_real = df_prueba_externa_sub['Ansiedad'].values.astype(int)
                y_prueba_externa_pred = (((prob_test_A + prob_test_B + prob_test_C) / 3.0) >= 0.50).astype(int)
                
                test_acc = accuracy_score(y_prueba_externa_real, y_prueba_externa_pred)
                test_prec = precision_score(y_prueba_externa_real, y_prueba_externa_pred, zero_division=0)
                test_sens = recall_score(y_prueba_externa_real, y_prueba_externa_pred, zero_division=0)
                test_f1 = f1_score(y_prueba_externa_real, y_prueba_externa_pred, zero_division=0)
                matriz_confusion_test = confusion_matrix(y_prueba_externa_real, y_prueba_externa_pred, labels=[0, 1])
                tn_t, fp_t, _, _ = matriz_confusion_test.ravel()
                test_spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
            else:
                test_acc = test_prec = test_sens = test_spec = test_f1 = np.nan
            
            cs_val = np.nan
            fisher_p = np.nan
            fisher_rmsea = np.nan
            if dag_final_A is not None:
                try:
                    cs_val = CorrelationScore(ci_test='pearsonr', significance_level=0.05).evaluate(X=df_train_final_A, causal_graph=dag_final_A)
                except Exception:
                    pass
                try:
                    fisher_p, fisher_rmsea = FisherC(ci_test='pearsonr', compute_rmsea=True, show_progress=False).evaluate(X=df_train_final_A, causal_graph=dag_final_A)
                except Exception:
                    pass

            alg_desc = 'PC (fisher_z)' if algoritmo == "PC" else 'GES (bic-cg)'
            resultados_loso.append({
                'Estructura': alg_desc,
                'Top': n_caracteristicas,
                'Exact. (Val)': val_acc,
                'Exact. (Test)': test_acc,
                'Prec. (Val)': val_prec,
                'Prec. (Test)': test_prec,
                'Sens. (Val)': val_sens,
                'Sens. (Test)': test_sens,
                'Esp. (Val)': val_spec,
                'Esp. (Test)': test_spec,
                'F1 (Val)': val_f1,
                'F1 (Test)': test_f1,
                'CS': cs_val,
                'FisherC_p': fisher_p,
                'FisherC_RMSEA': fisher_rmsea
            })
            
            if dag_final_A is not None:
                try:
                    graficar_dag(
                        dag_final_A, 
                        f"Red Causal Mixta - Trial {trial} ({alg_desc}, Top-{n_caracteristicas})", 
                        os.path.join(dir_resultados, f"Estructura_Trial{trial}_{algoritmo}_top{n_caracteristicas}.png")
                    )
                except Exception as e:
                    print(f"Error graficando DAG {algoritmo} top {n_caracteristicas} para trial {trial}: {e}")

    df_resultados = pd.DataFrame(resultados_loso).round(4)
    if tarea:
        ruta_csv = os.path.join(dir_resultados, f'Resultados_Red_Mixta_Trial_{trial}_{tarea}.csv')
    else:
        ruta_csv = os.path.join(dir_resultados, f'Resultados_Red_Mixta_Trial_{trial}.csv')
        
    df_resultados.to_csv(ruta_csv, index=False)
    
    print(f"\n=== RESULTADOS RED CAUSAL MIXTA ({label_ejecucion}) ===")
    print(df_resultados.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entrenamiento de Red Causal Mixta por Trials (Segmentación Temporal)")
    parser.add_argument('--trial', type=int, choices=[1, 2, 3],
                        help="Número del trial a evaluar (1, 2 o 3)")
    parser.add_argument('--tarea', type=str, choices=['Aritmetica', 'Espejo', 'Stroop'],
                        help="Nombre de la tarea a evaluar (opcional)")
    args = parser.parse_args()
    
    if args.trial:
        principal(args.trial, args.tarea)
    else:
        print("No se especificó un trial con --trial. Ejecutando de forma secuencial los Trials 1, 2 y 3...")
        for tr in [1, 2, 3]:
            print("\n==========================================")
            print(f"Iniciando ejecución de: Trial {tr} (Global)")
            print("==========================================")
            principal(tr, args.tarea)
