import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)

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

CALL_COUNTER = 0

class RobustBICCondGauss(BICCondGauss):
    def _local_score(self, variable, parents):
        try:
            score = super()._local_score(variable, parents)
            if np.isnan(score) or np.isinf(score):
                return -1e9
            return score
        except Exception:
            return -1e9

def graficar_dag(dag, titulo, nombre_archivo):
    plt.figure(figsize=(10, 8))
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
        node_size=2000, 
        node_color=colores_nodos, 
        edgecolors='navy', 
        linewidths=1.5
    )
    nx.draw_networkx_labels(
        G, pos, 
        font_size=9, 
        font_family='sans-serif', 
        font_weight='bold'
    )
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=list(G.edges()), 
        node_size=2000,
        edge_color='navy', 
        arrowstyle='-|>',
        arrowsize=20, 
        width=1.5, 
        alpha=0.8
    )
    
    plt.title(titulo, fontsize=12, fontweight='bold', pad=15)
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
    
    for col in caracteristicas_seleccionadas:
         datos_mixed[col] = pd.to_numeric(datos_mixed[col], errors='coerce').fillna(0).astype(float)
    datos_mixed[columna_objetivo] = datos_mixed[columna_objetivo].astype(int)
    
    desc_bar = f"LOSO Regional {algoritmo}"
    
    for train_idx, test_idx in tqdm(particion_grupos.split(datos_mixed, y_real, grupos), desc=desc_bar, leave=False):
        df_train = datos_mixed.iloc[train_idx].copy()
        df_prueba = datos_mixed.iloc[test_idx].copy()
        
        indices_prueba = df_prueba.index
        
        df_control = df_train[df_train[columna_objetivo] == 0]
        df_ansiedad = df_train[df_train[columna_objetivo] == 1]
        
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
                est_A = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
                est_A.fit(df_train_A)
                dag_A = est_A.causal_graph_
            else:
                scoring_fn = RobustBICCondGauss(df_train_A)
                est_A = GES(scoring_method=scoring_fn, return_type='dag')
                est_A.fit(df_train_A)
                dag_A = est_A.causal_graph_
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
                est_B = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
                est_B.fit(df_train_B)
                dag_B = est_B.causal_graph_
            else:
                scoring_fn = RobustBICCondGauss(df_train_B)
                est_B = GES(scoring_method=scoring_fn, return_type='dag')
                est_B.fit(df_train_B)
                dag_B = est_B.causal_graph_
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
                est_C = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
                est_C.fit(df_train_C)
                dag_C = est_C.causal_graph_
            else:
                scoring_fn = RobustBICCondGauss(df_train_C)
                est_C = GES(scoring_method=scoring_fn, return_type='dag')
                est_C.fit(df_train_C)
                dag_C = est_C.causal_graph_
            if columna_objetivo in dag_C.nodes():
                manta_C = dag_C.get_markov_blanket(columna_objetivo)
        except Exception:
            pass
        if len(manta_C) == 0:
            manta_C = caracteristicas_seleccionadas
            
        clf_C = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        clf_C.fit(df_train_C[manta_C], df_train_C[columna_objetivo])
        prob_C = clf_C.predict_proba(df_prueba[manta_C])[:, 1]
        
        y_pred[test_idx] = (prob_A + prob_B + prob_C) / 3.0
            
    umbral = 0.5
    y_pred_bin = (y_pred >= umbral).astype(int)
    y_real_bin = y_real.astype(int)
    
    return y_real_bin, y_pred_bin

def principal():
    dir_resultados = os.path.join('Resultados', 'Red Bayesiana', 'Global', 'Regional')
    os.makedirs(dir_resultados, exist_ok=True)
    
    archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    
    if not os.path.exists(archivo_datos):
        print("Error: No se encontró el archivo de datos normalizados.")
        return
        
    df_datos = pd.read_parquet(archivo_datos)
    df_datos = df_datos[(df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1)].copy()
    df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0 if x == 0 else 1)
    
    # 1. Definición de canales y cálculo de las 5 regiones cerebrales
    regiones_canales = {
        'Reg_Frontal': ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8'],
        'Reg_Central': ['Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'C3', 'C4'],
        'Reg_Temporal': ['FT9', 'FT10', 'T7', 'T8', 'CP5', 'CP6'],
        'Reg_Parietal': ['CP1', 'CP2', 'P3', 'P4', 'P7', 'P8', 'Pz'],
        'Reg_Occipital': ['O1', 'O2', 'Oz', 'PO9', 'PO10']
    }
    
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    print("Calculando potencias espectrales promedio por regiones de la corteza...")
    for region, canales in regiones_canales.items():
        columnas_absolutas = [f"{canal}_{banda}" for canal in canales for banda in bandas]
        df_datos[region] = df_datos[columnas_absolutas].mean(axis=1)
        
    caracteristicas = list(regiones_canales.keys())
    print(f"Nodos de regiones causales definidos: {caracteristicas}")
    
    # 2. División de sujetos (idéntica al modelo global por canales)
    sujetos_unicos = np.sort(df_datos['Sujeto'].unique())
    np.random.seed(42)
    sujetos_mezclados = np.random.permutation(sujetos_unicos)
    sujetos_desarrollo = sujetos_mezclados[:32]
    sujetos_prueba_externa = sujetos_mezclados[32:]
    
    df_desarrollo = df_datos[df_datos['Sujeto'].isin(sujetos_desarrollo)].copy()
    df_prueba_externa = df_datos[df_datos['Sujeto'].isin(sujetos_prueba_externa)].copy()
    
    grupos_desarrollo = df_desarrollo['Sujeto'].values
    
    resultados_regional = []
    
    # 3. Modelado y validación de la Red Regional
    for algoritmo in ["PC", "GES"]:
        print(f"\nEntrenando Red Mixta Regional usando algoritmo: {algoritmo}...")
        y_real_val, y_pred_val = ejecutar_loso(df_desarrollo, caracteristicas, 'Ansiedad', grupos_desarrollo, algoritmo)
        
        val_acc = accuracy_score(y_real_val, y_pred_val)
        val_prec = precision_score(y_real_val, y_pred_val, zero_division=0)
        val_sens = recall_score(y_real_val, y_pred_val, zero_division=0)
        val_f1 = f1_score(y_real_val, y_pred_val, zero_division=0)
        matriz_confusion_val = confusion_matrix(y_real_val, y_pred_val, labels=[0, 1])
        tn, fp, _, _ = matriz_confusion_val.ravel()
        val_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
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
        prob_test_A = clf_final_A.predict_proba(df_prueba_externa_sub[manta_final_A])[:, 1]
        
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
        prob_test_B = clf_final_B.predict_proba(df_prueba_externa_sub[manta_final_B])[:, 1]
        
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
        prob_test_C = clf_final_C.predict_proba(df_prueba_externa_sub[manta_final_C])[:, 1]
        
        y_prueba_externa_pred = (((prob_test_A + prob_test_B + prob_test_C) / 3.0) >= 0.50).astype(int)
        
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

        y_prueba_externa_real = df_prueba_externa_sub['Ansiedad'].values.astype(int)
        test_acc = accuracy_score(y_prueba_externa_real, y_prueba_externa_pred)
        test_prec = precision_score(y_prueba_externa_real, y_prueba_externa_pred, zero_division=0)
        test_sens = recall_score(y_prueba_externa_real, y_prueba_externa_pred, zero_division=0)
        test_f1 = f1_score(y_prueba_externa_real, y_prueba_externa_pred, zero_division=0)
        matriz_confusion_test = confusion_matrix(y_prueba_externa_real, y_prueba_externa_pred, labels=[0, 1])
        tn_t, fp_t, _, _ = matriz_confusion_test.ravel()
        test_spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
        
        resultados_regional.append({
            'Estructura': f"{algoritmo} (Regional)",
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
        
        # 4. Guardar gráficos del DAG del sub-modelo A
        if dag_final_A is not None:
            nombre_imagen = os.path.join(dir_resultados, f"Red_Mixta_Regional_{algoritmo}.png")
            print(f"Guardando grafo causal regional en: {nombre_imagen}")
            graficar_dag(dag_final_A, f"Red Causal Mixta Regional - {algoritmo} (Sub-modelo A)", nombre_imagen)
            
    # 5. Guardar tabla de resultados
    df_res = pd.DataFrame(resultados_regional)
    archivo_csv = os.path.join(dir_resultados, 'Resultados_Red_Regional.csv')
    df_res.to_csv(archivo_csv, index=False)
    
    print("\n=== RESULTADOS RED CAUSAL MIXTA REGIONAL ===")
    print(df_res.to_string(index=False))
    print(f"\nResultados csv guardados en: {archivo_csv}")

if __name__ == "__main__":
    principal()
