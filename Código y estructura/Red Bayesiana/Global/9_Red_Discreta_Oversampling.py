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
from pgmpy.estimators import BayesianEstimator
from pgmpy.metrics import CorrelationScore, FisherC
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import KBinsDiscretizer
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

def make_dag_clean(edges):
    clean_edges = []
    seen = set()
    for u, v in edges:
        if (u, v) in seen or (v, u) in seen:
            continue
        clean_edges.append((u, v))
        seen.add((u, v))
    
    G = nx.DiGraph(clean_edges)
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation='original')
            G.remove_edge(cycle[-1][0], cycle[-1][1])
        except Exception:
            break
            
    return list(G.edges())

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

def ejecutar_loso(df_datos, caracteristicas_seleccionadas, columna_objetivo, grupos, algoritmo, n_intervalos):
    particion_grupos = LeaveOneGroupOut()
    y_real = df_datos[columna_objetivo].values
    y_pred = np.zeros(len(df_datos))
    
    total_pliegues = particion_grupos.get_n_splits(df_datos, y_real, grupos)
    barra_progreso = tqdm(
        particion_grupos.split(df_datos, y_real, grupos), 
        total=total_pliegues, 
        desc=f"LOSO Oversampling {algoritmo} (bins={n_intervalos})", 
        unit='fold', 
        leave=False
    )
    
    for indices_entrenamiento, indices_prueba in barra_progreso:
        df_entrenamiento_crudo = df_datos.iloc[indices_entrenamiento]
        df_prueba_crudo = df_datos.iloc[indices_prueba]
        
        discretizador = KBinsDiscretizer(n_bins=n_intervalos, encode='ordinal', strategy='quantile')
        
        caracteristicas_entrenamiento_disc = discretizador.fit_transform(df_entrenamiento_crudo[caracteristicas_seleccionadas])
        caracteristicas_prueba_disc = discretizador.transform(df_prueba_crudo[caracteristicas_seleccionadas])
        
        df_entrenamiento = pd.DataFrame(caracteristicas_entrenamiento_disc, columns=caracteristicas_seleccionadas, index=df_entrenamiento_crudo.index).astype(int)
        df_entrenamiento[columna_objetivo] = df_entrenamiento_crudo[columna_objetivo].values.astype(int)
        
        df_prueba = pd.DataFrame(caracteristicas_prueba_disc, columns=caracteristicas_seleccionadas, index=df_prueba_crudo.index).astype(int)
        
        # 1. Separar clases
        df_control_train = df_entrenamiento[df_entrenamiento[columna_objetivo] == 0]
        df_ansiedad_train = df_entrenamiento[df_entrenamiento[columna_objetivo] == 1]
        
        # 2. Ajustar Generador Causal en Control (clase minoritaria)
        df_control_features = df_control_train[caracteristicas_seleccionadas]
        global CALL_COUNTER
        try:
            CALL_COUNTER = 0
            if algoritmo == "PC":
                est_ctrl = PC(variant='stable', ci_test='chi_square', significance_level=0.05, return_type='dag', show_progress=False)
            else:
                est_ctrl = GES(scoring_method='bic-d', return_type='dag')
            est_ctrl.fit(df_control_features)
            dag_ctrl = est_ctrl.causal_graph_
        except Exception:
            # Fallback a DAG vacío
            dag_ctrl = DiscreteBayesianNetwork().causal_graph_
            
        model_ctrl = DiscreteBayesianNetwork(make_dag_clean(dag_ctrl.edges()))
        model_ctrl.add_nodes_from(caracteristicas_seleccionadas)
        try:
            est_param_ctrl = BayesianEstimator(model_ctrl, df_control_features)
            model_ctrl.add_cpds(*est_param_ctrl.get_parameters(prior_type="K2"))
        except Exception:
            pass
            
        # 3. Sobremuestreo
        n_diff = len(df_ansiedad_train) - len(df_control_train)
        if n_diff > 0:
            try:
                sampler = BayesianModelSampling(model_ctrl)
                df_synthetic = sampler.forward_sample(size=n_diff, show_progress=False)
                df_synthetic[columna_objetivo] = 0
                df_synthetic = df_synthetic[df_entrenamiento.columns]
                df_train_balanced = pd.concat([df_entrenamiento, df_synthetic], ignore_index=True)
            except Exception:
                df_train_balanced = df_entrenamiento.copy()
        else:
            df_train_balanced = df_entrenamiento.copy()
            
        # 4. Entrenar Modelo Final
        prob = None
        col_1 = f"{columna_objetivo}_1"
        try:
            CALL_COUNTER = 0
            if algoritmo == "PC":
                estimador = PC(variant='stable', ci_test='chi_square', significance_level=0.05, return_type='dag', show_progress=False)
                estimador.fit(df_train_balanced)
                dag_final = estimador.causal_graph_
            else:
                estimador = GES(scoring_method='bic-d', return_type='dag')
                estimador.fit(df_train_balanced)
                dag_final = estimador.causal_graph_
                
            if len(dag_final.edges()) > 0:
                modelo_final = DiscreteBayesianNetwork(make_dag_clean(dag_final.edges()))
                modelo_final.add_nodes_from(dag_final.nodes())
                
                est_param = BayesianEstimator(modelo_final, df_train_balanced)
                cpds = est_param.get_parameters(prior_type="K2")
                modelo_final.add_cpds(*cpds)
                
                y_prob = modelo_final.predict_probability(df_prueba)
                if col_1 in y_prob.columns:
                    prob = y_prob[col_1].values
        except Exception:
            pass
            
        if prob is not None:
            y_pred[indices_prueba] = prob
        else:
            y_pred[indices_prueba] = df_train_balanced[columna_objetivo].mean()
            
    umbral_optimo = 0.50
    y_pred_bin = (y_pred >= umbral_optimo).astype(int)
    y_real_bin = y_real.astype(int)
    
    return y_real_bin, y_pred_bin, umbral_optimo

def principal():
    dir_resultados = os.path.join('Resultados', 'Red Bayesiana', 'Global', 'Discreta_Oversampling')
    os.makedirs(dir_resultados, exist_ok=True)
    
    archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')
    
    if not os.path.exists(archivo_datos) or not os.path.exists(archivo_ranking):
        print("Error: No se encontraron los archivos necesarios en Resultados.")
        return
        
    df_datos = pd.read_parquet(archivo_datos)
    df_datos = df_datos[(df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1)].copy()
    df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0 if x == 0 else 1)
    
    sujetos_unicos = np.sort(df_datos['Sujeto'].unique())
    np.random.seed(42)
    sujetos_mezclados = np.random.permutation(sujetos_unicos)
    sujetos_desarrollo = sujetos_mezclados[:32]
    sujetos_prueba_externa = sujetos_mezclados[32:]
    
    df_desarrollo = df_datos[df_datos['Sujeto'].isin(sujetos_desarrollo)].copy()
    df_prueba_externa = df_datos[df_datos['Sujeto'].isin(sujetos_prueba_externa)].copy()
    
    grupos_desarrollo = df_desarrollo['Sujeto'].values
    df_ranking = pd.read_csv(archivo_ranking)
    
    tamanos_intervalos = [2, 3]
    top_tamanos = [10, 15]
    resultados_loso = []
    
    for n_intervalos in tamanos_intervalos:
        for n_caracteristicas in top_tamanos:
            for algoritmo in ["PC", "GES"]:
                # Selección híbrida de características (mRMR para PC Top 10, DT para los demás)
                if algoritmo == "PC" and n_caracteristicas == 10:
                    df_ranking_temp = df_ranking.sort_values(by='mRMR_40_Rank', ascending=True)
                else:
                    df_ranking_temp = df_ranking.sort_values(by='Importancia_DT', ascending=False)
                    
                caracteristicas = df_ranking_temp['Caracteristica'].tolist()[:n_caracteristicas]
                
                y_real_val, y_pred_val, umbral_optimo = ejecutar_loso(df_desarrollo, caracteristicas, 'Ansiedad', grupos_desarrollo, algoritmo, n_intervalos)
                
                val_acc = accuracy_score(y_real_val, y_pred_val)
                val_prec = precision_score(y_real_val, y_pred_val, zero_division=0)
                val_sens = recall_score(y_real_val, y_pred_val, zero_division=0)
                val_f1 = f1_score(y_real_val, y_pred_val, zero_division=0)
                matriz_confusion_val = confusion_matrix(y_real_val, y_pred_val, labels=[0, 1])
                tn, fp, _, _ = matriz_confusion_val.ravel()
                val_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                # Evaluación en Test Externo
                discretizador = KBinsDiscretizer(n_bins=n_intervalos, encode='ordinal', strategy='quantile')
                caract_dev_disc = discretizador.fit_transform(df_desarrollo[caracteristicas])
                caract_prueba_ext_disc = discretizador.transform(df_prueba_externa[caracteristicas])
                
                df_desarrollo_disc = pd.DataFrame(caract_dev_disc, columns=caracteristicas, index=df_desarrollo.index).astype(int)
                df_desarrollo_disc['Ansiedad'] = df_desarrollo['Ansiedad'].values.astype(int)
                df_prueba_externa_disc = pd.DataFrame(caract_prueba_ext_disc, columns=caracteristicas, index=df_prueba_externa.index).astype(int)
                
                # Separar desarrollo para entrenar el generador final
                df_control_final = df_desarrollo_disc[df_desarrollo_disc['Ansiedad'] == 0]
                df_ansiedad_final = df_desarrollo_disc[df_desarrollo_disc['Ansiedad'] == 1]
                
                # Entrenar generador en control
                df_control_features_final = df_control_final[caracteristicas]
                try:
                    global CALL_COUNTER
                    CALL_COUNTER = 0
                    if algoritmo == "PC":
                        est_ctrl_final = PC(variant='stable', ci_test='chi_square', significance_level=0.05, return_type='dag', show_progress=False)
                    else:
                        est_ctrl_final = GES(scoring_method='bic-d', return_type='dag')
                    est_ctrl_final.fit(df_control_features_final)
                    dag_ctrl_final = est_ctrl_final.causal_graph_
                except Exception:
                    dag_ctrl_final = DiscreteBayesianNetwork().causal_graph_
                    
                model_ctrl_final = DiscreteBayesianNetwork(make_dag_clean(dag_ctrl_final.edges()))
                model_ctrl_final.add_nodes_from(caracteristicas)
                try:
                    est_param_ctrl_final = BayesianEstimator(model_ctrl_final, df_control_features_final)
                    model_ctrl_final.add_cpds(*est_param_ctrl_final.get_parameters(prior_type="K2"))
                except Exception:
                    pass
                    
                # Generar muestras sintéticas control
                n_diff_final = len(df_ansiedad_final) - len(df_control_final)
                if n_diff_final > 0:
                    try:
                        sampler_final = BayesianModelSampling(model_ctrl_final)
                        df_synthetic_final = sampler_final.forward_sample(size=n_diff_final, show_progress=False)
                        df_synthetic_final['Ansiedad'] = 0
                        df_synthetic_final = df_synthetic_final[df_desarrollo_disc.columns]
                        df_train_balanced_final = pd.concat([df_desarrollo_disc, df_synthetic_final], ignore_index=True)
                    except Exception:
                        df_train_balanced_final = df_desarrollo_disc.copy()
                else:
                    df_train_balanced_final = df_desarrollo_disc.copy()
                    
                # Entrenar clasificador final
                dag_final = None
                prob_test = None
                col_1 = 'Ansiedad_1'
                try:
                    CALL_COUNTER = 0
                    if algoritmo == "PC":
                        est_final = PC(variant='stable', ci_test='chi_square', significance_level=0.05, return_type='dag', show_progress=False)
                        est_final.fit(df_train_balanced_final)
                        dag_final = est_final.causal_graph_
                    else:
                        est_final = GES(scoring_method='bic-d', return_type='dag')
                        est_final.fit(df_train_balanced_final)
                        dag_final = est_final.causal_graph_
                        
                    if len(dag_final.edges()) > 0:
                        modelo_final = DiscreteBayesianNetwork(make_dag_clean(dag_final.edges()))
                        modelo_final.add_nodes_from(dag_final.nodes())
                        
                        est_param_final = BayesianEstimator(modelo_final, df_train_balanced_final)
                        cpds_final = est_param_final.get_parameters(prior_type="K2")
                        modelo_final.add_cpds(*cpds_final)
                        
                        y_prob_test = modelo_final.predict_probability(df_prueba_externa_disc)
                        if col_1 in y_prob_test.columns:
                            prob_test = y_prob_test[col_1].values
                except Exception:
                    pass
                    
                if prob_test is not None:
                    y_prueba_externa_pred = (prob_test >= umbral_optimo).astype(int)
                else:
                    y_prueba_externa_pred = np.zeros(len(df_prueba_externa_disc))
                    
                cs_val = np.nan
                fisher_p = np.nan
                fisher_rmsea = np.nan
                if dag_final is not None:
                    try:
                        cs_val = CorrelationScore(ci_test='chi_square', significance_level=0.05).evaluate(X=df_desarrollo_disc, causal_graph=dag_final)
                    except Exception:
                        pass
                    try:
                        fisher_p, fisher_rmsea = FisherC(ci_test='chi_square', compute_rmsea=True, show_progress=False).evaluate(X=df_desarrollo_disc, causal_graph=dag_final)
                    except Exception:
                        pass
                        
                y_prueba_externa_real = df_prueba_externa['Ansiedad'].values.astype(int)
                test_acc = accuracy_score(y_prueba_externa_real, y_prueba_externa_pred)
                test_prec = precision_score(y_prueba_externa_real, y_prueba_externa_pred, zero_division=0)
                test_sens = recall_score(y_prueba_externa_real, y_prueba_externa_pred, zero_division=0)
                test_f1 = f1_score(y_prueba_externa_real, y_prueba_externa_pred, zero_division=0)
                matriz_confusion_test = confusion_matrix(y_prueba_externa_real, y_prueba_externa_pred, labels=[0, 1])
                tn_t, fp_t, _, _ = matriz_confusion_test.ravel()
                test_spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
                
                val_bal_acc = (val_sens + val_spec) / 2
                test_bal_acc = (test_sens + test_spec) / 2
                
                alg_desc = 'PC (chi_square)' if algoritmo == "PC" else 'GES (bic-d)'
                resultados_loso.append({
                    'Estructura': alg_desc,
                    'n_bins': n_intervalos,
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
                
                if dag_final is not None:
                    try:
                        graficar_dag(
                            dag_final, 
                            f"Red Discreta Causal (Oversampling, {alg_desc}, Top-{n_caracteristicas}, Bins-{n_intervalos})", 
                            os.path.join(dir_resultados, f"Estructura_Global_{algoritmo}_top{n_caracteristicas}_bins{n_intervalos}.png")
                        )
                    except Exception as e:
                        print(f"Error graficando DAG {algoritmo} top {n_caracteristicas} bins {n_intervalos}: {e}")

    df_resultados = pd.DataFrame(resultados_loso).round(4)
    ruta_salida = os.path.join(dir_resultados, 'Resultados_Red_Discreta_Oversampling.csv')
    df_resultados.to_csv(ruta_salida, index=False)
    
    print("\n=== RESULTADOS RED CAUSAL DISCRETA (OVERSAMPLING CAUSAL) ===")
    print(df_resultados.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    principal()
