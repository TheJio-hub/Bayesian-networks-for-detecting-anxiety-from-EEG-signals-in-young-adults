import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import BayesianEstimator, TreeSearch
from pgmpy.metrics import CorrelationScore, FisherC
from pgmpy.models import DiscreteBayesianNetwork
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

def split_ansiedad_into_three(df_ansiedad, n_control, seed=42):
    df_ansiedad_shuffled = df_ansiedad.sample(frac=1.0, random_state=seed)
    sub_A = df_ansiedad_shuffled.iloc[0:n_control]
    sub_B = df_ansiedad_shuffled.iloc[n_control:2*n_control]
    n_remaining = len(df_ansiedad_shuffled) - 2 * n_control
    remaining = df_ansiedad_shuffled.iloc[2*n_control:]
    if n_remaining < n_control:
        n_to_fill = n_control - n_remaining
        fill = df_ansiedad_shuffled.iloc[0:2*n_control].sample(n=n_to_fill, replace=False, random_state=seed+1)
        sub_C = pd.concat([remaining, fill])
    else:
        sub_C = remaining.iloc[0:n_control]
    return sub_A, sub_B, sub_C

def ejecutar_loso(df_datos, caracteristicas_seleccionadas, columna_objetivo, grupos, n_intervalos):
    particion_grupos = LeaveOneGroupOut()
    y_real = df_datos[columna_objetivo].values
    y_pred = np.zeros(len(df_datos))
    
    total_pliegues = particion_grupos.get_n_splits(df_datos, y_real, grupos)
    barra_progreso = tqdm(
        particion_grupos.split(df_datos, y_real, grupos), 
        total=total_pliegues, 
        desc=f"LOSO TAN (bins={n_intervalos})", 
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
        
        try:
            est = TreeSearch(df_entrenamiento, root_node=caracteristicas_seleccionadas[0])
            dag = est.estimate(estimator_type="tan", class_node=columna_objetivo, show_progress=False)
            
            if len(dag.edges()) == 0:
                raise ValueError("Grafo vacío")

            df_control = df_entrenamiento[df_entrenamiento[columna_objetivo] == 0]
            df_ansiedad = df_entrenamiento[df_entrenamiento[columna_objetivo] == 1]
            
            n_control = len(df_control)
            df_ansiedad_A, df_ansiedad_B, df_ansiedad_C = split_ansiedad_into_three(df_ansiedad, n_control, seed=42)
            
            df_train_A = pd.concat([df_control, df_ansiedad_A])
            df_train_B = pd.concat([df_control, df_ansiedad_B])
            df_train_C = pd.concat([df_control, df_ansiedad_C])
            
            prob_A = None
            prob_B = None
            prob_C = None
            col_1 = f"{columna_objetivo}_1"
            
            # Model A
            try:
                modelo_A = DiscreteBayesianNetwork(dag.edges())
                modelo_A.add_nodes_from(dag.nodes())
                est_param_A = BayesianEstimator(modelo_A, df_train_A)
                modelo_A.add_cpds(*est_param_A.get_parameters(prior_type="K2"))
                y_prob_A = modelo_A.predict_probability(df_prueba)
                if col_1 in y_prob_A.columns:
                    prob_A = y_prob_A[col_1].values
            except Exception:
                pass
                
            # Model B
            try:
                modelo_B = DiscreteBayesianNetwork(dag.edges())
                modelo_B.add_nodes_from(dag.nodes())
                est_param_B = BayesianEstimator(modelo_B, df_train_B)
                modelo_B.add_cpds(*est_param_B.get_parameters(prior_type="K2"))
                y_prob_B = modelo_B.predict_probability(df_prueba)
                if col_1 in y_prob_B.columns:
                    prob_B = y_prob_B[col_1].values
            except Exception:
                pass

            # Model C
            try:
                modelo_C = DiscreteBayesianNetwork(dag.edges())
                modelo_C.add_nodes_from(dag.nodes())
                est_param_C = BayesianEstimator(modelo_C, df_train_C)
                modelo_C.add_cpds(*est_param_C.get_parameters(prior_type="K2"))
                y_prob_C = modelo_C.predict_probability(df_prueba)
                if col_1 in y_prob_C.columns:
                    prob_C = y_prob_C[col_1].values
            except Exception:
                pass
                
            probs_validas = [p for p in [prob_A, prob_B, prob_C] if p is not None]
            if len(probs_validas) > 0:
                y_pred[indices_prueba] = np.mean(probs_validas, axis=0)
            else:
                y_pred[indices_prueba] = df_entrenamiento[columna_objetivo].mean()
            
        except Exception:
            y_pred[indices_prueba] = df_entrenamiento[columna_objetivo].mean()
            
    umbral = 0.50
    y_pred_bin = (y_pred >= umbral).astype(int)
    y_real_bin = y_real.astype(int)
    
    return y_real_bin, y_pred_bin

def principal():
    dir_resultados = os.path.join('Resultados', 'Red Bayesiana', 'Global', 'TAN')
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
    df_ranking = df_ranking.sort_values(by=['Importancia_DT', 'Mutual_Info'], ascending=[False, False])
    mejores_caracteristicas = df_ranking['Caracteristica'].tolist()
    
    tamanos_intervalos = [2, 3]
    top_tamanos = [10, 15]
    resultados_loso = []
    
    for n_intervalos in tamanos_intervalos:
        for n_caracteristicas in top_tamanos:
            caracteristicas = mejores_caracteristicas[:n_caracteristicas]
            
            y_real_val, y_pred_val = ejecutar_loso(df_desarrollo, caracteristicas, 'Ansiedad', grupos_desarrollo, n_intervalos)
            
            val_acc = accuracy_score(y_real_val, y_pred_val)
            val_prec = precision_score(y_real_val, y_pred_val, zero_division=0)
            val_sens = recall_score(y_real_val, y_pred_val, zero_division=0)
            val_f1 = f1_score(y_real_val, y_pred_val, zero_division=0)
            matriz_confusion_val = confusion_matrix(y_real_val, y_pred_val, labels=[0, 1])
            tn, fp, _, _ = matriz_confusion_val.ravel()
            val_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            discretizador = KBinsDiscretizer(n_bins=n_intervalos, encode='ordinal', strategy='quantile')
            
            caract_dev_disc = discretizador.fit_transform(df_desarrollo[caracteristicas])
            caract_prueba_ext_disc = discretizador.transform(df_prueba_externa[caracteristicas])
            
            df_desarrollo_disc = pd.DataFrame(caract_dev_disc, columns=caracteristicas, index=df_desarrollo.index).astype(int)
            df_desarrollo_disc['Ansiedad'] = df_desarrollo['Ansiedad'].values.astype(int)
            
            df_prueba_externa_disc = pd.DataFrame(caract_prueba_ext_disc, columns=caracteristicas, index=df_prueba_externa.index).astype(int)
            
            dag_final = None
            try:
                est_final = TreeSearch(df_desarrollo_disc, root_node=caracteristicas[0])
                dag_final = est_final.estimate(estimator_type="tan", class_node="Ansiedad", show_progress=False)
            except Exception:
                dag_final = None
                
            try:
                if dag_final is None or len(dag_final.edges()) == 0:
                    raise ValueError("Grafo vacío")
                    
                df_control_final = df_desarrollo_disc[df_desarrollo_disc['Ansiedad'] == 0]
                df_ansiedad_final = df_desarrollo_disc[df_desarrollo_disc['Ansiedad'] == 1]
                
                n_control_final = len(df_control_final)
                df_ansiedad_final_A, df_ansiedad_final_B, df_ansiedad_final_C = split_ansiedad_into_three(df_ansiedad_final, n_control_final, seed=42)
                
                df_train_final_A = pd.concat([df_control_final, df_ansiedad_final_A])
                df_train_final_B = pd.concat([df_control_final, df_ansiedad_final_B])
                df_train_final_C = pd.concat([df_control_final, df_ansiedad_final_C])
                
                prob_test_A = None
                prob_test_B = None
                prob_test_C = None
                col_1 = 'Ansiedad_1'
                
                try:
                    modelo_final_A = DiscreteBayesianNetwork(dag_final.edges())
                    modelo_final_A.add_nodes_from(dag_final.nodes())
                    est_param_final_A = BayesianEstimator(modelo_final_A, df_train_final_A)
                    modelo_final_A.add_cpds(*est_param_final_A.get_parameters(prior_type="K2"))
                    y_prob_test_A = modelo_final_A.predict_probability(df_prueba_externa_disc)
                    if col_1 in y_prob_test_A.columns:
                        prob_test_A = y_prob_test_A[col_1].values
                except Exception:
                    pass
                    
                try:
                    modelo_final_B = DiscreteBayesianNetwork(dag_final.edges())
                    modelo_final_B.add_nodes_from(dag_final.nodes())
                    est_param_final_B = BayesianEstimator(modelo_final_B, df_train_final_B)
                    modelo_final_B.add_cpds(*est_param_final_B.get_parameters(prior_type="K2"))
                    y_prob_test_B = modelo_final_B.predict_probability(df_prueba_externa_disc)
                    if col_1 in y_prob_test_B.columns:
                        prob_test_B = y_prob_test_B[col_1].values
                except Exception:
                    pass

                try:
                    modelo_final_C = DiscreteBayesianNetwork(dag_final.edges())
                    modelo_final_C.add_nodes_from(dag_final.nodes())
                    est_param_final_C = BayesianEstimator(modelo_final_C, df_train_final_C)
                    modelo_final_C.add_cpds(*est_param_final_C.get_parameters(prior_type="K2"))
                    y_prob_test_C = modelo_final_C.predict_probability(df_prueba_externa_disc)
                    if col_1 in y_prob_test_C.columns:
                        prob_test_C = y_prob_test_C[col_1].values
                except Exception:
                    pass
                    
                probs_test_validas = [p for p in [prob_test_A, prob_test_B, prob_test_C] if p is not None]
                if len(probs_test_validas) > 0:
                    prob_final_test = np.mean(probs_test_validas, axis=0)
                    y_prueba_externa_pred = (prob_final_test >= 0.50).astype(int)
                else:
                    y_prueba_externa_pred = np.zeros(len(df_prueba_externa_disc))
            except Exception:
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
            
            alg_desc = 'TAN (Tree Search)'
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
                        f"Red Causal TAN (Top-{n_caracteristicas}, Bins-{n_intervalos})", 
                        os.path.join(dir_resultados, f"Estructura_Global_TAN_top{n_caracteristicas}_bins{n_intervalos}.png")
                    )
                except Exception as e:
                    print(f"Error graficando DAG TAN top {n_caracteristicas} bins {n_intervalos}: {e}")

    df_resultados = pd.DataFrame(resultados_loso).round(4)
    ruta_salida = os.path.join(dir_resultados, 'Resultados_Red_TAN.csv')
    df_resultados.to_csv(ruta_salida, index=False)
    
    print("\n=== RESULTADOS RED CAUSAL TAN ===")
    print(df_resultados.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    principal()
