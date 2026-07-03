import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.auto import tqdm as _tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pgmpy.causal_discovery import GES, PC
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.base import DAG
from pgmpy.prediction import NaiveAdjustmentRegressor
from pgmpy.metrics import CorrelationScore, FisherC

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

def construir_modelo_regresor(dag, mb_features, columna_objetivo):
    if len(mb_features) == 0:
        return None
        
    prediction_subgraph = dag.subgraph(mb_features + [columna_objetivo])
    parents = list(dag.get_parents(columna_objetivo))
    
    if len(parents) == 0:
        exposures = [mb_features[0]]
        spouses_and_children = [node for node in mb_features if node != mb_features[0]]
    else:
        exposures = [parents[0]]
        spouses_and_children = [node for node in mb_features if node != parents[0]]
        
    causal_graph_model = DAG(
        prediction_subgraph.edges(),
        roles={
            "exposures": exposures,
            "outcomes": [columna_objetivo],
            "adjustment": spouses_and_children
        }
    )
    return causal_graph_model

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
            colores_nodos.append('salmon')
        else:
            colores_nodos.append('lightblue')
            
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=1500, 
        node_color=colores_nodos, 
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

def ejecutar_loso(df_datos, caracteristicas_seleccionadas, columna_objetivo, grupos, algoritmo):
    particion_grupos = LeaveOneGroupOut()
    y_real = df_datos[columna_objetivo].values
    y_pred = np.zeros(len(df_datos))
    
    columnas_subconjunto = caracteristicas_seleccionadas + [columna_objetivo]
    datos_loso = df_datos[columnas_subconjunto].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    
    total_pliegues = particion_grupos.get_n_splits(datos_loso, y_real, grupos)
    barra_progreso = tqdm(
        particion_grupos.split(datos_loso, y_real, grupos), 
        total=total_pliegues, 
        desc=f"LOSO Regresor Causal {algoritmo}", 
        unit='fold', 
        leave=False
    )
    
    for indices_entrenamiento, indices_prueba in barra_progreso:
        df_entrenamiento = datos_loso.iloc[indices_entrenamiento]
        df_prueba = datos_loso.iloc[indices_prueba]
        
        global CALL_COUNTER
        CALL_COUNTER = 0
        
        try:
            if algoritmo == "PC":
                estimador = PC(
                    variant='stable', 
                    ci_test='fisher_z', 
                    significance_level=0.05, 
                    return_type='dag', 
                    show_progress=False
                )
                estimador.fit(df_entrenamiento)
                dag = estimador.causal_graph_
            else:
                estimador = GES(
                    scoring_method='bic-g', 
                    return_type='dag'
                )
                estimador.fit(df_entrenamiento)
                dag = estimador.causal_graph_
            
            mb_features = list(dag.get_markov_blanket(columna_objetivo))
            causal_graph_model = construir_modelo_regresor(dag, mb_features, columna_objetivo)
            
            if causal_graph_model is None or len(causal_graph_model.edges()) == 0:
                raise ValueError("Grafo causal sin relaciones suficientes")
                
            df_control = df_entrenamiento[df_entrenamiento[columna_objetivo] == 0]
            df_ansiedad = df_entrenamiento[df_entrenamiento[columna_objetivo] == 1]
            
            n_control = len(df_control)
            df_ansiedad_A, df_ansiedad_B, df_ansiedad_C = split_ansiedad_into_three(df_ansiedad, n_control, seed=42)
            
            df_train_A = pd.concat([df_control, df_ansiedad_A])
            df_train_B = pd.concat([df_control, df_ansiedad_B])
            df_train_C = pd.concat([df_control, df_ansiedad_C])
            
            pred_A = None
            pred_B = None
            pred_C = None
            
            # Model A
            try:
                regressor_A = NaiveAdjustmentRegressor(causal_graph=causal_graph_model)
                regressor_A.fit(df_train_A[mb_features], df_train_A[columna_objetivo])
                pred_A = regressor_A.predict(df_prueba[mb_features])
            except Exception:
                pass
                
            # Model B
            try:
                regressor_B = NaiveAdjustmentRegressor(causal_graph=causal_graph_model)
                regressor_B.fit(df_train_B[mb_features], df_train_B[columna_objetivo])
                pred_B = regressor_B.predict(df_prueba[mb_features])
            except Exception:
                pass

            # Model C
            try:
                regressor_C = NaiveAdjustmentRegressor(causal_graph=causal_graph_model)
                regressor_C.fit(df_train_C[mb_features], df_train_C[columna_objetivo])
                pred_C = regressor_C.predict(df_prueba[mb_features])
            except Exception:
                pass
                
            preds_validas = [p for p in [pred_A, pred_B, pred_C] if p is not None]
            if len(preds_validas) > 0:
                y_pred[indices_prueba] = np.mean(preds_validas, axis=0)
            else:
                y_pred[indices_prueba] = df_entrenamiento[columna_objetivo].mean()
            
        except Exception:
            y_pred[indices_prueba] = df_entrenamiento[columna_objetivo].mean()
            
    umbral = 0.50
    y_pred_bin = (y_pred >= umbral).astype(int)
    y_real_bin = y_real.astype(int)
    
    return y_real_bin, y_pred_bin

def principal():
    dir_resultados = os.path.join('Resultados', 'Red Bayesiana', 'Regresion_Causal')
    os.makedirs(dir_resultados, exist_ok=True)
    
    archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')
    
    if not os.path.exists(archivo_datos) or not os.path.exists(archivo_ranking):
        print("Error: No se encontraron los archivos necesarios en Resultados.")
        return
        
    df_datos = pd.read_parquet(archivo_datos)
    df_datos = df_datos[(df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1)].copy()
    df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0.0 if x == 0 else 1.0)
    
    df_ranking = pd.read_csv(archivo_ranking)
    df_ranking = df_ranking.sort_values(by=['Importancia_DT', 'Mutual_Info'], ascending=[False, False])
    mejores_caracteristicas = df_ranking['Caracteristica'].tolist()
    
    sujetos_unicos = np.sort(df_datos['Sujeto'].unique())
    np.random.seed(42)
    sujetos_mezclados = np.random.permutation(sujetos_unicos)
    sujetos_desarrollo = sujetos_mezclados[:32]
    sujetos_prueba_externa = sujetos_mezclados[32:]
    
    df_desarrollo = df_datos[df_datos['Sujeto'].isin(sujetos_desarrollo)].copy()
    df_prueba_externa = df_datos[df_datos['Sujeto'].isin(sujetos_prueba_externa)].copy()
    
    grupos_desarrollo = df_desarrollo['Sujeto'].values
    
    tamanos_top = [10, 15, 20]
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
            val_bal_acc = (val_sens + val_spec) / 2
            
            columnas_desarrollo = caracteristicas + ['Ansiedad']
            df_desarrollo_sub = df_desarrollo[columnas_desarrollo].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
            df_prueba_externa_sub = df_prueba_externa[columnas_desarrollo].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
            
            dag_final = None
            global CALL_COUNTER
            CALL_COUNTER = 0
            
            try:
                if algoritmo == "PC":
                    est_final = PC(variant='stable', ci_test='pearsonr', significance_level=0.05, return_type='dag', show_progress=False)
                    est_final.fit(df_desarrollo_sub)
                    dag_final = est_final.causal_graph_
                else:
                    est_final = GES(scoring_method='bic-g', return_type='dag')
                    est_final.fit(df_desarrollo_sub)
                    dag_final = est_final.causal_graph_
            except Exception:
                dag_final = None
                
            try:
                if dag_final is None or len(dag_final.edges()) == 0:
                    raise ValueError("Grafo vacío")
                    
                mb_features_final = list(dag_final.get_markov_blanket('Ansiedad'))
                causal_model_final = construir_modelo_regresor(dag_final, mb_features_final, 'Ansiedad')
                
                if causal_model_final is None or len(causal_model_final.edges()) == 0:
                    raise ValueError("Grafo causal final vacío")
                    
                df_control_final = df_desarrollo_sub[df_desarrollo_sub['Ansiedad'] == 0]
                df_ansiedad_final = df_desarrollo_sub[df_desarrollo_sub['Ansiedad'] == 1]
                
                n_control_final = len(df_control_final)
                df_ansiedad_final_A, df_ansiedad_final_B, df_ansiedad_final_C = split_ansiedad_into_three(df_ansiedad_final, n_control_final, seed=42)
                
                df_train_final_A = pd.concat([df_control_final, df_ansiedad_final_A])
                df_train_final_B = pd.concat([df_control_final, df_ansiedad_final_B])
                df_train_final_C = pd.concat([df_control_final, df_ansiedad_final_C])
                
                pred_test_A = None
                pred_test_B = None
                pred_test_C = None
                
                # Model A
                try:
                    regressor_final_A = NaiveAdjustmentRegressor(causal_graph=causal_model_final)
                    regressor_final_A.fit(df_train_final_A[mb_features_final], df_train_final_A['Ansiedad'])
                    pred_test_A = regressor_final_A.predict(df_prueba_externa_sub[mb_features_final])
                except Exception:
                    pass
                    
                # Model B
                try:
                    regressor_final_B = NaiveAdjustmentRegressor(causal_graph=causal_model_final)
                    regressor_final_B.fit(df_train_final_B[mb_features_final], df_train_final_B['Ansiedad'])
                    pred_test_B = regressor_final_B.predict(df_prueba_externa_sub[mb_features_final])
                except Exception:
                    pass

                # Model C
                try:
                    regressor_final_C = NaiveAdjustmentRegressor(causal_graph=causal_model_final)
                    regressor_final_C.fit(df_train_final_C[mb_features_final], df_train_final_C['Ansiedad'])
                    pred_test_C = regressor_final_C.predict(df_prueba_externa_sub[mb_features_final])
                except Exception:
                    pass
                    
                preds_test_validas = [p for p in [pred_test_A, pred_test_B, pred_test_C] if p is not None]
                if len(preds_test_validas) > 0:
                    prob_final_test = np.mean(preds_test_validas, axis=0)
                    y_prueba_externa_pred = (prob_final_test >= 0.50).astype(int)
                else:
                    y_prueba_externa_pred = np.zeros(len(df_prueba_externa_sub))
            except Exception:
                y_prueba_externa_pred = np.zeros(len(df_prueba_externa_sub))
                
            cs_val = np.nan
            fisher_p = np.nan
            fisher_rmsea = np.nan
            if dag_final is not None:
                try:
                    cs_val = CorrelationScore(ci_test='pearsonr', significance_level=0.05).evaluate(X=df_desarrollo_sub, causal_graph=dag_final)
                except Exception:
                    pass
                try:
                    fisher_p, fisher_rmsea = FisherC(ci_test='pearsonr', compute_rmsea=True, show_progress=False).evaluate(X=df_desarrollo_sub, causal_graph=dag_final)
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
            test_bal_acc = (test_sens + test_spec) / 2
            
            alg_desc = 'PC (fisher_z)' if algoritmo == "PC" else 'GES (bic-g)'
            resultados_loso.append({
                'Algoritmo_Estructura': alg_desc,
                'Top_Features': n_caracteristicas,
                'Val_Exactitud': val_acc,
                'Val_Exactitud_Balanceada': val_bal_acc,
                'Val_Precisión': val_prec,
                'Val_Sensibilidad': val_sens,
                'Val_Especificidad': val_spec,
                'Val_F1_Score': val_f1,
                'Test_Exactitud': test_acc,
                'Test_Exactitud_Balanceada': test_bal_acc,
                'Test_Precisión': test_prec,
                'Test_Sensibilidad': test_sens,
                'Test_Especificidad': test_spec,
                'Test_F1_Score': test_f1,
                'CorrelationScore': cs_val,
                'FisherC_p_value': fisher_p,
                'FisherC_RMSEA': fisher_rmsea
            })
            
            if dag_final is not None:
                try:
                    graficar_dag(
                        dag_final, 
                        f"Red Regresion Causal ({alg_desc}, Top-{n_caracteristicas})", 
                        os.path.join(dir_resultados, f"Estructura_Global_{algoritmo}_top{n_caracteristicas}.png")
                    )
                except Exception as e:
                    print(f"Error graficando DAG {algoritmo} top {n_caracteristicas}: {e}")

    df_resultados = pd.DataFrame(resultados_loso)
    archivo_csv = os.path.join(dir_resultados, 'Resultados_Comparativa.csv')
    df_resultados.to_csv(archivo_csv, index=False)
    
    print("\n=== RESULTADOS REGRESIÓN CAUSAL (MARKOV BLANKET) ===")
    print(df_resultados.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    principal()
