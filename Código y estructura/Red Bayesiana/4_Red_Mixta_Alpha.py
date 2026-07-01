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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pgmpy.causal_discovery import GES, PC
from pgmpy.structure_score import BICCondGauss
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
        desc=f"LOSO Alpha {algoritmo}", 
        unit='fold', 
        leave=False
    )
    
    for indices_entrenamiento, indices_prueba in barra_progreso:
        df_entrenamiento = datos_mixed.iloc[indices_entrenamiento]
        df_prueba = datos_mixed.iloc[indices_prueba]
        
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
                scoring_fn = RobustBICCondGauss(df_entrenamiento)
                estimador = GES(scoring_method=scoring_fn, return_type='dag')
                estimador.fit(df_entrenamiento)
                dag = estimador.causal_graph_
            
            if columna_objetivo in dag.nodes():
                manta_markov = dag.get_markov_blanket(columna_objetivo)
            else:
                manta_markov = []
                
            if len(manta_markov) == 0:
                raise ValueError("Manta de Markov vacía")

            clf = LogisticRegression(class_weight='balanced', random_state=42)
            clf.fit(df_entrenamiento[manta_markov], df_entrenamiento[columna_objetivo])
            
            predicciones = clf.predict(df_prueba[manta_markov])
            y_pred[indices_prueba] = predicciones
            
        except Exception:
            y_pred[indices_prueba] = df_entrenamiento[columna_objetivo].mode().values[0]
            
    y_pred_bin = y_pred.astype(int)
    y_real_bin = y_real.astype(int)
    
    return y_real_bin, y_pred_bin

def principal():
    dir_resultados = os.path.join('Resultados', 'Red Bayesiana', 'Mixta_Alpha')
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
    
    df_ranking = df_ranking[df_ranking['Caracteristica'].str.contains('Alpha', case=False)]
    df_ranking = df_ranking.sort_values(by=['Importancia_DT', 'Mutual_Info'], ascending=[False, False])
    mejores_caracteristicas = df_ranking['Caracteristica'].tolist()
    
    print(f"Features de la banda Alpha disponibles: {len(mejores_caracteristicas)}")
    
    tamanos_top = [10, 15]
    resultados_loso = []
    
    for n_caracteristicas in tamanos_top:
        if len(mejores_caracteristicas) < n_caracteristicas:
            print(f"Advertencia: No hay suficientes características Alpha para el top {n_caracteristicas}.")
            continue
            
        caracteristicas = mejores_caracteristicas[:n_caracteristicas]
        
        for algoritmo in ["PC", "GES"]:
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
            
            dag_final = None
            try:
                if algoritmo == "PC":
                    est_final = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
                    est_final.fit(df_desarrollo_sub)
                    dag_final = est_final.causal_graph_
                else:
                    scoring_fn = RobustBICCondGauss(df_desarrollo_sub)
                    est_final = GES(scoring_method=scoring_fn, return_type='dag')
                    est_final.fit(df_desarrollo_sub)
                    dag_final = est_final.causal_graph_
            except Exception:
                dag_final = None
                
            try:
                if dag_final is None or len(dag_final.edges()) == 0:
                    raise ValueError("Grafo vacío")
                    
                if 'Ansiedad' in dag_final.nodes():
                    manta_markov_final = dag_final.get_markov_blanket('Ansiedad')
                else:
                    manta_markov_final = []
                    
                if len(manta_markov_final) == 0:
                    raise ValueError("Manta de Markov vacía")
                    
                clf_final = LogisticRegression(class_weight='balanced', random_state=42)
                clf_final.fit(df_desarrollo_sub[manta_markov_final], df_desarrollo_sub['Ansiedad'])
                
                y_prueba_externa_pred = clf_final.predict(df_prueba_externa_sub[manta_markov_final])
            except Exception:
                y_prueba_externa_pred = np.repeat(df_desarrollo_sub['Ansiedad'].mode().values[0], len(df_prueba_externa_sub))
                
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
            
            val_bal_acc = (val_sens + val_spec) / 2
            test_bal_acc = (test_sens + test_spec) / 2
            
            alg_desc = 'PC (fisher_z)' if algoritmo == "PC" else 'GES (bic-cg)'
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
                        f"Red Causal Mixta Alpha ({alg_desc}, Top-{n_caracteristicas})", 
                        os.path.join(dir_resultados, f"Estructura_Global_Alpha_{algoritmo}_top{n_caracteristicas}.png")
                    )
                except Exception as e:
                    print(f"Error graficando DAG {algoritmo} top {n_caracteristicas}: {e}")

    df_resultados = pd.DataFrame(resultados_loso)
    ruta_salida = os.path.join(dir_resultados, 'Resultados_Red_Mixta_Alpha.csv')
    df_resultados.to_csv(ruta_salida, index=False)
    
    print("\n=== RESULTADOS RED CAUSAL MIXTA BANDA ALPHA ===")
    print(df_resultados.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    principal()
