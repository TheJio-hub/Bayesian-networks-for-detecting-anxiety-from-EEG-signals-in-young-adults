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
        raise RuntimeError("Bucle infinito detectado")
    return original_to_pdag(self, *args, **kwargs)


def patched_to_cpdag(self, *args, **kwargs):
    global CALL_COUNTER
    CALL_COUNTER += 1
    if CALL_COUNTER > 150:
        raise RuntimeError("Bucle infinito detectado")
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
            
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=colores_nodos, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, width=1.5, node_size=2000)
    
    plt.title(titulo, fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
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


def entrenar_modelo_mixto_probabilidades(df_entrenamiento, df_prueba, caracteristicas_seleccionadas, columna_objetivo='Ansiedad', algoritmo='GES', white_list=None):
    columnas_subconjunto = caracteristicas_seleccionadas + [columna_objetivo]
    train_sub = df_entrenamiento[columnas_subconjunto].copy()
    test_sub = df_prueba[columnas_subconjunto].copy()
    
    train_sub[columna_objetivo] = train_sub[columna_objetivo].astype(int)
    test_sub[columna_objetivo] = test_sub[columna_objetivo].astype(int)
    
    for col in caracteristicas_seleccionadas:
        train_sub[col] = pd.to_numeric(train_sub[col], errors='coerce').fillna(0).astype(float)
        test_sub[col] = pd.to_numeric(test_sub[col], errors='coerce').fillna(0).astype(float)

    df_control = train_sub[train_sub[columna_objetivo] == 0]
    df_ansiedad = train_sub[train_sub[columna_objetivo] == 1]
    
    if len(df_control) == 0 or len(df_ansiedad) == 0:
        return np.zeros(len(test_sub)), []
        
    n_control = len(df_control)
    sub_A, sub_B, sub_C = split_ansiedad_into_three(df_ansiedad, n_control, seed=42)
    
    df_train_A = pd.concat([df_control, sub_A])
    df_train_B = pd.concat([df_control, sub_B])
    df_train_C = pd.concat([df_control, sub_C])
    
    probs = []
    dags = []
    
    for idx_sub, df_sub in enumerate([df_train_A, df_train_B, df_train_C]):
        manta = []
        dag = None
        try:
            global CALL_COUNTER
            CALL_COUNTER = 0
            if algoritmo == "PC":
                est = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
                if white_list is not None and len(white_list) > 0:
                    est.fit(df_sub, fixed_edges=white_list)
                else:
                    est.fit(df_sub)
                dag = est.causal_graph_
            else:
                scoring_fn = RobustBICCondGauss(df_sub)
                est = GES(scoring_method=scoring_fn, return_type='dag')
                if white_list is not None and len(white_list) > 0:
                    est.fit(df_sub, fixed_edges=white_list)
                else:
                    est.fit(df_sub)
                dag = est.causal_graph_
                
            if columna_objetivo in dag.nodes():
                manta = dag.get_markov_blanket(columna_objetivo)
            dags.append(dag)
        except Exception:
            pass
            
        if len(manta) == 0:
            manta = caracteristicas_seleccionadas
            
        clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        clf.fit(df_sub[manta], df_sub[columna_objetivo])
        prob_sub = clf.predict_proba(test_sub[manta])[:, 1]
        probs.append(prob_sub)
        
    prob_promedio = np.mean(probs, axis=0)
    return prob_promedio, dags


def ejecutar_loso_ensamble(df_datos, caracteristicas_seleccionadas, grupos_desarrollo, algoritmo='GES'):
    particion = LeaveOneGroupOut()
    
    y_real = df_datos['Ansiedad'].values
    n_muestras = len(df_datos)
    
    prob_global = np.zeros(n_muestras)
    prob_aritmetica = np.zeros(n_muestras)
    prob_espejo = np.zeros(n_muestras)
    prob_stroop = np.zeros(n_muestras)
    prob_transf_estructural = np.zeros(n_muestras)
    
    tareas = ['Aritmetica', 'Espejo', 'Stroop']
    
    total_pliegues = particion.get_n_splits(df_datos, y_real, grupos_desarrollo)
    
    barra = tqdm(
        particion.split(df_datos, y_real, grupos_desarrollo),
        total=total_pliegues,
        desc=f"LOSO Ensamble Red Bayesiana ({algoritmo})",
        unit="fold"
    )
    
    for idx_train, idx_test in barra:
        df_train = df_datos.iloc[idx_train].copy()
        df_test = df_datos.iloc[idx_test].copy()
        
        # 1. Entrenar Modelo Global
        p_glob, dags_glob = entrenar_modelo_mixto_probabilidades(df_train, df_test, caracteristicas_seleccionadas, 'Ansiedad', algoritmo)
        prob_global[idx_test] = p_glob
        
        # Extraer arcos globales para transferencia estructural.
        arcos_globales = set()
        for d in dags_glob:
            if d is not None:
                arcos_globales.update(d.edges())
                
        # 3. Entrenar Modelos Específicos por Tarea
        p_tareas_test = {}
        p_transf_list = []
        
        for tarea in tareas:
            df_train_tarea = df_train[(df_train['Tarea'] == 'Relajacion') | (df_train['Tarea'] == tarea)].copy()
            
            p_t, _ = entrenar_modelo_mixto_probabilidades(df_train_tarea, df_test, caracteristicas_seleccionadas, 'Ansiedad', algoritmo)
            p_tareas_test[tarea] = p_t
            
            mascara_test_tarea = (df_test['Tarea'] == tarea)
            if mascara_test_tarea.any():
                if tarea == 'Aritmetica': prob_aritmetica[idx_test[mascara_test_tarea]] = p_t[mascara_test_tarea]
                elif tarea == 'Espejo': prob_espejo[idx_test[mascara_test_tarea]] = p_t[mascara_test_tarea]
                elif tarea == 'Stroop': prob_stroop[idx_test[mascara_test_tarea]] = p_t[mascara_test_tarea]
                
            p_t_prior, _ = entrenar_modelo_mixto_probabilidades(df_train_tarea, df_test, caracteristicas_seleccionadas, 'Ansiedad', algoritmo, white_list=arcos_globales)
            p_transf_list.append(p_t_prior)
            
        mascara_control = (df_test['Tarea'] == 'Relajacion')
        if mascara_control.any():
            p_control_mean = (p_tareas_test['Aritmetica'] + p_tareas_test['Espejo'] + p_tareas_test['Stroop']) / 3.0
            prob_aritmetica[idx_test[mascara_control]] = p_control_mean[mascara_control]
            prob_espejo[idx_test[mascara_control]] = p_control_mean[mascara_control]
            prob_stroop[idx_test[mascara_control]] = p_control_mean[mascara_control]

        prob_transf_estructural[idx_test] = np.mean(p_transf_list, axis=0)

    # El stacking queda fuera de la evaluación: el metaclasificador no dispone
    # aquí de una segunda capa de predicciones fuera de muestra independiente.

    resultados_ponderados = []
    prob_tarea_asignada = np.zeros(n_muestras)
    for i in range(n_muestras):
        t = df_datos.iloc[i]['Tarea']
        if t == 'Aritmetica': prob_tarea_asignada[i] = prob_aritmetica[i]
        elif t == 'Espejo': prob_tarea_asignada[i] = prob_espejo[i]
        elif t == 'Stroop': prob_tarea_asignada[i] = prob_stroop[i]
        else: prob_tarea_asignada[i] = (prob_aritmetica[i] + prob_espejo[i] + prob_stroop[i]) / 3.0
        
    pesos_w = np.linspace(0.0, 1.0, 11)
    
    for w in pesos_w:
        p_ensamble_w = w * prob_tarea_asignada + (1.0 - w) * prob_global
        y_pred_w = (p_ensamble_w >= 0.5).astype(int)
        
        acc = accuracy_score(y_real, y_pred_w)
        sens = recall_score(y_real, y_pred_w, zero_division=0)
        f1 = f1_score(y_real, y_pred_w, zero_division=0)
        prec = precision_score(y_real, y_pred_w, zero_division=0)
        tn, fp, _, _ = confusion_matrix(y_real, y_pred_w, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        resultados_ponderados.append({
            'Peso_w': round(w, 2),
            'Exactitud': acc,
            'Sensibilidad': sens,
            'Especificidad': spec,
            'Precisión': prec,
            'F1_Score': f1
        })

    res_stacking = None

    y_pred_transf = (prob_transf_estructural >= 0.5).astype(int)
    acc_transf = accuracy_score(y_real, y_pred_transf)
    sens_transf = recall_score(y_real, y_pred_transf, zero_division=0)
    prec_transf = precision_score(y_real, y_pred_transf, zero_division=0)
    f1_transf = f1_score(y_real, y_pred_transf, zero_division=0)
    tn_t, fp_t, _, _ = confusion_matrix(y_real, y_pred_transf, labels=[0, 1]).ravel()
    spec_transf = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0

    res_transferencia = {
        'Enfoque': 'Transferencia_Estructural_B',
        'Exactitud': acc_transf,
        'Sensibilidad': sens_transf,
        'Especificidad': spec_transf,
        'Precisión': prec_transf,
        'F1_Score': f1_transf
    }

    return pd.DataFrame(resultados_ponderados), res_stacking, res_transferencia


def principal():
    dir_resultados = os.path.join('Resultados', 'Red Bayesiana', 'Ensamble')
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
    sujetos_prueba = sujetos_mezclados[32:]
    
    df_desarrollo = df_datos[df_datos['Sujeto'].isin(sujetos_desarrollo)].copy()
    df_prueba_externa = df_datos[df_datos['Sujeto'].isin(sujetos_prueba)].copy()
    grupos_desarrollo = df_desarrollo['Sujeto'].values
    
    df_ranking = pd.read_csv(archivo_ranking)
    df_ranking = df_ranking.sort_values(by='Importancia_DT', ascending=False)
    mejores_caracteristicas = df_ranking['Caracteristica'].tolist()
    
    resultados_globales = []
    
    for n_features in [10, 15]:
        top_features = mejores_caracteristicas[:n_features]
        for algoritmo in ["GES", "PC"]:
            df_ponderado, res_stack, res_transf = ejecutar_loso_ensamble(df_desarrollo, top_features, grupos_desarrollo, algoritmo=algoritmo)
            
            p_glob_test, dags_glob_test = entrenar_modelo_mixto_probabilidades(df_desarrollo, df_prueba_externa, top_features, 'Ansiedad', algoritmo)
            if len(dags_glob_test) > 0 and dags_glob_test[0] is not None:
                graficar_dag(
                    dags_glob_test[0], 
                    f'Red Bayesiana Ensamble Global ({algoritmo} - Top {n_features})', 
                    os.path.join(dir_resultados, f'Red_Ensamble_Global_Top{n_features}_{algoritmo}.png')
                )
            p_tarea_aligned_test = np.zeros(len(df_prueba_externa))
            probabilidades_tarea_test = []
            for tarea in ['Aritmetica', 'Espejo', 'Stroop']:
                df_train_t = df_desarrollo[(df_desarrollo['Tarea'] == 'Relajacion') | (df_desarrollo['Tarea'] == tarea)].copy()
                p_t_ext, _ = entrenar_modelo_mixto_probabilidades(df_train_t, df_prueba_externa, top_features, 'Ansiedad', algoritmo)
                probabilidades_tarea_test.append(p_t_ext)
                mask_t = (df_prueba_externa['Tarea'] == tarea)
                if mask_t.any():
                    p_tarea_aligned_test[mask_t] = p_t_ext[mask_t]
                    
            mask_ctrl_ext = (df_prueba_externa['Tarea'] == 'Relajacion')
            if mask_ctrl_ext.any():
                p_control_mean = np.mean(probabilidades_tarea_test, axis=0)
                p_tarea_aligned_test[mask_ctrl_ext] = p_control_mean[mask_ctrl_ext]
            
            idx_w_opt = df_ponderado['F1_Score'].idxmax()
            w_opt = df_ponderado.loc[idx_w_opt, 'Peso_w']
            val_acc_opt = df_ponderado.loc[idx_w_opt, 'Exactitud']
            val_sens_opt = df_ponderado.loc[idx_w_opt, 'Sensibilidad']
            val_spec_opt = df_ponderado.loc[idx_w_opt, 'Especificidad']
            val_prec_opt = df_ponderado.loc[idx_w_opt, 'Precisión']
            val_f1_opt = df_ponderado.loc[idx_w_opt, 'F1_Score']
            
            # Aplicar a prueba externa exactamente la misma regla optimizada
            # en LOGO: combinación tarea-específica y global con un solo peso.
            p_test_w_opt = w_opt * p_tarea_aligned_test + (1.0 - w_opt) * p_glob_test
            y_test_real = df_prueba_externa['Ansiedad'].values.astype(int)
            y_test_pred = (p_test_w_opt >= 0.50).astype(int)
            
            test_acc = accuracy_score(y_test_real, y_test_pred)
            test_sens = recall_score(y_test_real, y_test_pred, zero_division=0)
            test_prec = precision_score(y_test_real, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test_real, y_test_pred, zero_division=0)
            tn_t, fp_t, _, _ = confusion_matrix(y_test_real, y_test_pred, labels=[0, 1]).ravel()
            test_spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
            
            alg_desc = f'Ensamble Ponderado w={w_opt} ({algoritmo})'
            resultados_globales.append({
                'Estructura': alg_desc,
                'Top': n_features,
                'Exact. (Val)': val_acc_opt,
                'Exact. (Test)': test_acc,
                'Prec. (Val)': val_prec_opt,
                'Prec. (Test)': test_prec,
                'Sens. (Val)': val_sens_opt,
                'Sens. (Test)': test_sens,
                'Esp. (Val)': val_spec_opt,
                'Esp. (Test)': test_spec,
                'F1 (Val)': val_f1_opt,
                'F1 (Test)': test_f1,
                'CS': np.nan,
                'FisherC_p': np.nan,
                'FisherC_RMSEA': np.nan
            })

    df_consolidado = pd.DataFrame(resultados_globales)
    csv_consolidado = os.path.join(dir_resultados, "Resultados_Red_Ensamble.csv")
    df_consolidado.to_csv(csv_consolidado, index=False)
    print(f"=== Archivo de resultados guardado exitosamente en: {csv_consolidado} ===")


if __name__ == "__main__":
    principal()
