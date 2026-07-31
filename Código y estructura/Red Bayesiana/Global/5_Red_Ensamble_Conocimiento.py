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
    """
    Entrena una Red Bayesiana mixta con sub-muestreo balanceado en 3 partes,
    obtiene la manta de Markov y devuelve las probabilidades estimadas para df_prueba.
    """
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
                est.fit(df_sub)
                dag = est.causal_graph_
            else:
                scoring_fn = RobustBICCondGauss(df_sub)
                est = GES(scoring_method=scoring_fn, return_type='dag')
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
    """
    Ejecuta la validación LOSO construyendo modelos globales y específicos por tarea,
    y evalúa las 3 estrategias de ensamble (Ponderado, Transferencia Estructural y Stacking).
    """
    particion = LeaveOneGroupOut()
    
    y_real = df_datos['Ansiedad'].values
    n_muestras = len(df_datos)
    
    prob_global = np.zeros(n_muestras)
    prob_aritmetica = np.zeros(n_muestras)
    prob_espejo = np.zeros(n_muestras)
    prob_stroop = np.zeros(n_muestras)
    
    prob_stacking = np.zeros(n_muestras)
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
        
        # Extracción de arcos globales para Transferencia Estructural (Enfoque B)
        arcos_globales = set()
        for d in dags_glob:
            if d is not None:
                arcos_globales.update(d.edges())
                
        # 2. Entrenar Modelos Específicos por Tarea
        p_tareas_test = {}
        p_tareas_train = {}
        
        for tarea in tareas:
            df_train_tarea = df_train[(df_train['Tarea'] == 'Relajacion') | (df_train['Tarea'] == tarea)].copy()
            
            p_t, _ = entrenar_modelo_mixto_probabilidades(df_train_tarea, df_test, caracteristicas_seleccionadas, 'Ansiedad', algoritmo)
            p_tareas_test[tarea] = p_t
            
            mascara_test_tarea = (df_test['Tarea'] == tarea)
            if mascara_test_tarea.any():
                if tarea == 'Aritmetica': prob_aritmetica[idx_test[mascara_test_tarea]] = p_t[mascara_test_tarea]
                elif tarea == 'Espejo': prob_espejo[idx_test[mascara_test_tarea]] = p_t[mascara_test_tarea]
                elif tarea == 'Stroop': prob_stroop[idx_test[mascara_test_tarea]] = p_t[mascara_test_tarea]
                
            p_t_train, _ = entrenar_modelo_mixto_probabilidades(df_train_tarea, df_train, caracteristicas_seleccionadas, 'Ansiedad', algoritmo)
            p_tareas_train[tarea] = p_t_train
            
        mascara_control = (df_test['Tarea'] == 'Relajacion')
        if mascara_control.any():
            p_control_mean = (p_tareas_test['Aritmetica'] + p_tareas_test['Espejo'] + p_tareas_test['Stroop']) / 3.0
            prob_aritmetica[idx_test[mascara_control]] = p_control_mean[mascara_control]
            prob_espejo[idx_test[mascara_control]] = p_control_mean[mascara_control]
            prob_stroop[idx_test[mascara_control]] = p_control_mean[mascara_control]

        # --- Enfoque C: Meta-Clasificador Jerárquico (Stacking) ---
        p_glob_train, _ = entrenar_modelo_mixto_probabilidades(df_train, df_train, caracteristicas_seleccionadas, 'Ansiedad', algoritmo)
        X_meta_train = np.column_stack([
            p_glob_train,
            p_tareas_train['Aritmetica'],
            p_tareas_train['Espejo'],
            p_tareas_train['Stroop']
        ])
        y_meta_train = df_train['Ansiedad'].values
        
        meta_clf = LogisticRegression(penalty='l2', C=1.0, random_state=42)
        meta_clf.fit(X_meta_train, y_meta_train)
        
        X_meta_test = np.column_stack([
            p_glob,
            p_tareas_test['Aritmetica'],
            p_tareas_test['Espejo'],
            p_tareas_test['Stroop']
        ])
        prob_stacking[idx_test] = meta_clf.predict_proba(X_meta_test)[:, 1]

        # --- Enfoque B: Transferencia Estructural (Priorización de DAG) ---
        p_transf_list = []
        for tarea in tareas:
            df_train_tarea = df_train[(df_train['Tarea'] == 'Relajacion') | (df_train['Tarea'] == tarea)].copy()
            p_t_prior, _ = entrenar_modelo_mixto_probabilidades(df_train_tarea, df_test, caracteristicas_seleccionadas, 'Ansiedad', algoritmo, white_list=arcos_globales)
            p_transf_list.append(p_t_prior)
            
        prob_transf_estructural[idx_test] = np.mean(p_transf_list, axis=0)

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

    y_pred_stack = (prob_stacking >= 0.5).astype(int)
    acc_stack = accuracy_score(y_real, y_pred_stack)
    sens_stack = recall_score(y_real, y_pred_stack, zero_division=0)
    prec_stack = precision_score(y_real, y_pred_stack, zero_division=0)
    f1_stack = f1_score(y_real, y_pred_stack, zero_division=0)
    tn_s, fp_s, _, _ = confusion_matrix(y_real, y_pred_stack, labels=[0, 1]).ravel()
    spec_stack = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0
    
    res_stacking = {
        'Enfoque': 'Stacking_Jerarquico_C',
        'Exactitud': acc_stack,
        'Sensibilidad': sens_stack,
        'Especificidad': spec_stack,
        'Precisión': prec_stack,
        'F1_Score': f1_stack
    }

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
    print("=== Iniciando Entrenamiento y Evaluación de la Red de Ensamble de Conocimiento (Punto 4) ===")
    
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
    
    df_desarrollo = df_datos[df_datos['Sujeto'].isin(sujetos_desarrollo)].copy()
    grupos_desarrollo = df_desarrollo['Sujeto'].values
    
    df_ranking = pd.read_csv(archivo_ranking)
    df_ranking = df_ranking.sort_values(by='Importancia_DT', ascending=False)
    mejores_caracteristicas = df_ranking['Caracteristica'].tolist()
    
    top_features = mejores_caracteristicas[:15]
    
    for algoritmo in ["GES", "PC"]:
        print(f"\n--- Evaluando Ensamble de Conocimiento con algoritmo {algoritmo} ---")
        df_ponderado, res_stack, res_transf = ejecutar_loso_ensamble(df_desarrollo, top_features, grupos_desarrollo, algoritmo=algoritmo)
        
        csv_pond = os.path.join(dir_resultados, f"Resultados_Ensamble_Ponderado_{algoritmo}.csv")
        df_ponderado.to_csv(csv_pond, index=False)
        
        df_otros = pd.DataFrame([res_stack, res_transf])
        csv_otros = os.path.join(dir_resultados, f"Resultados_Ensamble_Stacking_Transferencia_{algoritmo}.csv")
        df_otros.to_csv(csv_otros, index=False)
        
        plt.figure(figsize=(9, 5))
        plt.plot(df_ponderado['Peso_w'], df_ponderado['Exactitud'], marker='o', label='Exactitud', color='navy', linewidth=2)
        plt.plot(df_ponderado['Peso_w'], df_ponderado['Sensibilidad'], marker='s', label='Sensibilidad', color='darkgreen', linewidth=2)
        plt.plot(df_ponderado['Peso_w'], df_ponderado['Especificidad'], marker='^', label='Especificidad', color='darkred', linewidth=2)
        plt.plot(df_ponderado['Peso_w'], df_ponderado['F1_Score'], marker='d', label='F1-Score', color='purple', linewidth=2)
        plt.axvline(x=0.0, color='gray', linestyle='--', alpha=0.7, label='Solo Modelo Global (w=0)')
        plt.axvline(x=1.0, color='orange', linestyle='--', alpha=0.7, label='Solo Modelo por Tareas (w=1)')
        
        plt.title(f'Desempeño del Ensamble Ponderado vs Peso w ({algoritmo})', fontsize=12, fontweight='bold')
        plt.xlabel('Peso w (Aporte de Redes por Tarea)', fontsize=10)
        plt.ylabel('Puntaje de Métrica', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=9, loc='lower right')
        plt.tight_layout()
        
        grafico_out = os.path.join(dir_resultados, f"Grafico_Ensamble_Ponderado_{algoritmo}.png")
        plt.savefig(grafico_out, dpi=300)
        plt.close()
        
        print(f"Resultados guardados exitosamente en '{dir_resultados}'.")

if __name__ == "__main__":
    principal()
