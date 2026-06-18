import os
import warnings
warnings.filterwarnings('ignore')  # Silenciar advertencias de dependencias externas

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Forzar backend no interactivo para guardado de plots sin GUI
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.auto import tqdm as _tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pgmpy.causal_discovery import PC, GES
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.parameter_estimator import LinearGaussianMLE

# Ajustar visualización de la barra de progreso
def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)


# GRAFICADO DE REDES BAYESIANAS 
def graficar_dag(dag, titulo, nombre_archivo):
    """Genera y guarda el diagrama de la red causal utilizando NetworkX y Matplotlib."""
    plt.figure(figsize=(12, 10))
    G = nx.DiGraph(dag.edges())
    G.add_nodes_from(dag.nodes())
    pos = nx.spring_layout(G, seed=42, k=1.5)
    
    # Dibujar nodos de la red
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=1500, 
        node_color='lightblue', 
        edgecolors='darkblue', 
        linewidths=1.5
    )
    # Dibujar etiquetas de los canales/variables
    nx.draw_networkx_labels(
        G, pos, 
        font_size=8, 
        font_family='sans-serif', 
        font_weight='bold'
    )
    # Dibujar aristas con dirección
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=list(G.edges()), 
        edge_color='navy', 
        arrowstyle='->',
        arrowsize=20, 
        width=1.5, 
        alpha=0.8
    )
    
    plt.title(titulo, fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()


# VALIDACIÓN INTERNA CON LEAVE-ONE-SUBJECT-OUT (LOSO)
def ejecutar_loso(df_datos, caracteristicas_seleccionadas, columna_objetivo, grupos, algoritmo):
    """Ejecuta la validación cruzada interna LOSO para medir desempeño del algoritmo seleccionado."""
    particion_grupos = LeaveOneGroupOut()
    y_real = df_datos[columna_objetivo].values
    y_pred = np.zeros(len(df_datos))
    
    # Filtrar solo características seleccionadas y variable objetivo, previniendo nulos
    columnas_subconjunto = caracteristicas_seleccionadas + [columna_objetivo]
    datos_loso = df_datos[columnas_subconjunto].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    
    total_pliegues = particion_grupos.get_n_splits(datos_loso, y_real, grupos)
    barra_progreso = tqdm(
        particion_grupos.split(datos_loso, y_real, grupos), 
        total=total_pliegues, 
        desc=f"LOSO {algoritmo}", 
        unit='fold', 
        leave=False
    )
    
    # Ciclo de entrenamiento y prueba por cada sujeto de desarrollo
    for indices_entrenamiento, indices_prueba in barra_progreso:
        df_entrenamiento = datos_loso.iloc[indices_entrenamiento]
        df_prueba = datos_loso.iloc[indices_prueba]
        
        try:
            # Aprendizaje de estructura causal (DAG)
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
            else:  # GES
                estimador = GES(
                    scoring_method='bic-g', 
                    return_type='dag'
                )
                estimador.fit(df_entrenamiento)
                dag = estimador.causal_graph_
            
            # Ajuste de parámetros cuantitativos (MLE) e inferencia
            modelo = LinearGaussianBayesianNetwork(dag.edges())
            modelo.add_nodes_from(dag.nodes())
            modelo.fit(df_entrenamiento)
            
            X_prueba = df_prueba.drop(columns=[columna_objetivo])
            predicciones = modelo.predict(X_prueba)
            y_pred[indices_prueba] = predicciones[columna_objetivo].values
            
        except Exception:
            # Fallback en caso de desconexión del grafo o error de ajuste
            y_pred[indices_prueba] = df_entrenamiento[columna_objetivo].mean()
            
    # Binarización de la predicción continua con umbral 0.5
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_real_bin = y_real.astype(int)
    
    return y_real_bin, y_pred_bin


# ORQUESTACIÓN Y PIPELINE PRINCIPAL DE EVALUACIÓN
def principal():
    # Estructura de directorios
    dir_resultados = os.path.join('Resultados', 'Red Bayesiana')
    os.makedirs(dir_resultados, exist_ok=True)
    
    archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')
    
    if not os.path.exists(archivo_datos) or not os.path.exists(archivo_ranking):
        print("Error: No se encontraron los archivos necesarios en Resultados.")
        return
        
    df_datos = pd.read_parquet(archivo_datos)
    # Filtrar registros correspondientes a ansiedad (grupo control = 0, y casos >= 1)
    df_datos = df_datos[(df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1)].copy()
    df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0.0 if x == 0 else 1.0)
    
    # Partición determinista por sujetos (80% Desarrollo / 20% Test Externo)
    sujetos_unicos = np.sort(df_datos['Sujeto'].unique())
    np.random.seed(42)
    sujetos_mezclados = np.random.permutation(sujetos_unicos)
    sujetos_desarrollo = sujetos_mezclados[:32]
    sujetos_prueba_externa = sujetos_mezclados[32:]
    
    df_desarrollo = df_datos[df_datos['Sujeto'].isin(sujetos_desarrollo)].copy()
    df_prueba_externa = df_datos[df_datos['Sujeto'].isin(sujetos_prueba_externa)].copy()
    
    grupos_desarrollo = df_desarrollo['Sujeto'].values
    
    # Cargar y ordenar ranking multicriterio de características
    df_ranking = pd.read_csv(archivo_ranking)
    df_ranking = df_ranking.sort_values(by=['Importancia_DT', 'Mutual_Info'], ascending=[False, False])
    mejores_caracteristicas = df_ranking['Caracteristica'].tolist()
    
    tamanos_top = [10, 15]
    resultados_loso = []
    
    # Bucle principal de evaluación cruzada y evaluación externa
    for n_caracteristicas in tamanos_top:
        caracteristicas = mejores_caracteristicas[:n_caracteristicas]
        
        # Algoritmo PC (Constraint-based)
        # Validación interna en Desarrollo (LOSO)
        y_real_pc, y_pred_pc = ejecutar_loso(df_desarrollo, caracteristicas, 'Ansiedad', grupos_desarrollo, "PC")
        
        val_acc_pc = accuracy_score(y_real_pc, y_pred_pc)
        val_prec_pc = precision_score(y_real_pc, y_pred_pc, zero_division=0)
        val_sens_pc = recall_score(y_real_pc, y_pred_pc, zero_division=0)
        val_f1_pc = f1_score(y_real_pc, y_pred_pc, zero_division=0)
        matriz_confusion_pc = confusion_matrix(y_real_pc, y_pred_pc, labels=[0, 1])
        tn_pc, fp_pc, _, _ = matriz_confusion_pc.ravel()
        val_spec_pc = tn_pc / (tn_pc + fp_pc) if (tn_pc + fp_pc) > 0 else 0.0
        
        # Test final en conjunto Holdout (Prueba Externa)
        columnas_desarrollo = caracteristicas + ['Ansiedad']
        df_desarrollo_sub = df_desarrollo[columnas_desarrollo].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
        df_prueba_externa_sub = df_prueba_externa[columnas_desarrollo].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
        
        try:
            pc_final = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
            pc_final.fit(df_desarrollo_sub)
            dag_pc = pc_final.causal_graph_
            
            modelo_pc = LinearGaussianBayesianNetwork(dag_pc.edges())
            modelo_pc.add_nodes_from(dag_pc.nodes())
            modelo_pc.fit(df_desarrollo_sub)
            
            predicciones_pc = modelo_pc.predict(df_prueba_externa_sub.drop(columns=['Ansiedad']))
            y_prueba_externa_pred_pc = (predicciones_pc['Ansiedad'].values >= 0.5).astype(int)
        except Exception:
            y_prueba_externa_pred_pc = (np.repeat(df_desarrollo_sub['Ansiedad'].mean(), len(df_prueba_externa_sub)) >= 0.5).astype(int)
            
        y_prueba_externa_real = df_prueba_externa_sub['Ansiedad'].values.astype(int)
        test_acc_pc = accuracy_score(y_prueba_externa_real, y_prueba_externa_pred_pc)
        test_prec_pc = precision_score(y_prueba_externa_real, y_prueba_externa_pred_pc, zero_division=0)
        test_sens_pc = recall_score(y_prueba_externa_real, y_prueba_externa_pred_pc, zero_division=0)
        test_f1_pc = f1_score(y_prueba_externa_real, y_prueba_externa_pred_pc, zero_division=0)
        matriz_confusion_test_pc = confusion_matrix(y_prueba_externa_real, y_prueba_externa_pred_pc, labels=[0, 1])
        tn_tpc, fp_tpc, _, _ = matriz_confusion_test_pc.ravel()
        test_spec_pc = tn_tpc / (tn_tpc + fp_tpc) if (tn_tpc + fp_tpc) > 0 else 0.0
        
        resultados_loso.append({
            'Algoritmo_Estructura': 'PC (fisher_z)',
            'Top_Features': n_caracteristicas,
            'Val_Exactitud': val_acc_pc,
            'Val_Precisión': val_prec_pc,
            'Val_Sensibilidad': val_sens_pc,
            'Val_Especificidad': val_spec_pc,
            'Val_F1_Score': val_f1_pc,
            'Test_Exactitud': test_acc_pc,
            'Test_Precisión': test_prec_pc,
            'Test_Sensibilidad': test_sens_pc,
            'Test_Especificidad': test_spec_pc,
            'Test_F1_Score': test_f1_pc
        })
        
        # Algoritmo GES (Score-based)
        # Validación interna en Desarrollo (LOSO)
        y_real_ges, y_pred_ges = ejecutar_loso(df_desarrollo, caracteristicas, 'Ansiedad', grupos_desarrollo, "GES")
        
        val_acc_ges = accuracy_score(y_real_ges, y_pred_ges)
        val_prec_ges = precision_score(y_real_ges, y_pred_ges, zero_division=0)
        val_sens_ges = recall_score(y_real_ges, y_pred_ges, zero_division=0)
        val_f1_ges = f1_score(y_real_ges, y_pred_ges, zero_division=0)
        matriz_confusion_ges = confusion_matrix(y_real_ges, y_pred_ges, labels=[0, 1])
        tn_ges, fp_ges, _, _ = matriz_confusion_ges.ravel()
        val_spec_ges = tn_ges / (tn_ges + fp_ges) if (tn_ges + fp_ges) > 0 else 0.0
        
        # Test final en conjunto Holdout (Prueba Externa)
        try:
            ges_final = GES(scoring_method='bic-g', return_type='dag')
            ges_final.fit(df_desarrollo_sub)
            dag_ges = ges_final.causal_graph_
            
            modelo_ges = LinearGaussianBayesianNetwork(dag_ges.edges())
            modelo_ges.add_nodes_from(dag_ges.nodes())
            modelo_ges.fit(df_desarrollo_sub)
            
            predicciones_ges = modelo_ges.predict(df_prueba_externa_sub.drop(columns=['Ansiedad']))
            y_prueba_externa_pred_ges = (predicciones_ges['Ansiedad'].values >= 0.5).astype(int)
        except Exception:
            y_prueba_externa_pred_ges = (np.repeat(df_desarrollo_sub['Ansiedad'].mean(), len(df_prueba_externa_sub)) >= 0.5).astype(int)
            
        test_acc_ges = accuracy_score(y_prueba_externa_real, y_prueba_externa_pred_ges)
        test_prec_ges = precision_score(y_prueba_externa_real, y_prueba_externa_pred_ges, zero_division=0)
        test_sens_ges = recall_score(y_prueba_externa_real, y_prueba_externa_pred_ges, zero_division=0)
        test_f1_ges = f1_score(y_prueba_externa_real, y_prueba_externa_pred_ges, zero_division=0)
        matriz_confusion_test_ges = confusion_matrix(y_prueba_externa_real, y_prueba_externa_pred_ges, labels=[0, 1])
        tn_tges, fp_tges, _, _ = matriz_confusion_test_ges.ravel()
        test_spec_ges = tn_tges / (tn_tges + fp_tges) if (tn_tges + fp_tges) > 0 else 0.0
        
        resultados_loso.append({
            'Algoritmo_Estructura': 'GES (bic-g)',
            'Top_Features': n_caracteristicas,
            'Val_Exactitud': val_acc_ges,
            'Val_Precisión': val_prec_ges,
            'Val_Sensibilidad': val_sens_ges,
            'Val_Especificidad': val_spec_ges,
            'Val_F1_Score': val_f1_ges,
            'Test_Exactitud': test_acc_ges,
            'Test_Precisión': test_prec_ges,
            'Test_Sensibilidad': test_sens_ges,
            'Test_Especificidad': test_spec_ges,
            'Test_F1_Score': test_f1_ges
        })
        
        # Exportación de Gráficos de Redes Globales
        try:
            graficar_dag(
                dag_pc, 
                f"Red Causal Continua (PC, Top-{n_caracteristicas})", 
                os.path.join(dir_resultados, f"Estructura_Global_PC_top{n_caracteristicas}.png")
            )
        except Exception:
            pass
            
        try:
            graficar_dag(
                dag_ges, 
                f"Red Causal Continua (GES, Top-{n_caracteristicas})", 
                os.path.join(dir_resultados, f"Estructura_Global_GES_top{n_caracteristicas}.png")
            )
        except Exception:
            pass

    # Guardado y visualización de resultados tabulares
    df_resultados = pd.DataFrame(resultados_loso)
    ruta_salida = os.path.join(dir_resultados, 'Resultados_Comparativa_RB.csv')
    df_resultados.to_csv(ruta_salida, index=False)
    
    print(df_resultados.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    principal()
