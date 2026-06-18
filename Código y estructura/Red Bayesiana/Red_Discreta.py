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
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pgmpy.causal_discovery import PC, HillClimbSearch
from pgmpy.structure_score import BIC
from pgmpy.models import DiscreteBayesianNetwork

# Ajustar visualización de la barra de progreso
def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)


# GRAFICADO DE REDES BAYESIANAS (DAGs)
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
        node_color='lightgreen', 
        edgecolors='darkgreen', 
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
        edge_color='darkgreen', 
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


# VALIDACIÓN INTERNA CON LEAVE-ONE-SUBJECT-OUT (LOSO) Y DISCRETIZACIÓN
def ejecutar_loso(df_datos, caracteristicas_seleccionadas, columna_objetivo, grupos, algoritmo, n_intervalos):
    """Ejecuta validación interna LOSO discretizando dinámicamente en cada pliegue para evitar fuga."""
    particion_grupos = LeaveOneGroupOut()
    y_real = df_datos[columna_objetivo].values
    y_pred = np.zeros(len(df_datos))
    
    total_pliegues = particion_grupos.get_n_splits(df_datos, y_real, grupos)
    barra_progreso = tqdm(
        particion_grupos.split(df_datos, y_real, grupos), 
        total=total_pliegues, 
        desc=f"LOSO {algoritmo} (bins={n_intervalos})", 
        unit='fold', 
        leave=False
    )
    
    # Ciclo de validación cruzada por sujeto
    for indices_entrenamiento, indices_prueba in barra_progreso:
        df_entrenamiento_crudo = df_datos.iloc[indices_entrenamiento]
        df_prueba_crudo = df_datos.iloc[indices_prueba]
        
        # Ajustar discretizador únicamente con datos de entrenamiento del fold
        discretizador = KBinsDiscretizer(n_bins=n_intervalos, encode='ordinal', strategy='quantile')
        
        caracteristicas_entrenamiento_disc = discretizador.fit_transform(df_entrenamiento_crudo[caracteristicas_seleccionadas])
        caracteristicas_prueba_disc = discretizador.transform(df_prueba_crudo[caracteristicas_seleccionadas])
        
        # Construcción de dataframes discretizados e indexados
        df_entrenamiento = pd.DataFrame(caracteristicas_entrenamiento_disc, columns=caracteristicas_seleccionadas, index=df_entrenamiento_crudo.index).astype(int)
        df_entrenamiento[columna_objetivo] = df_entrenamiento_crudo[columna_objetivo].values.astype(int)
        
        df_prueba = pd.DataFrame(caracteristicas_prueba_disc, columns=caracteristicas_seleccionadas, index=df_prueba_crudo.index).astype(int)
        
        try:
            # Aprendizaje de estructura del DAG
            if algoritmo == "PC":
                estimador = PC(
                    variant='stable', 
                    ci_test='chi_square', 
                    significance_level=0.05, 
                    return_type='dag', 
                    show_progress=False
                )
                estimador.fit(df_entrenamiento)
                dag = estimador.causal_graph_
            else:  # HC
                hc = HillClimbSearch(scoring_method=BIC(df_entrenamiento), show_progress=False)
                hc.fit(df_entrenamiento)
                dag = hc.causal_graph_
            
            # Ajuste de tablas CPT del modelo discreto
            modelo = DiscreteBayesianNetwork(dag.edges())
            modelo.add_nodes_from(dag.nodes())
            modelo.fit(df_entrenamiento)
            
            predicciones = modelo.predict(df_prueba)
            y_pred[indices_prueba] = predicciones[columna_objetivo].values
            
        except Exception:
            # Fallback a la moda en caso de fallos de inferencia o desconexión
            y_pred[indices_prueba] = df_entrenamiento[columna_objetivo].mode().values[0]
            
    y_pred_bin = y_pred.astype(int)
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
    df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0 if x == 0 else 1)
    
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
    
    tamanos_intervalos = [2, 3]
    top_tamanos = [10, 15]
    resultados_loso = []
    
    # Bucle principal sobre número de intervalos y top características
    for n_intervalos in tamanos_intervalos:
        for n_caracteristicas in top_tamanos:
            caracteristicas = mejores_caracteristicas[:n_caracteristicas]
            
            # Algoritmo PC (Constraint-based)
            y_real_pc, y_pred_pc = ejecutar_loso(df_desarrollo, caracteristicas, 'Ansiedad', grupos_desarrollo, "PC", n_intervalos)
            
            val_acc_pc = accuracy_score(y_real_pc, y_pred_pc)
            val_prec_pc = precision_score(y_real_pc, y_pred_pc, zero_division=0)
            val_sens_pc = recall_score(y_real_pc, y_pred_pc, zero_division=0)
            val_f1_pc = f1_score(y_real_pc, y_pred_pc, zero_division=0)
            matriz_confusion_pc = confusion_matrix(y_real_pc, y_pred_pc, labels=[0, 1])
            tn_pc, fp_pc, _, _ = matriz_confusion_pc.ravel()
            val_spec_pc = tn_pc / (tn_pc + fp_pc) if (tn_pc + fp_pc) > 0 else 0.0
            
            # PC: Test en conjunto Holdout (Prueba Externa)
            discretizador_pc = KBinsDiscretizer(n_bins=n_intervalos, encode='ordinal', strategy='quantile')
            
            caract_dev_disc = discretizador_pc.fit_transform(df_desarrollo[caracteristicas])
            caract_prueba_ext_disc = discretizador_pc.transform(df_prueba_externa[caracteristicas])
            
            df_desarrollo_disc = pd.DataFrame(caract_dev_disc, columns=caracteristicas, index=df_desarrollo.index).astype(int)
            df_desarrollo_disc['Ansiedad'] = df_desarrollo['Ansiedad'].values.astype(int)
            
            df_prueba_externa_disc = pd.DataFrame(caract_prueba_ext_disc, columns=caracteristicas, index=df_prueba_externa.index).astype(int)
            
            try:
                pc_final = PC(variant='stable', ci_test='chi_square', significance_level=0.05, return_type='dag', show_progress=False)
                pc_final.fit(df_desarrollo_disc)
                dag_pc = pc_final.causal_graph_
                
                modelo_pc = DiscreteBayesianNetwork(dag_pc.edges())
                modelo_pc.add_nodes_from(dag_pc.nodes())
                modelo_pc.fit(df_desarrollo_disc)
                
                predicciones_pc = modelo_pc.predict(df_prueba_externa_disc)
                y_prueba_externa_pred_pc = predicciones_pc['Ansiedad'].values.astype(int)
            except Exception:
                y_prueba_externa_pred_pc = np.repeat(df_desarrollo_disc['Ansiedad'].mode().values[0], len(df_prueba_externa_disc))
                
            y_prueba_externa_real = df_prueba_externa['Ansiedad'].values.astype(int)
            test_acc_pc = accuracy_score(y_prueba_externa_real, y_prueba_externa_pred_pc)
            test_prec_pc = precision_score(y_prueba_externa_real, y_prueba_externa_pred_pc, zero_division=0)
            test_sens_pc = recall_score(y_prueba_externa_real, y_prueba_externa_pred_pc, zero_division=0)
            test_f1_pc = f1_score(y_prueba_externa_real, y_prueba_externa_pred_pc, zero_division=0)
            matriz_confusion_test_pc = confusion_matrix(y_prueba_externa_real, y_prueba_externa_pred_pc, labels=[0, 1])
            tn_tpc, fp_tpc, _, _ = matriz_confusion_test_pc.ravel()
            test_spec_pc = tn_tpc / (tn_tpc + fp_tpc) if (tn_tpc + fp_tpc) > 0 else 0.0
            
            resultados_loso.append({
                'Algoritmo_Estructura': 'PC (chi_square)',
                'n_bins': n_intervalos,
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
            
            # Algoritmo HC (Score-based Hill Climbing)
            y_real_hc, y_pred_hc = ejecutar_loso(df_desarrollo, caracteristicas, 'Ansiedad', grupos_desarrollo, "HC", n_intervalos)
            
            val_acc_hc = accuracy_score(y_real_hc, y_pred_hc)
            val_prec_hc = precision_score(y_real_hc, y_pred_hc, zero_division=0)
            val_sens_hc = recall_score(y_real_hc, y_pred_hc, zero_division=0)
            val_f1_hc = f1_score(y_real_hc, y_pred_hc, zero_division=0)
            matriz_confusion_hc = confusion_matrix(y_real_hc, y_pred_hc, labels=[0, 1])
            tn_hc, fp_hc, _, _ = matriz_confusion_hc.ravel()
            val_spec_hc = tn_hc / (tn_hc + fp_hc) if (tn_hc + fp_hc) > 0 else 0.0
            
            # Test en conjunto externo
            try:
                hc_final = HillClimbSearch(scoring_method=BIC(df_desarrollo_disc), show_progress=False)
                hc_final.fit(df_desarrollo_disc)
                dag_hc = hc_final.causal_graph_
                
                modelo_hc = DiscreteBayesianNetwork(dag_hc.edges())
                modelo_hc.add_nodes_from(dag_hc.nodes())
                modelo_hc.fit(df_desarrollo_disc)
                
                predicciones_hc = modelo_hc.predict(df_prueba_externa_disc)
                y_prueba_externa_pred_hc = predicciones_hc['Ansiedad'].values.astype(int)
            except Exception:
                y_prueba_externa_pred_hc = np.repeat(df_desarrollo_disc['Ansiedad'].mode().values[0], len(df_prueba_externa_disc))
                
            test_acc_hc = accuracy_score(y_prueba_externa_real, y_prueba_externa_pred_hc)
            test_prec_hc = precision_score(y_prueba_externa_real, y_prueba_externa_pred_hc, zero_division=0)
            test_sens_hc = recall_score(y_prueba_externa_real, y_prueba_externa_pred_hc, zero_division=0)
            test_f1_hc = f1_score(y_prueba_externa_real, y_prueba_externa_pred_hc, zero_division=0)
            matriz_confusion_test_hc = confusion_matrix(y_prueba_externa_real, y_prueba_externa_pred_hc, labels=[0, 1])
            tn_thc, fp_thc, _, _ = matriz_confusion_test_hc.ravel()
            test_spec_hc = tn_thc / (tn_thc + fp_thc) if (tn_thc + fp_thc) > 0 else 0.0
            
            resultados_loso.append({
                'Algoritmo_Estructura': 'HC (BIC)',
                'n_bins': n_intervalos,
                'Top_Features': n_caracteristicas,
                'Val_Exactitud': val_acc_hc,
                'Val_Precisión': val_prec_hc,
                'Val_Sensibilidad': val_sens_hc,
                'Val_Especificidad': val_spec_hc,
                'Val_F1_Score': val_f1_hc,
                'Test_Exactitud': test_acc_hc,
                'Test_Precisión': test_prec_hc,
                'Test_Sensibilidad': test_sens_hc,
                'Test_Especificidad': test_spec_hc,
                'Test_F1_Score': test_f1_hc
            })
            
            # Exportación de Gráficos de Redes Globales
            try:
                graficar_dag(
                    dag_pc, 
                    f"Red Causal Discreta (PC, Top-{n_caracteristicas}, Bins-{n_intervalos})", 
                    os.path.join(dir_resultados, f"Estructura_Global_Discreta_PC_top{n_caracteristicas}_bins{n_intervalos}.png")
                )
            except Exception:
                pass
                
            try:
                graficar_dag(
                    dag_hc, 
                    f"Red Causal Discreta (HC, Top-{n_caracteristicas}, Bins-{n_intervalos})", 
                    os.path.join(dir_resultados, f"Estructura_Global_Discreta_HC_top{n_caracteristicas}_bins{n_intervalos}.png")
                )
            except Exception:
                pass

    # Guardado y visualización de resultados tabulares
    df_resultados = pd.DataFrame(resultados_loso)
    ruta_salida = os.path.join(dir_resultados, 'Resultados_Comparativa_RB_Discreta.csv')
    df_resultados.to_csv(ruta_salida, index=False)
    
    print(df_resultados.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    principal()
