import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Evitar interfaces gráficas en entornos sin pantalla
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.auto import tqdm as _tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pgmpy.causal_discovery import PC, GES
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.parameter_estimator import LinearGaussianMLE

def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)

def plot_dag(dag, title, filename):
    plt.figure(figsize=(12, 10))
    
    G = nx.DiGraph(dag.edges())
    G.add_nodes_from(dag.nodes())
    
    pos = nx.spring_layout(G, seed=42, k=1.5)
    
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=1500, 
        node_color='lightblue', 
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
        edge_color='navy', 
        arrowstyle='->',
        arrowsize=20, 
        width=1.5, 
        alpha=0.8
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def ejecutar_loso(df_datos, selected_features, target_col, grupos, variant_name):
    logo = LeaveOneGroupOut()
    y_real = df_datos[target_col].values
    y_pred = np.zeros(len(df_datos))
    
    subset_cols = selected_features + [target_col]
    data_loso = df_datos[subset_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    
    total_folds = logo.get_n_splits(data_loso, y_real, grupos)
    
    pbar = tqdm(
        logo.split(data_loso, y_real, grupos), 
        total=total_folds, 
        desc=f"LOSO {variant_name}", 
        unit='fold', 
        leave=False
    )
    
    for train_idx, test_idx in pbar:
        df_train = data_loso.iloc[train_idx]
        df_test = data_loso.iloc[test_idx]
        
        try:
            # Búsqueda de Estructura
            if variant_name == "PC":
                est = PC(
                    variant='stable', 
                    ci_test='fisher_z', 
                    significance_level=0.05, 
                    return_type='dag', 
                    show_progress=False
                )
                est.fit(df_train)
                dag = est.causal_graph_
            else:  # GES
                est = GES(
                    scoring_method='bic-g', 
                    return_type='dag'
                )
                est.fit(df_train)
                dag = est.causal_graph_
            
            # Ajuste de Parámetros
            model = LinearGaussianBayesianNetwork(dag.edges())
            model.add_nodes_from(dag.nodes())
            model.fit(df_train)
            
            # Predicción en conjunto de prueba
            X_test = df_test.drop(columns=[target_col])
            pred = model.predict(X_test)
            y_pred[test_idx] = pred[target_col].values
            
        except Exception as e:
            # Fallback en caso de singularidades o errores numéricos
            y_pred[test_idx] = df_train[target_col].mean()
            
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_real_bin = y_real.astype(int)
    
    return y_real_bin, y_pred_bin

def main():
    # Definir rutas
    dir_resultados = os.path.join('Resultados', 'Red Bayesiana')
    os.makedirs(dir_resultados, exist_ok=True)
    
    archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')
    
    if not os.path.exists(archivo_datos) or not os.path.exists(archivo_ranking):
        print("Error: No se encontraron los archivos necesarios en Resultados.")
        return
        
    df_datos = pd.read_parquet(archivo_datos)
    # Filtrar: 0 para relajación, >= 1 para ansiedad
    df_datos = df_datos[(df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1)].copy()
    df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0.0 if x == 0 else 1.0)
    grupos = df_datos['Sujeto'].values
    
    # Cargar ranking y obtener características DT
    df_ranking = pd.read_csv(archivo_ranking)
    df_ranking = df_ranking.sort_values(by=['Importancia_DT', 'Mutual_Info'], ascending=[False, False])
    mejores_caracteristicas = df_ranking['Caracteristica'].tolist()
    
    top_tamanos = [10, 15, 20, 30, 40]
    resultados_loso = []
    
    # 2. Bucle de Validación Cruzada LOSO
    for n_features in top_tamanos:
        features = mejores_caracteristicas[:n_features]
        
        y_real_pc, y_pred_pc = ejecutar_loso(df_datos, features, 'Ansiedad', grupos, "PC")
        
        # Calcular métricas para PC
        acc_pc = accuracy_score(y_real_pc, y_pred_pc)
        prec_pc = precision_score(y_real_pc, y_pred_pc, zero_division=0)
        sens_pc = recall_score(y_real_pc, y_pred_pc, zero_division=0)
        f1_pc = f1_score(y_real_pc, y_pred_pc, zero_division=0)
        
        cm_pc = confusion_matrix(y_real_pc, y_pred_pc, labels=[0, 1])
        tn_pc, fp_pc, _, _ = cm_pc.ravel()
        spec_pc = tn_pc / (tn_pc + fp_pc) if (tn_pc + fp_pc) > 0 else 0.0
        
        print(f"  [PC]  Exactitud: {acc_pc:.4f} | Sensibilidad: {sens_pc:.4f} | Especificidad: {spec_pc:.4f} | F1: {f1_pc:.4f}")
        resultados_loso.append({
            'Algoritmo_Estructura': 'PC (fisher_z)',
            'Top_Features': n_features,
            'Exactitud': acc_pc,
            'Precisión': prec_pc,
            'Sensibilidad (Ansiedad)': sens_pc,
            'Especificidad (Relajación)': spec_pc,
            'F1_Score': f1_pc
        })
        
        y_real_ges, y_pred_ges = ejecutar_loso(df_datos, features, 'Ansiedad', grupos, "GES")
        
        # Calcular métricas para GES
        acc_ges = accuracy_score(y_real_ges, y_pred_ges)
        prec_ges = precision_score(y_real_ges, y_pred_ges, zero_division=0)
        sens_ges = recall_score(y_real_ges, y_pred_ges, zero_division=0)
        f1_ges = f1_score(y_real_ges, y_pred_ges, zero_division=0)
        
        cm_ges = confusion_matrix(y_real_ges, y_pred_ges, labels=[0, 1])
        tn_ges, fp_ges, _, _ = cm_ges.ravel()
        spec_ges = tn_ges / (tn_ges + fp_ges) if (tn_ges + fp_ges) > 0 else 0.0
        
        resultados_loso.append({
            'Algoritmo_Estructura': 'GES (bic-g)',
            'Top_Features': n_features,
            'Exactitud': acc_ges,
            'Precisión': prec_ges,
            'Sensibilidad (Ansiedad)': sens_ges,
            'Especificidad (Relajación)': spec_ges,
            'F1_Score': f1_ges
        })
        
        data_global = df_datos[features + ['Ansiedad']].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Estructura Global PC
        try:
            pc_global = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
            pc_global.fit(data_global)
            dag_pc = pc_global.causal_graph_
            plot_dag(
                dag_pc, 
                f"Red Bayesiana Causal (PC, Top-{n_features})", 
                os.path.join(dir_resultados, f"Estructura_Global_PC_top{n_features}.png")
            )
        except Exception:
            pass
            
        try:
            ges_global = GES(scoring_method='bic-g', return_type='dag')
            ges_global.fit(data_global)
            dag_ges = ges_global.causal_graph_
            plot_dag(
                dag_ges, 
                f"Red Bayesiana Causal (GES, Top-{n_features})", 
                os.path.join(dir_resultados, f"Estructura_Global_GES_top{n_features}.png")
            )
        except Exception:
            pass

    df_resultados = pd.DataFrame(resultados_loso)
    ruta_salida = os.path.join(dir_resultados, 'Resultados_Comparativa_RB.csv')
    df_resultados.to_csv(ruta_salida, index=False)
    
    print(df_resultados.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    main()
