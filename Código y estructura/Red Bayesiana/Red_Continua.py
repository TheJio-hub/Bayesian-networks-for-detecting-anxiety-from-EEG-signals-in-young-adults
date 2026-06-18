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
            
            model = LinearGaussianBayesianNetwork(dag.edges())
            model.add_nodes_from(dag.nodes())
            model.fit(df_train)
            
            X_test = df_test.drop(columns=[target_col])
            pred = model.predict(X_test)
            y_pred[test_idx] = pred[target_col].values
            
        except Exception:
            y_pred[test_idx] = df_train[target_col].mean()
            
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_real_bin = y_real.astype(int)
    
    return y_real_bin, y_pred_bin

def main():
    dir_resultados = os.path.join('Resultados', 'Red Bayesiana')
    os.makedirs(dir_resultados, exist_ok=True)
    
    archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')
    
    if not os.path.exists(archivo_datos) or not os.path.exists(archivo_ranking):
        print("Error: No se encontraron los archivos necesarios en Resultados.")
        return
        
    df_datos = pd.read_parquet(archivo_datos)
    df_datos = df_datos[(df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1)].copy()
    df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0.0 if x == 0 else 1.0)
    
    unique_subjects = np.sort(df_datos['Sujeto'].unique())
    np.random.seed(42)
    shuffled_subjects = np.random.permutation(unique_subjects)
    dev_subjects = shuffled_subjects[:32]
    holdout_subjects = shuffled_subjects[32:]
    
    df_dev = df_datos[df_datos['Sujeto'].isin(dev_subjects)].copy()
    df_holdout = df_datos[df_datos['Sujeto'].isin(holdout_subjects)].copy()
    
    grupos_dev = df_dev['Sujeto'].values
    
    df_ranking = pd.read_csv(archivo_ranking)
    df_ranking = df_ranking.sort_values(by=['Importancia_DT', 'Mutual_Info'], ascending=[False, False])
    mejores_caracteristicas = df_ranking['Caracteristica'].tolist()
    
    top_tamanos = [10, 15]
    resultados_loso = []
    
    for n_features in top_tamanos:
        features = mejores_caracteristicas[:n_features]
        
        # PC: Validación Cruzada en Dev
        y_real_pc, y_pred_pc = ejecutar_loso(df_dev, features, 'Ansiedad', grupos_dev, "PC")
        
        val_acc_pc = accuracy_score(y_real_pc, y_pred_pc)
        val_prec_pc = precision_score(y_real_pc, y_pred_pc, zero_division=0)
        val_sens_pc = recall_score(y_real_pc, y_pred_pc, zero_division=0)
        val_f1_pc = f1_score(y_real_pc, y_pred_pc, zero_division=0)
        cm_pc = confusion_matrix(y_real_pc, y_pred_pc, labels=[0, 1])
        tn_pc, fp_pc, _, _ = cm_pc.ravel()
        val_spec_pc = tn_pc / (tn_pc + fp_pc) if (tn_pc + fp_pc) > 0 else 0.0
        
        # PC: Ajuste Final en Dev y Test en Holdout
        dev_cols = features + ['Ansiedad']
        df_dev_sub = df_dev[dev_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
        df_holdout_sub = df_holdout[dev_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
        
        try:
            pc_final = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
            pc_final.fit(df_dev_sub)
            dag_pc = pc_final.causal_graph_
            
            model_pc = LinearGaussianBayesianNetwork(dag_pc.edges())
            model_pc.add_nodes_from(dag_pc.nodes())
            model_pc.fit(df_dev_sub)
            
            pred_pc = model_pc.predict(df_holdout_sub.drop(columns=['Ansiedad']))
            y_holdout_pred_pc = (pred_pc['Ansiedad'].values >= 0.5).astype(int)
        except Exception:
            y_holdout_pred_pc = (np.repeat(df_dev_sub['Ansiedad'].mean(), len(df_holdout_sub)) >= 0.5).astype(int)

        # 8 sujetos de validacion externa     
        y_holdout_real = df_holdout_sub['Ansiedad'].values.astype(int)
        test_acc_pc = accuracy_score(y_holdout_real, y_holdout_pred_pc)
        test_prec_pc = precision_score(y_holdout_real, y_holdout_pred_pc, zero_division=0)
        test_sens_pc = recall_score(y_holdout_real, y_holdout_pred_pc, zero_division=0)
        test_f1_pc = f1_score(y_holdout_real, y_holdout_pred_pc, zero_division=0)
        cm_test_pc = confusion_matrix(y_holdout_real, y_holdout_pred_pc, labels=[0, 1])
        tn_tpc, fp_tpc, _, _ = cm_test_pc.ravel()
        test_spec_pc = tn_tpc / (tn_tpc + fp_tpc) if (tn_tpc + fp_tpc) > 0 else 0.0
        
        print(f"  [PC]  Val_Acc: {val_acc_pc:.4f} | Test_Acc: {test_acc_pc:.4f}")
        resultados_loso.append({
            'Algoritmo_Estructura': 'PC (fisher_z)',
            'Top_Features': n_features,
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
        
        # 2. GES: Validación Cruzada en Dev
        y_real_ges, y_pred_ges = ejecutar_loso(df_dev, features, 'Ansiedad', grupos_dev, "GES")
        
        val_acc_ges = accuracy_score(y_real_ges, y_pred_ges)
        val_prec_ges = precision_score(y_real_ges, y_pred_ges, zero_division=0)
        val_sens_ges = recall_score(y_real_ges, y_pred_ges, zero_division=0)
        val_f1_ges = f1_score(y_real_ges, y_pred_ges, zero_division=0)
        cm_ges = confusion_matrix(y_real_ges, y_pred_ges, labels=[0, 1])
        tn_ges, fp_ges, _, _ = cm_ges.ravel()
        val_spec_ges = tn_ges / (tn_ges + fp_ges) if (tn_ges + fp_ges) > 0 else 0.0
        
        # GES: Ajuste Final en Dev y Test en Holdout
        try:
            ges_final = GES(scoring_method='bic-g', return_type='dag')
            ges_final.fit(df_dev_sub)
            dag_ges = ges_final.causal_graph_
            
            model_ges = LinearGaussianBayesianNetwork(dag_ges.edges())
            model_ges.add_nodes_from(dag_ges.nodes())
            model_ges.fit(df_dev_sub)
            
            pred_ges = model_ges.predict(df_holdout_sub.drop(columns=['Ansiedad']))
            y_holdout_pred_ges = (pred_ges['Ansiedad'].values >= 0.5).astype(int)
        except Exception:
            y_holdout_pred_ges = (np.repeat(df_dev_sub['Ansiedad'].mean(), len(df_holdout_sub)) >= 0.5).astype(int)

        # 8 sujetos de validacion externa    
        test_acc_ges = accuracy_score(y_holdout_real, y_holdout_pred_ges)
        test_prec_ges = precision_score(y_holdout_real, y_holdout_pred_ges, zero_division=0)
        test_sens_ges = recall_score(y_holdout_real, y_holdout_pred_ges, zero_division=0)
        test_f1_ges = f1_score(y_holdout_real, y_holdout_pred_ges, zero_division=0)
        cm_test_ges = confusion_matrix(y_holdout_real, y_holdout_pred_ges, labels=[0, 1])
        tn_tges, fp_tges, _, _ = cm_test_ges.ravel()
        test_spec_ges = tn_tges / (tn_tges + fp_tges) if (tn_tges + fp_tges) > 0 else 0.0
        
        print(f"  [GES] Val_Acc: {val_acc_ges:.4f} | Test_Acc: {test_acc_ges:.4f}")
        resultados_loso.append({
            'Algoritmo_Estructura': 'GES (bic-g)',
            'Top_Features': n_features,
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
        
        # Graficar la estructura global 
        try:
            plot_dag(
                dag_pc, 
                f"Red Causal Continua (PC, Top-{n_features})", 
                os.path.join(dir_resultados, f"Estructura_Global_PC_top{n_features}.png")
            )
        except Exception:
            pass
            
        try:
            plot_dag(
                dag_ges, 
                f"Red Causal Continua (GES, Top-{n_features})", 
                os.path.join(dir_resultados, f"Estructura_Global_GES_top{n_features}.png")
            )
        except Exception:
            pass

    df_resultados = pd.DataFrame(resultados_loso)
    ruta_salida = os.path.join(dir_resultados, 'Resultados_Comparativa_RB.csv')
    df_resultados.to_csv(ruta_salida, index=False)
    
    print("\n" + "=" * 80)
    print("Resultados de Validación y Test Externo (Continuo):")
    print(df_resultados.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f"\nResultados guardados en: {ruta_salida}\n")

if __name__ == "__main__":
    main()
