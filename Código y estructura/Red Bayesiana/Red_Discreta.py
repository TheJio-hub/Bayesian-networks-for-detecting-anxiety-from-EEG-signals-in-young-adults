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
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pgmpy.causal_discovery import PC, HillClimbSearch
from pgmpy.structure_score import BIC
from pgmpy.models import DiscreteBayesianNetwork

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
        edge_color='darkgreen', 
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

def ejecutar_loso(df_datos, selected_features, target_col, grupos, variant_name, n_bins):
    logo = LeaveOneGroupOut()
    y_real = df_datos[target_col].values
    y_pred = np.zeros(len(df_datos))
    
    total_folds = logo.get_n_splits(df_datos, y_real, grupos)
    pbar = tqdm(
        logo.split(df_datos, y_real, grupos), 
        total=total_folds, 
        desc=f"LOSO {variant_name} (bins={n_bins})", 
        unit='fold', 
        leave=False
    )
    
    for train_idx, test_idx in pbar:
        df_train_raw = df_datos.iloc[train_idx]
        df_test_raw = df_datos.iloc[test_idx]
        
        # Discretización (fit en train, transform en train y test)
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        
        train_features_disc = discretizer.fit_transform(df_train_raw[selected_features])
        test_features_disc = discretizer.transform(df_test_raw[selected_features])
        
        df_train = pd.DataFrame(train_features_disc, columns=selected_features, index=df_train_raw.index).astype(int)
        df_train[target_col] = df_train_raw[target_col].values.astype(int)
        
        df_test = pd.DataFrame(test_features_disc, columns=selected_features, index=df_test_raw.index).astype(int)
        
        try:
            if variant_name == "PC":
                est = PC(
                    variant='stable', 
                    ci_test='chi_square', 
                    significance_level=0.05, 
                    return_type='dag', 
                    show_progress=False
                )
                est.fit(df_train)
                dag = est.causal_graph_
            else:  # HC
                hc = HillClimbSearch(scoring_method=BIC(df_train), show_progress=False)
                hc.fit(df_train)
                dag = hc.causal_graph_
            
            model = DiscreteBayesianNetwork(dag.edges())
            model.add_nodes_from(dag.nodes())
            model.fit(df_train)
            
            pred = model.predict(df_test)
            y_pred[test_idx] = pred[target_col].values
            
        except Exception:
            y_pred[test_idx] = df_train[target_col].mode().values[0]
            
    y_pred_bin = y_pred.astype(int)
    y_real_bin = y_real.astype(int)
    
    return y_real_bin, y_pred_bin

def main():
    # Definir rutas
    dir_resultados = os.path.join('Resultados', 'Red Bayesiana')
    os.makedirs(dir_resultados, exist_ok=True)
    
    archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')
        
    df_datos = pd.read_parquet(archivo_datos)
    df_datos = df_datos[(df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1)].copy()
    df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0 if x == 0 else 1)
    
    # Partición 80/20
    unique_subjects = np.sort(df_datos['Sujeto'].unique())
    np.random.seed(42)
    shuffled_subjects = np.random.permutation(unique_subjects)
    dev_subjects = shuffled_subjects[:32]
    holdout_subjects = shuffled_subjects[32:]
    
    df_dev = df_datos[df_datos['Sujeto'].isin(dev_subjects)].copy()
    df_holdout = df_datos[df_datos['Sujeto'].isin(holdout_subjects)].copy()
    
    grupos_dev = df_dev['Sujeto'].values
    
    # Cargar ranking y obtener características DT
    df_ranking = pd.read_csv(archivo_ranking)
    df_ranking = df_ranking.sort_values(by=['Importancia_DT', 'Mutual_Info'], ascending=[False, False])
    mejores_caracteristicas = df_ranking['Caracteristica'].tolist()
    
    bins_tamanos = [2, 3]
    top_tamanos = [10, 15]
    resultados_loso = []
    
    # Bucle de Validación Cruzada LOSO
    for n_bins in bins_tamanos:
        for n_features in top_tamanos:
            features = mejores_caracteristicas[:n_features]
            
            # PC: Validación 
            y_real_pc, y_pred_pc = ejecutar_loso(df_dev, features, 'Ansiedad', grupos_dev, "PC", n_bins)
            
            val_acc_pc = accuracy_score(y_real_pc, y_pred_pc)
            val_prec_pc = precision_score(y_real_pc, y_pred_pc, zero_division=0)
            val_sens_pc = recall_score(y_real_pc, y_pred_pc, zero_division=0)
            val_f1_pc = f1_score(y_real_pc, y_pred_pc, zero_division=0)
            cm_pc = confusion_matrix(y_real_pc, y_pred_pc, labels=[0, 1])
            tn_pc, fp_pc, _, _ = cm_pc.ravel()
            val_spec_pc = tn_pc / (tn_pc + fp_pc) if (tn_pc + fp_pc) > 0 else 0.0
            
            # PC: Test 
            discretizer_pc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            
            dev_features_disc = discretizer_pc.fit_transform(df_dev[features])
            holdout_features_disc = discretizer_pc.transform(df_holdout[features])
            
            df_dev_disc = pd.DataFrame(dev_features_disc, columns=features, index=df_dev.index).astype(int)
            df_dev_disc['Ansiedad'] = df_dev['Ansiedad'].values.astype(int)
            
            df_holdout_disc = pd.DataFrame(holdout_features_disc, columns=features, index=df_holdout.index).astype(int)
            
            try:
                pc_final = PC(variant='stable', ci_test='chi_square', significance_level=0.05, return_type='dag', show_progress=False)
                pc_final.fit(df_dev_disc)
                dag_pc = pc_final.causal_graph_
                
                model_pc = DiscreteBayesianNetwork(dag_pc.edges())
                model_pc.add_nodes_from(dag_pc.nodes())
                model_pc.fit(df_dev_disc)
                
                pred_pc = model_pc.predict(df_holdout_disc)
                y_holdout_pred_pc = pred_pc['Ansiedad'].values.astype(int)
            except Exception:
                y_holdout_pred_pc = np.repeat(df_dev_disc['Ansiedad'].mode().values[0], len(df_holdout_disc))
                
            # 8 sujetos de validacion externa     
            y_holdout_real = df_holdout['Ansiedad'].values.astype(int)
            test_acc_pc = accuracy_score(y_holdout_real, y_holdout_pred_pc)
            test_prec_pc = precision_score(y_holdout_real, y_holdout_pred_pc, zero_division=0)
            test_sens_pc = recall_score(y_holdout_real, y_holdout_pred_pc, zero_division=0)
            test_f1_pc = f1_score(y_holdout_real, y_holdout_pred_pc, zero_division=0)
            cm_test_pc = confusion_matrix(y_holdout_real, y_holdout_pred_pc, labels=[0, 1])
            tn_tpc, fp_tpc, _, _ = cm_test_pc.ravel()
            test_spec_pc = tn_tpc / (tn_tpc + fp_tpc) if (tn_tpc + fp_tpc) > 0 else 0.0
            
            resultados_loso.append({
                'Algoritmo_Estructura': 'PC (chi_square)',
                'n_bins': n_bins,
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
            
            # HC: Validación
            y_real_hc, y_pred_hc = ejecutar_loso(df_dev, features, 'Ansiedad', grupos_dev, "HC", n_bins)
            
            val_acc_hc = accuracy_score(y_real_hc, y_pred_hc)
            val_prec_hc = precision_score(y_real_hc, y_pred_hc, zero_division=0)
            val_sens_hc = recall_score(y_real_hc, y_pred_hc, zero_division=0)
            val_f1_hc = f1_score(y_real_hc, y_pred_hc, zero_division=0)
            cm_hc = confusion_matrix(y_real_hc, y_pred_hc, labels=[0, 1])
            tn_hc, fp_hc, _, _ = cm_hc.ravel()
            val_spec_hc = tn_hc / (tn_hc + fp_hc) if (tn_hc + fp_hc) > 0 else 0.0
            
            # HC: Test
            try:
                hc_final = HillClimbSearch(scoring_method=BIC(df_dev_disc), show_progress=False)
                hc_final.fit(df_dev_disc)
                dag_hc = hc_final.causal_graph_
                
                model_hc = DiscreteBayesianNetwork(dag_hc.edges())
                model_hc.add_nodes_from(dag_hc.nodes())
                model_hc.fit(df_dev_disc)
                
                pred_hc = model_hc.predict(df_holdout_disc)
                y_holdout_pred_hc = pred_hc['Ansiedad'].values.astype(int)
            except Exception:
                y_holdout_pred_hc = np.repeat(df_dev_disc['Ansiedad'].mode().values[0], len(df_holdout_disc))
            # 8 sujetos de validacion externa    
            test_acc_hc = accuracy_score(y_holdout_real, y_holdout_pred_hc)
            test_prec_hc = precision_score(y_holdout_real, y_holdout_pred_hc, zero_division=0)
            test_sens_hc = recall_score(y_holdout_real, y_holdout_pred_hc, zero_division=0)
            test_f1_hc = f1_score(y_holdout_real, y_holdout_pred_hc, zero_division=0)
            cm_test_hc = confusion_matrix(y_holdout_real, y_holdout_pred_hc, labels=[0, 1])
            tn_thc, fp_thc, _, _ = cm_test_hc.ravel()
            test_spec_hc = tn_thc / (tn_thc + fp_thc) if (tn_thc + fp_thc) > 0 else 0.0
            
            resultados_loso.append({
                'Algoritmo_Estructura': 'HC (BIC)',
                'n_bins': n_bins,
                'Top_Features': n_features,
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
            
            # Graficar estructura global 
            try:
                plot_dag(
                    dag_pc, 
                    f"Red Causal Discreta (PC, Top-{n_features}, Bins-{n_bins})", 
                    os.path.join(dir_resultados, f"Estructura_Global_Discreta_PC_top{n_features}_bins{n_bins}.png")
                )
            except Exception:
                pass
                
            try:
                plot_dag(
                    dag_hc, 
                    f"Red Causal Discreta (HC, Top-{n_features}, Bins-{n_bins})", 
                    os.path.join(dir_resultados, f"Estructura_Global_Discreta_HC_top{n_features}_bins{n_bins}.png")
                )
            except Exception:
                pass

    # Guardar resultados en CSV
    df_resultados = pd.DataFrame(resultados_loso)
    ruta_salida = os.path.join(dir_resultados, 'Resultados_Comparativa_RB_Discreta.csv')
    df_resultados.to_csv(ruta_salida, index=False)
    
    print(df_resultados.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    main()
