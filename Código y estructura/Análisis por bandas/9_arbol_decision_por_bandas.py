import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


def columnas_ratio_por_banda(columnas, banda):
    seleccionadas = []
    for col in columnas:
        if not col.startswith('Ratio_'):
            continue
        partes = col.split('_')
        if len(partes) < 3:
            continue
        tipo_ratio = partes[1]
        if banda in tipo_ratio:
            seleccionadas.append(col)
    return seleccionadas


def evaluar_arboles_por_bandas():
    archivo_entrada = os.path.join('Resultados', 'Exploratorio', 'datos_bandas_normalizados.parquet')
    archivo_asimetria = os.path.join('Resultados', 'Exploratorio', 'datos_asimetria_normalizados.parquet')
    archivo_ratios = os.path.join('Resultados', 'Exploratorio', 'datos_ratios_normalizados.parquet')
    directorio_salida = os.path.join('Resultados', 'Análisis por bandas', 'Modelos por banda')
    directorio_rankings = os.path.join('Resultados', 'Análisis por bandas', 'Ranking de características')
    
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        
    if not os.path.exists(archivo_entrada):
        archivo_entrada = os.path.join('Resultados', 'datos_bandas.parquet')
    
    if not os.path.exists(archivo_entrada):
        return

    df = pd.read_parquet(archivo_entrada)

    df_asim = pd.read_parquet(archivo_asimetria) if os.path.exists(archivo_asimetria) else pd.DataFrame()
    df_ratio = pd.read_parquet(archivo_ratios) if os.path.exists(archivo_ratios) else pd.DataFrame()
    
    if 'Puntaje' in df.columns:
        df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
        y = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
        grupos = df['Sujeto'].values
    else:
        return

    if not df_asim.empty and 'Puntaje' in df_asim.columns:
        df_asim = df_asim[(df_asim['Puntaje'] == 0) | (df_asim['Puntaje'] >= 5)].copy()
        y_asim = df_asim['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
    else:
        y_asim = np.array([])

    if not df_ratio.empty and 'Puntaje' in df_ratio.columns:
        df_ratio = df_ratio[(df_ratio['Puntaje'] == 0) | (df_ratio['Puntaje'] >= 5)].copy()
        y_ratio = df_ratio['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
    else:
        y_ratio = np.array([])

    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    todas_caracteristicas = [c for c in df.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df[c])]
    
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    resultados_metricas = []

    for banda in bandas:
        cols_banda = [c for c in todas_caracteristicas if c.endswith(f"_{banda}")]
        
        if not cols_banda:
            continue
            
        X_banda = df[cols_banda]
        
        clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        
        logo = LeaveOneGroupOut()
        
        scoring = {
            'accuracy': 'accuracy', 
            'precision_0': make_scorer(precision_score, pos_label=0, zero_division=0),
            'recall_0':    make_scorer(recall_score, pos_label=0, zero_division=0),
            'f1_0':        make_scorer(f1_score, pos_label=0, zero_division=0),
            'precision_1': make_scorer(precision_score, pos_label=1, zero_division=0),
            'recall_1':    make_scorer(recall_score, pos_label=1, zero_division=0),
            'f1_1':        make_scorer(f1_score, pos_label=1, zero_division=0)
        }
        
        scores = cross_validate(clf, X_banda, y, cv=logo, scoring=scoring, groups=grupos)
        
        mean_acc = np.mean(scores['test_accuracy'])
        mean_f1_0 = np.mean(scores['test_f1_0'])
        mean_f1_1 = np.mean(scores['test_f1_1'])
        
        resultados_metricas.append({
            'Banda': banda,
            'Accuracy': mean_acc,
            'Precision_Relajacion_0': np.mean(scores['test_precision_0']),
            'Recall_Relajacion_0':    np.mean(scores['test_recall_0']),
            'F1_Relajacion_0':        mean_f1_0,
            'Precision_Ansiedad_1': np.mean(scores['test_precision_1']),
            'Recall_Ansiedad_1':    np.mean(scores['test_recall_1']),
            'F1_Ansiedad_1':        mean_f1_1
        })
        
        clf.fit(X_banda, y)
        importancias = clf.feature_importances_
        df_imp = pd.DataFrame({'Caracteristica': cols_banda, 'Importancia': importancias})
        df_imp = df_imp.sort_values(by='Importancia', ascending=False)
        
        df_imp.to_csv(os.path.join(directorio_salida, f'Importancia_Arbol_{banda}.csv'), index=False)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        rf.fit(X_banda, y)
        df_rf = pd.DataFrame({'Caracteristica': cols_banda, 'Importancia': rf.feature_importances_})
        df_rf = df_rf.sort_values(by='Importancia', ascending=False)
        df_rf.to_csv(os.path.join(directorio_salida, f'Importancia_RandomForest_{banda}.csv'), index=False)

        # DT para asimetria por banda
        dir_asim = os.path.join(directorio_rankings, banda, 'Asimetria')
        os.makedirs(dir_asim, exist_ok=True)
        if not df_asim.empty:
            cols_asim = [c for c in df_asim.columns if c.startswith('Asym_') and c.endswith(f'_{banda}')]
            if cols_asim:
                X_asim = df_asim[cols_asim]
                dt_asim = DecisionTreeClassifier(random_state=42, class_weight='balanced')
                dt_asim.fit(X_asim, y_asim)
                df_dt_asim = pd.DataFrame({'Caracteristica': cols_asim, 'Importancia_DT': dt_asim.feature_importances_})
                df_dt_asim = df_dt_asim.sort_values(by='Importancia_DT', ascending=False)
                df_dt_asim.to_csv(os.path.join(dir_asim, f'DT_Asimetria_{banda}.csv'), index=False)
            else:
                pd.DataFrame(columns=['Caracteristica', 'Importancia_DT']).to_csv(
                    os.path.join(dir_asim, f'DT_Asimetria_{banda}.csv'), index=False
                )

        # DT para ratios por banda
        dir_ratios = os.path.join(directorio_rankings, banda, 'Ratios')
        os.makedirs(dir_ratios, exist_ok=True)
        if not df_ratio.empty:
            cols_ratio = columnas_ratio_por_banda(df_ratio.columns.tolist(), banda)
            if cols_ratio:
                X_ratio = df_ratio[cols_ratio]
                dt_ratio = DecisionTreeClassifier(random_state=42, class_weight='balanced')
                dt_ratio.fit(X_ratio, y_ratio)
                df_dt_ratio = pd.DataFrame({'Caracteristica': cols_ratio, 'Importancia_DT': dt_ratio.feature_importances_})
                df_dt_ratio = df_dt_ratio.sort_values(by='Importancia_DT', ascending=False)
                df_dt_ratio.to_csv(os.path.join(dir_ratios, f'DT_Ratios_{banda}.csv'), index=False)
            else:
                pd.DataFrame(columns=['Caracteristica', 'Importancia_DT']).to_csv(
                    os.path.join(dir_ratios, f'DT_Ratios_{banda}.csv'), index=False
                )

    df_resumen = pd.DataFrame(resultados_metricas)
    archivo_resumen = os.path.join(directorio_salida, 'Resumen_Metricas_Arbol_Por_Bandas.csv')
    df_resumen.to_csv(archivo_resumen, index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_resumen, x='Banda', y='Accuracy', palette='viridis')
    plt.ylim(0, 1.05)
    plt.title('Accuracy del Árbol de Decisión por Banda')
    plt.ylabel('Accuracy Promedio (CV)')
    for index, row in df_resumen.iterrows():
        plt.text(index, row.Accuracy + 0.01, f"{row.Accuracy:.4f}", color='black', ha="center")
    
    plt.savefig(os.path.join(directorio_salida, 'Comparativa_Accuracy_Arbol_Bandas.png'))
    plt.close()

if __name__ == "__main__":
    evaluar_arboles_por_bandas()
