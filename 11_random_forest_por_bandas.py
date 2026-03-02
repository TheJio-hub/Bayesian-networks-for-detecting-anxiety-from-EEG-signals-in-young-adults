import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


def evaluar_random_forest_por_bandas():
    archivo_entrada = os.path.join('Resultados', 'datos_bandas_normalizados.parquet')
    directorio_salida = os.path.join('Resultados', 'Modelos')
    
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        
    if not os.path.exists(archivo_entrada):
        archivo_entrada = os.path.join('Resultados', 'datos_bandas.parquet')
    
    if not os.path.exists(archivo_entrada):
        return

    df = pd.read_parquet(archivo_entrada)
    
    if 'Puntaje' in df.columns:
        df = df[ (df['Puntaje'] == 0) | (df['Puntaje'] >= 5) ].copy()
        y = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
        grupos = df['Sujeto'].values
    else:
        return

    cols_meta = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje', 'Grupo', 'Ensayo']
    todas_caracteristicas = [c for c in df.columns if c not in cols_meta and pd.api.types.is_numeric_dtype(df[c])]
    
    bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    resultados_metricas = []

    for banda in bandas:
        cols_banda = [c for c in todas_caracteristicas if c.endswith(f"_{banda}")]
        
        if not cols_banda:
            continue
            
        X_banda = df[cols_banda]
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        
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
        
        df_imp.to_csv(os.path.join(directorio_salida, f'Importancia_RandomForest_{banda}.csv'), index=False)

    df_resumen = pd.DataFrame(resultados_metricas)
    archivo_resumen = os.path.join(directorio_salida, 'Resumen_Metricas_RandomForest_Por_Bandas.csv')
    df_resumen.to_csv(archivo_resumen, index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_resumen, x='Banda', y='Accuracy', palette='magma')
    plt.ylim(0, 1.05)
    plt.title('Exactitud (Accuracy) del Random Forest por Banda')
    plt.ylabel('Exactitud Promedio (CV)')
    for index, row in df_resumen.iterrows():
        plt.text(index, row.Accuracy + 0.01, f"{row.Accuracy:.4f}", color='black', ha="center")
    
    plt.savefig(os.path.join(directorio_salida, 'Comparativa_Accuracy_RandomForest_Bandas.png'))
    plt.close()

if __name__ == "__main__":
    evaluar_random_forest_por_bandas()
