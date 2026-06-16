import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

def obtener_top_n_por_criterio(df_ranking, criterio, n=30):
    if criterio == 'mRMR':
        if n >= 40 and 'mRMR_40_Rank' in df_ranking.columns:
            mrmr_col = 'mRMR_40_Rank'
        elif n >= 30 and 'mRMR_30_Rank' in df_ranking.columns:
            mrmr_col = 'mRMR_30_Rank'
        elif n >= 20 and 'mRMR_20_Rank' in df_ranking.columns:
            mrmr_col = 'mRMR_20_Rank'
        elif n >= 15 and 'mRMR_20_Rank' in df_ranking.columns:
            mrmr_col = 'mRMR_20_Rank'
        elif 'mRMR_10_Rank' in df_ranking.columns:
            mrmr_col = 'mRMR_10_Rank'
        else:
            mrmr_col = 'mRMR_Rank'
        sub = df_ranking.dropna(subset=[mrmr_col]).sort_values(mrmr_col, ascending=True)
        return sub['Caracteristica'].tolist()[:n]
    elif criterio == 'Fisher':
        sub = df_ranking.sort_values('Fisher_Score', ascending=False)
        return sub['Caracteristica'].tolist()[:n]
    elif criterio == 'Mutual_Info':
        sub = df_ranking.sort_values('Mutual_Info', ascending=False)
        return sub['Caracteristica'].tolist()[:n]
    elif criterio == 'DT':
        sub = df_ranking.sort_values('Importancia_DT', ascending=False)
        return sub['Caracteristica'].tolist()[:n]
    return []

#  LOGO, funciona como un Leave-One-Subject-Out, pero con la flexibilidad de definir grupos personalizados (en este caso, sujetos)
# Especificamente lo que hace esta funcion es entrenar el modelo con todos los datos excepto los de un sujeto específico, y luego evalúa el modelo en los datos de ese sujeto. Esto se repite para cada sujeto, lo que permite obtener una evaluación robusta del modelo en términos de su capacidad para generalizar a nuevos sujetos. Al final todas las metricas de los entrenamientos se promedian para obtener una evaluación global del modelo.

def entrenar_y_evaluar(X, y, grupos, modelo):
    validacion = LeaveOneGroupOut()
    y_real_total = []
    y_predicha_total = []
    y_train_total = []
    y_train_pred_total = []
    total_folds = validacion.get_n_splits(X, y, grupos)
    
    for indice_entrenamiento, indice_prueba in tqdm(validacion.split(X, y, grupos), total=total_folds, desc='LOGO DT', unit='fold', leave=False):
        X_entrenamiento, X_prueba = X.iloc[indice_entrenamiento], X.iloc[indice_prueba]
        y_entrenamiento, y_prueba = y[indice_entrenamiento], y[indice_prueba]
        
        modelo.fit(X_entrenamiento, y_entrenamiento)
        prediccion = modelo.predict(X_prueba)

        y_real_total.extend(y_prueba)
        y_predicha_total.extend(prediccion)
        
        # Métricas de entrenamiento: usar 5-fold CV interno para obtener predicciones no vistas
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for idx_train_cv, idx_val_cv in kfold.split(X_entrenamiento):
            X_cv_train, X_cv_val = X_entrenamiento.iloc[idx_train_cv], X_entrenamiento.iloc[idx_val_cv]
            y_cv_train, y_cv_val = y_entrenamiento[idx_train_cv], y_entrenamiento[idx_val_cv]
            modelo.fit(X_cv_train, y_cv_train)
            train_pred = modelo.predict(X_cv_val)
            y_train_total.extend(y_cv_val)
            y_train_pred_total.extend(train_pred)
        
    # Métricas de validación (LOGO)
    precision, sensibilidad, f1, _ = precision_recall_fscore_support(y_real_total, y_predicha_total, labels=[0, 1], zero_division=0)
    exactitud = accuracy_score(y_real_total, y_predicha_total)
    cm = confusion_matrix(y_real_total, y_predicha_total, labels=[0, 1])
    total = cm.sum()
    especificidades = []
    for clase_idx in range(cm.shape[0]):
        tp = cm[clase_idx, clase_idx]
        fp = cm[:, clase_idx].sum() - tp
        fn = cm[clase_idx, :].sum() - tp
        tn = total - tp - fp - fn
        especificidades.append(tn / (tn + fp) if (tn + fp) else 0.0)

    # Métricas de entrenamiento (acumuladas sobre folds)
    precision_tr, sensibilidad_tr, f1_tr, _ = precision_recall_fscore_support(y_train_total, y_train_pred_total, labels=[0, 1], zero_division=0)
    exactitud_tr = accuracy_score(y_train_total, y_train_pred_total)
    cm_tr = confusion_matrix(y_train_total, y_train_pred_total, labels=[0, 1])
    total_tr = cm_tr.sum()
    especificidades_tr = []
    for clase_idx in range(cm_tr.shape[0]):
        tp = cm_tr[clase_idx, clase_idx]
        fp = cm_tr[:, clase_idx].sum() - tp
        fn = cm_tr[clase_idx, :].sum() - tp
        tn = total_tr - tp - fp - fn
        especificidades_tr.append(tn / (tn + fp) if (tn + fp) else 0.0)

    # Devuelve filas separadas para validación y entrenamiento por clase
    return pd.DataFrame([
        {'Precision': precision[0], 'Sensibilidad': sensibilidad[0], 'Especificidad': especificidades[0], 'Puntaje_F1': f1[0], 'Exactitud': exactitud},
        {'Precision': precision[1], 'Sensibilidad': sensibilidad[1], 'Especificidad': especificidades[1], 'Puntaje_F1': f1[1], 'Exactitud': exactitud},
        {'Precision': precision_tr[0], 'Sensibilidad': sensibilidad_tr[0], 'Especificidad': especificidades_tr[0], 'Puntaje_F1': f1_tr[0], 'Exactitud': exactitud_tr},
        {'Precision': precision_tr[1], 'Sensibilidad': sensibilidad_tr[1], 'Especificidad': especificidades_tr[1], 'Puntaje_F1': f1_tr[1], 'Exactitud': exactitud_tr}
    ], index=['Clase 0 (Validación)', 'Clase 1 (Validación)', 'Clase 0 (Entrenamiento)', 'Clase 1 (Entrenamiento)'])

def principal():
    archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')
    if not os.path.exists(archivo_ranking):
        archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Ranking_Multicriterio_Completo.csv')
    archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
    dir_salida = os.path.join('Resultados', 'Análisis global', 'Modelos generados')
    
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)
        
    if not os.path.exists(archivo_datos) or not os.path.exists(archivo_ranking):
        return

    df_ranking = pd.read_csv(archivo_ranking)
    df_datos = pd.read_parquet(archivo_datos)
    
    df_datos = df_datos[ (df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1) ].copy()
    y = df_datos['Puntaje'].apply(lambda x: 0 if x == 0 else 1).values
    grupos = df_datos['Sujeto'].values
    
    modelo = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5)
    nombre_modelo = 'DT'
    
    top_5 = [40, 30, 20, 15, 10]
    configuraciones_por_criterio = {
        'Fisher': top_5,
        'Mutual_Info': top_5,
        'mRMR': top_5,
        'DT': [20, 15, 10],
    }

    for criterio in tqdm(configuraciones_por_criterio.keys(), desc='Criterios DT', unit='criterio'):
        for n_caracteristicas in tqdm(configuraciones_por_criterio[criterio], desc=f'Top para {criterio}', unit='top', leave=False):
            nombre_carpeta = f'Top {n_caracteristicas}'
            directorio_top = os.path.join(dir_salida, nombre_carpeta)
            os.makedirs(directorio_top, exist_ok=True)

            mejores_caracteristicas = obtener_top_n_por_criterio(df_ranking, criterio, n=n_caracteristicas)
            
            X_subconjunto = df_datos[mejores_caracteristicas]
            df_resultados = entrenar_y_evaluar(X_subconjunto, y, grupos, modelo)
            
            nombre_archivo = f"Resultados_{nombre_modelo}_Ranking_{criterio}_top{n_caracteristicas}.csv"
            archivo_salida = os.path.join(directorio_top, nombre_archivo)
            df_resultados.to_csv(archivo_salida)
            print(f"Guardado: {archivo_salida}")

if __name__ == "__main__":
    principal()
