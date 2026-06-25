import pandas as pd
import numpy as np
from pgmpy.causal_discovery import PC
from pgmpy.metrics import CorrelationScore, FisherC
from pgmpy.base import DAG
import os
import traceback

archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')

df_datos = pd.read_parquet(archivo_datos)
df_datos = df_datos[(df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1)].copy()
df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0.0 if x == 0 else 1.0)

df_ranking = pd.read_csv(archivo_ranking)
df_ranking = df_ranking.sort_values(by=['Importancia_DT', 'Mutual_Info'], ascending=[False, False])
mejores_caracteristicas = df_ranking['Caracteristica'].tolist()

caracteristicas = mejores_caracteristicas[:10]
columnas_desarrollo = caracteristicas + ['Ansiedad']

sujetos_unicos = np.sort(df_datos['Sujeto'].unique())
np.random.seed(42)
sujetos_mezclados = np.random.permutation(sujetos_unicos)
sujetos_desarrollo = sujetos_mezclados[:32]
df_desarrollo = df_datos[df_datos['Sujeto'].isin(sujetos_desarrollo)].copy()

df_desarrollo_sub = df_desarrollo[columnas_desarrollo].copy().apply(pd.to_numeric, errors='coerce').fillna(0)

pc_final = PC(variant='stable', ci_test='fisher_z', significance_level=0.05, return_type='dag', show_progress=False)
pc_final.fit(df_desarrollo_sub)
dag_pc = pc_final.causal_graph_

def evaluar_dag_metricas(dag, df_datos):
    from pgmpy.metrics import CorrelationScore, FisherC
    from pgmpy.base import DAG
    
    if not isinstance(dag, DAG):
        dag_val = DAG()
        dag_val.add_nodes_from(dag.nodes())
        dag_val.add_edges_from(dag.edges())
        dag = dag_val
        
    try:
        from scipy.stats import pearsonr
        def test_pearson_umbral(X, Y, Z=[], significance_level=0.01):
            r, _ = pearsonr(df_datos[X], df_datos[Y])
            return abs(r) < 0.20
        print("Tratando de evaluar CorrelationScore...")
        cs = CorrelationScore(ci_test=test_pearson_umbral, significance_level=0.01).evaluate(X=df_datos, causal_graph=dag)
    except Exception as e:
        print("Excepción en CorrelationScore:")
        traceback.print_exc()
        cs = np.nan
        
    try:
        print("Tratando de evaluar FisherC...")
        p_val, rmsea = FisherC(ci_test="pearsonr", compute_rmsea=True, show_progress=False).evaluate(X=df_datos, causal_graph=dag)
    except Exception as e:
        print("Excepción en FisherC:")
        traceback.print_exc()
        p_val, rmsea = np.nan, np.nan
        
    return cs, p_val, rmsea

cs, p_val, rmsea = evaluar_dag_metricas(dag_pc, df_desarrollo_sub)
print("Resultado final:", cs, p_val, rmsea)
