import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.causal_discovery import PC, GES
from pgmpy.metrics import CorrelationScore, FisherC
from pgmpy.base import DAG
import os

archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')

df_datos = pd.read_parquet(archivo_datos)
df_datos = df_datos[(df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1)].copy()
df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0 if x == 0 else 1)

df_ranking = pd.read_csv(archivo_ranking)
df_ranking = df_ranking.sort_values(by=['Importancia_DT', 'Mutual_Info'], ascending=[False, False])
mejores_caracteristicas = df_ranking['Caracteristica'].tolist()

caracteristicas = mejores_caracteristicas[:10] + ['Ansiedad']
discretizador = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
df_disc = pd.DataFrame(discretizador.fit_transform(df_datos[caracteristicas]), columns=caracteristicas).astype(int)

# PC
estimador_pc = PC(variant='stable', ci_test='chi_square', significance_level=0.05, return_type='dag', show_progress=False)
estimador_pc.fit(df_disc)
dag_pc = estimador_pc.causal_graph_

print("--- PC (Top 10) ---")
print("Num aristas:", len(dag_pc.edges()))
cs_pc = CorrelationScore(ci_test='chi_square', significance_level=0.05).evaluate(X=df_disc, causal_graph=dag_pc)
p_pc, rm_pc = FisherC(ci_test='chi_square', compute_rmsea=True, show_progress=False).evaluate(X=df_disc, causal_graph=dag_pc)
print(f"CorrelationScore: {cs_pc}")
print(f"FisherC p-val: {p_pc}, RMSEA: {rm_pc}")

# GES
estimador_ges = GES(scoring_method='bic-d', return_type='dag')
estimador_ges.fit(df_disc)
dag_ges = estimador_ges.causal_graph_

print("\n--- GES (Top 10) ---")
print("Num aristas:", len(dag_ges.edges()))
cs_ges = CorrelationScore(ci_test='chi_square', significance_level=0.05).evaluate(X=df_disc, causal_graph=dag_ges)
p_ges, rm_ges = FisherC(ci_test='chi_square', compute_rmsea=True, show_progress=False).evaluate(X=df_disc, causal_graph=dag_ges)
print(f"CorrelationScore: {cs_ges}")
print(f"FisherC p-val: {p_ges}, RMSEA: {rm_ges}")
