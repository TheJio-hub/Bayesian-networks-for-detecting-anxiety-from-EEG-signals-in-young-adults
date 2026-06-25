import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.causal_discovery import GES
import os

archivo_datos = os.path.join('Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')
df_datos = pd.read_parquet(archivo_datos)
df_datos = df_datos[(df_datos['Puntaje'] == 0) | (df_datos['Puntaje'] >= 1)].copy()
df_datos['Ansiedad'] = df_datos['Puntaje'].apply(lambda x: 0 if x == 0 else 1)

caracteristicas = ['P3_Alpha', 'P4_Alpha', 'Fz_Delta', 'Ansiedad']
discretizador = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
df_disc = pd.DataFrame(discretizador.fit_transform(df_datos[caracteristicas]), columns=caracteristicas).astype(int)

print("Entrenando GES con bic-d en datos discretos...")
ges = GES(scoring_method='bic-d', return_type='dag')
ges.fit(df_disc)
print("DAG resultante aristas:")
print(ges.causal_graph_.edges())
