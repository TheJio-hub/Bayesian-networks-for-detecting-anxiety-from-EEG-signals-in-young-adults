import runpy
import os
import pandas as pd
import numpy as np
import sys

# Paths
base = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(base, 'Código y estructura', 'Red Bayesiana', 'Global', '4_Red_Mixta_Regional.py')
parquet_path = os.path.join(base, 'Resultados', 'Exploratorio', 'datos_completos_normalizados.parquet')

print('Cargando módulo original desde:', module_path)
mod_globals = runpy.run_path(module_path)

if 'ejecutar_loso' not in mod_globals:
    print('No se encontró ejecutar_loso en el módulo.')
    sys.exit(1)

ejecutar_loso = mod_globals['ejecutar_loso']

print('Leyendo datos:', parquet_path)
df = pd.read_parquet(parquet_path)
# Recompute regional means as in original script
regiones_canales = {
    'Reg_Frontal': ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8'],
    'Reg_Central': ['Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'C3', 'C4'],
    'Reg_Temporal': ['FT9', 'FT10', 'T7', 'T8', 'CP5', 'CP6'],
    'Reg_Parietal': ['CP1', 'CP2', 'P3', 'P4', 'P7', 'P8', 'Pz'],
    'Reg_Occipital': ['O1', 'O2', 'Oz', 'PO9', 'PO10']
}
bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
for region, canales in regiones_canales.items():
    columnas_absolutas = [f"{canal}_{banda}" for canal in canales for banda in bandas]
    # if columns present
    existing = [c for c in columnas_absolutas if c in df.columns]
    if not existing:
        print(f'Advertencia: no se encontraron columnas para {region}. Columnas esperadas (algunas faltan).')
    df[region] = df[existing].mean(axis=1)

# Filter and prepare
df = df[(df['Puntaje'] == 0) | (df['Puntaje'] >= 1)].copy()
df['Ansiedad'] = df['Puntaje'].apply(lambda x: 0 if x == 0 else 1)

sujetos = np.sort(df['Sujeto'].unique())
np.random.seed(42)
sujetos = np.random.permutation(sujetos)
# Reduce to fewer sujetos for debugging
sujetos_desarrollo = sujetos[:12]
print('Usando sujetos de desarrollo (reducido):', sujetos_desarrollo)

df_dev = df[df['Sujeto'].isin(sujetos_desarrollo)].copy()
grupos = df_dev['Sujeto'].values

caracteristicas = list(regiones_canales.keys())
print('Caracteristicas usadas:', caracteristicas)

print('Iniciando ejecución de ejecutar_loso con algoritmo GES...')
y_real, y_pred = ejecutar_loso(df_dev, caracteristicas, 'Ansiedad', grupos, 'GES')

print('Finalizado. Resultados:')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Acc:', accuracy_score(y_real, y_pred))
print('Prec:', precision_score(y_real, y_pred, zero_division=0))
print('Sens:', recall_score(y_real, y_pred, zero_division=0))
print('F1:', f1_score(y_real, y_pred, zero_division=0))
