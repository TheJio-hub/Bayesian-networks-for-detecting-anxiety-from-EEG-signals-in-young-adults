import pandas as pd
import os

archivo_ranking = os.path.join('Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')
df_ranking = pd.read_csv(archivo_ranking)
print(df_ranking['Caracteristica'].head().tolist())
