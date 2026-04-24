import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)



def normalizar_z_score_relajacion(df: pd.DataFrame) -> pd.DataFrame:
    columnas_metadatos = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']
    columnas_caracteristicas = [
        c for c in df.columns
        if c not in columnas_metadatos and np.issubdtype(df[c].dtype, np.number)
    ]

    df_norm = df.copy()
    sujetos_unicos = df_norm['Sujeto'].unique()

    for sujeto in tqdm(sujetos_unicos, desc='Normalizando por sujeto', unit='sujeto'):
        mascara_sujeto = df_norm['Sujeto'] == sujeto
        datos_sujeto = df_norm.loc[mascara_sujeto, columnas_caracteristicas]

        mascara_relajacion = (df_norm['Sujeto'] == sujeto) & (df_norm['Tarea'] == 'Relajacion')
        datos_relajacion = df_norm.loc[mascara_relajacion, columnas_caracteristicas]

        if datos_relajacion.empty:
            continue

        mu_relajacion = datos_relajacion.mean()
        sigma_relajacion = datos_relajacion.std().replace(0, 1.0)

        df_norm.loc[mascara_sujeto, columnas_caracteristicas] = (
            datos_sujeto - mu_relajacion
        ) / sigma_relajacion

    return df_norm


def cargar_df(ruta_parquet: str, ruta_csv: str) -> pd.DataFrame:
    if os.path.exists(ruta_parquet):
        return pd.read_parquet(ruta_parquet)
    if os.path.exists(ruta_csv):
        return pd.read_csv(ruta_csv)
    return pd.DataFrame()


def guardar_df(df: pd.DataFrame, ruta_parquet: str, ruta_csv: str) -> None:
    df.to_parquet(ruta_parquet, index=False)
    df.to_csv(ruta_csv, index=False)


def ejecutar_normalizacion() -> None:
    directorio = os.path.join('Resultados', 'Exploratorio')
    os.makedirs(directorio, exist_ok=True)

    df_bandas = cargar_df(
        os.path.join(directorio, 'datos_bandas_log10.parquet'),
        os.path.join(directorio, 'datos_bandas_log10.csv'),
    )
    if df_bandas.empty:
        df_bandas = cargar_df(
            os.path.join(directorio, 'potencias_log10.parquet'),
            os.path.join(directorio, 'potencias_log10.csv'),
        )

    df_asim = cargar_df(
        os.path.join(directorio, 'datos_asimetria_log10.parquet'),
        os.path.join(directorio, 'datos_asimetria_log10.csv'),
    )
    df_ratios = cargar_df(
        os.path.join(directorio, 'datos_ratios_log10.parquet'),
        os.path.join(directorio, 'datos_ratios_log10.csv'),
    )

    if df_bandas.empty:
        raise FileNotFoundError('No se encontro el dataset log10 de bandas en Resultados/Exploratorio.')

    df_bandas_norm = normalizar_z_score_relajacion(df_bandas)
    guardar_df(
        df_bandas_norm,
        os.path.join(directorio, 'datos_bandas_normalizados.parquet'),
        os.path.join(directorio, 'datos_bandas_normalizados.csv'),
    )

    if not df_asim.empty:
        df_asim_norm = normalizar_z_score_relajacion(df_asim)
        guardar_df(
            df_asim_norm,
            os.path.join(directorio, 'datos_asimetria_normalizados.parquet'),
            os.path.join(directorio, 'datos_asimetria_normalizados.csv'),
        )
    else:
        df_asim_norm = pd.DataFrame()

    if not df_ratios.empty:
        df_ratios_norm = normalizar_z_score_relajacion(df_ratios)
        guardar_df(
            df_ratios_norm,
            os.path.join(directorio, 'datos_ratios_normalizados.parquet'),
            os.path.join(directorio, 'datos_ratios_normalizados.csv'),
        )
    else:
        df_ratios_norm = pd.DataFrame()

    columnas_join = ['Sujeto', 'Tarea', 'Trial', 'Epoca', 'Puntaje']

    if not df_asim_norm.empty and not df_ratios_norm.empty:
        df_merged = pd.merge(df_bandas_norm, df_asim_norm, on=columnas_join, how='inner')
        df_merged = pd.merge(df_merged, df_ratios_norm, on=columnas_join, how='inner')
    elif not df_asim_norm.empty:
        df_merged = pd.merge(df_bandas_norm, df_asim_norm, on=columnas_join, how='inner')
    elif not df_ratios_norm.empty:
        df_merged = pd.merge(df_bandas_norm, df_ratios_norm, on=columnas_join, how='inner')
    else:
        df_merged = df_bandas_norm.copy()

    guardar_df(
        df_merged,
        os.path.join(directorio, 'datos_completos_normalizados.parquet'),
        os.path.join(directorio, 'datos_completos_normalizados.csv'),
    )


if __name__ == '__main__':
    ejecutar_normalizacion()
