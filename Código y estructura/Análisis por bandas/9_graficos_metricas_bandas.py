import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)


def leer_metricas_por_csv(ruta_csv: str) -> dict:
    df = pd.read_csv(ruta_csv, index_col=0)
    # Selección segura de filas
    if 'Clase 1 (Validación)' in df.index:
        row_val = df.loc['Clase 1 (Validación)']
    elif 'Clase 1' in df.index:
        row_val = df.loc['Clase 1']
    else:
        row_val = df.iloc[1] if len(df) > 1 else df.iloc[0]

    if 'Clase 1 (Entrenamiento)' in df.index:
        row_train = df.loc['Clase 1 (Entrenamiento)']
    else:
        row_train = None

    def to_f(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    return {
        'Exactitud': to_f(row_val.get('Exactitud')),
        'Sensibilidad': to_f(row_val.get('Sensibilidad')),
        'Especificidad': to_f(row_val.get('Especificidad')),
        'Exactitud_train': to_f(row_train.get('Exactitud')) if row_train is not None else np.nan,
        'Sensibilidad_train': to_f(row_train.get('Sensibilidad')) if row_train is not None else np.nan,
        'Especificidad_train': to_f(row_train.get('Especificidad')) if row_train is not None else np.nan,
    }


def construir_dataframe_bandas():
    raiz = os.path.join('Resultados', 'Análisis por bandas', 'Modelos por banda')
    bloques = ["Alpha", "Beta", "Delta", "Asimetria", "Ratios"]
    metodos = ['Fisher', 'Mutual_Info', 'mRMR', 'DT']
    clasificadores = ['DT', 'KNN', 'SVM', 'XGB']

    filas = []
    for bloque in tqdm(bloques, desc='Bloques', unit='bloque'):
        # determinar tops a usar según bloque
        if bloque == 'Asimetria':
            tops = [(8, 'Top 8')]
        else:
            tops = [(8, 'Top 8'), (32, 'Top 32')]

        for top_num, top_nombre in tops:
            for metodo in metodos:
                for clf in clasificadores:
                    ruta = os.path.join(raiz, top_nombre, bloque, f'Resultados_{clf}_Ranking_{metodo}_top{top_num}.csv')
                    if not os.path.exists(ruta):
                        continue
                    metricas = leer_metricas_por_csv(ruta)
                    if np.isnan(metricas['Exactitud']) or np.isnan(metricas['Sensibilidad']) or np.isnan(metricas['Especificidad']):
                        continue

                    filas.append({
                        'Bloque': bloque,
                        'Top': top_nombre,
                        'Top_Orden': top_num,
                        'Metodo': metodo,
                        'Clasificador': clf,
                        'Conjunto': 'Validación',
                        'Exactitud': metricas['Exactitud'],
                        'Sensibilidad': metricas['Sensibilidad'],
                        'Especificidad': metricas['Especificidad'],
                    })

                    if not np.isnan(metricas['Exactitud_train']):
                        filas.append({
                            'Bloque': bloque,
                            'Top': top_nombre,
                            'Top_Orden': top_num,
                            'Metodo': metodo,
                            'Clasificador': clf,
                            'Conjunto': 'Entrenamiento',
                            'Exactitud': metricas['Exactitud_train'],
                            'Sensibilidad': metricas['Sensibilidad_train'],
                            'Especificidad': metricas['Especificidad_train'],
                        })

    return pd.DataFrame(filas)


def calcular_ylim_dinamico(df_filtrado, metrica):
    valores = df_filtrado[metrica].dropna()
    if valores.empty:
        return (0.5, 1.0)
    y_min = valores.min()
    y_max = valores.max()
    rango = y_max - y_min
    margen = max(rango * 0.05, 0.015)
    # redondeo a 0.02
    y_min_aj = np.floor((y_min - margen) / 0.02) * 0.02
    y_max_aj = np.ceil((y_max + margen) / 0.02) * 0.02
    if y_max_aj - y_min_aj < 0.06:
        centro = (y_min_aj + y_max_aj) / 2
        y_min_aj = centro - 0.03
        y_max_aj = centro + 0.03
    return (max(0.3, y_min_aj), min(1.0, y_max_aj))


def generar_graficas_bandas():
    df = construir_dataframe_bandas()
    if df.empty:
        print('No hay datos por bandas para graficar.')
        return

    salida = os.path.join('Resultados', 'Análisis por bandas', 'Gráficas comparativas')
    os.makedirs(salida, exist_ok=True)

    bloques = df['Bloque'].unique()
    metodos = ['Fisher', 'Mutual_Info', 'mRMR', 'DT']
    clasificadores = ['DT', 'KNN', 'SVM', 'XGB']
    palette = ['#1f2937', '#374151', '#6b7280', '#0f766e']
    estilos = {
        'DT': {'marker': 'o', 'linestyle': '-', 'linewidth': 2.2, 'markersize': 6},
        'KNN': {'marker': '^', 'linestyle': '-.', 'linewidth': 2.0, 'markersize': 6},
        'SVM': {'marker': 'D', 'linestyle': ':', 'linewidth': 2.2, 'markersize': 6},
        'XGB': {'marker': 'P', 'linestyle': '-', 'linewidth': 2.2, 'markersize': 6},
    }

    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams.update({'font.family': 'serif', 'axes.titlesize': 14, 'axes.labelsize': 12})

    for bloque in tqdm(bloques, desc='Graficando por bandas', unit='bloque'):
        df_b = df[df['Bloque'] == bloque]
        # ordenar tops disponibles
        orden_top = sorted(df_b['Top_Orden'].unique())
        etiquetas_top = [f'Top {t}' for t in orden_top]

        # generar figuras para Exactitud, Sensibilidad, Especificidad
        for conjunto in ['Validación', 'Entrenamiento']:
            df_conj = df_b[df_b['Conjunto'] == conjunto]
            # Exactitud
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
            axes_flat = axes.flatten()
            for i, metodo in enumerate(metodos):
                ax = axes_flat[i]
                df_m = df_conj[df_conj['Metodo'] == metodo]
                if df_m.empty:
                    ax.axis('off')
                    continue
                ylim = calcular_ylim_dinamico(df_m, 'Exactitud')
                for idx, clf in enumerate(clasificadores):
                    df_clf = df_m[df_m['Clasificador'] == clf].sort_values('Top_Orden')
                    if df_clf.empty:
                        continue
                    ax.plot(df_clf['Top'], df_clf['Exactitud'], label=clf, color=palette[idx], **estilos.get(clf, {}))
                    for x, y in zip(df_clf['Top'], df_clf['Exactitud']):
                        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0,6), ha='center', fontsize=9, fontweight='bold')
                ax.set_ylim(ylim)
                ax.set_xticks([f'Top {t}' for t in orden_top])
                ax.set_ylabel('Exactitud')
                ax.grid(True, axis='y', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            fig.suptitle(f'{bloque} - Exactitud ({conjunto})', fontsize=16, fontweight='bold')
            fig.legend(loc='lower center', ncol=4, frameon=True)
            plt.tight_layout(rect=[0, 0.06, 1, 0.93])
            ruta = os.path.join(salida, f'Comparativa_{bloque}_Exactitud_{conjunto}.png')
            plt.savefig(ruta, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Sensibilidad
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
            axes_flat = axes.flatten()
            for i, metodo in enumerate(metodos):
                ax = axes_flat[i]
                df_m = df_conj[df_conj['Metodo'] == metodo]
                if df_m.empty:
                    ax.axis('off')
                    continue
                ylim = calcular_ylim_dinamico(df_m, 'Sensibilidad')
                for idx, clf in enumerate(clasificadores):
                    df_clf = df_m[df_m['Clasificador'] == clf].sort_values('Top_Orden')
                    if df_clf.empty:
                        continue
                    ax.plot(df_clf['Top'], df_clf['Sensibilidad'], label=clf, color=palette[idx], **estilos.get(clf, {}))
                    for x, y in zip(df_clf['Top'], df_clf['Sensibilidad']):
                        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0,6), ha='center', fontsize=9, fontweight='bold')
                ax.set_ylim(ylim)
                ax.set_xticks([f'Top {t}' for t in orden_top])
                ax.set_ylabel('Sensibilidad')
                ax.grid(True, axis='y', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            fig.suptitle(f'{bloque} - Sensibilidad ({conjunto})', fontsize=16, fontweight='bold')
            fig.legend(loc='lower center', ncol=4, frameon=True)
            plt.tight_layout(rect=[0, 0.06, 1, 0.93])
            ruta = os.path.join(salida, f'Comparativa_{bloque}_Sensibilidad_{conjunto}.png')
            plt.savefig(ruta, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Especificidad
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
            axes_flat = axes.flatten()
            for i, metodo in enumerate(metodos):
                ax = axes_flat[i]
                df_m = df_conj[df_conj['Metodo'] == metodo]
                if df_m.empty:
                    ax.axis('off')
                    continue
                ylim = calcular_ylim_dinamico(df_m, 'Especificidad')
                for idx, clf in enumerate(clasificadores):
                    df_clf = df_m[df_m['Clasificador'] == clf].sort_values('Top_Orden')
                    if df_clf.empty:
                        continue
                    ax.plot(df_clf['Top'], df_clf['Especificidad'], label=clf, color=palette[idx], **estilos.get(clf, {}))
                    for x, y in zip(df_clf['Top'], df_clf['Especificidad']):
                        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0,6), ha='center', fontsize=9, fontweight='bold')
                ax.set_ylim(ylim)
                ax.set_xticks([f'Top {t}' for t in orden_top])
                ax.set_ylabel('Especificidad')
                ax.grid(True, axis='y', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            fig.suptitle(f'{bloque} - Especificidad ({conjunto})', fontsize=16, fontweight='bold')
            fig.legend(loc='lower center', ncol=4, frameon=True)
            plt.tight_layout(rect=[0, 0.06, 1, 0.93])
            ruta = os.path.join(salida, f'Comparativa_{bloque}_Especificidad_{conjunto}.png')
            plt.savefig(ruta, dpi=300, bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    generar_graficas_bandas()
