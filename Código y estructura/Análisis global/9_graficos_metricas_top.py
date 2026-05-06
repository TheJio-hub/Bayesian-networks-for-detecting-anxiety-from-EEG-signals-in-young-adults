import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)



def leer_metrica_csv(ruta_csv):
    df = pd.read_csv(ruta_csv, index_col=0)
    if 'Clase 1 (Validación)' in df.index:
        fila_validacion = df.loc['Clase 1 (Validación)']
    elif 'Clase 1' in df.index:
        fila_validacion = df.loc['Clase 1']
    else:
        fila_validacion = df.iloc[1] if len(df) > 1 else df.iloc[0]

    if 'Clase 1 (Entrenamiento)' in df.index:
        fila_entrenamiento = df.loc['Clase 1 (Entrenamiento)']
    else:
        filas_entrenamiento = [idx for idx in df.index if 'Entrenamiento' in str(idx)]
        if filas_entrenamiento:
            fila_entrenamiento = df.loc[filas_entrenamiento[0]]
        else:
            fila_entrenamiento = df.iloc[3] if len(df) > 3 else fila_validacion

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    return {
        'Exactitud': to_float(fila_validacion.get('Exactitud', np.nan)),
        'Sensibilidad': to_float(fila_validacion.get('Sensibilidad', np.nan)),
        'Especificidad': to_float(fila_validacion.get('Especificidad', np.nan)),
        'Exactitud_train': to_float(fila_entrenamiento.get('Exactitud', np.nan)),
        'Sensibilidad_train': to_float(fila_entrenamiento.get('Sensibilidad', np.nan)),
        'Especificidad_train': to_float(fila_entrenamiento.get('Especificidad', np.nan)),
    }


def construir_dataframe_base():
    base_dir = os.path.join('Resultados', 'Análisis global', 'Modelos generados')
    configuracion = [
        ('Top 40', 40),
        ('Top 30', 30),
        ('Top 20', 20),
        ('Top 15', 15),
        ('Top 10', 10),
    ]
    clasificadores = ['DT', 'RF', 'KNN', 'SVM', 'XGB']
    # Usamos solo cuatro métodos: Fisher, Mutual_Info, mRMR y DT (Importancia del árbol)
    metodos = ['Fisher', 'Mutual_Info', 'mRMR', 'DT']
    tops_permitidos_dt = {10, 15, 20}

    filas = []
    for nombre_top, sufijo in tqdm(configuracion, desc='Cargando métricas', unit='top'):
        for clasificador in clasificadores:
            for metodo in metodos:
                if clasificador == 'DT' and sufijo not in tops_permitidos_dt:
                    continue

                archivo = f'Resultados_{clasificador}_Ranking_{metodo}_top{sufijo}.csv'
                ruta_csv = os.path.join(base_dir, nombre_top, archivo)
                if not os.path.exists(ruta_csv):
                    continue

                metricas = leer_metrica_csv(ruta_csv)
                # Evitar añadir filas con métricas vacías (NaN)
                if np.isnan(metricas['Exactitud']) or np.isnan(metricas['Sensibilidad']) or np.isnan(metricas['Especificidad']):
                    continue

                filas.append({
                    'Top': nombre_top,
                    'Top_Orden': sufijo,
                    'Clasificador': clasificador,
                    'Metodo': metodo,
                    'Conjunto': 'Validación',
                    **metricas,
                })

                if np.isnan(metricas['Exactitud_train']) or np.isnan(metricas['Sensibilidad_train']) or np.isnan(metricas['Especificidad_train']):
                    continue

                filas.append({
                    'Top': nombre_top,
                    'Top_Orden': sufijo,
                    'Clasificador': clasificador,
                    'Metodo': metodo,
                    'Conjunto': 'Entrenamiento',
                    'Exactitud': metricas['Exactitud_train'],
                    'Sensibilidad': metricas['Sensibilidad_train'],
                    'Especificidad': metricas['Especificidad_train'],
                })

    return pd.DataFrame(filas)


def generar_graficos():
    df = construir_dataframe_base()
    if df.empty:
        print('No se encontraron CSV de métricas para graficar.')
        return

    base_salida = os.path.join('Resultados', 'Análisis global', 'Modelos generados', 'Gráficas comparativas')
    os.makedirs(base_salida, exist_ok=True)

    # Mostrar ticks de menor a mayor para claridad en los subgráficos
    orden_top = ['Top 10', 'Top 15', 'Top 20', 'Top 30', 'Top 40']
    orden_metodos = ['Fisher', 'Mutual_Info', 'mRMR', 'DT']
    nombres_metodo = {
        'Fisher': 'Fisher',
        'Mutual_Info': 'MI',
        'mRMR': 'mRMR',
        'DT': 'DT'
    }
    nombres_clasificador = {
        'DT': 'DT',
        'RF': 'RF',
        'KNN': 'KNN',
        'SVM': 'SVM',
        'XGB': 'XGB'
    }
    metricas = ['Exactitud', 'Sensibilidad', 'Especificidad']
    estilos = {
        'DT': {'marker': 'o', 'linestyle': '-', 'linewidth': 2.2, 'markersize': 5},
        'RF': {'marker': 's', 'linestyle': '--', 'linewidth': 2.0, 'markersize': 5},
        'KNN': {'marker': '^', 'linestyle': '-.', 'linewidth': 2.0, 'markersize': 5},
        'SVM': {'marker': 'D', 'linestyle': ':', 'linewidth': 2.2, 'markersize': 5},
        'XGB': {'marker': 'P', 'linestyle': '-', 'linewidth': 2.2, 'markersize': 5},
    }
    offset_por_clasificador = {
        'DT': (-10, 9),
        'RF': (10, 11),
        'KNN': (-10, -13),
        'SVM': (10, -15),
        'XGB': (-12, 12),
    }

    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })
    palette = ['#1f2937', '#374151', '#6b7280', '#0f766e', '#b45309']

    def dibujar_panel(ax, metodo, metrica, conjunto, ylim=(0.45, 1.0)):
        df_metodo = df[(df['Metodo'] == metodo) & (df['Conjunto'] == conjunto)].copy()
        if df_metodo.empty:
            ax.axis('off')
            return

        for idx, clasificador in enumerate(['DT', 'RF', 'KNN', 'SVM', 'XGB']):
            df_clf = df_metodo[df_metodo['Clasificador'] == clasificador].sort_values('Top_Orden')
            if df_clf.empty:
                continue
            ax.plot(
                df_clf['Top'],
                df_clf[metrica],
                label=nombres_clasificador[clasificador],
                color=palette[idx],
                **estilos[clasificador]
            )

            # Etiquetas numéricas sobre cada punto con offsets por clasificador
            for punto_idx, (x_val, y_val) in enumerate(zip(df_clf['Top'], df_clf[metrica])):
                dx, dy = offset_por_clasificador[clasificador]
                extra = punto_idx * (1 if dy >= 0 else -1)
                ax.annotate(
                    f'{y_val:.3f}',
                    (x_val, y_val),
                    textcoords='offset points',
                    xytext=(dx, dy + extra),
                    ha='center',
                    fontsize=9,
                    color=palette[idx],
                    bbox=dict(boxstyle='round,pad=0.14', facecolor='white', edgecolor='none', alpha=0.85),
                    clip_on=False,
                    zorder=6,
                )

        ax.set_title(f'{nombres_metodo[metodo]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Top')
        ax.set_ylabel(metrica)
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.001, 0.1))
        ax.set_xticks(orden_top)
        ax.set_xticklabels(orden_top)
        ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for conjunto, sufijo_salida in [('Validación', ''), ('Entrenamiento', '_train')]:
        n_rows = 2
        n_cols = 2
        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharex=True, sharey=True)
        axes1_flat = axes1.flatten()
        for i, metodo in enumerate(tqdm(orden_metodos, desc=f'Graficando exactitud {conjunto.lower()}', unit='metodo')):
            dibujar_panel(axes1_flat[i], metodo, 'Exactitud', conjunto, ylim=(0.6, 0.9))
        handles, labels = axes1_flat[0].get_legend_handles_labels()
        fig1.legend(handles, labels, loc='lower center', ncol=5, frameon=True, fontsize=13)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        ruta_exactitud = os.path.join(base_salida, f'Comparativa_Exactitud_2x2{sufijo_salida}.png')
        plt.savefig(ruta_exactitud, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f'Guardado: {ruta_exactitud}')

        n_rows2 = len(orden_metodos)
        fig2, axes2 = plt.subplots(n_rows2, 2, figsize=(14, 4 * n_rows2), sharex=True, sharey=True)
        for fila, metodo in enumerate(tqdm(orden_metodos, desc=f'Graficando sensibilidad/especificidad {conjunto.lower()}', unit='metodo')):
            dibujar_panel(axes2[fila, 0], metodo, 'Sensibilidad', conjunto, ylim=(0.4, 1.0))
            axes2[fila, 0].set_title(f'{nombres_metodo[metodo]} - Sensibilidad', fontsize=13, fontweight='bold')
            axes2[fila, 0].set_xlabel('Top')

            dibujar_panel(axes2[fila, 1], metodo, 'Especificidad', conjunto, ylim=(0.4, 1.0))
            axes2[fila, 1].set_title(f'{nombres_metodo[metodo]} - Especificidad', fontsize=13, fontweight='bold')
            axes2[fila, 1].set_xlabel('Top')

        handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
        fig2.legend(handles2, labels2, loc='lower center', ncol=5, frameon=True, fontsize=13)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        ruta_sens_espe = os.path.join(base_salida, f'Comparativa_Sensibilidad_Especificidad_4x2{sufijo_salida}.png')
        plt.savefig(ruta_sens_espe, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f'Guardado: {ruta_sens_espe}')


if __name__ == '__main__':
    generar_graficos()