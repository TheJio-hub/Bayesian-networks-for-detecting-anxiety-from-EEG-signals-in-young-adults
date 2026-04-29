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
    fila_clase_1 = df.loc['Clase 1']
    return {
        'Exactitud': float(fila_clase_1['Exactitud']),
        'Sensibilidad': float(fila_clase_1['Sensibilidad']),
        'Especificidad': float(fila_clase_1['Especificidad'])
    }


def construir_dataframe_base():
    base_dir = os.path.join('Resultados', 'Análisis global', 'Modelos generados')
    configuracion = [
        ('Top 15', 15),
        ('Top 20', 20),
        ('Top 30', 30),
    ]
    clasificadores = ['DT', 'RF', 'KNN', 'SVM']
    metodos = ['Fisher', 'Mutual_Info', 'mRMR', 'mRMR_30', 'mRMR_20', 'Random_Forest']

    filas = []
    for nombre_top, sufijo in tqdm(configuracion, desc='Cargando métricas', unit='top'):
        for clasificador in clasificadores:
            for metodo in metodos:
                archivo = f'Resultados_{clasificador}_Ranking_{metodo}_top{sufijo}.csv'
                ruta_csv = os.path.join(base_dir, nombre_top, archivo)
                if not os.path.exists(ruta_csv):
                    continue

                metricas = leer_metrica_csv(ruta_csv)
                filas.append({
                    'Top': nombre_top,
                    'Top_Orden': sufijo,
                    'Clasificador': clasificador,
                    'Metodo': metodo,
                    **metricas,
                })

    return pd.DataFrame(filas)


def generar_graficos():
    df = construir_dataframe_base()
    if df.empty:
        print('No se encontraron CSV de métricas para graficar.')
        return

    base_salida = os.path.join('Resultados', 'Análisis global', 'Modelos generados', 'Gráficas comparativas')
    os.makedirs(base_salida, exist_ok=True)

    orden_top = ['Top 30', 'Top 20', 'Top 15']
    orden_metodos = ['Fisher', 'Mutual_Info', 'mRMR', 'mRMR_30', 'mRMR_20', 'Random_Forest']
    nombres_metodo = {
        'Fisher': 'Fisher',
        'Mutual_Info': 'MI',
        'mRMR': 'mRMR(40)',
        'mRMR_30': 'mRMR(30)',
        'mRMR_20': 'mRMR(20)',
        'Random_Forest': 'RF'
    }
    nombres_clasificador = {
        'DT': 'DT',
        'RF': 'RF',
        'KNN': 'KNN',
        'SVM': 'SVM'
    }
    metricas = ['Exactitud', 'Sensibilidad', 'Especificidad']
    estilos = {
        'DT': {'marker': 'o', 'linestyle': '-', 'linewidth': 2.2, 'markersize': 5},
        'RF': {'marker': 's', 'linestyle': '--', 'linewidth': 2.0, 'markersize': 5},
        'KNN': {'marker': '^', 'linestyle': '-.', 'linewidth': 2.0, 'markersize': 5},
        'SVM': {'marker': 'D', 'linestyle': ':', 'linewidth': 2.2, 'markersize': 5},
    }
    offset_por_clasificador = {
        'DT': (-10, 9),
        'RF': (10, 11),
        'KNN': (-10, -13),
        'SVM': (10, -15),
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
    palette = ['#1f2937', '#374151', '#6b7280', '#0f766e']

    def dibujar_panel(ax, metodo, metrica, ylim=(0.4, 1.0)):
        df_metodo = df[df['Metodo'] == metodo].copy()
        if df_metodo.empty:
            ax.axis('off')
            return

        for idx, clasificador in enumerate(['DT', 'RF', 'KNN', 'SVM']):
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

    # Figura 1: Exactitud en 3x2 (métodos)
    n_metodos = len(orden_metodos)
    n_rows = 3
    n_cols = 2
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(14, 14), sharex=True, sharey=True)
    for i, metodo in enumerate(tqdm(orden_metodos, desc='Graficando exactitud', unit='metodo')):
        fila, col = divmod(i, n_cols)
        dibujar_panel(axes1[fila, col], metodo, 'Exactitud', ylim=(0.6, 0.9))
    handles, labels = axes1[0, 0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower center', ncol=4, frameon=True, fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    ruta_exactitud = os.path.join(base_salida, 'Comparativa_Exactitud_3x2.png')
    plt.savefig(ruta_exactitud, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f'Guardado: {ruta_exactitud}')

    # Figura 2: Sensibilidad y Especificidad en Nx2 (uno por método)
    n_rows2 = len(orden_metodos)
    fig2, axes2 = plt.subplots(n_rows2, 2, figsize=(14, 4 * n_rows2), sharex=True, sharey=True)
    for fila, metodo in enumerate(tqdm(orden_metodos, desc='Graficando sensibilidad/especificidad', unit='metodo')):
        dibujar_panel(axes2[fila, 0], metodo, 'Sensibilidad', ylim=(0.4, 1.0))
        axes2[fila, 0].set_title(f'{nombres_metodo[metodo]} - Sensibilidad', fontsize=13, fontweight='bold')

        dibujar_panel(axes2[fila, 1], metodo, 'Especificidad', ylim=(0.4, 1.0))
        axes2[fila, 1].set_title(f'{nombres_metodo[metodo]} - Especificidad', fontsize=13, fontweight='bold')

    handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='lower center', ncol=4, frameon=True, fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    ruta_sens_espe = os.path.join(base_salida, 'Comparativa_Sensibilidad_Especificidad_4x2.png')
    plt.savefig(ruta_sens_espe, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'Guardado: {ruta_sens_espe}')


if __name__ == '__main__':
    generar_graficos()