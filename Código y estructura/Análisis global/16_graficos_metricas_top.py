import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as pe


def leer_metrica_csv(ruta_csv):
    df = pd.read_csv(ruta_csv, index_col=0)
    fila_clase_1 = df.loc['Clase 1']
    return {
        'Exactitud': float(fila_clase_1['Exactitud']),
        'Sensibilidad': float(fila_clase_1['Sensibilidad']),
        'Especificidad': float(fila_clase_1['Especificidad'])
    }


def construir_dataframe_base():
    base_dir = os.path.join('Resultados', 'Análisis global', 'Modelos (Usando Rankings)')
    configuracion = [
        ('Top 15', 15),
        ('Top 20', 20),
        ('Top 30', 30),
    ]
    clasificadores = ['DT', 'RF', 'KNN', 'SVM']
    metodos = ['Fisher', 'Mutual_Info', 'mRMR', 'Random_Forest']

    filas = []
    for nombre_top, sufijo in configuracion:
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

    base_salida = os.path.join('Resultados', 'Análisis global', 'Modelos (Usando Rankings)', 'Gráficas comparativas')
    os.makedirs(base_salida, exist_ok=True)

    orden_top = ['Top 15', 'Top 20', 'Top 30']
    orden_metodos = ['Fisher', 'Mutual_Info', 'mRMR', 'Random_Forest']
    nombres_metodo = {
        'Fisher': 'Fisher',
        'Mutual_Info': 'MI',
        'mRMR': 'mRMR',
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
        'DT': {'marker': 'o', 'linestyle': '-', 'linewidth': 2},
        'RF': {'marker': 's', 'linestyle': '-', 'linewidth': 2},
        'KNN': {'marker': '^', 'linestyle': '-', 'linewidth': 2},
        'SVM': {'marker': 'D', 'linestyle': '-', 'linewidth': 2},
    }
    offset_por_clasificador = {
        'DT': (-14, 14),
        'RF': (14, 22),
        'KNN': (-14, -20),
        'SVM': (14, -32),
    }

    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
    })
    palette = sns.color_palette('Set2', 4)

    def dibujar_panel(ax, metodo, metrica, mostrar_leyenda=False):
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

            for punto_idx, (x_val, y_val) in enumerate(zip(df_clf['Top'], df_clf[metrica])):
                base_dx, base_dy = offset_por_clasificador[clasificador]
                desplazamiento_x = base_dx + (punto_idx * 2 if base_dx > 0 else -punto_idx * 2)
                desplazamiento_y = base_dy + (punto_idx * 3 if punto_idx % 2 == 0 else -punto_idx * 3)
                ax.annotate(
                    f'{y_val:.3f}',
                    (x_val, y_val),
                    textcoords='offset points',
                    xytext=(desplazamiento_x, desplazamiento_y),
                    ha='center',
                    fontsize=6.5,
                    color=palette[idx],
                    bbox=dict(boxstyle='round,pad=0.16', facecolor='white', edgecolor='none', alpha=0.9),
                    clip_on=False,
                    zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
                )

        ax.set_title(metrica, fontsize=11, fontweight='bold')
        ax.set_xlabel('Top')
        ax.set_ylabel('Valor')
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        ax.set_yticks(np.arange(0.4, 1.01, 0.05), minor=True)
        ax.set_xticks(orden_top)
        ax.set_xticklabels(orden_top)
        ax.margins(x=0.12, y=0.08)
        ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.45)
        ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if mostrar_leyenda:
            ax.legend(frameon=True, fontsize=8, loc='upper left')

    for metodo in orden_metodos:
        df_metodo = df[df['Metodo'] == metodo].copy()
        if df_metodo.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        fig.suptitle(f'Rendimiento por Top para {nombres_metodo[metodo]}', fontsize=16, fontweight='bold')

        for ax, metrica in zip(axes, metricas):
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
                for punto_idx, (x_val, y_val) in enumerate(zip(df_clf['Top'], df_clf[metrica])):
                    base_dx, base_dy = offset_por_clasificador[clasificador]
                    desplazamiento_x = base_dx + (punto_idx * 2 if base_dx > 0 else -punto_idx * 2)
                    desplazamiento_y = base_dy + (punto_idx * 3 if punto_idx % 2 == 0 else -punto_idx * 3)
                    ax.annotate(
                        f'{y_val:.3f}',
                        (x_val, y_val),
                        textcoords='offset points',
                        xytext=(desplazamiento_x, desplazamiento_y),
                        ha='center',
                        fontsize=7,
                        color=palette[idx],
                        bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.9),
                        clip_on=False,
                        zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
                    )

            ax.set_title(metrica)
            ax.set_xlabel('Top')
            ax.set_ylabel('Valor')
            ax.set_ylim(0.4, 1.0)
            ax.set_yticks(np.arange(0.4, 1.01, 0.1))
            ax.set_yticks(np.arange(0.4, 1.01, 0.05), minor=True)
            ax.set_xticks(orden_top)
            ax.set_xticklabels(orden_top)
            ax.margins(x=0.12, y=0.08)
            ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.45)
            ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5, alpha=0.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(frameon=True, fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        ruta_png = os.path.join(base_salida, f'{metodo}_metricas_top.png')
        plt.savefig(ruta_png, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f'Guardado: {ruta_png}')

    fig, axes = plt.subplots(4, 3, figsize=(20, 16), sharex=True)
    for fila, metodo in enumerate(orden_metodos):
        for col, metrica in enumerate(metricas):
            mostrar_leyenda = (fila == 0 and col == 2)
            dibujar_panel(axes[fila, col], metodo, metrica, mostrar_leyenda=mostrar_leyenda)
        axes[fila, 0].set_ylabel(f'{nombres_metodo[metodo]}\nValor')

    for col, metrica in enumerate(metricas):
        axes[0, col].set_title(metrica, fontsize=12, fontweight='bold')

    for ax in axes[-1, :]:
        ax.set_xlabel('Top')

    fig.suptitle('Comparativa unificada de métricas por método y clasificador', fontsize=16, fontweight='bold')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    ruta_unificada = os.path.join(base_salida, 'Comparativa_Unificada_4x3.png')
    plt.savefig(ruta_unificada, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Guardado: {ruta_unificada}')


if __name__ == '__main__':
    generar_graficos()