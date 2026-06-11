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
    tops_permitidos_metodo_dt = {10, 15, 20}  # Método DT solo en estos tops

    filas = []
    for nombre_top, sufijo in tqdm(configuracion, desc='Cargando métricas', unit='top'):
        for clasificador in clasificadores:
            for metodo in metodos:
                # DT como MÉTODO solo tiene Top 10, 15, 20
                if metodo == 'DT' and sufijo not in tops_permitidos_metodo_dt:
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

    # Para visualización, mostrar todos los Tops disponibles (Top 40, Top 30, Top 20, Top 15, Top 10).
    # Los datos completos se conservan en el DataFrame original para las tablas.
    df_grafico = df.copy()

    base_salida = os.path.join('Resultados', 'Análisis global', 'Modelos generados', 'Gráficas comparativas')
    os.makedirs(base_salida, exist_ok=True)

    # Mostrar ticks de menor a mayor para claridad en los subgráficos
    orden_top = ['Top 40', 'Top 30', 'Top 20', 'Top 15', 'Top 10']
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
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
    })
    palette = ['#1f2937', '#374151', '#6b7280', '#0f766e', '#b45309']

    def calcular_ylim_dinamico(df_filtrado, metrica):
        """Calcula límites Y dinámicos basados en los datos, muy ajustados."""
        valores = df_filtrado[metrica].dropna()
        if valores.empty:
            return (0.5, 1.0)
        
        y_min = valores.min()
        y_max = valores.max()
        rango = y_max - y_min
        
        # Margen muy pequeño: 5% del rango o 0.015, lo que sea mayor
        margen = max(rango * 0.05, 0.015)
        
        # Redondear a múltiplos de 0.02 para precisión y valores más limpios
        y_min_ajustado = np.floor((y_min - margen) / 0.02) * 0.02
        y_max_ajustado = np.ceil((y_max + margen) / 0.02) * 0.02
        
        # Asegurar que el rango mínimo sea 0.06
        if y_max_ajustado - y_min_ajustado < 0.06:
            centro = (y_min_ajustado + y_max_ajustado) / 2
            y_min_ajustado = centro - 0.03
            y_max_ajustado = centro + 0.03
        
        return (max(0.3, y_min_ajustado), min(1.0, y_max_ajustado))

    def dibujar_panel(ax, metodo, metrica, conjunto):
        df_metodo = df_grafico[(df_grafico['Metodo'] == metodo) & (df_grafico['Conjunto'] == conjunto)].copy()
        if df_metodo.empty:
            ax.axis('off')
            return

        # Calcular límites Y dinámicos
        ylim = calcular_ylim_dinamico(df_metodo, metrica)

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

            # Etiquetas numéricas por punto (más visibles para reporte)
            for punto_idx, (x_val, y_val) in enumerate(zip(df_clf['Top'], df_clf[metrica])):
                dx, dy = offset_por_clasificador[clasificador]
                ajuste = punto_idx * (1 if dy >= 0 else -1)
                ax.annotate(
                    f'{y_val:.3f}',
                    (x_val, y_val),
                    textcoords='offset points',
                    xytext=(dx, dy + ajuste),
                    ha='center',
                    fontsize=10,
                    fontweight='bold',
                    color=palette[idx],
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.88),
                    clip_on=False,
                    zorder=6,
                )

        # Título por panel removido (se usa título global de figura para evitar redundancia)
        ax.set_title(nombres_metodo.get(metodo, metodo), fontsize=14, fontweight='bold', pad=8)
        ax.set_xlabel('Top')
        ax.set_ylabel(metrica)
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.001, 0.05))
        ax.set_xticks(orden_top)
        ax.set_xticklabels(orden_top)
        ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Marcar etiquetas de ticks en negrita para mayor legibilidad
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')

    for conjunto, sufijo_salida in [('Validación', ''), ('Entrenamiento', '_train')]:
        n_rows = 2
        n_cols = 2
        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(12, 12), sharex=True)
        axes1_flat = axes1.flatten()
        for i, metodo in enumerate(tqdm(orden_metodos, desc=f'Graficando exactitud {conjunto.lower()}', unit='metodo')):
            dibujar_panel(axes1_flat[i], metodo, 'Exactitud', conjunto)
        # Título global de la figura (evita títulos redundantes por panel)
        fig1.suptitle('Exactitud', fontsize=16, fontweight='bold')
        handles, labels = axes1_flat[0].get_legend_handles_labels()
        fig1.legend(handles, labels, loc='lower center', ncol=5, frameon=True, fontsize=13)
        plt.tight_layout(rect=[0, 0.06, 1, 0.93])
        ruta_exactitud = os.path.join(base_salida, f'Comparativa_Exactitud_2x2{sufijo_salida}.png')
        plt.savefig(ruta_exactitud, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f'Guardado: {ruta_exactitud}')

        # Figura separada para Sensibilidad (2x2)
        n_rows_s = 2
        n_cols_s = 2
        fig_sens, axes_sens = plt.subplots(n_rows_s, n_cols_s, figsize=(14, 11), sharex=True)
        axes_sens_flat = axes_sens.flatten()
        for i, metodo in enumerate(tqdm(orden_metodos, desc=f'Graficando sensibilidad {conjunto.lower()}', unit='metodo')):
            dibujar_panel(axes_sens_flat[i], metodo, 'Sensibilidad', conjunto)
            axes_sens_flat[i].set_xlabel('Top')

        # Título global para la figura de sensibilidad
        fig_sens.suptitle('Sensibilidad', fontsize=16, fontweight='bold')
        handles2, labels2 = axes_sens_flat[0].get_legend_handles_labels()
        fig_sens.legend(handles2, labels2, loc='lower center', ncol=5, frameon=True, fontsize=13)
        plt.tight_layout(rect=[0, 0.06, 1, 0.93])
        ruta_sens = os.path.join(base_salida, f'Comparativa_Sensibilidad_2x2{sufijo_salida}.png')
        plt.savefig(ruta_sens, dpi=300, bbox_inches='tight')
        plt.close(fig_sens)
        print(f'Guardado: {ruta_sens}')

        # Figura separada para Especificidad (2x2)
        n_rows_e = 2
        n_cols_e = 2
        fig_espe, axes_espe = plt.subplots(n_rows_e, n_cols_e, figsize=(14, 11), sharex=True)
        axes_espe_flat = axes_espe.flatten()
        for i, metodo in enumerate(tqdm(orden_metodos, desc=f'Graficando especificidad {conjunto.lower()}', unit='metodo')):
            dibujar_panel(axes_espe_flat[i], metodo, 'Especificidad', conjunto)
            axes_espe_flat[i].set_xlabel('Top')

        # Título global para la figura de especificidad
        fig_espe.suptitle('Especificidad', fontsize=16, fontweight='bold')
        handles3, labels3 = axes_espe_flat[0].get_legend_handles_labels()
        fig_espe.legend(handles3, labels3, loc='lower center', ncol=5, frameon=True, fontsize=13)
        plt.tight_layout(rect=[0, 0.06, 1, 0.93])
        ruta_espe = os.path.join(base_salida, f'Comparativa_Especificidad_2x2{sufijo_salida}.png')
        plt.savefig(ruta_espe, dpi=300, bbox_inches='tight')
        plt.close(fig_espe)
        print(f'Guardado: {ruta_espe}')


def generar_tablas_csv():
    """Genera tablas CSV leyendo directamente de archivos para obtener ambas clases."""
    base_dir = os.path.join('Resultados', 'Análisis global', 'Modelos generados')
    base_salida = os.path.join('Resultados', 'Análisis global', 'Modelos generados', 'Gráficas comparativas')
    os.makedirs(base_salida, exist_ok=True)
    
    configuracion = [('Top 30', 30), ('Top 20', 20), ('Top 15', 15)]
    clasificadores = ['DT', 'RF', 'KNN', 'SVM', 'XGB']
    metodos = ['Fisher', 'Mutual_Info', 'mRMR', 'DT']
    tops_permitidos_metodo_dt = {10, 15, 20}
    
    orden_clf = ['DT', 'RF', 'KNN', 'SVM', 'XGB']
    
    nombres_metodo_archivo = {
        'Fisher': 'Fisher',
        'Mutual_Info': 'Mutual_Info',
        'mRMR': 'mRMR',
        'DT': 'DT'
    }
    
    for metodo in tqdm(metodos, desc='Generando tablas CSV', unit='metodo'):
        filas = []
        
        for nombre_top, sufijo in configuracion:
            for clf in orden_clf:
                # DT como MÉTODO solo tiene Top 10, 15, 20
                if metodo == 'DT' and sufijo not in tops_permitidos_metodo_dt:
                    continue
                
                archivo = f'Resultados_{clf}_Ranking_{metodo}_top{sufijo}.csv'
                ruta_csv = os.path.join(base_dir, nombre_top, archivo)
                
                if not os.path.exists(ruta_csv):
                    continue
                
                # Leer CSV para obtener métricas de ambas clases
                df_csv = pd.read_csv(ruta_csv, index_col=0)
                
                # Extraer Clase 0 y Clase 1 de Validación
                if 'Clase 0 (Validación)' in df_csv.index and 'Clase 1 (Validación)' in df_csv.index:
                    row_c0 = df_csv.loc['Clase 0 (Validación)']
                    row_c1 = df_csv.loc['Clase 1 (Validación)']
                    
                    # Guardar Clase 0
                    filas.append({
                        'Top': nombre_top,
                        'Clasificador': clf,
                        'Clase': 'Clase 0',
                        'Exactitud': round(float(row_c0['Exactitud']), 3),
                        'Sensibilidad': round(float(row_c0['Sensibilidad']), 3),
                        'Especificidad': round(float(row_c0['Especificidad']), 3),
                    })
                    
                    # Guardar Clase 1
                    filas.append({
                        'Top': nombre_top,
                        'Clasificador': clf,
                        'Clase': 'Clase 1',
                        'Exactitud': round(float(row_c1['Exactitud']), 3),
                        'Sensibilidad': round(float(row_c1['Sensibilidad']), 3),
                        'Especificidad': round(float(row_c1['Especificidad']), 3),
                    })
        
        if filas:
            df_tabla = pd.DataFrame(filas)
            archivo_salida = os.path.join(base_salida, f'Tabla_{nombres_metodo_archivo[metodo]}.csv')
            df_tabla.to_csv(archivo_salida, index=False)
            print(f'Guardado: {archivo_salida}')



if __name__ == '__main__':
    generar_graficos()
    generar_tablas_csv()