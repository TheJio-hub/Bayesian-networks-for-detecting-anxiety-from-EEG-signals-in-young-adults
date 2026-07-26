#!/usr/bin/env python3
"""
figura_graficas.py
Genera las gráficas Top-8 para mRMR (alpha), DT (delta) y Top-20 de DT en el
ranking global, y guarda PNGs en `Artículo científico/Figuras y tablas/`.

Uso:
    python figura_graficas.py

Requisitos: pandas, matplotlib
"""
import os
import sys
import argparse
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(os.path.dirname(__file__), 'Figuras y tablas')
ALPHA_CSV = os.path.join(ROOT, 'Resultados', 'Análisis por bandas', 'Selección de características', 'Alpha', 'mRMR_Alpha.csv')
DELTA_CSV = os.path.join(ROOT, 'Resultados', 'Análisis por bandas', 'Selección de características', 'Delta', 'DT_Delta.csv')
GLOBAL_RANKING_CSV = os.path.join(ROOT, 'Resultados', 'Análisis global', 'Selección de características', 'Ranking_Multicriterio_Completo.csv')
BAR_STYLES = {
    'alpha': '#4F9ACD',
    'delta': '#E38A3A',
    'global': '#4F8F73',
}


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe: {path}")
    return pd.read_csv(path)


def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)


def apply_bar_style(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0, colors='#444444')
    ax.set_facecolor('white')


def plot_horizontal_bar(df: pd.DataFrame, x_col: str, y_col: str, out_fn: str, xlabel: str = 'Importancia', color: str = 'tab:blue', annotate: bool = True, fig_height: Optional[float] = None):
    df_plot = df.copy()
    # ordenar ascendente para que la mayor importancia quede arriba en barh
    df_plot = df_plot.sort_values(x_col, ascending=True)

    fig_height = fig_height if fig_height is not None else max(4, 0.42 * len(df_plot) + 1.1)
    fig, ax = plt.subplots(figsize=(8, fig_height), constrained_layout=True)
    y_pos = list(range(len(df_plot)))
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot[y_col], fontsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_color('#222222')
        label.set_fontsize(12)
    apply_bar_style(ax)

    maxv = df_plot[x_col].max() if not df_plot[x_col].empty else 1.0
    ax.barh(y_pos, df_plot[x_col], color=color, height=0.72)

    ax.set_xlim(0, maxv * 1.22)

    # Ajustar tamaño de ticks y etiquetas del eje x
    ax.tick_params(axis='x', labelsize=11, colors='#444444')
    ax.tick_params(axis='y', labelsize=12)

    if annotate:
        for i, width in enumerate(df_plot[x_col]):
            text_x = width + maxv * 0.015
            ax.text(text_x, i, f'{width:.2f}', va='center', ha='left', fontsize=11, color='#222222', fontweight='bold')

    fig.savefig(out_fn, dpi=300)
    plt.close(fig)


def generate_top20_dt_global(ranking_csv: str = GLOBAL_RANKING_CSV, out_dir: Optional[str] = None):
    out_dir = out_dir or OUT_DIR
    ranking = load_csv(ranking_csv)

    if 'Importancia_DT' not in ranking.columns or 'Caracteristica' not in ranking.columns:
        raise RuntimeError('CSV global no contiene las columnas esperadas: Importancia_DT, Caracteristica')

    dt_top20 = ranking.nlargest(20, 'Importancia_DT')
    fn_dt = os.path.join(out_dir, 'Top20_DT_RF.png')
    plot_horizontal_bar(dt_top20, 'Importancia_DT', 'Caracteristica', fn_dt, xlabel='Importancia', color=BAR_STYLES['global'], fig_height=5)

    return fn_dt


def generate(alpha_csv: str = ALPHA_CSV, delta_csv: str = DELTA_CSV, ranking_csv: str = GLOBAL_RANKING_CSV, out_dir: Optional[str] = None):
    ensure_outdir()
    out_dir = out_dir or OUT_DIR

    alpha = load_csv(alpha_csv)
    delta = load_csv(delta_csv)

    # mRMR Alpha: columna Relevancia_Original y Caracteristica
    if 'Relevancia_Original' not in alpha.columns or 'Caracteristica' not in alpha.columns:
        raise RuntimeError('CSV de Alpha no contiene las columnas esperadas: Relevancia_Original, Caracteristica')

    alpha_top = alpha.nlargest(8, 'Relevancia_Original')
    fn_alpha = os.path.join(out_dir, 'Top8_mRMR_alpha.png')
    plot_horizontal_bar(alpha_top, 'Relevancia_Original', 'Caracteristica', fn_alpha, xlabel='Importancia', color=BAR_STYLES['alpha'])

    # DT Delta: detectar columna de importancia
    dt_key = None
    for candidate in ['Importancia_DT', 'Importancia', 'importancia', 'Importance', 'importance']:
        if candidate in delta.columns:
            dt_key = candidate
            break
    if dt_key is None:
        raise RuntimeError('No se encontró columna de importancia en DT_Delta.csv')
    if 'Caracteristica' not in delta.columns:
        # intentar columnas alternativas
        possible_y = [c for c in delta.columns if 'caract' in c.lower() or 'feature' in c.lower()]
        if possible_y:
            y_col = possible_y[0]
        else:
            raise RuntimeError('No se encontró columna `Caracteristica` o alternativa en DT_Delta.csv')
    else:
        y_col = 'Caracteristica'

    delta_top = delta.nlargest(8, dt_key)
    fn_delta = os.path.join(out_dir, 'Top8_DT_delta.png')
    plot_horizontal_bar(delta_top, dt_key, y_col, fn_delta, xlabel='Importancia', color=BAR_STYLES['delta'])

    fn_dt_global = generate_top20_dt_global(ranking_csv, out_dir)

    return fn_alpha, fn_delta, fn_dt_global


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera Top-8 figuras para mRMR alpha y DT delta, y Top-20 global para DT')
    parser.add_argument('--alpha', help='Ruta al CSV mRMR alpha', default=ALPHA_CSV)
    parser.add_argument('--delta', help='Ruta al CSV DT delta', default=DELTA_CSV)
    parser.add_argument('--ranking', help='Ruta al CSV global de ranking multicriterio', default=GLOBAL_RANKING_CSV)
    parser.add_argument('--out', help='Directorio de salida para PNGs', default=OUT_DIR)
    args = parser.parse_args()

    try:
        a, b, c = generate(args.alpha, args.delta, args.ranking, args.out)
        print('Generadas:', a, b, c)
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
