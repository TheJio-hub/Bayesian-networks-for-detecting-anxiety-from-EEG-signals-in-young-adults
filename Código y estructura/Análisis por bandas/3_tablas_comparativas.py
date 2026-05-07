from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)



BLOQUES = ["Alpha", "Beta", "Delta", "Asimetria", "Ratios"]
METODOS = ["Fisher", "Mutual_Info", "mRMR", "DT"]
TOPS = [(32, "Top 32"), (8, "Top 8")]
CLASIFICADORES = ["DT", "KNN", "SVM", "XGB"]
METRICAS = ["Exactitud", "Sensibilidad", "Especificidad"]


def raiz_proyecto() -> Path:
    return Path(__file__).resolve().parents[2]


def leer_metricas(ruta_csv: Path) -> dict[str, float]:
    df = pd.read_csv(ruta_csv, index_col=0)
    if df.empty:
        return {m: float("nan") for m in METRICAS}

    if "Clase 1" in df.index:
        fila = df.loc["Clase 1"]
    else:
        fila = df.iloc[-1]

    return {
        "Exactitud": float(fila["Exactitud"]),
        "Sensibilidad": float(fila["Sensibilidad"]),
        "Especificidad": float(fila["Especificidad"]),
    }


def construir_tabla_bloque(raiz: Path, bloque: str) -> pd.DataFrame:
    base = raiz / "Resultados" / "Análisis por bandas" / "Modelos por banda"

    # Asimetria solo tiene Top 8; Ratios tiene ambos (Top 32 y Top 8)
    tops_a_usar = [(8, "Top 8")] if bloque == "Asimetria" else TOPS

    filas = []
    for metodo in tqdm(METODOS, desc=f'Tabla {bloque}', unit='metodo', leave=False):
        for top_num, top_nombre in tops_a_usar:
            fila = {"Metodo": metodo, "Top": top_nombre}
            for clasificador in CLASIFICADORES:
                ruta_csv = base / top_nombre / bloque / f"Resultados_{clasificador}_Ranking_{metodo}_top{top_num}.csv"
                if ruta_csv.exists():
                    metricas = leer_metricas(ruta_csv)
                else:
                    metricas = {m: float("nan") for m in METRICAS}

                for metrica in METRICAS:
                    fila[(clasificador, metrica)] = metricas[metrica]

            filas.append(fila)

    df = pd.DataFrame(filas)
    df = df.set_index(["Metodo", "Top"])
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Clasificador", "Metrica"])
    return df


def guardar_tabla(df: pd.DataFrame, ruta_xlsx: Path, ruta_csv: Path) -> None:
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_csv, float_format="%.3g")


def main() -> None:
    raiz = raiz_proyecto()
    salida = raiz / "Resultados" / "Análisis por bandas" / "Tablas comparativas"
    salida.mkdir(parents=True, exist_ok=True)

    for bloque in tqdm(BLOQUES, desc='Tablas comparativas', unit='bloque'):
        df = construir_tabla_bloque(raiz, bloque)
        ruta_csv = salida / f"Tabla_Comparativa_{bloque}.csv"
        guardar_tabla(df, None, ruta_csv)
        print(f"Guardado: {ruta_csv}")


if __name__ == "__main__":
    main()
