from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault('mininterval', 1.5)
    kwargs.setdefault('miniters', 1)
    return _tqdm(*args, **kwargs)


from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold, LeaveOneGroupOut

from xgboost import XGBClassifier


BLOQUES = ["Alpha", "Beta", "Delta", "Asimetria", "Ratios"]
METODOS = ["Fisher", "Mutual_Info", "mRMR", "DT"]
TOPS = [(8, "Top 8"), (32, "Top 32")]


def raiz_proyecto() -> Path:
    return Path(__file__).resolve().parents[2]


def cargar_dataframe(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_parquet(ruta)


def filtrar_clases_extremas(df: pd.DataFrame) -> pd.DataFrame:
    if "Puntaje" not in df.columns:
        return df.copy()
    return df[(df["Puntaje"] == 0) | (df["Puntaje"] >= 5)].copy()


def preparar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = filtrar_clases_extremas(df)
    if df.empty or "Puntaje" not in df.columns or "Sujeto" not in df.columns:
        return pd.DataFrame(), np.array([]), np.array([])

    y = df["Puntaje"].apply(lambda valor: 0 if valor == 0 else 1).values
    grupos = df["Sujeto"].values
    return df, y, grupos


def columnas_base(df: pd.DataFrame, banda: str) -> list[str]:
    meta = ["Sujeto", "Tarea", "Trial", "Epoca", "Puntaje", "Grupo", "Ensayo"]
    columnas = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in columnas if c.endswith(f"_{banda}")]


def columnas_asimetria(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Asym_")]


def columnas_ratios(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Ratio_")]


def ruta_ranking_csv(raiz: Path, banda: str, bloque: str, metodo: str) -> Path | None:
    base = raiz / "Resultados" / "Análisis por bandas" / "Ranking de características"

    if bloque in {"Alpha", "Beta", "Delta"}:
        if metodo == "Fisher":
            ruta = base / bloque / f"Fisher_{bloque}.csv"
        elif metodo == "Mutual_Info":
            ruta = base / bloque / f"MI_{bloque}.csv"
            if not ruta.exists():
                ruta = base / bloque / f"MMI_{bloque}.csv"
        elif metodo == "mRMR":
            ruta = base / bloque / f"mRMR_{bloque}.csv"
        elif metodo == "DT":
            ruta = base / bloque / f"DT_{bloque}.csv"
        else:
            return None
        return ruta if ruta.exists() else None

    if bloque in {"Asimetria", "Ratios"}:
        bloque_dir = base / banda / bloque
        if metodo == "Fisher":
            ruta = bloque_dir / f"Fisher_{bloque}_{banda}.csv"
        elif metodo == "Mutual_Info":
            ruta = bloque_dir / f"MI_{bloque}_{banda}.csv"
            if not ruta.exists():
                ruta = bloque_dir / f"MMI_{bloque}_{banda}.csv"
        elif metodo == "mRMR":
            ruta = bloque_dir / f"mRMR_{bloque}_{banda}.csv"
        elif metodo == "DT":
            ruta = bloque_dir / f"DT_{bloque}_{banda}.csv"
        else:
            return None
        return ruta if ruta.exists() else None

    return None


def columna_valor_ranking(df: pd.DataFrame, metodo: str) -> str:
    candidatos = {
        "Fisher": ["Fisher_Score"],
        "Mutual_Info": ["MI_Score", "MMI_Score", "Mutual_Info"],
        "mRMR": ["Relevancia_Original", "Orden_Seleccion", "mRMR_Rank"],
        "DT": ["Importancia_DT", "Importancia"],
    }
    for columna in candidatos.get(metodo, []):
        if columna in df.columns:
            return columna
    return ""


def normalizar_ranking(df: pd.DataFrame, metodo: str) -> pd.DataFrame:
    if df.empty or "Caracteristica" not in df.columns:
        return pd.DataFrame(columns=["Caracteristica", "Valor_Ranking"])

    if metodo == "mRMR" and "Orden_Seleccion" in df.columns:
        base = df[["Caracteristica", "Orden_Seleccion"]].copy()
        base = base.rename(columns={"Orden_Seleccion": "Valor_Ranking"})
        base = base.dropna(subset=["Caracteristica", "Valor_Ranking"])
        base["Caracteristica"] = base["Caracteristica"].astype(str)
        base["Valor_Ranking"] = -pd.to_numeric(base["Valor_Ranking"], errors="coerce")
        base = base.dropna(subset=["Valor_Ranking"])
        return base

    columna_valor = columna_valor_ranking(df, metodo)
    if not columna_valor:
        return pd.DataFrame(columns=["Caracteristica", "Valor_Ranking"])

    base = df[["Caracteristica", columna_valor]].copy()
    base = base.rename(columns={columna_valor: "Valor_Ranking"})
    base = base.dropna(subset=["Caracteristica", "Valor_Ranking"])
    base["Caracteristica"] = base["Caracteristica"].astype(str)
    base["Valor_Ranking"] = pd.to_numeric(base["Valor_Ranking"], errors="coerce")
    base = base.dropna(subset=["Valor_Ranking"])
    return base


def cargar_ranking_bloque(raiz: Path, bloque: str, metodo: str) -> pd.DataFrame:
    if bloque in {"Alpha", "Beta", "Delta"}:
        ruta = ruta_ranking_csv(raiz, bloque, bloque, metodo)
        if ruta is None:
            return pd.DataFrame(columns=["Caracteristica", "Valor_Ranking"])
        return normalizar_ranking(pd.read_csv(ruta), metodo)

    frames = []
    for banda in ["Alpha", "Beta", "Delta"]:
        ruta = ruta_ranking_csv(raiz, banda, bloque, metodo)
        if ruta is None:
            continue
        df = pd.read_csv(ruta)
        df_norm = normalizar_ranking(df, metodo)
        if not df_norm.empty:
            frames.append(df_norm)

    if not frames:
        return pd.DataFrame(columns=["Caracteristica", "Valor_Ranking"])

    return pd.concat(frames, ignore_index=True)


def seleccionar_caracteristicas(df_ranking: pd.DataFrame, n: int) -> list[str]:
    if df_ranking.empty:
        return []

    df = df_ranking.sort_values("Valor_Ranking", ascending=False).drop_duplicates(subset=["Caracteristica"])
    return df["Caracteristica"].tolist()[:n]


def crear_modelo_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def entrenar_y_evaluar(X: pd.DataFrame, y: np.ndarray, grupos: np.ndarray, modelo) -> pd.DataFrame:
    logo = LeaveOneGroupOut()
    y_real_total = []
    y_predicha_total = []
    y_train_total = []
    y_train_pred_total = []
    total_folds = logo.get_n_splits(X, y, grupos)

    for indice_entrenamiento, indice_prueba in tqdm(
        logo.split(X, y, grupos),
        total=total_folds,
        desc='LOGO XGB',
        unit='fold',
        leave=False,
    ):
        X_entrenamiento, X_prueba = X.iloc[indice_entrenamiento], X.iloc[indice_prueba]
        y_entrenamiento, y_prueba = y[indice_entrenamiento], y[indice_prueba]

        modelo.fit(X_entrenamiento, y_entrenamiento)
        prediccion = modelo.predict(X_prueba)

        y_real_total.extend(y_prueba)
        y_predicha_total.extend(prediccion)

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for idx_train_cv, idx_val_cv in kfold.split(X_entrenamiento):
            X_cv_train, X_cv_val = X_entrenamiento.iloc[idx_train_cv], X_entrenamiento.iloc[idx_val_cv]
            y_cv_train, y_cv_val = y_entrenamiento[idx_train_cv], y_entrenamiento[idx_val_cv]
            modelo.fit(X_cv_train, y_cv_train)
            train_pred = modelo.predict(X_cv_val)
            y_train_total.extend(y_cv_val)
            y_train_pred_total.extend(train_pred)

    precision, sensibilidad, f1, _ = precision_recall_fscore_support(
        y_real_total,
        y_predicha_total,
        labels=[0, 1],
        zero_division=0,
    )
    exactitud = accuracy_score(y_real_total, y_predicha_total)
    cm = confusion_matrix(y_real_total, y_predicha_total, labels=[0, 1])
    total = cm.sum()
    especificidades = []
    for clase_idx in range(cm.shape[0]):
        tp = cm[clase_idx, clase_idx]
        fp = cm[:, clase_idx].sum() - tp
        fn = cm[clase_idx, :].sum() - tp
        tn = total - tp - fp - fn
        especificidades.append(tn / (tn + fp) if (tn + fp) else 0.0)

    precision_tr, sensibilidad_tr, f1_tr, _ = precision_recall_fscore_support(
        y_train_total,
        y_train_pred_total,
        labels=[0, 1],
        zero_division=0,
    )
    exactitud_tr = accuracy_score(y_train_total, y_train_pred_total)
    cm_tr = confusion_matrix(y_train_total, y_train_pred_total, labels=[0, 1])
    total_tr = cm_tr.sum()
    especificidades_tr = []
    for clase_idx in range(cm_tr.shape[0]):
        tp = cm_tr[clase_idx, clase_idx]
        fp = cm_tr[:, clase_idx].sum() - tp
        fn = cm_tr[clase_idx, :].sum() - tp
        tn = total_tr - tp - fp - fn
        especificidades_tr.append(tn / (tn + fp) if (tn + fp) else 0.0)

    return pd.DataFrame(
        [
            {'Precision': precision[0], 'Sensibilidad': sensibilidad[0], 'Especificidad': especificidades[0], 'Puntaje_F1': f1[0], 'Exactitud': exactitud},
            {'Precision': precision[1], 'Sensibilidad': sensibilidad[1], 'Especificidad': especificidades[1], 'Puntaje_F1': f1[1], 'Exactitud': exactitud},
            {'Precision': precision_tr[0], 'Sensibilidad': sensibilidad_tr[0], 'Especificidad': especificidades_tr[0], 'Puntaje_F1': f1_tr[0], 'Exactitud': exactitud_tr},
            {'Precision': precision_tr[1], 'Sensibilidad': sensibilidad_tr[1], 'Especificidad': especificidades_tr[1], 'Puntaje_F1': f1_tr[1], 'Exactitud': exactitud_tr},
        ],
        index=['Clase 0 (Validación)', 'Clase 1 (Validación)', 'Clase 0 (Entrenamiento)', 'Clase 1 (Entrenamiento)'],
    )


def columnas_disponibles(df: pd.DataFrame, bloque: str) -> list[str]:
    if bloque in {"Alpha", "Beta", "Delta"}:
        return columnas_base(df, bloque)
    if bloque == "Asimetria":
        return columnas_asimetria(df)
    if bloque == "Ratios":
        return columnas_ratios(df)
    return []


def evaluar_bloque(
    raiz: Path,
    bloque: str,
    df_base: pd.DataFrame,
    df_asim: pd.DataFrame,
    df_ratio: pd.DataFrame,
    y_base: np.ndarray,
    grupos_base: np.ndarray,
    y_asim: np.ndarray,
    grupos_asim: np.ndarray,
    y_ratio: np.ndarray,
    grupos_ratio: np.ndarray,
) -> None:
    if bloque in {"Alpha", "Beta", "Delta"}:
        df = df_base
        y = y_base
        grupos = grupos_base
    elif bloque == "Asimetria":
        df = df_asim
        y = y_asim
        grupos = grupos_asim
    else:
        df = df_ratio
        y = y_ratio
        grupos = grupos_ratio

    if df.empty or y.size == 0 or grupos.size == 0:
        return

    columnas_validas = set(columnas_disponibles(df, bloque))
    if not columnas_validas:
        return

    modelo = crear_modelo_xgb()

    for n_top, nombre_top in tqdm(TOPS, desc=f'Evaluando {bloque}', unit='top', leave=False):
        dir_top = raiz / "Resultados" / "Análisis por bandas" / "Modelos por banda" / nombre_top / bloque
        dir_top.mkdir(parents=True, exist_ok=True)

        for metodo in tqdm(METODOS, desc=f'Metodos {bloque} top{n_top}', unit='metodo', leave=False):
            df_ranking = cargar_ranking_bloque(raiz, bloque, metodo)
            if df_ranking.empty:
                continue

            caracteristicas = seleccionar_caracteristicas(df_ranking, n_top)
            caracteristicas = [c for c in caracteristicas if c in columnas_validas and c in df.columns]
            if not caracteristicas:
                continue

            X = df[caracteristicas].copy().apply(pd.to_numeric, errors="coerce")
            X = X.dropna(axis=1, how="all").fillna(0)
            if X.empty:
                continue

            resultados = entrenar_y_evaluar(X, y, grupos, modelo)
            archivo_salida = dir_top / f"Resultados_XGB_Ranking_{metodo}_top{n_top}.csv"
            resultados.to_csv(archivo_salida)
            print(f"Guardado: {archivo_salida}")


def main() -> None:
    raiz = raiz_proyecto()

    ruta_bandas = raiz / "Resultados" / "Exploratorio" / "datos_bandas_normalizados.parquet"
    ruta_asimetria = raiz / "Resultados" / "Exploratorio" / "datos_asimetria_normalizados.parquet"
    ruta_ratios = raiz / "Resultados" / "Exploratorio" / "datos_ratios_normalizados.parquet"

    df_base = cargar_dataframe(ruta_bandas)
    df_asim = cargar_dataframe(ruta_asimetria)
    df_ratio = cargar_dataframe(ruta_ratios)

    df_base, y_base, grupos_base = preparar_datos(df_base)
    df_asim, y_asim, grupos_asim = preparar_datos(df_asim)
    df_ratio, y_ratio, grupos_ratio = preparar_datos(df_ratio)

    if df_base.empty and df_asim.empty and df_ratio.empty:
        raise FileNotFoundError("No se encontraron los conjuntos de datos necesarios en Resultados/Exploratorio.")

    for bloque in tqdm(BLOQUES, desc='Bloques por banda', unit='bloque'):
        evaluar_bloque(
            raiz,
            bloque,
            df_base,
            df_asim,
            df_ratio,
            y_base,
            grupos_base,
            y_asim,
            grupos_asim,
            y_ratio,
            grupos_ratio,
        )


if __name__ == "__main__":
    main()