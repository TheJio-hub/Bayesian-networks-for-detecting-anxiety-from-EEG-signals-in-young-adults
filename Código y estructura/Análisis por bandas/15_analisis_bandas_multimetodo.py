import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix


def fisher_scores_binary(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Compute Fisher score per feature for a binary target."""
    c0 = X[y == 0]
    c1 = X[y == 1]
    mu0 = c0.mean()
    mu1 = c1.mean()
    var0 = c0.var(ddof=0)
    var1 = c1.var(ddof=0)
    denom = (var0 + var1).replace(0, np.nan)
    score = ((mu0 - mu1) ** 2) / denom
    return score.fillna(0.0)


def dt_loso_metrics(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """Return (accuracy, specificity, sensitivity) under LOSO for DT."""
    logo = LeaveOneGroupOut()
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in logo.split(X, y, groups):
        clf = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=5)
        clf.fit(X.iloc[train_idx], y[train_idx])
        y_pred = clf.predict(X.iloc[test_idx])
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return acc, specificity, sensitivity


def main() -> None:
    root = os.path.join("Resultados")
    data_path = os.path.join(root, "datos_bandas_normalizados.parquet")
    ranking_dir = os.path.join(root, "Ranking")
    out_dir = os.path.join(root, "Analisis_Bandas")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No existe {data_path}")

    df = pd.read_parquet(data_path)
    df = df[(df["Puntaje"] == 0) | (df["Puntaje"] >= 5)].copy()
    y = df["Puntaje"].apply(lambda v: 0 if v == 0 else 1).values
    groups = df["Sujeto"].values

    meta_cols = ["Sujeto", "Tarea", "Trial", "Epoca", "Puntaje", "Grupo", "Ensayo"]
    all_features = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

    rows_top1 = []
    rows_metrics = []

    for band in bands:
        cols_band = [c for c in all_features if c.endswith(f"_{band}")]
        if not cols_band:
            continue

        Xb = df[cols_band]

        fisher = fisher_scores_binary(Xb, y).sort_values(ascending=False)
        fisher_top1 = fisher.index[0]
        fisher_val = float(fisher.iloc[0])

        mi_vals = mutual_info_classif(Xb, y, discrete_features=False, random_state=42)
        mi = pd.Series(mi_vals, index=Xb.columns).sort_values(ascending=False)
        mi_top1 = mi.index[0]
        mi_val = float(mi.iloc[0])

        mrmr_path = os.path.join(ranking_dir, f"mRMR_{band}.csv")
        if os.path.exists(mrmr_path):
            mrmr_df = pd.read_csv(mrmr_path)
            mrmr_top1 = str(mrmr_df.iloc[0]["Caracteristica"])
            mrmr_val = float(mrmr_df.iloc[0]["Relevancia_Original"])
        else:
            mrmr_top1 = "NA"
            mrmr_val = np.nan

        dt = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=5)
        dt.fit(Xb, y)
        dt_imp = pd.Series(dt.feature_importances_, index=Xb.columns).sort_values(ascending=False)
        dt_top1 = dt_imp.index[0]
        dt_val = float(dt_imp.iloc[0])

        acc, spec, sens = dt_loso_metrics(Xb, y, groups)

        rows_top1.append(
            {
                "Banda": band,
                "Fisher_Top1": fisher_top1,
                "Fisher_Score": fisher_val,
                "MI_Top1": mi_top1,
                "MI_Score": mi_val,
                "mRMR_Top1": mrmr_top1,
                "mRMR_Relevancia": mrmr_val,
                "DT_Top1": dt_top1,
                "DT_Importancia": dt_val,
            }
        )

        rows_metrics.append(
            {
                "Banda": band,
                "DT_Exactitud": acc,
                "DT_Especificidad": spec,
                "DT_Sensibilidad": sens,
            }
        )

    df_top1 = pd.DataFrame(rows_top1)
    df_metrics = pd.DataFrame(rows_metrics)

    df_top1.to_csv(os.path.join(out_dir, "Resumen_Top1_Metodos_Bandas.csv"), index=False)
    df_metrics.to_csv(os.path.join(out_dir, "Resumen_Metricas_DT_Por_Banda.csv"), index=False)

    print("Generados:")
    print(os.path.join(out_dir, "Resumen_Top1_Metodos_Bandas.csv"))
    print(os.path.join(out_dir, "Resumen_Metricas_DT_Por_Banda.csv"))


if __name__ == "__main__":
    main()
