"""Comparación validada por sujeto de clasificadores basados en redes bayesianas.

Este es el ejecutor de referencia para los resultados predictivos de la tesis.
Recalcula los rankings DT y mRMR únicamente con los datos de entrenamiento de
cada pliegue, compara Top 10/15/20/30 y reserva ocho sujetos para una evaluación
final única.
Las flechas aprendidas se interpretan como estructura estadística, no como
efectos causales identificados.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.causal_discovery import GES, PC
from pgmpy.estimators import BayesianEstimator, TreeSearch
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = PROJECT_ROOT / "Resultados" / "Exploratorio" / "datos_completos_normalizados.parquet"
OUTPUT_ROOT = PROJECT_ROOT / "Resultados" / "Red Bayesiana" / "Comparativa_Seleccion_BN_CV"

TARGET = "Ansiedad"
TOP_SIZES = (10, 15, 20, 30, 40)
BAND_TOP_SIZES = (10, 15, 20, 30)
DT_GLOBAL_TOP_SIZES = (10, 15, 20)
RANK_METHODS = ("DT", "Fisher", "Mutual_Info", "mRMR")
ARTICLE_BANDS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")
ARTICLE_RATIOS = ("ThetaBeta", "ThetaAlpha", "AlphaBeta")
MODEL_NAMES = ("PC", "GES", "TAN", "NaiveBayes")
BIN_COUNTS = (2, 3)
SEED = 42
N_DEVELOPMENT_SUBJECTS = 32
OUTER_FOLDS = 4
INNER_FOLDS = 2
BOOTSTRAP_REPLICATES = 1000


def article_features(columns: pd.Index) -> list[str]:
    """Retorna y valida las 306 variables incluidas en el artículo."""
    columns = list(columns)
    powers = [
        c for c in columns
        if not c.startswith(("Asym_", "Ratio_"))
        and any(c.endswith(f"_{band}") for band in ARTICLE_BANDS)
    ]
    asymmetries = [
        c for c in columns
        if c.startswith("Asym_")
        and any(c.endswith(f"_{band}") for band in ARTICLE_BANDS)
    ]
    ratios = [
        c for c in columns
        if c.startswith("Ratio_")
        and any(c.startswith(f"Ratio_{ratio}_") for ratio in ARTICLE_RATIOS)
    ]
    selected = powers + asymmetries + ratios
    counts = (len(powers), len(asymmetries), len(ratios))
    if counts != (160, 50, 96) or len(selected) != 306:
        raise ValueError(
            "El conjunto no coincide con las 306 variables del artículo: "
            f"potencias={counts[0]}, asimetrías={counts[1]}, "
            f"razones={counts[2]}, total={len(selected)}."
        )
    if len(set(selected)) != len(selected):
        raise ValueError("Hay nombres duplicados en el conjunto de 306 variables.")
    return selected


def load_data() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"No se encontró el conjunto normalizado: {DATA_FILE}")
    data = pd.read_parquet(DATA_FILE)
    required = {"Sujeto", "Tarea", "Trial", "Epoca", "Puntaje"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    # Conserva el criterio acordado con la directora: control = SAM 0; caso = SAM >= 1.
    data = data[(data["Puntaje"] == 0) | (data["Puntaje"] >= 1)].copy()
    data[TARGET] = (data["Puntaje"] >= 1).astype(int)
    features = article_features(data.columns)
    data = data.dropna(subset=["Sujeto", TARGET]).reset_index(drop=True)
    return data, features


def mrmr_rank(X: pd.DataFrame, y: np.ndarray, n_features: int = 30) -> list[str]:
    """Ranking mRMR secuencial; solo debe recibir filas del entrenamiento."""
    if X.shape[1] == 0:
        raise ValueError("No hay características disponibles para mRMR.")
    if np.unique(y).size != 2:
        raise ValueError("mRMR requiere ambas clases en el conjunto de entrenamiento.")

    relevance = mutual_info_classif(
        X, y, discrete_features=False, random_state=SEED
    )
    n_features = min(n_features, X.shape[1])
    chosen = [int(np.argmax(relevance))]
    candidates = np.ones(X.shape[1], dtype=bool)
    candidates[chosen[0]] = False
    redundancy_sum = np.zeros(X.shape[1], dtype=float)

    while len(chosen) < n_features:
        last = X.iloc[:, chosen[-1]].to_numpy()
        redundancy_sum += mutual_info_regression(
            X, last, discrete_features=False, random_state=SEED
        )
        scores = relevance - redundancy_sum / len(chosen)
        scores[~candidates] = -np.inf
        next_index = int(np.argmax(scores))
        if not np.isfinite(scores[next_index]):
            break
        chosen.append(next_index)
        candidates[next_index] = False

    return [X.columns[i] for i in chosen]


def rank_features(
    train: pd.DataFrame, features: list[str], method: str, n_features: int = 30
) -> list[str]:
    """Ajusta el ranking solo con filas del entrenamiento del pliegue."""
    X, _ = prepare_numeric(train, train, features)
    y = train[TARGET].to_numpy(dtype=int)
    if method == "mRMR":
        return mrmr_rank(X, y, n_features)
    if method == "DT":
        selector = DecisionTreeClassifier(
            random_state=SEED, class_weight="balanced", max_depth=5
        )
        selector.fit(X, y)
        order = np.argsort(-selector.feature_importances_, kind="stable")
        return [X.columns[index] for index in order[:n_features]]
    if method == "Mutual_Info":
        scores = mutual_info_classif(X, y, discrete_features=False, random_state=SEED)
        order = np.argsort(-scores, kind="stable")
        return [X.columns[index] for index in order[:n_features]]
    if method == "Fisher":
        class_0, class_1 = X.loc[y == 0], X.loc[y == 1]
        numerator = (class_0.mean() - class_1.mean()) ** 2
        denominator = class_0.var(ddof=0) + class_1.var(ddof=0) + 1e-12
        scores = numerator / denominator
        return scores.sort_values(ascending=False, kind="stable").index[:n_features].tolist()
    raise ValueError(f"Método de selección no reconocido: {method}")


def prepare_numeric(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    X_train = train[features].apply(pd.to_numeric, errors="coerce")
    X_test = test[features].apply(pd.to_numeric, errors="coerce")
    medians = X_train.median().fillna(0)
    return X_train.fillna(medians), X_test.fillna(medians)


def discretize_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_bins: int,
):
    X_train, X_test = prepare_numeric(train, test, features)
    discretizer = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        train_values = discretizer.fit_transform(X_train)
        test_values = discretizer.transform(X_test)
    train_disc = pd.DataFrame(
        train_values, columns=features, index=train.index
    ).astype(int)
    test_disc = pd.DataFrame(
        test_values, columns=features, index=test.index
    ).astype(int)
    train_disc[TARGET] = train[TARGET].astype(int).to_numpy()
    return train_disc, test_disc


def learn_structure(
    model_name: str, train_disc: pd.DataFrame, features: list[str]
) -> DiscreteBayesianNetwork:
    if model_name == "PC":
        estimator = PC(
            variant="stable",
            ci_test="chi_square",
            significance_level=0.05,
            max_cond_vars=2,
            return_type="dag",
            show_progress=False,
        )
        estimator.fit(train_disc)
        graph = estimator.causal_graph_
    elif model_name == "GES":
        estimator = GES(scoring_method="bic-d", return_type="dag")
        estimator.fit(train_disc)
        graph = estimator.causal_graph_
    elif model_name == "TAN":
        estimator = TreeSearch(train_disc, root_node=features[0])
        graph = estimator.estimate(
            estimator_type="tan", class_node=TARGET, show_progress=False
        )
    elif model_name == "NaiveBayes":
        graph = DiscreteBayesianNetwork([(TARGET, feature) for feature in features])
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    model = DiscreteBayesianNetwork(list(graph.edges()))
    model.add_nodes_from([TARGET, *features])
    if not nx.is_directed_acyclic_graph(nx.DiGraph(model.edges())):
        raise ValueError(f"{model_name} produjo una estructura que no es DAG.")
    return model


def posterior_probability(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_bins: int,
):
    train_disc, test_disc = discretize_pair(train, test, features, n_bins)
    model = learn_structure(model_name, train_disc, features)
    estimator = BayesianEstimator(model, train_disc)
    model.add_cpds(
        *estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=5)
    )
    if not model.check_model():
        raise ValueError("La red aprendida no pasó la validación de CPD.")

    probabilities = model.predict_probability(test_disc)
    positive_column = f"{TARGET}_1"
    if positive_column not in probabilities:
        raise ValueError(
            f"No se obtuvo el posterior {positive_column}; columnas: "
            f"{list(probabilities.columns)}"
        )
    return probabilities[positive_column].to_numpy(), model


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    if np.unique(y_true).size != 2:
        return 0.5
    _, _, thresholds = roc_curve(y_true, probability)
    thresholds = np.unique(np.r_[0.5, thresholds[np.isfinite(thresholds)]])
    scored = [
        (balanced_accuracy_score(y_true, probability >= threshold), threshold)
        for threshold in thresholds
    ]
    best_score = max(score for score, _ in scored)
    tied = [threshold for score, threshold in scored if np.isclose(score, best_score)]
    return float(min(tied, key=lambda value: abs(value - 0.5)))


def calculate_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    metrics = {
        "Exactitud": accuracy_score(y_true, prediction),
        "Exactitud_balanceada": balanced_accuracy_score(y_true, prediction),
        "Sensibilidad": recall_score(y_true, prediction, zero_division=0),
        "Especificidad": specificity,
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "Umbral": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    if np.unique(y_true).size == 2:
        metrics["ROC_AUC"] = roc_auc_score(y_true, probability)
        metrics["PR_AUC"] = average_precision_score(y_true, probability)
    else:
        metrics["ROC_AUC"] = np.nan
        metrics["PR_AUC"] = np.nan
    metrics["Brier"] = brier_score_loss(y_true, probability)
    return metrics, prediction


def tune_model(
    train: pd.DataFrame,
    features_all: list[str],
    selection_method: str,
    top_size: int,
    n_splits: int,
    rank_cache: dict[tuple[str, ...], list[str]],
):
    y = train[TARGET].to_numpy(dtype=int)
    groups = train["Sujeto"].to_numpy()
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Se requieren al menos dos sujetos para validación agrupada.")

    splitter = GroupKFold(n_splits=n_splits)
    candidate_rows = []
    candidate_predictions = {}
    candidate_graphs = {}
    candidate_failures = []

    for model_name in MODEL_NAMES:
        for n_bins in BIN_COUNTS:
            key = (model_name, n_bins)
            oof_probability = np.full(len(train), np.nan, dtype=float)
            fold_graphs = []
            elapsed = 0.0
            for train_idx, valid_idx in splitter.split(train, y, groups):
                inner_train = train.iloc[train_idx]
                inner_valid = train.iloc[valid_idx]
                try:
                    rank_key = (selection_method, *sorted(inner_train["Sujeto"].astype(str).unique()))
                    if rank_key not in rank_cache:
                        rank_cache[rank_key] = rank_features(
                            inner_train, features_all, selection_method, max(TOP_SIZES)
                        )
                    ranking = rank_cache[rank_key]
                    selected = ranking[:top_size]
                    started = time.perf_counter()
                    probability, graph = posterior_probability(
                        model_name,
                        inner_train,
                        inner_valid,
                        selected,
                        n_bins,
                    )
                    elapsed += time.perf_counter() - started
                    oof_probability[valid_idx] = probability
                    fold_graphs.append(list(graph.edges()))
                except Exception as exc:
                    candidate_failures.append({
                        "Seleccion": selection_method,
                        "Top": top_size,
                        "Modelo": model_name,
                        "Bins": n_bins,
                        "Fase": "Validacion_interna",
                        "Error": f"{type(exc).__name__}: {exc}",
                    })
                    oof_probability[valid_idx] = np.nan
                    break

            if np.isnan(oof_probability).any():
                continue
            threshold = choose_threshold(y, oof_probability)
            metrics, _ = calculate_metrics(y, oof_probability, threshold)
            mean_edges = float(np.mean([len(edges) for edges in fold_graphs]))
            candidate_rows.append({
                "Seleccion": selection_method,
                "Top": top_size,
                "Modelo": model_name,
                "Bins": n_bins,
                "Tiempo_modelado_seg": elapsed,
                "Arcos_promedio": mean_edges,
                **metrics,
            })
            candidate_predictions[key] = (oof_probability, threshold)
            candidate_graphs[key] = fold_graphs

    if not candidate_rows:
        details = " | ".join(row["Error"] for row in candidate_failures[-6:])
        raise RuntimeError(f"Fallaron todas las configuraciones Top {top_size}: {details}")

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["Exactitud_balanceada", "PR_AUC", "Arcos_promedio", "Tiempo_modelado_seg"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    winner = candidates.iloc[0].to_dict()
    key = (winner["Modelo"], int(winner["Bins"]))
    return {
        "selection_method": selection_method,
        "winner": winner,
        "threshold": float(winner["Umbral"]),
        "candidate_rows": candidate_rows,
        "candidate_failures": candidate_failures,
        "winner_inner_probability": candidate_predictions[key][0],
        "winner_inner_graphs": candidate_graphs[key],
    }


def bootstrap_subject_ci(
    frame: pd.DataFrame,
    metric: str,
    threshold: float,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
):
    subjects = frame["Sujeto"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(SEED)
    values = []
    by_subject = {s: frame[frame["Sujeto"] == s] for s in subjects}
    for _ in range(n_bootstrap):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        boot = pd.concat([by_subject[s] for s in sampled], ignore_index=True)
        y = boot[TARGET].to_numpy(dtype=int)
        probability = boot["Probabilidad"].to_numpy(dtype=float)
        if metric in ("Exactitud_balanceada", "F1") and "Prediccion" in boot:
            prediction = boot["Prediccion"].to_numpy(dtype=int)
            if metric == "Exactitud_balanceada":
                value = balanced_accuracy_score(y, prediction)
            else:
                value = f1_score(y, prediction, zero_division=0)
        else:
            scores, _ = calculate_metrics(y, probability, threshold)
            value = scores[metric]
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def save_graph(edges, nodes, path: Path, title: str):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    plt.figure(figsize=(15, 11))
    pos = nx.spring_layout(graph, seed=SEED, k=1.5 / np.sqrt(max(1, len(nodes))))
    node_colors = ["#F28E2B" if node == TARGET else "#A8DADC" for node in graph.nodes]
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, node_size=1900,
        edgecolors="#243746", linewidths=1.2
    )
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="medium")
    nx.draw_networkx_edges(
        graph, pos, edge_color="#3B5368", arrows=True,
        arrowstyle="-|>", arrowsize=16, width=1.3,
        connectionstyle="arc3,rad=0.06"
    )
    plt.title(title, fontsize=14, weight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def run_scope(
    data: pd.DataFrame,
    features_all: list[str],
    scope_name: str,
    output_dir: Path,
    development_subjects: np.ndarray,
    external_subjects: np.ndarray,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    scope_top_sizes = BAND_TOP_SIZES if scope_name.startswith('Banda_') else TOP_SIZES
    development = data[data["Sujeto"].isin(development_subjects)].copy()
    external = data[data["Sujeto"].isin(external_subjects)].copy()
    if development["Sujeto"].nunique() < INNER_FOLDS + 1:
        raise ValueError(f"{scope_name}: hay muy pocos sujetos en desarrollo.")

    if development[TARGET].nunique() != 2 or external[TARGET].nunique() != 2:
        raise ValueError(f"{scope_name}: desarrollo y prueba deben contener ambas clases.")

    outer_splitter = GroupKFold(n_splits=min(OUTER_FOLDS, len(development_subjects)))
    oof_records = []
    outer_config_records = []
    inner_config_records = []
    failures = []
    feature_counts = Counter()
    edge_counts = Counter()
    edge_denominators = Counter()
    rank_cache: dict[tuple[str, ...], list[str]] = {}

    y_dev = development[TARGET].to_numpy(dtype=int)
    group_dev = development["Sujeto"].to_numpy()

    for selection_method in RANK_METHODS:
      method_top_sizes = (
          scope_top_sizes
          if scope_name.startswith('Banda_') or selection_method != 'DT'
          else DT_GLOBAL_TOP_SIZES
      )
      for top_size in method_top_sizes:
        for fold_number, (train_idx, valid_idx) in enumerate(
            outer_splitter.split(development, y_dev, group_dev), start=1
        ):
            outer_train = development.iloc[train_idx].copy()
            outer_valid = development.iloc[valid_idx].copy()
            try:
                tuned = tune_model(
                    outer_train, features_all, selection_method, top_size,
                    INNER_FOLDS, rank_cache
                )
            except Exception as exc:
                failures.append({
                    "Seleccion": selection_method,
                    "Top": top_size,
                    "Modelo": "Configuraciones_internas",
                    "Bins": np.nan,
                    "Fase": f"Pliegue_externo_{fold_number}",
                    "Error": f"{type(exc).__name__}: {exc}",
                })
                pd.DataFrame(
                    failures,
                    columns=["Seleccion", "Top", "Modelo", "Bins", "Fase", "Error"],
                ).to_csv(output_dir / "Fallos_modelos.csv", index=False)
                raise
            print(
                f"[{scope_name}] selección={selection_method}, Top={top_size}, "
                f"pliegue externo={fold_number}/{OUTER_FOLDS}: configuración interna lista",
                flush=True,
            )
            inner_config_records.extend(tuned["candidate_rows"])
            failures.extend(tuned["candidate_failures"])
            winner = tuned["winner"]
            model_name = str(winner["Modelo"])
            n_bins = int(winner["Bins"])
            threshold = float(tuned["threshold"])

            rank_key = (selection_method, *sorted(outer_train["Sujeto"].astype(str).unique()))
            if rank_key not in rank_cache:
                rank_cache[rank_key] = rank_features(
                    outer_train, features_all, selection_method, max(TOP_SIZES)
                )
            ranking = rank_cache[rank_key]
            selected = ranking[:top_size]
            for feature in selected:
                feature_counts[(top_size, selection_method, feature)] += 1

            try:
                probability, graph = posterior_probability(
                    model_name, outer_train, outer_valid, selected, n_bins
                )
            except Exception as exc:
                failures.append({
                    "Seleccion": selection_method,
                    "Top": top_size,
                    "Modelo": model_name,
                    "Bins": n_bins,
                    "Fase": f"Pliegue_externo_{fold_number}",
                    "Error": f"{type(exc).__name__}: {exc}",
                })
                pd.DataFrame(
                    failures,
                    columns=["Seleccion", "Top", "Modelo", "Bins", "Fase", "Error"],
                ).to_csv(output_dir / "Fallos_modelos.csv", index=False)
                raise RuntimeError(
                    f"Falló la configuración seleccionada en el pliegue externo "
                    f"{fold_number}, Top {top_size}: {exc}"
                ) from exc

            _, prediction = calculate_metrics(
                outer_valid[TARGET].to_numpy(dtype=int), probability, threshold
            )
            for edge in graph.edges:
                edge_counts[(top_size, selection_method, model_name, n_bins, edge[0], edge[1])] += 1
            edge_denominators[(top_size, selection_method, model_name, n_bins)] += 1

            for row_idx, (_, row) in enumerate(outer_valid.iterrows()):
                oof_records.append({
                    "Alcance": scope_name,
                    "Top": top_size,
                    "Seleccion": selection_method,
                    "Pliegue": fold_number,
                    "Sujeto": row["Sujeto"],
                    "Tarea": row["Tarea"],
                    "Trial": row["Trial"],
                    "Epoca": row["Epoca"],
                    TARGET: int(row[TARGET]),
                    "Probabilidad": float(probability[row_idx]),
                    "Prediccion": int(prediction[row_idx]),
                    "Umbral": threshold,
                    "Modelo": model_name,
                    "Bins": n_bins,
                })
            outer_config_records.append({
                "Alcance": scope_name,
                "Seleccion": selection_method,
                "Top": top_size,
                "Pliegue": fold_number,
                "Modelo_seleccionado": model_name,
                "Bins": n_bins,
                "Umbral": threshold,
                "Exactitud_balanceada_interna": winner["Exactitud_balanceada"],
                "Arcos_promedio_interno": winner["Arcos_promedio"],
                "Tiempo_modelado_interno_seg": winner["Tiempo_modelado_seg"],
                "Caracteristicas": ";".join(selected),
            })

    oof = pd.DataFrame(oof_records)
    summary_rows = []
    for selection_method in RANK_METHODS:
      method_top_sizes = (
          scope_top_sizes
          if scope_name.startswith('Banda_') or selection_method != 'DT'
          else DT_GLOBAL_TOP_SIZES
      )
      for top_size in method_top_sizes:
        subset = oof[(oof["Top"] == top_size) & (oof["Seleccion"] == selection_method)].copy()
        metric_values, _ = calculate_metrics(
            subset[TARGET].to_numpy(dtype=int),
            subset["Probabilidad"].to_numpy(dtype=float),
            0.5,
        )
        # Umbrales distintos por pliegue ya están fijados solo con validación interna.
        pred = subset["Prediccion"].to_numpy(dtype=int)
        true = subset[TARGET].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
        metric_values["Exactitud"] = accuracy_score(true, pred)
        metric_values["Exactitud_balanceada"] = balanced_accuracy_score(true, pred)
        metric_values["Sensibilidad"] = recall_score(true, pred, zero_division=0)
        metric_values["Especificidad"] = tn / (tn + fp) if tn + fp else 0.0
        metric_values["Precision"] = precision_score(true, pred, zero_division=0)
        metric_values["F1"] = f1_score(true, pred, zero_division=0)
        metric_values["Umbral_promedio"] = subset["Umbral"].mean()
        metric_values["TN"], metric_values["FP"] = int(tn), int(fp)
        metric_values["FN"], metric_values["TP"] = int(fn), int(tp)
        for metric_name in ("Exactitud_balanceada", "F1", "ROC_AUC", "PR_AUC"):
            low, high = bootstrap_subject_ci(subset, metric_name, 0.5)
            metric_values[f"{metric_name}_IC95_inf"] = low
            metric_values[f"{metric_name}_IC95_sup"] = high
        summary_rows.append({"Alcance": scope_name, "Seleccion": selection_method,
                             "Top": top_size, **metric_values})

    summary = pd.DataFrame(summary_rows)
    # El desempate favorece menos variables: prioridad por exactitud balanceada, luego Top menor.
    selected_pipeline = summary.sort_values(
        ["Exactitud_balanceada", "Top", "Seleccion"],
        ascending=[False, True, True], kind="stable"
    ).iloc[0]
    chosen_top = int(selected_pipeline["Top"])
    chosen_selector = str(selected_pipeline["Seleccion"])

    # Selección final dentro de los 32 sujetos de desarrollo; la prueba externa
    # se toca una sola vez después de fijar Top, estructura, bins y umbral.
    final_tuned = tune_model(
        development, features_all, chosen_selector, chosen_top,
        INNER_FOLDS + 1, rank_cache
    )
    inner_config_records.extend(final_tuned["candidate_rows"])
    failures.extend(final_tuned["candidate_failures"])
    final_winner = final_tuned["winner"]
    final_model_name = str(final_winner["Modelo"])
    final_bins = int(final_winner["Bins"])
    final_threshold = float(final_tuned["threshold"])
    final_rank_key = (chosen_selector, *sorted(development["Sujeto"].astype(str).unique()))
    if final_rank_key not in rank_cache:
        rank_cache[final_rank_key] = rank_features(
            development, features_all, chosen_selector, max(TOP_SIZES)
        )
    final_ranking = rank_cache[final_rank_key]
    final_features = final_ranking[:chosen_top]
    try:
        external_probability, final_graph = posterior_probability(
            final_model_name, development, external, final_features, final_bins
        )
    except Exception as exc:
        failures.append({
            "Seleccion": chosen_selector,
            "Top": chosen_top,
            "Modelo": final_model_name,
            "Bins": final_bins,
            "Fase": "Prueba_externa",
            "Error": f"{type(exc).__name__}: {exc}",
        })
        pd.DataFrame(
            failures, columns=["Seleccion", "Top", "Modelo", "Bins", "Fase", "Error"]
        ).to_csv(output_dir / "Fallos_modelos.csv", index=False)
        raise RuntimeError(f"Falló el ajuste final de prueba externa: {exc}") from exc
    external_metrics, external_prediction = calculate_metrics(
        external[TARGET].to_numpy(dtype=int), external_probability, final_threshold
    )

    external_output = external[
        ["Sujeto", "Tarea", "Trial", "Epoca", "Puntaje", TARGET]
    ].copy()
    external_output["Probabilidad"] = external_probability
    external_output["Prediccion"] = external_prediction
    external_output["Umbral"] = final_threshold
    external_output["Top"] = chosen_top
    external_output["Seleccion"] = chosen_selector
    external_output["Modelo"] = final_model_name
    external_output["Bins"] = final_bins

    subject_metrics = []
    for subject, subject_rows in external_output.groupby("Sujeto"):
        scores, _ = calculate_metrics(
            subject_rows[TARGET].to_numpy(dtype=int),
            subject_rows["Probabilidad"].to_numpy(dtype=float),
            final_threshold,
        )
        subject_metrics.append({"Sujeto": subject, **scores})

    baseline_probability = np.full(len(external), development[TARGET].mean())
    baseline_metrics, baseline_prediction = calculate_metrics(
        external[TARGET].to_numpy(dtype=int), baseline_probability, 0.5
    )
    external_result_rows = [
        {
            "Alcance": scope_name,
            "Metodo": "Red bayesiana seleccionada en desarrollo",
            "Top": chosen_top,
            "Seleccion": chosen_selector,
            "Modelo": final_model_name,
            "Bins": final_bins,
            **external_metrics,
        },
        {
            "Alcance": scope_name,
            "Metodo": "Referencia: clase mayoritaria de desarrollo",
            "Top": 0,
            "Seleccion": "No aplica",
            "Modelo": "Baseline",
            "Bins": np.nan,
            **baseline_metrics,
        },
    ]

    summary.to_csv(output_dir / "Resumen_CV_anidada_por_Top.csv", index=False)
    oof.to_csv(output_dir / "Predicciones_CV_anidada.csv", index=False)
    oof.groupby(["Seleccion", "Top", "Modelo", "Bins"], dropna=False).size().rename(
        "Numero_predicciones"
    ).reset_index().to_csv(output_dir / "Conteo_predicciones_CV.csv", index=False)
    pd.DataFrame(outer_config_records).to_csv(
        output_dir / "Configuracion_seleccionada_por_pliegue.csv", index=False
    )
    pd.DataFrame(inner_config_records).to_csv(
        output_dir / "Diagnostico_configuraciones_internas.csv", index=False
    )
    pd.DataFrame(failures, columns=["Seleccion", "Top", "Modelo", "Bins", "Fase", "Error"]).to_csv(
        output_dir / "Fallos_modelos.csv", index=False
    )
    pd.DataFrame(external_result_rows).to_csv(
        output_dir / "Resultado_prueba_externa_una_sola_configuracion.csv", index=False
    )
    external_output.to_csv(output_dir / "Predicciones_prueba_externa.csv", index=False)
    pd.DataFrame(subject_metrics).to_csv(
        output_dir / "Metricas_por_sujeto_externo.csv", index=False
    )
    pd.DataFrame({"Orden": range(1, len(final_ranking) + 1), "Caracteristica": final_ranking}).to_csv(
        output_dir / f"Ranking_{chosen_selector}_final_desarrollo.csv", index=False
    )

    feature_rows = []
    for (top, selection_method, feature), count in sorted(feature_counts.items()):
        feature_rows.append({
            "Top": top,
            "Seleccion": selection_method,
            "Caracteristica": feature,
            "Pliegues_incluida": count,
            "Frecuencia": count / OUTER_FOLDS,
        })
    pd.DataFrame(feature_rows).to_csv(
        output_dir / "Estabilidad_caracteristicas_seleccion.csv", index=False
    )

    edge_rows = []
    for (top, selection_method, model_name, bins, source, target), count in sorted(edge_counts.items()):
        denominator = edge_denominators[(top, selection_method, model_name, bins)]
        edge_rows.append({
            "Top": top,
            "Seleccion": selection_method,
            "Modelo": model_name,
            "Bins": bins,
            "Origen": source,
            "Destino": target,
            "Pliegues_con_este_modelo": denominator,
            "Pliegues_con_arco": count,
            "Frecuencia_arco": count / denominator if denominator else np.nan,
        })
    pd.DataFrame(edge_rows).to_csv(output_dir / "Estabilidad_arcos_CV.csv", index=False)

    final_edges = list(final_graph.edges())
    pd.DataFrame(final_edges, columns=["Origen", "Destino"]).to_csv(
        output_dir / "Arcos_red_final.csv", index=False
    )
    save_graph(
        final_edges,
        [TARGET, *final_features],
        output_dir / "Red_final_prueba_externa.png",
        f"Red seleccionada en desarrollo: {final_model_name}, Top {chosen_top}, {final_bins} bins",
    )
    with open(output_dir / "Protocolo_ejecucion.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "SAM_umbral": ">=1; control=0",
                "variables_candidatas": len(features_all),
                "metodos_seleccion_comparados": RANK_METHODS,
                "seleccion": f"{chosen_selector}, recalculado dentro de cada conjunto de entrenamiento",
                "tops_globales_evaluados": TOP_SIZES,
                "tops_globales_DT_evaluados": DT_GLOBAL_TOP_SIZES,
                "tops_por_banda_evaluados": BAND_TOP_SIZES,
                "validacion": f"CV anidada agrupada por sujeto: {OUTER_FOLDS} pliegues externos, {INNER_FOLDS} internos por pliegue y {INNER_FOLDS + 1} internos para selección final",
                "PC_max_cond_vars": 2,
                "sujetos_desarrollo": list(map(str, development_subjects)),
                "sujetos_prueba_externa": list(map(str, external_subjects)),
                "modelo_final": final_model_name,
                "metodo_seleccion_final": chosen_selector,
                "bins_finales": final_bins,
                "top_final": chosen_top,
                "umbral_final": final_threshold,
                "prior_parametros": "BDeu, equivalent_sample_size=5",
                "mecanismo_balanceo": "sin sobremuestreo; prevalencia estimada en datos de entrenamiento",
                "precision_sujeto": len(external_subjects),
                "nota": "Las filas de época están agrupadas por Sujeto. La normalización utiliza el baseline de relajación de cada sujeto; la interpretación asume disponibilidad de ese baseline al predecir.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Análisis terminado: {scope_name}")
    print(summary[["Seleccion", "Top", "Exactitud_balanceada", "F1", "PR_AUC", "ROC_AUC"]].to_string(index=False))
    print(
        f"Prueba externa reservada: {chosen_selector}, Top {chosen_top}, {final_model_name}, "
        f"{final_bins} bins. Resultados en: {output_dir}"
    )


def build_scopes(data: pd.DataFrame, features: list[str], analysis: str):
    tasks = ("Aritmetica", "Espejo", "Stroop")
    scopes = {}
    if analysis in ("global", "all"):
        scopes["Global"] = (data, features)
    if analysis in ("tasks", "all"):
        for task in tasks:
            subset = data[data["Tarea"].isin(["Relajacion", task])].copy()
            if not subset.empty:
                scopes[f"Tarea_{task}"] = (subset, features)
    if analysis in ("trials", "all"):
        for trial in sorted(data["Trial"].dropna().unique()):
            subset = data[data["Trial"] == trial].copy()
            if not subset.empty:
                scopes[f"Trial_{trial}"] = (subset, features)
    if analysis in ("bands", "all"):
        for band in ARTICLE_BANDS:
            band_features = [
                feature for feature in features
                if (
                    (not feature.startswith(("Asym_", "Ratio_")) and feature.endswith(f"_{band}"))
                    or (feature.startswith("Asym_") and feature.endswith(f"_{band}"))
                    or (
                        feature.startswith("Ratio_")
                        and band in feature.split("_")[1]
                    )
                )
            ]
            if len(band_features) < max(BAND_TOP_SIZES):
                raise ValueError(
                    f"La banda {band} tiene {len(band_features)} variables; "
                    f"no alcanza el Top {max(BAND_TOP_SIZES)}."
                )
            scopes[f"Banda_{band}"] = (data, band_features)
    return scopes


def main():
    parser = argparse.ArgumentParser(
        description="CV anidada por sujeto para redes bayesianas y cuatro métodos de selección."
    )
    parser.add_argument(
        "--analysis",
        choices=("global", "bands", "tasks", "trials", "all"),
        default="global",
        help="Análisis global (por defecto), por bandas, tareas, trials o todos.",
    )
    args = parser.parse_args()

    data, features = load_data()
    subjects = np.sort(data["Sujeto"].unique())
    if len(subjects) != 40:
        raise ValueError(
            f"Se esperaban 40 sujetos SAM-40 y se encontraron {len(subjects)}."
        )
    # Réplica exactamente la partición aleatoria del análisis previo: 32/8.
    shuffled_subjects = np.random.RandomState(SEED).permutation(subjects)
    development_subjects = shuffled_subjects[:N_DEVELOPMENT_SUBJECTS]
    external_subjects = shuffled_subjects[N_DEVELOPMENT_SUBJECTS:]
    scopes = build_scopes(data, features, args.analysis)
    if not scopes:
        raise ValueError(f"No hay subconjuntos disponibles para --analysis {args.analysis}.")
    print(
        f"Filas: {len(data)} | Sujetos: {data['Sujeto'].nunique()} | "
        f"Prevalencia SAM>=1: {data[TARGET].mean():.3f} | "
        f"Características: {len(features)} | Rankings: {', '.join(RANK_METHODS)}"
    )
    for scope_name, (subset, scope_features) in scopes.items():
        output_dir = OUTPUT_ROOT / scope_name
        run_scope(
            subset,
            scope_features,
            scope_name,
            output_dir,
            development_subjects,
            external_subjects,
        )


if __name__ == "__main__":
    main()
