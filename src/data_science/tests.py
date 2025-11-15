from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

__all__ = (
    'ClusteringMetrics',
    'evaluate_clustering_performance',
    'plot_clustering_metrics',
)


class ClusteringMetrics(TypedDict):
    adjusted_rand_score: float
    normalized_mutual_info: float
    homogeneity: float
    completeness: float
    v_measure: float
    silhouette: float


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    try:
        unique_labels = np.unique(labels)
        if unique_labels.size < 2 or unique_labels.size >= labels.size:
            return np.nan
        return silhouette_score(X, labels)
    except Exception:
        return np.nan


def evaluate_clustering_performance(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    X_data: np.ndarray,
) -> ClusteringMetrics:
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    silhouette = _safe_silhouette(X_data, predicted_labels)

    return {
        'adjusted_rand_score': ari,
        'normalized_mutual_info': nmi,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'silhouette': silhouette,
    }


def plot_clustering_metrics(
    results: list[tuple[str, ClusteringMetrics]],
    *,
    title: str | None = None,
    metrics: Literal[
        'adjusted_rand_score',
        'normalized_mutual_info',
        'homogeneity',
        'completeness',
        'v_measure',
        'silhouette',
    ]
    | None = None,
    palette: Any = None,
    figsize: tuple[int, int] = (14, 8),
    dpi: int = 120,
    show_values: bool = True,
    value_fmt: str = '{:.3f}',
    value_min: float = 0.01,
) -> tuple[np.ndarray, plt.Figure, plt.Axes]:

    records = []

    metrics = metrics or [
        'adjusted_rand_score',
        'normalized_mutual_info',
        'homogeneity',
        'completeness',
        'v_measure',
        'silhouette',
    ]

    metric_labels = {
        'adjusted_rand_score': 'ARI',
        'normalized_mutual_info': 'NMI',
        'homogeneity': 'Homogeneity',
        'completeness': 'Completeness',
        'v_measure': 'V-Measure',
        'silhouette': 'Silhouette',
    }

    for run_label, res in results:
        for m in metrics:
            records.append({
                'run_label': run_label,
                'metric': m,
                'metric_short': metric_labels[m],
                'value': res[m],
            })

    dfc = pd.DataFrame.from_records(records)
    metric_order = [metric_labels[m] for m in metrics]
    dfc['metric_short'] = pd.Categorical(dfc['metric_short'], categories=metric_order, ordered=True)

    # plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.barplot(
        data=dfc,
        x='metric_short',
        y='value',
        hue='run_label',
        palette=palette,
        dodge=True,
        errorbar=None,
        ax=ax,
    )

    for p in ax.patches:
        p.set_linewidth(0.4)
        p.set_alpha(0.8)
        p.set_edgecolor(p.get_facecolor())

    if show_values:
        for p in ax.patches:
            v = p.get_height()
            if np.isfinite(v) and (v > value_min):
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    v + 0.01,
                    value_fmt.format(v),
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    # fontweight='bold',
                )

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel('Evaluation Metrics', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    ax.set_xticks(ax.get_xticks(), labels=ax.get_xticklabels(), fontsize=14)
    ax.set_yticks(ax.get_yticks(), labels=ax.get_yticklabels(), fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(False)
    ax.legend(
        title=None,
        fontsize=14,
        frameon=True,
        fancybox=True,
        loc='upper right',
    )

    vals = dfc['value'].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if vals.size > 0:
        ax.set_ylim(0, float(np.nanmax(vals)) * 1.15)
    plt.tight_layout()

    return dfc, fig, ax
