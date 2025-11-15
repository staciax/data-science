from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, Normalize, to_rgba
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from sklearn.decomposition import PCA

__all__ = (
    'plot_clusters',
    'plot_confusion_matrix',
    'plot_cumulative_explained_variance',
    'plot_dendrogram',
    'plot_elbow',
    'plot_pca_scatter',
)


def plot_confusion_matrix(
    cm: np.typing.ArrayLike,
    classes: list[str],
    *,
    normalize: bool = False,
    title: str | None = None,
    cmap: str | Colormap = 'Greens',
):
    cm = np.asarray(cm)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm.astype(float) / row_sums

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    vmax = 1.0 if normalize else cm.max(initial=1)
    norm = Normalize(vmin=0.0, vmax=vmax)

    plt.figure(figsize=(7, 7), dpi=120)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        norm=norm,
        square=True,
        cbar=True,
        # cbar_kws={"shrink": 0.9, "aspect": 20},
        cbar_kws={
            'shrink': 0.7,
            'aspect': 10,
        },
        linewidths=0.5,
        linecolor='white',
    )

    if title:
        ax.set_title(title, fontsize=16, pad=15)

    ax.set_xlabel('Predicted label', fontsize=14)
    ax.set_ylabel('True label', fontsize=14)

    ax.set_xticks(np.arange(len(classes)) + 0.5)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=14)
    ax.set_yticks(np.arange(len(classes)) + 0.5)
    ax.set_yticklabels(classes, rotation=0, fontsize=14)

    texts = ax.texts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            r, g, b, _ = to_rgba(cmap(norm(val)))
            # perceived luminance
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            color = 'white' if lum < 0.5 else 'black'
            texts[i * cm.shape[1] + j].set_color(color)
            texts[i * cm.shape[1] + j].set_fontsize(16)

    plt.tight_layout()
    plt.grid(False)
    plt.show()


#


def plot_cumulative_explained_variance(
    pca: PCA,
    *,
    threshold: float = 0.90,
    figsize: tuple[int, int] = (8, 5),
    dpi: int = 120,
    title: (str | None) = 'Cumulative Explained Variance vs Number of Principal Components',
) -> None:
    cumvar: np.typing.NDArray[np.floating] = np.cumsum(pca.explained_variance_ratio_)
    n_components = len(cumvar)

    if threshold > 1.0:
        threshold = threshold / 100.0

    effective_threshold = min(threshold, float(cumvar[-1]))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(
        range(1, n_components + 1),
        cumvar,
        marker='o',
        linestyle='--',
    )

    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance')
    if title is not None:
        ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)

    # threshold line
    ax.axhline(y=effective_threshold, color='r', linestyle='-', linewidth=1.75)

    # find the first index that reaches the threshold
    idx_first = int(np.searchsorted(cumvar, effective_threshold)) + 1  # 1-based

    if idx_first <= n_components:
        x_pt = idx_first
        y_pt = float(cumvar[idx_first - 1])

        # จุด
        ax.scatter([x_pt], [y_pt], color='red', zorder=5)

        # 1) Horizontal: if near right edge, shift left a bit
        is_near_right = (x_pt / n_components) > 0.8
        if is_near_right:
            x_offset = -28
            ha = 'right'
        else:
            x_offset = 8
            ha = 'left'

        # 2) Vertical: if near top, shift down instead of up
        is_near_top = y_pt > 0.92
        if is_near_top:
            y_offset = -14
            va = 'top'
        else:
            y_offset = 10
            va = 'bottom'

        ax.annotate(
            f'{effective_threshold * 100:.0f}% at PC {idx_first}',
            xy=(x_pt, y_pt),
            xycoords='data',
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            fontsize=10,
            color='red',
            ha=ha,
            va=va,
            bbox={
                'facecolor': 'white',
                'edgecolor': 'none',
                'alpha': 0.6,
                'pad': 1.5,
            },
            clip_on=True,
        )
    else:
        # case: threshold not reached
        ax.text(
            0.5,
            0.9,
            f'{int(threshold * 100)}% not reached',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=10,
            color='red',
        )

    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    plt.show()


#


def plot_pca_scatter(
    X_pca: np.ndarray,
    label: np.typing.ArrayLike,
    *,
    figsize=(10, 8),
    dpi: int = 120,
    title: str = 'Data Distribution after PCA',
    xlabel: str = 'Principal Component 1',
    ylabel: str = 'Principal Component 2',
    palette: Any = None,
    alpha: float = 0.8,
    markerscale: float = 2.0,
) -> None:
    dfc = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'label': label,
    })

    plt.figure(figsize=figsize, dpi=dpi)
    sns.scatterplot(
        data=dfc,
        x='PC1',
        y='PC2',
        hue='label',
        palette=palette,
        legend=True,
        s=25,
        alpha=alpha,
        edgecolor=None,
        zorder=3,
        # hue_order=["red", "white"],
        # edgecolor=None,
        # edgecolor="w",
    )

    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    fontsize_by_figure = max(figsize) * dpi // 100

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    # ncol calculation
    n_labels = len(labels) - 1  # exclude title
    if n_labels <= 5:
        ncol = 1
    elif n_labels <= 10:
        ncol = 2
    else:
        ncol = 3

    legend = ax.legend(
        handles,
        labels,
        title='True Labels',
        fontsize=fontsize_by_figure,
        title_fontsize=fontsize_by_figure + 2,
        markerscale=markerscale,
        handlelength=1.8,
        borderpad=1.0,
        borderaxespad=0.6,
        ncol=ncol,
        # handlelength=1.8,
        # borderpad=1.0,
        # labelspacing=0.8,
        # borderaxespad=0.6,
    )

    if legend is not None:
        for h in legend.legend_handles:
            if hasattr(h, 'set_sizes'):  # scatter -> PathCollection
                h.set_sizes([180])

    plt.grid(True, linestyle='--', alpha=0.65)
    plt.tight_layout()
    plt.show()


#


def plot_elbow(
    Ks: Sequence[int],
    wcss: Sequence[float],
    *,
    title: str = 'Elbow Method',
    show: bool = True,
) -> int | tuple[plt.Figure, plt.Axes, int | None]:
    fig = plt.figure(figsize=(6, 4), dpi=120)
    plt.plot(Ks, wcss, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title(
        title,
        fontdict={
            'fontsize': 16,
        },
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

    kl = KneeLocator(Ks, wcss, curve='convex', direction='decreasing')
    elbow_k: int | None = kl.elbow

    if elbow_k is not None:
        plt.axvline(
            x=elbow_k,
            color='r',
            linestyle='--',
            label=f'Elbow at K={elbow_k}',
            linewidth=1.5,
        )

        elbow_y = wcss[elbow_k - 1]
        plt.scatter([elbow_k], [elbow_y], zorder=5)

        plt.annotate(
            f'K={elbow_k}',
            xy=(elbow_k, elbow_y),
            xytext=(elbow_k + 0.8, elbow_y * 1.05),
            arrowprops={'arrowstyle': '->', 'lw': 1},
            fontsize=10,
            bbox={'boxstyle': 'round, pad=0.2', 'fc': 'white', 'alpha': 0.7},
        )

        plt.legend()
    else:
        plt.text(
            0.5,
            0.9,
            'No clear elbow detected',
            transform=plt.gca().transAxes,
            ha='center',
            va='center',
        )

    plt.grid(True, alpha=0.5, linestyle='--')

    if show:
        plt.show()
        return elbow_k
    else:
        return fig, ax, elbow_k


# clustering


def plot_dendrogram(
    X_rep: np.ndarray,
    method: Literal['single', 'complete', 'average', 'ward'] = 'complete',
    *,
    size: int = 50,
    no_labels: bool = False,
    p: int = 30,
    palette: str | list = 'muted',
    color_threshold: float | None = None,
    above_threshold_color: str = '#CCCCCC',
    random_state: int | None = None,
    x_label: str = 'Sample Index',
    y_label: str = 'Distance',
    figsize: tuple[int, int] = (10, 4.5),
    dpi: int = 120,
) -> None:

    if size > len(X_rep):
        raise ValueError(f'size ({size}) > n_samples ({len(X_rep)})')

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_rep), size=size, replace=False)
    Z = linkage(X_rep[idx], method=method, metric='euclidean')

    pal_full = sns.color_palette(palette, n_colors=12).as_hex()

    set_link_color_palette(pal_full)
    try:
        plt.figure(figsize=figsize, dpi=dpi)

        D = dendrogram(
            Z,
            p=p,
            no_labels=no_labels,
            color_threshold=color_threshold,
            above_threshold_color=above_threshold_color,
            leaf_font_size=10,
        )

        if not no_labels:
            ax = plt.gca()
            plt.gcf().canvas.draw()
            for tick, leaf_color in zip(ax.get_xmajorticklabels(), D['leaves_color_list'], strict=True):
                tick.set_color(leaf_color)
                tick.set_fontfamily('Prompt')
                tick.set_fontweight('bold')
                tick.set_fontsize(10)
                if size > 40:
                    tick.set_rotation(90)

        plt.title(f'Dendrogram - {method} linkage (sample={size})', fontsize=18)
        plt.xlabel(x_label, fontsize=16)
        plt.ylabel(y_label, fontsize=16)
        plt.tight_layout()
        plt.grid(False)
        plt.show()
    finally:
        set_link_color_palette(None)


def plot_clusters(
    labels: np.ndarray,
    X_data: np.ndarray,
    *,
    show_centroids: bool = True,
    palette='muted',
    title: str | None = None,
) -> None:
    dfc = pd.DataFrame({
        'PC1': X_data[:, 0],
        'PC2': X_data[:, 1],
        'cluster': labels,
    })

    cluster_ids = sorted(c for c in pd.unique(dfc['cluster']) if c != -1)
    # palette = sns.color_palette(palette, len(cluster_ids))
    # color_map = dict(zip(cluster_ids, palette, strict=True))

    n_clusters = len(np.unique(labels))

    # if isinstance(palette, str):
    #     palette = sns.color_palette(palette, n_clusters)

    plt.figure(figsize=(10, 8), dpi=120)
    ax = plt.gca()

    sns.scatterplot(
        data=dfc,
        x='PC1',
        y='PC2',
        hue=dfc['cluster'].astype('category'),
        hue_order=cluster_ids,
        palette=palette,
        s=25,
        alpha=0.9,
        edgecolor=None,
        ax=ax,
        legend=True,
        zorder=3,
    )

    if show_centroids:
        centroids = dfc[dfc['cluster'] != -1].groupby('cluster')[['PC1', 'PC2']].mean().reset_index()

        for _, row in centroids.iterrows():
            _cid = int(row['cluster'])
            x, y = row['PC1'], row['PC2']
            ax.scatter(
                x,
                y,
                marker='X',
                s=100,
                # linewidths=1.5,
                edgecolors='k',
                # c=[color_map[cid]],
                c='red',
                alpha=0.8,
                zorder=4,
            )

        from matplotlib.lines import Line2D

        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            Line2D(
                [],
                [],
                marker='X',
                linestyle='None',
                markersize=8,
                markeredgecolor='k',
                markerfacecolor='red',
                label='Centroid',
            )
        )
        labels.append('Centroid')
        # ncol calculation
        n_labels = n_clusters + 1  # include centroids
        if n_labels <= 5:
            ncol = 1
        elif n_labels <= 10:
            ncol = 2
        else:
            ncol = 3

        ax.legend(
            handles=handles,
            labels=labels,
            title='Cluster Groups',
            # loc="upper right",
            # bbox_to_anchor=(1.02, 1),
            ncol=ncol,
            fontsize=10,
            title_fontsize=12,
            frameon=True,
            framealpha=0.6,
            markerscale=1.8,
            handlelength=1.4,
            borderpad=0.6,
        )

    if title:
        ax.set_title(title, fontsize=18)
    ax.set_xlabel('Principal Component 1', fontsize=16)
    ax.set_ylabel('Principal Component 2', fontsize=16)
    ax.grid(True, which='both', linestyle='--', alpha=0.65)
    plt.tight_layout()

    leg = ax.get_legend()
    if leg:
        for i, text in enumerate(leg.get_texts()):
            if text.get_text() in {'Centroid'} or not text.get_text().isdigit():
                continue
            text.set_text(f'Cluster {i + 1}')

    plt.show()
