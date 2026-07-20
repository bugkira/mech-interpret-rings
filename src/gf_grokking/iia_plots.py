"""Publication-style IIA bar charts (Zheng Table 2 protocol)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Muted, print-friendly palette
COLOR_RAW = "#2c5f8a"
COLOR_SUB = "#c44e52"
COLOR_RND = "#b8b8b8"

HIERARCHY_LABELS: dict[str, str] = {
    "random_init": "Random",
    "grokked": "Grokked",
    "random_subspace": "Rand.\nsub.",
}

COMPONENT_LABELS: dict[str, str] = {
    "L_diag1": r"$L_{\mathrm{diag}_1}$",
    "L_diag2": r"$L_{\mathrm{diag}_2}$",
    "L_rad": r"$L_{\mathrm{rad}}$",
    "R_L_field": r"$R_{L}$",
    "R_R_field": r"$R_{R}$",
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _format_hierarchy_tick(label: str, test_acc: float | None = None) -> str:
    return HIERARCHY_LABELS.get(label, label.replace("_", " "))


def _format_component_tick(name: str) -> str:
    return COMPONENT_LABELS.get(name, name.replace("_", " "))


def plot_grouped_iia_bars(
    ax: plt.Axes,
    categories: Sequence[str],
    raw: Sequence[float],
    sub: Sequence[float],
    rnd: Sequence[float],
    *,
    title: str,
    show_legend: bool = True,
    bar_width: float = 0.22,
) -> None:
    x = np.arange(len(categories))
    w = bar_width
    ax.bar(
        x - w,
        raw,
        width=w,
        color=COLOR_RAW,
        edgecolor="white",
        linewidth=0.6,
        label="Raw IIA",
        zorder=3,
    )
    ax.bar(
        x,
        sub,
        width=w,
        color=COLOR_SUB,
        edgecolor="white",
        linewidth=0.6,
        label="Probe subspace",
        zorder=3,
    )
    ax.bar(
        x + w,
        rnd,
        width=w,
        color=COLOR_RND,
        edgecolor="white",
        linewidth=0.6,
        label="Random subspace",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("IIA score")
    ax.set_title(title, pad=10)
    ax.yaxis.grid(True, linestyle="-", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            edgecolor="#cccccc",
            fancybox=False,
        )


def save_iia_figure(fig: plt.Figure, path: Path, *, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_hierarchy_panel(
    ax: plt.Axes,
    rows: list[dict],
    *,
    title: str = "(a) Four-level hierarchy",
) -> None:
    labels = [
        _format_hierarchy_tick(r["label"], r.get("test_accuracy"))
        for r in rows
    ]
    raw = [r["mean_raw_iia"] for r in rows]
    sub = [
        0.0 if r.get("mean_subspace_iia") is None else float(r.get("mean_subspace_iia", 0.0))
        for r in rows
    ]
    rnd = [r.get("mean_random_subspace_iia", 0.0) for r in rows]
    plot_grouped_iia_bars(ax, labels, raw, sub, rnd, title=title, show_legend=False)


def plot_component_panel(
    ax: plt.Axes,
    per_component: list[dict],
    *,
    title: str = "(b) Per Wedderburn component",
) -> None:
    labels = [_format_component_tick(c["irrep"]) for c in per_component]
    raw = [c["raw_iia"] for c in per_component]
    sub = [c["subspace_iia"] for c in per_component]
    rnd = [c["random_subspace_iia"] for c in per_component]
    plot_grouped_iia_bars(ax, labels, raw, sub, rnd, title=title, show_legend=False)


def plot_paper_iia_figure(
    hierarchy_rows: list[dict],
    per_component: list[dict],
    out_path: Path,
    *,
    ring_latex: str = r"$T_2(\mathbb{F}_3)\times\mathbb{F}_3^2$",
) -> None:
    """Two-panel figure for workshop paper."""
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    fig.subplots_adjust(bottom=0.30, wspace=0.32, top=0.88)

    plot_hierarchy_panel(axes[0], hierarchy_rows, title="(a) Four-level hierarchy")
    plot_component_panel(axes[1], per_component, title="(b) Per-component IIA")

    legend_handles = [
        Patch(facecolor=COLOR_RAW, edgecolor="white", label="Raw IIA"),
        Patch(facecolor=COLOR_SUB, edgecolor="white", label="Probe subspace"),
        Patch(facecolor=COLOR_RND, edgecolor="white", label="Random subspace"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    save_iia_figure(fig, out_path)
