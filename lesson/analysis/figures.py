"""Publication-quality figure generation for LESSON-Bench analysis.

Each function creates and returns a matplotlib Figure. No side effects —
no plt.show(), no file I/O. The caller decides whether to display or save.

All figures use a consistent dark-on-white academic style suitable for
PDF reports and Kaggle notebook embedding.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from lesson.analysis.pipeline import (
    get_model_label,
    CORE_CONDITIONS,
    MODEL_META,
)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
}

CONDITION_COLORS = {
    "correction": "#2196F3",
    "practice_only": "#FF9800",
    "error_only": "#9C27B0",
    "no_feedback": "#607D8B",
    "explanation": "#4CAF50",
    "misleading": "#F44336",
    "clean_context": "#00BCD4",
    "prompted_correction": "#795548",
    "structured_correction": "#FF5722",
    "reformatted_correction": "#009688",
}

CONDITION_LABELS = {
    "correction": "Correction",
    "practice_only": "Practice Only",
    "error_only": "Error Only",
    "no_feedback": "No Feedback",
    "explanation": "Explanation",
    "misleading": "Misleading",
    "clean_context": "Clean Context",
    "prompted_correction": "Prompted Correction",
    "structured_correction": "Structured Correction",
    "reformatted_correction": "Reformatted Correction",
}

TRAINING_TYPE_COLORS = {
    "code-tuned": "#2196F3",
    "chat-tuned": "#FF9800",
    "reasoning-rl": "#9C27B0",
    "base": "#607D8B",
    "unknown": "#BDBDBD",
}


def _apply_style():
    """Apply the LESSON-Bench figure style."""
    plt.rcParams.update(STYLE)


def _add_watermark(fig: plt.Figure, text: str = "LESSON-Bench Pilot"):
    """Add a subtle watermark to the figure."""
    fig.text(0.99, 0.01, text, fontsize=7, color="0.7",
             ha="right", va="bottom", style="italic")


# ---------------------------------------------------------------------------
# Figure 1: Gap Chart
# ---------------------------------------------------------------------------

def fig_gap_chart(
    gap_data: List[Dict[str, Any]],
    human_gap: Optional[float] = None,
) -> plt.Figure:
    """Horizontal bar chart of feedback learning gap per model.

    gap_data: list of {model, label, gap, ci_lo, ci_hi} sorted by gap desc.
    human_gap: optional human baseline gap value for reference line.
    """
    _apply_style()

    labels = [d["label"] for d in gap_data]
    gaps = [d["gap"] for d in gap_data]
    ci_lo = [d["gap"] - d["ci_lo"] for d in gap_data]
    ci_hi = [d["ci_hi"] - d["gap"] for d in gap_data]
    errors = [ci_lo, ci_hi]

    colors = ["#2196F3" if g >= 0 else "#F44336" for g in gaps]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.6)))
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, gaps, xerr=errors, height=0.6, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    ax.axvline(x=0, color="black", linewidth=0.8, zorder=5)
    ax.set_xlabel("Feedback Learning Gap (Correction − Practice Only, Turns 7-12)")
    ax.set_title("Feedback Learning Gap by Model")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Annotate values
    for i, (g, label) in enumerate(zip(gaps, labels)):
        x_pos = g + 0.01 if g >= 0 else g - 0.01
        ha = "left" if g >= 0 else "right"
        ax.text(x_pos, i, f"{g*100:+.1f}%", va="center", ha=ha, fontsize=8, fontweight="bold")

    if human_gap is not None:
        ax.axvline(x=human_gap, color="#4CAF50", linewidth=2, linestyle="--",
                   label=f"Human baseline ({human_gap*100:+.0f}%)")
        ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    _add_watermark(fig)
    return fig


# ---------------------------------------------------------------------------
# Figure 2: 2×2 Factorial Decomposition
# ---------------------------------------------------------------------------

def fig_factorial_2x2(
    factorial_data: Dict[str, Dict[str, Tuple[float, float, float]]],
) -> plt.Figure:
    """Grouped bar chart showing answer effect vs evaluation effect per model.

    factorial_data: {model: {"answer_effect": (pt, lo, hi), "evaluation_effect": ..., "interaction": ...}}
    """
    _apply_style()

    models = list(factorial_data.keys())
    labels = [get_model_label(m) for m in models]
    n = len(models)

    answer_pts = [factorial_data[m]["answer_effect"][0] for m in models]
    answer_lo = [factorial_data[m]["answer_effect"][0] - factorial_data[m]["answer_effect"][1] for m in models]
    answer_hi = [factorial_data[m]["answer_effect"][2] - factorial_data[m]["answer_effect"][0] for m in models]

    eval_pts = [factorial_data[m]["evaluation_effect"][0] for m in models]
    eval_lo = [factorial_data[m]["evaluation_effect"][0] - factorial_data[m]["evaluation_effect"][1] for m in models]
    eval_hi = [factorial_data[m]["evaluation_effect"][2] - factorial_data[m]["evaluation_effect"][0] for m in models]

    interaction_pts = [factorial_data[m]["interaction"][0] for m in models]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 6))
    x = np.arange(n)
    w = 0.35

    ax.bar(x - w/2, answer_pts, w, yerr=[answer_lo, answer_hi],
           label="Answer Effect", color="#FF9800", alpha=0.85, capsize=3,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, eval_pts, w, yerr=[eval_lo, eval_hi],
           label="Evaluation Effect", color="#9C27B0", alpha=0.85, capsize=3,
           edgecolor="white", linewidth=0.5)

    # Annotate interaction
    for i, intr in enumerate(interaction_pts):
        if abs(intr) > 0.01:
            ax.text(i, max(answer_pts[i], eval_pts[i]) + 0.03,
                    f"Int: {intr:+.2f}", ha="center", fontsize=7, color="0.4")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Effect Size")
    ax.set_title("2×2 Factorial Decomposition: Answer vs. Evaluation Effect")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    fig.tight_layout()
    _add_watermark(fig)
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Per-model trajectory plot
# ---------------------------------------------------------------------------

def fig_trajectory(
    turn_data: Dict[str, Dict[int, float]],
    model_name: str,
    conditions: Optional[List[str]] = None,
    sb1_baseline: Optional[float] = None,
) -> plt.Figure:
    """Line plot of accuracy by turn for each condition.

    turn_data: {condition: {turn_idx: accuracy}}
    """
    _apply_style()

    if conditions is None:
        conditions = [c for c in CORE_CONDITIONS if c in turn_data]
        # Add any extra conditions present
        for c in turn_data:
            if c not in conditions:
                conditions.append(c)

    fig, ax = plt.subplots(figsize=(10, 5))

    for cond in conditions:
        if cond not in turn_data:
            continue
        turns = sorted(turn_data[cond].keys())
        accs = [turn_data[cond][t] for t in turns]
        color = CONDITION_COLORS.get(cond, "#333333")
        label = CONDITION_LABELS.get(cond, cond)
        ax.plot([t + 1 for t in turns], accs, marker="o", markersize=4,
                linewidth=2, color=color, label=label, alpha=0.9)

    if sb1_baseline is not None:
        ax.axhline(y=sb1_baseline, color="#333333", linewidth=1.5,
                   linestyle="--", alpha=0.6, label=f"SB1 baseline ({sb1_baseline*100:.0f}%)")

    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Feedback Trajectory — {get_model_label(model_name)}")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    _add_watermark(fig)
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Trajectory grid (all models)
# ---------------------------------------------------------------------------

def fig_trajectory_grid(
    all_turn_data: Dict[str, Dict[str, Dict[int, float]]],
    models: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
) -> plt.Figure:
    """Multi-panel trajectory plot: one subplot per model.

    all_turn_data: {model: {condition: {turn_idx: accuracy}}}
    """
    _apply_style()

    if models is None:
        models = sorted(all_turn_data.keys())
    if conditions is None:
        conditions = CORE_CONDITIONS

    n = len(models)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                              squeeze=False, sharey=True)

    for idx, model in enumerate(models):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        model_data = all_turn_data.get(model, {})

        for cond in conditions:
            if cond not in model_data:
                continue
            turns = sorted(model_data[cond].keys())
            accs = [model_data[cond][t] for t in turns]
            color = CONDITION_COLORS.get(cond, "#333333")
            label = CONDITION_LABELS.get(cond, cond)
            ax.plot([t + 1 for t in turns], accs, marker="o", markersize=3,
                    linewidth=1.5, color=color, label=label, alpha=0.85)

        ax.set_title(get_model_label(model), fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        if r == rows - 1:
            ax.set_xlabel("Turn")
        if c == 0:
            ax.set_ylabel("Accuracy")

    # Hide empty subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    # Single legend at the bottom
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(conditions),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Feedback Trajectories Across Models", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _add_watermark(fig)
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Codex vs Chat comparison
# ---------------------------------------------------------------------------

def fig_codex_vs_chat(
    all_turn_data: Dict[str, Dict[str, Dict[int, float]]],
    codex_model: str = "gpt-5.3-codex",
    chat_model: str = "gpt-5.3-chat",
    conditions: Optional[List[str]] = None,
) -> plt.Figure:
    """Side-by-side panels comparing code-tuned vs chat-tuned models.

    Tests Hypothesis H2: code training enables error signal processing.
    """
    _apply_style()

    if conditions is None:
        conditions = CORE_CONDITIONS

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, model, title_suffix in [
        (ax1, codex_model, "(Code-Tuned)"),
        (ax2, chat_model, "(Chat-Tuned)"),
    ]:
        model_data = all_turn_data.get(model, {})
        for cond in conditions:
            if cond not in model_data:
                continue
            turns = sorted(model_data[cond].keys())
            accs = [model_data[cond][t] for t in turns]
            color = CONDITION_COLORS.get(cond, "#333333")
            label = CONDITION_LABELS.get(cond, cond)
            ax.plot([t + 1 for t in turns], accs, marker="o", markersize=4,
                    linewidth=2, color=color, label=label, alpha=0.9)

        ax.set_title(f"{get_model_label(model)}\n{title_suffix}", fontsize=11)
        ax.set_xlabel("Turn Number")
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    ax1.set_ylabel("Accuracy")

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(conditions),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Code Training and Feedback Sensitivity (Hypothesis H2)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _add_watermark(fig)
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Model grouping box plot
# ---------------------------------------------------------------------------

def fig_model_grouping_boxplot(
    grouping_data: Dict[str, List[Dict[str, Any]]],
) -> plt.Figure:
    """Box/strip plot of FLR values grouped by training type.

    grouping_data: {training_type: [{model, label, flr}, ...]}
    """
    _apply_style()

    # Sort groups by mean FLR descending
    group_order = sorted(grouping_data.keys(),
                         key=lambda g: np.mean([d["flr"] for d in grouping_data[g]]),
                         reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    positions = []
    box_data = []
    colors = []
    all_labels = []

    for i, group in enumerate(group_order):
        items = grouping_data[group]
        flrs = [d["flr"] for d in items]
        box_data.append(flrs)
        positions.append(i)
        colors.append(TRAINING_TYPE_COLORS.get(group, "#BDBDBD"))
        all_labels.append(group.replace("-", " ").title())

    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=False, medianprops={"color": "black", "linewidth": 1.5})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (group, items) in enumerate([(g, grouping_data[g]) for g in group_order]):
        flrs = [d["flr"] for d in items]
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(flrs))
        ax.scatter([i + j for j in jitter], flrs, s=40, color=colors[i],
                   edgecolors="black", linewidths=0.5, zorder=5, alpha=0.9)
        # Label each point
        for j, item in enumerate(items):
            ax.annotate(item["label"], (i + jitter[j], item["flr"]),
                        fontsize=6, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")

    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, fontsize=10)
    ax.set_ylabel("FLR (Feedback Learning Rate)")
    ax.set_title("FLR by Model Training Type")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    _add_watermark(fig)
    return fig


# ---------------------------------------------------------------------------
# Figure 7: Summary statistics table
# ---------------------------------------------------------------------------

def fig_summary_table(
    condition_means: Dict[str, Dict[str, float]],
    flr_data: List[Dict[str, Any]],
) -> plt.Figure:
    """Render a summary statistics table as a figure.

    Shows per-model accuracy by condition + FLR.
    """
    _apply_style()

    # Build table data
    models = sorted(condition_means.keys(), key=lambda m: get_model_label(m))
    conditions = CORE_CONDITIONS
    col_labels = ["Model"] + [CONDITION_LABELS.get(c, c) for c in conditions] + ["FLR"]

    flr_lookup = {d["model"]: d["flr"] for d in flr_data}

    cell_text = []
    cell_colors = []

    for model in models:
        row = [get_model_label(model)]
        row_colors = ["white"]
        for cond in conditions:
            val = condition_means.get(model, {}).get(cond)
            if val is not None:
                row.append(f"{val*100:.1f}%")
                # Color by accuracy (green = high, red = low)
                intensity = min(1.0, max(0.0, val))
                row_colors.append(plt.cm.RdYlGn(intensity * 0.7 + 0.15))
            else:
                row.append("—")
                row_colors.append("#f0f0f0")
        # FLR
        flr = flr_lookup.get(model)
        if flr is not None:
            row.append(f"{flr:+.3f}")
            # Color FLR (blue = positive, red = negative)
            if flr > 0.01:
                row_colors.append("#BBDEFB")
            elif flr < -0.01:
                row_colors.append("#FFCDD2")
            else:
                row_colors.append("white")
        else:
            row.append("—")
            row_colors.append("white")

        cell_text.append(row)
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(12, max(3, len(models) * 0.5 + 1.5)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header row
    for j, label in enumerate(col_labels):
        cell = table[0, j]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("Summary: Per-Condition Accuracy and FLR",
                 fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    _add_watermark(fig)
    return fig
