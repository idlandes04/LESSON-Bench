"""Analysis utilities for LESSON-Bench results.

Provides data loading, figure generation, and PDF report export.
Re-exports statistical functions from lesson.eval.stats and lesson.eval.analysis.

Usage::

    from lesson.analysis import generate_report, fig_gap_chart
    from lesson.analysis import compute_flr_with_ci, bootstrap_ci
"""

# Pipeline: data loading and computation
from lesson.analysis.pipeline import (
    load_all_turn_data,
    load_condition_means,
    compute_gap_data,
    compute_flr_data,
    compute_factorial_data,
    compute_grouping_data,
    get_model_label,
)

# Figures: publication-quality chart generation
from lesson.analysis.figures import (
    fig_gap_chart,
    fig_factorial_2x2,
    fig_trajectory,
    fig_trajectory_grid,
    fig_codex_vs_chat,
    fig_model_grouping_boxplot,
    fig_summary_table,
)

# Report: PDF generation
from lesson.analysis.report import generate_report, save_individual_figures

# Re-exports from stdlib analysis
from lesson.eval.analysis import (
    summarize_sb1_results,
    summarize_sb2_results,
    estimate_flr,
    compute_type_e_feasibility,
)

# Re-exports from numpy stats
from lesson.eval.stats import (
    bootstrap_ci,
    bootstrap_ci_difference,
    compute_aulc,
    compute_rii,
    compute_htr,
    compute_flr_with_ci,
    compute_condition_trajectory,
    factorial_decomposition,
    factorial_decomposition_with_ci,
    compute_model_profile,
    rank_correlation,
    fmt_ci,
    results_to_instance_accuracies,
)

__all__ = [
    # Pipeline
    "load_all_turn_data",
    "load_condition_means",
    "compute_gap_data",
    "compute_flr_data",
    "compute_factorial_data",
    "compute_grouping_data",
    "get_model_label",
    # Figures
    "fig_gap_chart",
    "fig_factorial_2x2",
    "fig_trajectory",
    "fig_trajectory_grid",
    "fig_codex_vs_chat",
    "fig_model_grouping_boxplot",
    "fig_summary_table",
    # Report
    "generate_report",
    "save_individual_figures",
    # Stdlib analysis
    "summarize_sb1_results",
    "summarize_sb2_results",
    "estimate_flr",
    "compute_type_e_feasibility",
    # Numpy stats
    "bootstrap_ci",
    "bootstrap_ci_difference",
    "compute_aulc",
    "compute_rii",
    "compute_htr",
    "compute_flr_with_ci",
    "compute_condition_trajectory",
    "factorial_decomposition",
    "factorial_decomposition_with_ci",
    "compute_model_profile",
    "rank_correlation",
    "fmt_ci",
    "results_to_instance_accuracies",
]
