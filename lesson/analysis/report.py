"""PDF report generator for LESSON-Bench analysis results.

Composes all figures from figures.py into a multi-page PDF using
matplotlib's PdfPages backend. Each figure gets a full page with
consistent styling and page numbering.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lesson.analysis.pipeline import (
    load_all_turn_data,
    load_condition_means,
    compute_gap_data,
    compute_flr_data,
    compute_factorial_data,
    compute_grouping_data,
)
from lesson.analysis.figures import (
    fig_gap_chart,
    fig_factorial_2x2,
    fig_trajectory_grid,
    fig_codex_vs_chat,
    fig_model_grouping_boxplot,
    fig_summary_table,
)


def _title_page() -> plt.Figure:
    """Create a title page figure."""
    fig = plt.figure(figsize=(10, 7))
    fig.text(0.5, 0.65, "LESSON-Bench", fontsize=36, fontweight="bold",
             ha="center", va="center", fontfamily="sans-serif")
    fig.text(0.5, 0.55, "Learning from Error Signals in Symbolic Operations",
             fontsize=16, ha="center", va="center", color="0.3")
    fig.text(0.5, 0.42, "Analysis Report", fontsize=20, ha="center",
             va="center", color="0.4")
    fig.text(0.5, 0.30, datetime.now().strftime("%B %d, %Y"),
             fontsize=12, ha="center", va="center", color="0.5")
    fig.text(0.5, 0.15, "Pilot Data — SB2 Multi-Turn Feedback Learning",
             fontsize=11, ha="center", va="center", color="0.5",
             style="italic")
    return fig


def generate_report(
    output_path: str = "results/lesson_bench_report.pdf",
    db_path: Optional[str] = None,
    title: str = "LESSON-Bench Analysis Report",
) -> str:
    """Generate a multi-page PDF report with all analysis figures.

    Args:
        output_path: Where to write the PDF.
        db_path: Path to SQLite DB (default: results/lesson_bench.db).
        title: Report title (used on title page).

    Returns:
        The output path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load all data once
    print("[report] Loading data from DB...")
    turn_data = load_all_turn_data(db_path)
    condition_means = load_condition_means(db_path)
    gap_data = compute_gap_data(db_path)
    flr_data = compute_flr_data(db_path)
    factorial_data = compute_factorial_data(db_path)
    grouping_data = compute_grouping_data(db_path)

    print(f"[report] {len(turn_data)} models loaded")
    print(f"[report] {len(gap_data)} models with gap data")
    print(f"[report] {len(factorial_data)} models with factorial data")

    figures = []

    # Page 1: Title
    print("[report] Generating title page...")
    figures.append(_title_page())

    # Page 2: Summary table
    print("[report] Generating summary table...")
    figures.append(fig_summary_table(condition_means, flr_data))

    # Page 3: Gap chart
    if gap_data:
        print("[report] Generating gap chart...")
        figures.append(fig_gap_chart(gap_data))

    # Page 4: 2×2 factorial
    if factorial_data:
        print("[report] Generating 2×2 factorial chart...")
        figures.append(fig_factorial_2x2(factorial_data))

    # Page 5: Codex vs Chat
    if "gpt-5.3-codex" in turn_data and "gpt-5.3-chat" in turn_data:
        print("[report] Generating Codex vs Chat comparison...")
        figures.append(fig_codex_vs_chat(turn_data))

    # Page 6: Trajectory grid
    if turn_data:
        print("[report] Generating trajectory grid...")
        figures.append(fig_trajectory_grid(turn_data))

    # Page 7: Model grouping
    if grouping_data and len(grouping_data) > 1:
        print("[report] Generating model grouping box plot...")
        figures.append(fig_model_grouping_boxplot(grouping_data))

    # Write PDF
    print(f"[report] Writing {len(figures)} pages to {out}...")
    with PdfPages(str(out)) as pdf:
        for i, fig in enumerate(figures):
            # Add page number
            fig.text(0.5, 0.01, f"— {i + 1} —", fontsize=8, ha="center",
                     color="0.6")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[report] Done. Report saved to {out}")
    return str(out)


def save_individual_figures(
    output_dir: str = "results/figures",
    db_path: Optional[str] = None,
    fmt: str = "png",
) -> list[str]:
    """Save each figure as an individual image file.

    Returns list of saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    turn_data = load_all_turn_data(db_path)
    condition_means = load_condition_means(db_path)
    gap_data = compute_gap_data(db_path)
    flr_data = compute_flr_data(db_path)
    factorial_data = compute_factorial_data(db_path)
    grouping_data = compute_grouping_data(db_path)

    saved: list[str] = []

    named_figs = [
        ("summary_table", fig_summary_table(condition_means, flr_data)),
    ]

    if gap_data:
        named_figs.append(("gap_chart", fig_gap_chart(gap_data)))

    if factorial_data:
        named_figs.append(("factorial_2x2", fig_factorial_2x2(factorial_data)))

    if "gpt-5.3-codex" in turn_data and "gpt-5.3-chat" in turn_data:
        named_figs.append(("codex_vs_chat", fig_codex_vs_chat(turn_data)))

    if turn_data:
        named_figs.append(("trajectory_grid", fig_trajectory_grid(turn_data)))

    if grouping_data and len(grouping_data) > 1:
        named_figs.append(("model_grouping", fig_model_grouping_boxplot(grouping_data)))

    for name, fig in named_figs:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(str(path), bbox_inches="tight", dpi=200)
        plt.close(fig)
        saved.append(str(path))
        print(f"  Saved {path}")

    return saved
