"""
charts.py
=========
Publication-quality figures for the equity factor model.

Figures
-------
1.  IC Decay Curves  — IC vs forecast horizon for each factor
2.  Factor IC Time Series  — rolling 12M IC for composite
3.  Factor Correlation Heatmap  — cross-factor redundancy
4.  IC Summary Table  — mean/t-stat/ICIR with significance markers
5.  Cumulative NAV  — portfolio vs SPY benchmark
6.  Drawdown  — underwater equity curve
7.  Factor Exposure History  — how much of each factor over time
8.  Rolling Sharpe  — 12M rolling Sharpe ratio
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FACTOR_COLORS = {
    "MOM":       "#2166ac",
    "LV":        "#d6604d",
    "SIZE":      "#4dac26",
    "VAL":       "#a6761d",
    "QUAL":      "#762a83",
    "COMPOSITE": "#1a1a2e",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "figure.dpi":       130,
})


def _save(fig, out: Path, name: str) -> None:
    p = out / name
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", p)


def fig1_ic_decay(decay_dict: Dict[str, pd.Series], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for factor, decay in decay_dict.items():
        if decay.empty: continue
        color = FACTOR_COLORS.get(factor, "#888")
        lw    = 3.0 if factor == "COMPOSITE" else 1.8
        ax.plot(decay.index, decay.values, color=color, lw=lw,
                marker="o", markersize=5, label=factor)
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xlabel("Forecast Horizon (Trading Days)")
    ax.set_ylabel("Mean IC (Spearman)")
    ax.set_title("Figure 1: IC Decay Curves by Factor\n"
                 "How predictive power degrades as forecast horizon increases", pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out, "fig1_ic_decay.png")


def fig2_ic_time_series(ic_series: pd.Series, out: Path) -> None:
    roll12 = ic_series.rolling(12, min_periods=6).mean()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax = axes[0]
    colors = ["#276749" if v > 0 else "#c1121f" for v in ic_series.values]
    ax.bar(ic_series.index, ic_series.values, color=colors, width=20, alpha=0.7)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Monthly IC")
    ax.set_title("Figure 2a: Composite Factor IC — Monthly", fontweight="bold")

    ax2 = axes[1]
    ax2.plot(roll12.index, roll12.values, color="#2166ac", lw=2.5, label="12M Rolling IC")
    ax2.fill_between(roll12.index, 0, roll12.values,
                     where=roll12 > 0, alpha=0.25, color="#276749")
    ax2.fill_between(roll12.index, 0, roll12.values,
                     where=roll12 < 0, alpha=0.25, color="#c1121f")
    ax2.axhline(0,    color="black", lw=0.7)
    ax2.axhline(0.05, color="#276749", lw=1, ls=":", label="IC=0.05 threshold")
    ax2.set_ylabel("Rolling IC")
    ax2.set_title("Figure 2b: 12-Month Rolling IC", fontweight="bold")
    ax2.legend(frameon=False, fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, out, "fig2_ic_time_series.png")


def fig3_factor_correlation(corr_df: pd.DataFrame, out: Path) -> None:
    if corr_df.empty: return
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.RdBu_r
    im   = ax.imshow(corr_df.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    factors = corr_df.columns.tolist()
    ax.set_xticks(range(len(factors))); ax.set_xticklabels(factors, rotation=45, ha="right")
    ax.set_yticks(range(len(factors))); ax.set_yticklabels(factors)
    for i in range(len(factors)):
        for j in range(len(factors)):
            val = corr_df.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if abs(val) > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Pearson Correlation")
    ax.set_title("Figure 3: Cross-Factor Correlation Matrix\n"
                 "|ρ| > 0.5 indicates factor redundancy", pad=10)
    fig.tight_layout()
    _save(fig, out, "fig3_factor_correlation.png")


def fig4_ic_summary_table(ic_summary: pd.DataFrame, out: Path) -> None:
    if ic_summary.empty: return
    sub = ic_summary[["IC_mean", "t_stat", "ICIR", "hit_rate", "significant"]].copy()
    sub = sub.reset_index()
    fig, ax = plt.subplots(figsize=(14, max(5, len(sub) * 0.45 + 1.5)))
    ax.axis("off")
    col_labels = ["Factor", "Horizon", "IC Mean", "t-stat", "ICIR", "Hit Rate", "Sig?"]
    cell_data  = [
        [row["factor"], f"{int(row['horizon_days'])}d",
         f"{row['IC_mean']:.4f}", f"{row['t_stat']:.2f}",
         f"{row['ICIR']:.3f}", f"{row['hit_rate']:.1%}",
         "✓" if row["significant"] else ""]
        for _, row in sub.iterrows()
    ]
    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#1a365d"); cell.set_text_props(color="white", fontweight="bold")
        elif cell_data[r-1][-1] == "✓" if r > 0 else False:
            cell.set_facecolor("#f0fff4")
    ax.set_title("Figure 4: IC Summary — Mean, t-stat, ICIR (Bonferroni-corrected)",
                 fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, out, "fig4_ic_summary.png")


def fig5_cumulative_nav(results: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(results.index, results["cum_port"],  color="#1a1a2e", lw=2.5, label="Factor Portfolio")
    if "cum_bench" in results.columns and results["cum_bench"].notna().any():
        ax.plot(results.index, results["cum_bench"], color="#888888", lw=1.8,
                ls="--", label="Benchmark (SPY)")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Figure 5: Cumulative NAV — Factor Portfolio vs Benchmark", pad=10)
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, out, "fig5_cumulative_nav.png")


def fig6_drawdown(results: pd.DataFrame, out: Path) -> None:
    nav  = results["cum_port"]
    peak = nav.expanding().max()
    dd   = (nav / peak - 1) * 100
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#c1121f", alpha=0.7)
    ax.plot(dd.index, dd.values, color="#c1121f", lw=1)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Figure 6: Portfolio Drawdown", pad=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, out, "fig6_drawdown.png")


def fig7_rolling_sharpe(results: pd.DataFrame, out: Path, window: int = 252) -> None:
    r     = results["port_ret"].dropna()
    rs    = r.rolling(window, min_periods=126).mean() / r.rolling(window, min_periods=126).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(rs.index, rs.values, color="#2166ac", lw=2)
    ax.axhline(0,   color="black", lw=0.7, ls="--")
    ax.axhline(0.5, color="#276749", lw=1, ls=":", label="Sharpe = 0.5")
    ax.fill_between(rs.index, 0, rs.values, where=rs > 0, alpha=0.2, color="#276749")
    ax.fill_between(rs.index, 0, rs.values, where=rs < 0, alpha=0.2, color="#c1121f")
    ax.set_ylabel("Rolling Sharpe Ratio (252-day)")
    ax.set_title("Figure 7: Rolling 12-Month Sharpe Ratio", pad=10)
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, out, "fig7_rolling_sharpe.png")


def save_all(
    decay_dict: Dict,
    ic_series:  pd.Series,
    corr_df:    pd.DataFrame,
    ic_summary: pd.DataFrame,
    results:    pd.DataFrame,
    out_dir:    Path,
) -> None:
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    fig1_ic_decay(decay_dict, fig_dir)
    fig2_ic_time_series(ic_series, fig_dir)
    fig3_factor_correlation(corr_df, fig_dir)
    fig4_ic_summary_table(ic_summary, fig_dir)
    fig5_cumulative_nav(results, fig_dir)
    fig6_drawdown(results, fig_dir)
    fig7_rolling_sharpe(results, fig_dir)

    ic_summary.to_csv(tbl_dir / "ic_summary.csv")
    corr_df.to_csv(tbl_dir / "factor_correlation.csv")
    results.to_csv(tbl_dir / "backtest_results.csv")
    logger.info("All outputs saved to %s", out_dir)
