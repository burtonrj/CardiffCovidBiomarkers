from matplotlib.pyplot import Axes
from statistics_utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def box_swarm(df: pd.DataFrame,
              x: str,
              y: str,
              marker_fill: str or None = None,
              xlabel: str or None = None,
              ylabel: str or None = None,
              ylim: tuple or None = None,
              xticklabels: list or None = None,
              legend_title: str or None = None,
              legend_pos: tuple or None = None,
              log_scale: bool = False,
              ax: Axes or None = None,
              wide: bool = False):
    if y not in df.columns:
        df = df[["patient_id", x, marker_fill,
                 "test_name", "test_result"]].drop_duplicates()
        df = df[df.test_name == y].dropna()
        y = "median_test_result"
    if not wide:
        df = df[["patient_id", x, y, marker_fill]].drop_duplicates()
    xlabel = xlabel or x
    ylabel = ylabel or y
    ax = sns.boxplot(x=x, y=y,
                     color="white",
                     data=df,
                     showfliers=False,
                     ax=ax)
    ax = sns.swarmplot(x=x, y=y,
                       hue=marker_fill,
                       data=df,
                       ax=ax,
                       linewidth=1,
                       edgecolor="#5c5c5c",
                       palette=["#f2f2f2", "#5c5c5c"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xticklabels = xticklabels or ax.get_xticklabels()
    ax.set_xticklabels(xticklabels)
    if log_scale:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    legend = ax.get_legend()
    legend_title = legend_title or marker_fill
    legend.set_title(legend_title)
    legend.legendHandles[0].set_linewidth(1)
    legend.legendHandles[1].set_linewidth(1)
    legend.legendHandles[0].set_edgecolor("#5c5c5c")
    legend.legendHandles[1].set_edgecolor("#5c5c5c")
    if legend_pos:
        legend.set_bbox_to_anchor(legend_pos)
    # Stats
    p = stat_test(df[df[x] == 0][y].values, df[df[x] == 1][y].values)
    print(p)
    p = stat_annotation(p)
    plt.gcf().text(.48, 0.95, p)
    trans = ax.get_xaxis_transform()
    ax.plot([0, 1], [1.05, 1.05], color="k", transform=trans, clip_on=False)
    return ax


def box_swarm_facet(df: pd.DataFrame,
                    y: str,
                    fill: str,
                    ylabel: str or None = None,
                    log_scale: bool = False,
                    wide: bool = False,
                    save: str or None = None):
    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]},
                                   figsize=(13, 6))
    if log_scale:
        ax0.set(yscale="log")
        ax1.set(yscale="log")
    df = df[["patient_id", fill, "age_groups", "gender",
             "test_name", "test_result"]].drop_duplicates()
    if not wide:
        df = df.drop_duplicates()
    df = df[df.test_name == y].dropna()
    for ax, x in zip([ax0, ax1], ["age_groups", "gender"]):
        sns.boxplot(x=x,
                    y="test_result",
                    hue=fill,
                    data=df,
                    showfliers=False,
                    ax=ax,
                    palette=["#f2f2f2", "#5c5c5c"])
        sns.swarmplot(x=x,
                      y="test_result",
                      hue=fill,
                      data=df,
                      ax=ax,
                      dodge=True,
                      edgecolor='#5c5c5c',
                      linewidth=1,
                      palette=["white", "black"])
    ax0.set_xlabel("Age")
    ylabel = ylabel or y
    ax0.set_ylabel(ylabel)
    handles, labels = ax0.get_legend_handles_labels()
    for h in handles:
        h.set_linewidth = 2
        h.set_edgecolor("black")
    ax1.set_xlabel("Gender")
    ax1.set_ylabel(ylabel)
    ax0.legend().remove()
    ax1.legend().remove()
    # Stats
    x, y = preprocess_faceted_stats(df=df, dep_var=fill)
    pvals = [stat_test(x[0], x[1]) for x in zip(x, y)]
    pvals = [stat_annotation(p) for p in pvals]
    trans = ax0.get_xaxis_transform()
    pos = [1, 2, 3, 4]
    lines = [(0.8, 1.2), (1.8, 2.2), (2.8, 3.2), (3.8, 4.2)]
    if len(pvals) == 7:
        pos = [0, 1, 2, 3, 4]
        lines = [(-0.2, 0.2), (0.8, 1.2), (1.8, 2.2), (2.8, 3.2), (3.8, 4.2)]
    for p, xp in zip(pvals[:len(pvals) - 2], pos):
        ax0.text(xp, 1.1, p, transform=trans, ha="center")
    for xl in lines:
        ax0.plot(xl, [1.05, 1.05], color="k", transform=trans, clip_on=False)
    trans = ax1.get_xaxis_transform()
    for p, xp in zip(pvals[4:], [0, 1]):
        ax1.text(xp, 1.1, p, transform=trans, ha="center")
    for xl in [(-.2, .2), (0.8, 1.2)]:
        ax1.plot(xl, [1.05, 1.05], color="k", transform=trans, clip_on=False)
    fig.tight_layout()
    if save:
        fig.savefig(save, facecolor="white")
    return fig, (ax0, ax1)
