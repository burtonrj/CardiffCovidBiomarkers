from statsmodels.stats.multitest import multipletests
from pingouin.effsize import compute_effsize
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from rpy2.rinterface import RRuntimeWarning

warnings.filterwarnings("ignore", category=RRuntimeWarning)
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

# Setup R objects
rpy2.robjects.numpy2ri.activate()
robjects.r['options'](warn=-1)
lawstat = rpackages.importr('lawstat')
exact_rank_tests = rpackages.importr("exactRankTests")
rstats = rpackages.importr('stats')


def stat_annotation(p: float):
    """
    Given a p-value, return the correct annotation

    Parameters
    -----------
    p: float
        Given p-value

    Returns
    --------
    str
        Annotation
    """
    p_vals = {"*": [0.05, 0.01],
              "**": [0.01, 0.001],
              "***": [0.001, 0.0001]}
    if p > 0.05:
        return "ns"
    for s, (ceil, floor) in p_vals.items():
        if ceil > p >= floor:
            return s
    return "****"


def shapiro(x: np.array,
            a: float = 0.05):
    """
    Wrapper for stats.shapiro test for testing normality. If True, then cannot reject
    null hypothesis and thus x is assumed to be normal, else False, and we assume
    x is not normal.

    Parameters
    ----------
    x: numpy.Array
    a: float (default = 0.05)
        Alpha
    Returns
    -------
    bool
    """
    return stats.shapiro(x)[1] > a


def symmetry_test(x: np.array, a: float = 0.05):
    """
    Port symmetry.test from lawstats in R
    """
    rvec = robjects.FloatVector(x)
    sym = robjects.r["symmetry.test"]
    return sym(rvec)[3][0] > a


def permutation_test(x: np.array,
                     y: np.array):
    x = robjects.FloatVector(x)
    y = robjects.FloatVector(y)
    perm = robjects.r["perm.test"]
    return perm(x, y, paired=False)[1][0]


def stat_test(x: np.array,
              y: np.array,
              a: float = 0.05):
    """
    Given two arrays, perform suitable statistical inference testing to determine whether
    their distributions significantly differ; were they drawn from the same distribution?
    Tests will be performed as follows:
    1. If both samples are gaussian (Shapiro-Wilk test) then perform a Welch two-sided t test
    2. Else, if both samples are symmetrical (Kolmogorov-Smirnov test) then perform a
    two-tailed Mann-Whitney U test.
    3. Otherwise, perform exact permutation testing with 1000 rounds of re-sampling
    """
    if shapiro(x, a=a) and shapiro(y, a=a):
        # Welch's two-tailed T test
        return stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")[1]
    if symmetry_test(x, a) and symmetry_test(y, a):
        # Mann-whitney two-tailed U test
        return stats.mannwhitneyu(x, y, alternative="two-sided")[1]
    return permutation_test(x, y)


def multi_stat_test(x: np.array,
                    y: np.array,
                    a: float = 0.05):
    """
    Perform statistical inference testing (see stat_test) but correct for multiple comparisons
    using Bonferroni p-value correction.
    """
    pvals = list()
    for x_, y_ in zip(x, y):
        pvals.append(stat_test(x_, y_, a))
    pvals = multipletests(pvals=pvals, alpha=a, method="bonferroni")[1]
    return pvals


def preprocess_faceted_stats(df: pd.DataFrame,
                             dep_var: str):
    """
    For a faceted box-plot, generate list of x and y arrays ready for
    being passed to stat_test
    """
    x = list()
    y = list()
    # Age comparisons
    for ag in ["<=50", "51-65", "66-75", "76-85", ">86"]:
        x_ = df[(df[dep_var] == False) &
                (df["age_groups"] == ag)]["test_result"].values
        y_ = df[(df[dep_var] == True) &
                (df["age_groups"] == ag)]["test_result"].values
        if len(x_) < 5 or len(y_) < 5:
            continue
        x.append(x_)
        y.append(y_)
    for g in ["F", "M"]:
        x_ = df[(df[dep_var] == False) &
                (df["gender"] == g)]["test_result"].values
        y_ = df[(df[dep_var] == True) &
                (df["gender"] == g)]["test_result"].values
        x.append(x_)
        y.append(y_)
    return x, y


def feature_scores(df: pd.DataFrame, features: list, endpoint: str, effect_size: str = "cles"):
    """
    Given the covid admissions dataframe generated earlier, iterate over the
    given features and determine the p-value when comparing patients for a given endpoint
    e.g. death or death and ICU admission.
    All p-values are corrected for multiple comparisons using bonferroni method and alpha = 0.05
    """
    pos = df[df[endpoint] == 1]
    neg = df[df[endpoint] == 0]
    results = {"feature": [],
               "p-values": [],
               "effect size": []}
    for f in features:
        x, y = pos[f].dropna().values, neg[f].dropna().values
        results["feature"].append(f)
        p = stat_test(x, y)
        results["p-values"].append(p)
        es = compute_effsize(x, y, paired=False, eftype=effect_size)
        results["effect size"].append(es)
    results = pd.DataFrame(results)
    results["p-values"] = multipletests(results["p-values"].values, method="bonferroni", alpha=0.05)[1]
    return results


def plot_pvals(df: pd.DataFrame,
               title: str,
               a: float = 0.01,
               save: str or None = None):
    df = df.copy()
    x = "-Log(p-value)"
    df[x] = -np.log(df["p-values"])
    fig, ax = plt.subplots(figsize=(8, 8))
    df = df.sort_values("-Log(p-value)")
    sns.barplot(x=df["feature"],
                y=df["-Log(p-value)"],
                ax=ax,
                color="#5a7abf")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.axhline(-np.log(a), ls='--', color="r")
    ax.annotate(f"p={a}", xy=(2, 6))
    ax.set_xlabel("")
    ax.set_title(title)
    plt.gcf().subplots_adjust(bottom=0.5)
    if save is not None:
        fig.savefig(save, facecolor="white", bbox_inches="tight")
    return fig, ax


def fishers(feature: str,
            df: pd.DataFrame,
            endpoint: str):
    contingency_table = pd.crosstab(df[feature],
                                    df[endpoint])
    res = rstats.fisher_test(contingency_table.values)
    return res[0][0]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

