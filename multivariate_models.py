from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn import metrics
from matplotlib.pyplot import Axes
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np


def inspect_class_imbalance(df: pd.DataFrame):
    """
    Inspect a single dataframe for bias and class imbalance e.g. complete case
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    sns.barplot(x="index",
                y="gender",
                ax=axes[0, 0],
                data=pd.DataFrame(df.gender.value_counts()).reset_index())
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_xlabel("Gender")
    axes[0, 0].set_xticklabels(["Male", "Female"])
    sns.distplot(df.age, ax=axes[0, 1], bins=20)
    axes[0, 1].set_xlabel("Age (Yrs)")
    axes[0, 1].set_xlim((df.age.min(), df.age.max()))
    sns.barplot(x="index",
                y="death",
                ax=axes[1, 0],
                data=pd.DataFrame(df.death.value_counts()).reset_index())
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_xlabel("28 Day Mortality")
    sns.barplot(x="index",
                y="composite",
                ax=axes[1, 1],
                data=pd.DataFrame(df.composite.value_counts()).reset_index())
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_xlabel("28 Day Mortality/ICU Admission")
    fig.tight_layout()
    return fig, axes


class SMLogitWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self._model = sm.Logit
        self.results = None

    def fit(self,
            X: np.array,
            y: np.array):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self._model = self._model(y, X)
        self.results = self._model.fit(disp=0)
        return self

    def predict(self, X: np.array, threshold: float = 0.5):
        if self.fit_intercept:
            X = sm.add_constant(X)
        proba = self.results.predict(X)
        return np.array(list(map(lambda x: 1 if x >= threshold else 0, proba)))

    def predict_proba(self, X: np.array):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results.predict(X)

    def get_params(self, deep=False):
        return {'fit_intercept': self.fit_intercept}

    def summary(self, version: int = 1):
        assert self.results is not None, "Must call fit prior to calling summary"
        if version == 1:
            return self.results.summary().tables
        return self.results.summary2().tables


def fit_and_predict(X_train, X_test, y_train, y_test):
    logreg = SMLogitWrapper()
    results = dict()
    results["Model"] = logreg.fit(X_train, y_train)
    results["Train Prediction"] = results["Model"].predict(X_train)
    results["Test Prediction"] = results["Model"].predict(X_test)
    results["Train true"] = y_train
    results["Train Probs"] = results["Model"].predict_proba(X_train)
    results["Test Probs"] = results["Model"].predict_proba(X_test)
    results["Test true"] = y_test
    # Training performance
    y_pred = results["Train Prediction"]
    y_score = results["Train Probs"]
    train_performance = {"F1 Score (Weighted)": metrics.f1_score(y_true=y_train,
                                                                 y_pred=y_pred,
                                                                 average="weighted"),
                         "Balanced Accuracy": metrics.balanced_accuracy_score(y_true=y_train,
                                                                              y_pred=y_pred),
                         "AUC Score": metrics.roc_auc_score(y_true=y_train,
                                                            y_score=y_score)}
    results["Training performance"] = train_performance
    # Testing performance
    y_pred = results["Test Prediction"]
    y_score = results["Test Probs"]
    test_performance = {"F1 Score (Weighted)": metrics.f1_score(y_true=y_test,
                                                                y_pred=y_pred,
                                                                average="weighted"),
                        "Balanced Accuracy": metrics.balanced_accuracy_score(y_true=y_test,
                                                                             y_pred=y_pred),
                        "AUC Score": metrics.roc_auc_score(y_true=y_test,
                                                           y_score=y_score)}
    results["Testing performance"] = test_performance
    return results


def plot_learning_curve(data: pd.DataFrame,
                        features: list,
                        label: str,
                        ax: Axes,
                        n_folds: int = 5,
                        n_jobs: int = -1,
                        scoring: str = "balanced_accuracy",
                        train_sizes: np.array = np.linspace(0.3, 1.0, 10)):
    logreg = SMLogitWrapper()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    train_sizes, train_scores, test_scores = \
        learning_curve(logreg,
                       data[features],
                       data[label],
                       cv=skf,
                       n_jobs=n_jobs,
                       scoring=scoring,
                       train_sizes=train_sizes,
                       return_times=False)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1,
                    color="g")
    ax.legend(loc="best")
    ax.set_xlabel("Training examples")
    ax.set_ylabel(scoring.replace("_", "").title())
    ax.set_ylim(0, 1)
    return ax


def roc_curve(y_true: list,
              y_score: list,
              ax: Axes,
              label: str,
              colour: str,
              plot_ci: bool = False):
    roc_metrics = [metrics.roc_curve(x.reshape(-1, 1), y) for x, y in zip(y_true, y_score)]
    fpr = [x[0] for x in roc_metrics]
    tpr = [x[1] for x in roc_metrics]
    base_fpr = np.linspace(0, 1, 101)
    tprs = list()
    for fpr_, tpr_ in zip(fpr, tpr):
        tpr = np.interp(base_fpr, fpr_, tpr_)
        tpr[0] = 0.0
        tprs.append(tpr)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    ax.plot(base_fpr, mean_tprs, c=colour, label=label)
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    if plot_ci:
        ax.fill_between(base_fpr, tprs_lower, tprs_upper, color=colour, alpha=0.3)
    return ax


def calc_spec(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn+fp)


def calc_npv(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn+fn)


def line_plot_ci(x, y, ax):
    avg = np.mean(y, axis=1)
    std = np.std(y, axis=1)
    upper = avg + std
    lower = avg - std
    ax.plot(x, avg)
    ax.fill_between(x, lower, upper, alpha=0.1)
    return ax


def threshold_performance_plot(y_score: list,
                               y_true: list,
                               figsize: tuple=(10, 15),
                               thresholds: np.array or None = None):
    sens = list()
    spec = list()
    ppv = list()
    npv = list()
    thresholds = thresholds or np.linspace(0.01, 1, 100)
    for threshold in thresholds:
        y_pred = [list(map(lambda x: 1 if x >= threshold else 0, ys)) for ys in y_score]
        sens.append([metrics.recall_score(yt, yp) for yt, yp in zip(y_true, y_pred)])
        ppv.append([metrics.precision_score(yt, yp) for yt, yp in zip(y_true, y_pred)])
        spec.append([calc_spec(yt, yp) for yt, yp in zip(y_true, y_pred)])
        npv.append([calc_npv(yt, yp) for yt, yp in zip(y_true, y_pred)])
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True, sharey=False)
    axes[0] = line_plot_ci(thresholds, sens, axes[0])
    axes[0].set_ylabel("Sensitivity")
    axes[1] = line_plot_ci(thresholds, spec, axes[1])
    axes[1].set_ylabel("Specificity")
    axes[2] = line_plot_ci(thresholds, ppv, axes[2])
    axes[2].set_ylabel("PPV")
    axes[3] = line_plot_ci(thresholds, npv, axes[3])
    axes[3].set_ylabel("NPV")
    axes[4].set_ylabel("Count")
    axes[4].set_xlabel("Threshold")
    for i in range(4):
        axes[i].set_ylim(0., 1.)
    sns.distplot(np.array(y_score).flatten(), kde=False)
    for a in axes:
        a.set_xlim(0, 1)
    return fig, axes


def box_swarm_models(models: dict,
                     metric: str,
                     ax,
                     box_color: str = "grey",
                     box_alpha: float = .2,
                     swarm_lw: int = 2,
                     swarm_colour: str = "black",
                     swarm_fill: str = "white",
                     swarm_size: int = 30,
                     ticks: int = 5,
                     ylim = None):
    plt.locator_params(nbins=ticks)
    df = dict()
    for x in models.keys():
        df[x] = [i["Testing performance"][metric] for i in models[x]]
    df = pd.DataFrame(df).melt(var_name="Model", value_name=metric)
    sns.boxplot(x="Model", y=metric, data=df, color=box_color, showfliers=False, ax=ax,
                boxprops=dict(alpha=box_alpha))
    sns.swarmplot(x="Model", y=metric, data=df,
                  color=swarm_fill, linewidth=swarm_lw, edgecolor=swarm_colour, ax=ax, s=swarm_size)
    xticklabels = ["Basic model",
                   "+ Troponin",
                   "+ LDH",
                   "+ Ferritin",
                   "+ PCT",
                   "+ D-dimer",
                   "+ Extended panel"]
    ax.set_xticklabels(xticklabels, rotation=90)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    return ax
