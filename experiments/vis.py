"""Visualisation routines and classes."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, r2_score


def iterate_scores(y_true, y_predict, scoring):
    scores = [scoring(y, ey) for y, ey in zip(y_true, y_predict)]
    return scores


def threshold_balanced_accuracy(
    y_true: np.ndarray, p_pred: np.ndarray, threshold: float = 0.5
) -> float:
    y_pred = (p_pred >= threshold).astype(int)
    return balanced_accuracy_score(y_true=y_true, y_pred=y_pred)


class _ScorePlot:
    def __init__(self):
        self.reset()

    def update(self, y_train, y_test, y_train_predict, y_test_predict):
        if len(y_test) != len(y_test_predict):
            raise ValueError("Test targets and predictions inconsistent.")
        if len(y_train) != len(y_train_predict):
            raise ValueError("Train targets and predictions inconsistent.")
        self.y_test.append(y_test)
        self.ey_test.append(y_test_predict)
        self.y_train.append(y_train)
        self.ey_train.append(y_train_predict)

    def reset(self):
        self.y_test = []
        self.ey_test = []
        self.y_train = []
        self.ey_train = []

    def _join_results(self):
        yr = np.concatenate(self.y_train)
        eyr = np.concatenate(self.ey_train)
        ys = np.concatenate(self.y_test)
        eys = np.concatenate(self.ey_test)
        return yr, eyr, ys, eys

    def _get_scores(self, scoring):
        train_scores = iterate_scores(self.y_train, self.ey_train, scoring)
        test_scores = iterate_scores(self.y_test, self.ey_test, scoring)
        return train_scores, test_scores


class CPplot(_ScorePlot):
    """Class probability plot."""

    def plot(self, scoring=threshold_balanced_accuracy):
        yr, eyr, ys, eys = self._join_results()
        train_scores, test_scores = self._get_scores(scoring)
        mean_tr = np.mean(train_scores)
        mean_ts = np.mean(test_scores)

        scorename = getattr(scoring, "__name__", "score")
        fig, axs = plt.subplots(
            nrows=2, ncols=1, dpi=150, figsize=(8, 5), sharex=True
        )

        for y, ey, ax, nm, s in zip(
            (yr, ys), (eyr, eys), axs, ("Train", "Test"), (mean_tr, mean_ts)
        ):
            df = pd.DataFrame(dict(Labels=y, Predictions=ey))
            sns.kdeplot(
                ax=ax,
                data=df,
                x="Predictions",
                hue="Labels",
                fill=True,
                common_norm=False,
                palette="crest",
                alpha=0.5,
                linewidth=0,
            )
            ax.set_title(f"{nm}: {scorename}: {s:.3f}")
        plt.tight_layout()
        return fig


class YYplot(_ScorePlot):
    """Make a YY-plot of training and test predictions, with scores."""

    def plot(self, scoring=r2_score, axis_name="fitness"):
        yr, eyr, ys, eys = self._join_results()
        train_scores, test_scores = self._get_scores(scoring)
        mean_tr = np.mean(train_scores)
        mean_ts = np.mean(test_scores)

        ymin = min(yr.min(), ys.min())
        ymax = max(yr.max(), ys.max())

        scorename = getattr(scoring, "__name__", "score")
        plt.figure(dpi=150, figsize=(6, 5))
        plt.plot(yr, eyr, ".", color="blue", label="train")
        plt.plot(ys, eys, ".", color="orange", label="test")
        plt.plot([ymin, ymax], [ymin, ymax], "k--")
        plt.grid()
        plt.xlabel(f"True {axis_name}")
        plt.ylabel(f"Predicted {axis_name}")
        plt.title(
            f"Y-Y plot, train {scorename} = {mean_tr:.3f}, "
            f"test {scorename} = {mean_ts:.3f}"
        )
        plt.legend()
        plt.show()
