from abc import ABC, abstractmethod

import numpy as np
from typing import NamedTuple, List

from .utils import normalize_array, multiply_row, to_matrix


UP = 'up'
DOWN = 'down'


class ExplanationData(NamedTuple):
    columns: List[str]
    values: List[float]
    contribution: List[float]
    intercept: float


class BaseExplainer(ABC):

    def __init__(self, clf, data, columns):
        self._clf = clf
        self._data = to_matrix(data)
        self._columns = columns

    def explain(self, instance, direction=UP, baseline=0):
        instance = to_matrix(instance)
        data = np.copy(self._data)
        instance = normalize_array(instance)

        if direction == UP:
            exp = self._explain_bottom_up(instance, data)
        else:
            exp = self._explain_top_down(instance, data)
        return exp

    @abstractmethod
    def _mean_predict(self, data):
        pass  # pragma: no cover

    @abstractmethod
    def _most_important(self, yhats_diff):
        pass  # pragma: no cover

    @abstractmethod
    def _init_ydiff(self, yhats_diff, default=0):
        pass  # pragma: no cover

    @abstractmethod
    def _least_important(self, yhats_diff):
        pass  # pragma: no cover

    def _format_result(self, instance, important_variables, mean_predictions,
                       baseline, position=0):
        var_names = np.array(self._columns)[important_variables]
        var_values = instance[0, important_variables]
        means = np.insert(mean_predictions, position, baseline, axis=0)
        contributions = np.diff(np.array(means), axis=0).reshape(1, -1)
        return ExplanationData(var_names, var_values, contributions, baseline)

    def _explain_bottom_up(self, instance, data):
        num_rows, num_features = data.shape
        new_data = multiply_row(instance, num_rows)

        baseline = self._mean_predict(data).reshape(1, -1)

        important_variables = []
        mean_predictions = np.zeros((num_features, baseline.shape[1]))
        relaxed_features = set()

        for i in range(num_features):
            yhats_mean = np.zeros((num_features, baseline.shape[1]))
            yhats_diff = np.zeros((num_features, baseline.shape[1]))
            self._init_ydiff(yhats_diff)

            for feature_idx in range(num_features):
                if feature_idx in relaxed_features:
                    continue

                tmp_data = np.copy(data)
                tmp_data[:, feature_idx] = new_data[:, feature_idx]
                yhats_mean[feature_idx] = self._mean_predict(tmp_data)
                yhats_diff[feature_idx] = abs(
                    baseline - yhats_mean[feature_idx]
                )

            most_important_idx = self._most_important(yhats_diff)
            important_variables.append(most_important_idx)
            mean_predictions[i] = yhats_mean[most_important_idx]
            data[:, most_important_idx] = new_data[:, most_important_idx]
            relaxed_features.add(most_important_idx)

        return self._format_result(
            instance, important_variables, mean_predictions, baseline)

    def _explain_top_down(self, instance, data):
        num_rows, num_features = data.shape
        new_data = multiply_row(instance, num_rows)

        baseline = self._mean_predict(instance).reshape(1, -1)

        important_variables = []
        mean_predictions = np.zeros((num_features, baseline.shape[1]))
        relaxed_features = set()

        for i in range(num_features):
            yhats_mean = np.zeros((num_features, baseline.shape[1]))
            yhats_diff = np.zeros((num_features, baseline.shape[1]))
            self._init_ydiff(yhats_diff, default=np.PINF)  # np.PINF

            for feature_idx in range(num_features):
                if feature_idx in relaxed_features:
                    continue

                tmp_data = np.copy(new_data)
                tmp_data[:, feature_idx] = data[:, feature_idx]
                yhats_mean[feature_idx] = self._mean_predict(tmp_data)
                yhats_diff[feature_idx] = abs(
                    baseline - yhats_mean[feature_idx]
                )

            least_important_idx = self._least_important(yhats_diff)
            important_variables.append(least_important_idx)

            mean_predictions[i] = yhats_mean[least_important_idx]
            new_data[:, least_important_idx] = data[:, least_important_idx]
            relaxed_features.add(least_important_idx)

        important_variables.reverse()
        mean_predictions = mean_predictions[::-1]
        return self._format_result(
            instance, important_variables, mean_predictions, baseline,
            position=-1)


class RegressionExplainer(BaseExplainer):

    def _mean_predict(self, data):
        return self._clf.predict(data).mean(axis=0)

    def _most_important(self, yhats_diff):
        most_important_idx = np.argmax(yhats_diff)
        return most_important_idx

    def _least_important(self, yhats_diff):
        most_important_idx = np.argmin(yhats_diff)
        return most_important_idx

    def _init_ydiff(self, yhats_diff, default=np.NINF):
        yhats_diff.fill(default)
        return yhats_diff


class ClassificationExplainer(BaseExplainer):

    def _mean_predict(self, data):
        return self._clf.predict_proba(data).mean(axis=0)

    def _most_important(self, yhats_diff):
        most_important_idx = np.argmax(np.linalg.norm(yhats_diff, axis=1))
        return most_important_idx

    def _init_ydiff(self, yhats_diff, default=None):
        return yhats_diff

    def _least_important(self, yhats_diff):
        most_important_idx = np.argmin(np.linalg.norm(yhats_diff, axis=1))
        return most_important_idx
