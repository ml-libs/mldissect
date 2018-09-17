import numpy as np
from typing import NamedTuple, List

from .utils import normalize_array, multiply_row, _get_means_from_yhats


UP = 'up'
DOWN = 'down'


class Explanation(NamedTuple):
    columns: List[str]
    values: List[float]
    contribution: List[float]
    intercept: float


class BaseExplainer:

    def __init__(self, clf, data, columns):
        self._clf = clf
        self._data = data
        self._columns = columns

    def explain(self, instance, direction=UP, baseline=0):
        data = np.copy(self._data)
        instance = normalize_array(instance)

        if direction == UP:
            exp = self._explain_bottom_up(instance, data)
        else:
            exp = self._explain_top_down(instance, data)
        return exp

    def _mean_predict(self, data):
        pass

    def _most_important(self, yhats_diff):
        pass

    def _init_ydiff(self, yhats_diff):
        pass

    def _format_result(self, instance, important_variables, mean_predictions,
                       baseline):
        var_names = np.array(self._columns)[important_variables]
        var_values = instance[0, important_variables]
        means = np.insert(mean_predictions, 0, baseline, axis=0)
        contributions = np.diff(np.array(means), axis=0).reshape(1, -1)
        return Explanation(var_names, var_values, contributions, baseline)

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


class RegressionExplainer(BaseExplainer):

    def _mean_predict(self, data):
        return self._clf.predict(data).mean(axis=0)

    def _most_important(self, yhats_diff):
        most_important_idx = np.argmax(yhats_diff)
        return most_important_idx

    def _init_ydiff(self, yhats_diff):
        yhats_diff.fill(np.NINF)
        return yhats_diff

    def _explain_top_down(self, observation, data):
        num_rows, num_features = data.shape
        new_data = multiply_row(observation, num_rows)

        mean_prediction = self._clf.predict(observation)

        open_variables = list(range(0, num_features))
        important_variables = []
        important_yhats = [None] * num_features

        for i in range(0, data.shape[1]):
            yhats = {}
            yhats_diff = np.repeat(np.PINF, num_features)

            for variable in open_variables:
                tmp_data = np.copy(new_data)
                tmp_data[:, variable] = data[:, variable]
                yhats[variable] = self._clf.predict(tmp_data)
                yhats_diff[variable] = abs(
                    mean_prediction - np.mean(yhats[variable])
                )

            amin = np.argmin(yhats_diff)
            important_variables.append(amin)
            important_yhats[i] = yhats[amin]
            new_data[:, amin] = data[:, amin]
            open_variables.remove(amin)

        important_variables.reverse()
        var_names = np.array(self._columns)[important_variables]
        var_values = observation[0, important_variables]

        means = _get_means_from_yhats(important_yhats)
        means.appendleft(mean_prediction[0])
        means.reverse()
        contributions = np.diff(means)
        return Explanation(
            var_names, var_values, contributions, mean_prediction)


class ClassificationExplainer(BaseExplainer):

    def _mean_predict(self, data):
        return self._clf.predict_proba(data).mean(axis=0)

    def _most_important(self, yhats_diff):
        most_important_idx = np.argmax(np.linalg.norm(yhats_diff, axis=1))
        return most_important_idx

    def _init_ydiff(self, yhats_diff):
        return yhats_diff

    def _explain_top_down(self, observation, data):
        num_rows, num_features = data.shape
        new_data = multiply_row(observation, num_rows)

        mean_prediction = self._clf.predict(observation)

        open_variables = list(range(0, num_features))
        important_variables = []
        important_yhats = [None] * num_features

        for i in range(0, data.shape[1]):
            yhats = {}
            yhats_diff = np.repeat(np.PINF, num_features)

            for variable in open_variables:
                tmp_data = np.copy(new_data)
                tmp_data[:, variable] = data[:, variable]
                yhats[variable] = self._clf.predict(tmp_data)
                yhats_diff[variable] = abs(
                    mean_prediction - np.mean(yhats[variable])
                )

            amin = np.argmin(yhats_diff)
            important_variables.append(amin)
            important_yhats[i] = yhats[amin]
            new_data[:, amin] = data[:, amin]
            open_variables.remove(amin)

        important_variables.reverse()
        var_names = np.array(self._columns)[important_variables]
        var_values = observation[0, important_variables]

        means = _get_means_from_yhats(important_yhats)
        means.appendleft(mean_prediction[0])
        means.reverse()
        contributions = np.diff(means)
        return Explanation(
            var_names, var_values, contributions, mean_prediction)
