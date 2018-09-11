import numpy as np

from .utils import normalize_array, multiply_row, _get_means_from_yhats


UP = 'up'
DOWN = 'down'

CLASSIFICATION = 'classification'
REGRESSION = 'regression'


class BreakDownExplainer:

    def __init__(self, clf, data, columns, objective):
        self._clf = clf
        self._data = data
        self._columns = columns
        self._objective = objective

    def explain(self, instance, direction=UP, baseline=0):
        data = np.copy(self._data)
        instance = normalize_array(instance)

        if direction == UP:
            exp = self._explain_bottom_up(instance, data)
        else:
            exp = self._explain_top_down(instance, data)
        return exp

    def _predict(self, data):
        if self._objective == REGRESSION:
            result = self._clf.predict(data)
        else:
            result = self._clf.predict_proba(data)
        return result

    def _explain_bottom_up(self, instance, data):
        num_rows, num_features = data.shape
        new_data = multiply_row(instance, num_rows)
        mean_prediction = np.mean(self._predict(data), axis=0)
        important_variables = []
        important_yhats = [None] * num_features
        relaxed_features = set()

        for i in range(num_features):
            yhats = {}
            yhats_diff = np.zeros((num_features, len(mean_prediction)))
            if self._objective == REGRESSION:
                yhats_diff.fill(np.NINF)

            for feature_idx in range(num_features):
                if feature_idx in relaxed_features:
                    continue

                tmp_data = np.copy(data)
                tmp_data[:, feature_idx] = new_data[:, feature_idx]
                yhats[feature_idx] = self._predict(tmp_data)
                yhats_diff[feature_idx] = abs(
                    mean_prediction - np.mean(yhats[feature_idx], axis=0)
                )

            most_important_idx = np.argmax(np.linalg.norm(yhats_diff, axis=1))

            important_variables.append(most_important_idx)
            important_yhats[i] = yhats[most_important_idx]
            data[:, most_important_idx] = new_data[:, most_important_idx]
            relaxed_features.add(most_important_idx)

        var_names = np.array(self._columns)[important_variables]
        var_values = instance[0, important_variables]
        means = _get_means_from_yhats(important_yhats)
        means.appendleft(mean_prediction)
        contributions = np.diff(np.array(means), axis=0)
        print(mean_prediction)
        return (var_names, var_values, contributions, mean_prediction, UP)

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
        return (var_names, var_values, contributions, mean_prediction, DOWN)
