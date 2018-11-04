from collections import namedtuple



import numpy as np
import pandas as pd
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from mldissect import RegressionExplainer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class ColumnsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns or []

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            return data[self.columns]
        return data[:, :len(self.columns)]


def add_fake_columns(df, n=1, seed=42):
    rng = np.random.RandomState(seed)
    s = rng.normal(size=df.shape[0])
    for i in range(n):
        df['fake_{}'.format(i)] = s
    return df


def shap_explainer(model, data, rows=10):
    #d = shap.kmeans(data.X_test, 1)
    explainer = shap.KernelExplainer(model.predict, data.X_test)
    values = explainer.shap_values(data.X_test.iloc[:rows, :])
    intercept = np.ones((values.shape[0], 1)) * explainer.expected_value
    return np.append(intercept, values, axis=1)


def lime_explanier(model, data, rows=10):
    categorical_features = []
    explainer = LimeTabularExplainer(
        data.X_train.values,
        feature_names=list(data.X_test.columns),
        class_names=['target'],
        categorical_features=categorical_features,
        verbose=True,
        mode=data.meta['objective'])
    results = []

    for instance in data.X_test.values[0:rows]:
        exp = explainer.explain_instance(
            instance, model.predict,
            num_features=data.X_train.shape[0])

        values = [e[1] for e in sorted(exp.local_exp[0], key=lambda v: v[0])]
        results.append(
            [exp.intercept[0]] + values
        )

    return np.array(results)


def breakdown_explainer(model, data, rows=10):
    columns = data.X_test.columns
    explainer = RegressionExplainer(model, data.X_train, columns)
    results = []
    for instance in data.X_test.values[0:rows]:
        result = explainer.explain(instance)
        columns, values, contribution, intercept, order = result
        original_order = np.argsort(order)
        c = contribution[original_order]
        results.append(np.append(np.array([intercept]), c, axis=0))

    return np.array(results)


Data = namedtuple("Data", ['X_train', 'X_test', 'y_train', 'y_test', 'meta'])


def boston_fake_features(num_features=5, seed=42):
    boston = load_boston()
    columns = list(boston.feature_names)
    X = pd.DataFrame(data=boston['data'], columns=columns)
    y = pd.Series(data=boston['target'], name='target')
    X = add_fake_columns(X, n=num_features, seed=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=seed
    )
    meta = {
        'objective': 'regression',
        'categorical_features': [],
        'columns': columns,
        'fake_columns': list(X_train.columns[-num_features:])
    }
    return Data(X_train, X_test, y_train, y_test, meta)


datasets = [
    ('boston', boston_fake_features),
]


def build_rf_regressor(data):
    pipeline = Pipeline(steps=[
        ('selector', ColumnsSelector(data.meta['columns'])),
        ('rf', RandomForestRegressor()),
    ])
    pipeline.fit(data.X_train, data.y_train)
    print(pipeline.score(data.X_test, data.y_test))
    return pipeline


def build_lasso_regressor(data):
    pipeline = Pipeline(steps=[
        ('selector', ColumnsSelector(data.meta['columns'])),
        ('lasso', LassoCV()),
    ])

    pipeline.fit(data.X_train, data.y_train)
    print(pipeline.score(data.X_test, data.y_test))
    return pipeline


def build_svr_regressor(data):
    pipeline = Pipeline(steps=[
        ('selector', ColumnsSelector(data.meta['columns'])),
        ('lasso', SVR(kernel='rbf', gamma=0.1)),
    ])

    pipeline.fit(data.X_train, data.y_train)
    print(pipeline.score(data.X_test, data.y_test))
    return pipeline


def main():
    np.set_printoptions(suppress=True)
    seed = 42
    num_features = 2
    for num_features in [5]:
        data = boston_fake_features(num_features=num_features, seed=seed)
        rf = build_rf_regressor(data)
        rows = -1
        exp1 = lime_explanier(rf, data, rows=rows)
        exp2 = shap_explainer(rf, data, rows=rows)
        exp3 = breakdown_explainer(rf, data, rows=rows)
        import ipdb
        ipdb.set_trace()
        print(exp3)

        fake_columns = data.X_train.columns[-num_features:]
        prefix = 'boston_rf_'

        plot_fake_box(exp1[:, -num_features:], fake_columns, 'lime', prefix)
        plot_fake_box(exp2[:, -num_features:], fake_columns, 'shap', prefix)
        plot_fake_box(exp3[:, -num_features:], fake_columns, 'breakdown', prefix)


def plot_fake_box(data, columns, method, prefix=''):
    sns_plot = sns.boxplot(data=data)
    sns_plot.set(xlabel='fake features', ylabel='values distribution')
    plt.xticks(plt.xticks()[0], columns)
    name = "{}{}_fake_feature_{}".format(prefix, method, len(columns))
    sns_plot.figure.savefig("{}.png".format(name))
    plt.clf()


if __name__ == '__main__':
    main()
