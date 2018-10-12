import numpy as np
import pandas as pd
import shap
from collections import namedtuple


from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV

from mldissect import ClassificationExplainer, RegressionExplainer
from mldissect.explanation import Explanation

from sklearn.base import BaseEstimator, TransformerMixin
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


def shap_explainer(model, data):
    explainer = shap.KernelExplainer(model.predict, data.X_test.iloc[:10, :])
    values = explainer.shap_values(data.X_test)
    return values


def lime_explanier(model, data):
    categorical_features = []
    explainer = LimeTabularExplainer(
        data.X_train.values,
        feature_names=list(data.X_test.columns),
        class_names=['target'],
        categorical_features=categorical_features,
        verbose=True,
        mode=data.meta['objective'])
    results = []

    for instance in data.X_test.values:
        exp = explainer.explain_instance(
            instance, model.predict,
            num_features=data.X_train.shape[0])

        values = [e[1] for e in sorted(exp.local_exp[0], key=lambda v: v[0])]
        results.append([exp.intercept] + values)
    return np.array(results)



def breakdown_explainer(model, X_train, X_text):
    pass


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
    seed = 42
    num_features = 5
    data = boston_fake_features(num_features=num_features, seed=seed)
    rf = build_rf_regressor(data)

    exp2 = shap_explainer(rf, data)
    exp1 = lime_explanier(rf, data)

    print(exp1)
    print(exp2)
    print(exp1 - exp2)


if __name__ == '__main__':
    main()
