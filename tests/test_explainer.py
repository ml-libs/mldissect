import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

from mldissect import ClassificationExplainer, RegressionExplainer
from mldissect.explanation import Explanation


expected_columns_up = [
    'LSTAT',
    'PTRATIO',
    'CRIM',
    'B',
    'RM',
    'INDUS',
    'CHAS',
    'NOX',
    'TAX',
    'AGE',
    'ZN',
    'DIS',
    'RAD',
]

expected_contributions_up = np.array(
    [
        3.68720313,
        1.36518986,
        0.2322529,
        0.11952246,
        0.0411346,
        0.03124897,
        0.,
        0.,
        -0.37937527,
        -0.51270827,
        -0.61432077,
        -1.07922686,
        -1.09927656,
    ]
)


@pytest.mark.parametrize('exp_columns, exp_contributions', [
    (expected_columns_up, expected_contributions_up),
])
def test_regression(seed, exp_columns, exp_contributions):
    boston = load_boston()
    columns = list(boston.feature_names)
    X, y = boston['data'], boston['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=seed
    )

    clf = LassoCV()
    clf.fit(X_train, y_train)

    observation = X_test[0]
    explainer = RegressionExplainer(clf, X_train, columns)
    result = explainer.explain(observation)
    columns, values, contribution, intercept, _ = result
    assert columns.tolist() == exp_columns
    assert np.allclose(contribution, exp_contributions, rtol=1e-05)
    prediction = clf.predict(X_test[0:1])[0]
    explained = np.sum(contribution) + intercept
    assert pytest.approx(prediction) == explained

    explanation = Explanation(result)
    explanation.print()


@pytest.mark.parametrize('exp_columns, exp_contributions', [
    (expected_columns_up, expected_contributions_up),
])
def test_regression_pandas(seed, exp_columns, exp_contributions):
    boston = load_boston()
    X, y = boston['data'], boston['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=0
    )

    columns = list(boston.feature_names)
    df_train = pd.DataFrame(data=X_train, columns=columns)
    df_train_target = pd.DataFrame(data=y_train, columns=['target'])

    df_test = pd.DataFrame(data=X_test, columns=columns)

    clf = LassoCV()
    clf.fit(df_train, df_train_target)
    observation = df_test.iloc[0]
    explainer = RegressionExplainer(clf, df_train, columns)
    result = explainer.explain(observation)
    columns, values, contribution, intercept, _ = result
    assert columns.tolist() == exp_columns
    assert np.allclose(contribution, exp_contributions, rtol=1e-05)


def test_classification(seed):
    iris = load_iris()
    columns = iris.feature_names
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed
    )
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    explainer = ClassificationExplainer(clf, X_train, columns)
    result = explainer.explain(X_test[0])
    columns, values, contribution, intercept, _ = result
    prediction = clf.predict_proba(X_test[0:1])
    explained = np.sum(contribution, axis=0) + intercept
    assert np.allclose(prediction, explained)

    explanation = Explanation(result)
    explanation.print()
