
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

from mldissect import BreakDownExplainer, REGRESSION, CLASSIFICATION
from mldissect.utils import multiply_row


def test_basic(seed):
    boston = load_boston()
    X, y = boston['data'], boston['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=seed
    )

    clf = LassoCV()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    explainer = BreakDownExplainer(
        clf, X_train, boston.feature_names, REGRESSION)
    result = explainer.explain(X_test[0], direction='up')
    columns, values, contribution, direction = result

    expected_columns = [
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

    expected_contributions = np.array(
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

    assert columns.tolist() == expected_columns
    assert np.allclose(contribution, expected_contributions, rtol=1e-05)

    result = explainer.explain(X_test[0], direction='down')
    columns, values, contribution, direction = result

    expected_columns = [
        'LSTAT',
        'RAD',
        'ZN',
        'DIS',
        'PTRATIO',
        'AGE',
        'CRIM',
        'TAX',
        'B',
        'RM',
        'INDUS',
        'NOX',
        'CHAS',
    ]

    expected_contributions = np.array(
        [
            3.68720313e+00,
            -1.09927656e+00,
            -6.14320765e-01,
            -1.07922686e+00,
            1.36518986e+00,
            -5.12708269e-01,
            2.32252898e-01,
            -3.79375268e-01,
            1.19522456e-01,
            4.11346046e-02,
            3.12489740e-02,
            0.00000000e+00,
            -3.55271368e-15,
        ]
    )

    assert columns.tolist() == expected_columns
    assert np.allclose(contribution, expected_contributions, rtol=1e-05)


def test_basic_pandas():
    boston = load_boston()
    X, y = boston['data'], boston['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=0
    )

    columns = list(boston.feature_names)
    df_train = pd.DataFrame(data=X_train, columns=columns)
    df_train_target = pd.DataFrame(data=y_train, columns=['target'])

    df_test = pd.DataFrame(data=X_test, columns=columns)
    df_test_target = pd.DataFrame(data=y_test, columns=['target'])

    clf = LassoCV()
    clf.fit(df_train, df_train_target)
    clf.score(df_test, df_test_target)

    explainer = BreakDownExplainer(
        clf, X_train, boston.feature_names, REGRESSION)
    result = explainer.explain(X_test[0], direction='up')
    assert result
    result = explainer.explain(X_test[0], direction='down')
    assert result


def test_multiply_row():
    row = np.array([[1, 2, 3]])
    data = multiply_row(row, 3)
    expected = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert np.allclose(data, expected)


def test_basic_classification(seed):
    iris = load_iris()
    columns = iris.feature_names
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    explainer = BreakDownExplainer(clf, X_train, columns, CLASSIFICATION)
    result = explainer.explain(X_test[1], direction='up')
    pred = clf.predict_proba(X_test[1:2])
    print(pred)
    assert result
    print(result)

    return
    result = explainer.explain(X_test[0], direction='down')
    assert result
    print(result)
