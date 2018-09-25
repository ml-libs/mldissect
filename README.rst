mldissect
=========
.. image:: https://travis-ci.com/ml-libs/mldissect.svg?branch=master
    :target: https://travis-ci.com/ml-libs/mldissect
.. image:: https://codecov.io/gh/ml-libs/mldissect/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/ml-libs/mldissect 
.. image:: https://api.codeclimate.com/v1/badges/bc29bc214f39b54ef30a/maintainability
   :target: https://codeclimate.com/github/ml-libs/mldissect/maintainability
   :alt: Maintainability


**mldissect** is model agnostic predictions explainer, library can show
contribution of each feature of your prediction value.

Features
========
* Supports predictions explanations for classification and regression
* Easy to use API.
* Works with ``pandas`` and ``numpy``


Installation
------------
Installation process is simple, just::

    $ pip install mldissect

Basic Usage
===========

.. code:: python

    # lets train a model
    boston = load_boston()
    columns = list(boston.feature_names)
    X, y = boston['data'], boston['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=seed
    )

    clf = LassoCV()
    clf.fit(X_train, y_train)

    # select first observation in test split
    observation = X_test[0]
    # RegressionExplainer uses training data or sample of training data
    # for large dataset to figure out contributions of each feature
    explainer = RegressionExplainer(clf, X_train, columns)
    result = explainer.explain(observation)
    # print/visualize explanation
    explanation = Explanation(result)
    explanation.print()


result::

    +----------+---------+--------------------+
    | Feature  | Value   | Contribution       |
    +----------+---------+--------------------+
    | baseline | -       | 22.611881188118804 |
    | LSTAT    | 7.34    | 3.6872             |
    | PTRATIO  | 16.9    | 1.3652             |
    | CRIM     | 0.06724 | 0.2323             |
    | B        | 375.21  | 0.1195             |
    | RM       | 6.333   | 0.0411             |
    | INDUS    | 3.24    | 0.0312             |
    | CHAS     | 0.0     | 0.0                |
    | NOX      | 0.46    | 0.0                |
    | TAX      | 430.0   | -0.3794            |
    | AGE      | 17.2    | -0.5127            |
    | ZN       | 0.0     | -0.6143            |
    | DIS      | 5.2146  | -1.0792            |
    | RAD      | 4.0     | -1.0993            |
    +----------+---------+--------------------+


Algorithm
=========
Algorithm is based on ideas describe in paper *"Explanations of model predictions
with live and breakDown packages"* https://arxiv.org/abs/1804.01955


Difference with pyBreakDown
===========================
``pyBreakDown`` is similar project, but there is key differences:

* `mldissect` is maintained
* Has tests and good code coverage.
* Classification is working properly.
* Multi class support.
* Top down approach is not implemented.
* Friendly license.


Requirements
------------

* Python_ 3.6+
* numpy_

.. _Python: https://www.python.org
.. _numpy: http://www.numpy.org/
