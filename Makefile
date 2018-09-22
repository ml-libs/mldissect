# Some simple testing tasks (sorry, UNIX only).

FLAGS=

flake: checkrst bandit pyroma
	flake8 mldissect tests examples setup.py demos

test: flake
	py.test -s -v $(FLAGS) ./tests/

vtest:
	py.test -s -vv $(FLAGS) ./tests/

checkrst:
	python setup.py check --restructuredtext

bandit:
	bandit -r ./mldissect

pyroma:
	pyroma -d .

mypy:
	mypy mldissect --ignore-missing-imports --disallow-untyped-calls --no-site-packages --strict

testloop:
	while true ; do \
        py.test -s -v $(FLAGS) ./tests/ ; \
    done

cov cover coverage: flake checkrst
	py.test -s -v --cov-report term --cov-report html --cov mldissect ./tests
	@echo "open file://`pwd`/htmlcov/index.html"

cov_only: flake
	py.test -s -v --cov-report term --cov-report html --cov mldissect ./tests
	@echo "open file://`pwd`/htmlcov/index.html"

ci: flake mypy
	py.test -s -v --cov-report term --cov-report html --cov mldissect ./tests
	@echo "open file://`pwd`/htmlcov/index.html"

clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -f `find . -type f -name '@*' `
	rm -f `find . -type f -name '#*#' `
	rm -f `find . -type f -name '*.orig' `
	rm -f `find . -type f -name '*.rej' `
	rm -f .coverage
	rm -rf coverage
	rm -rf build
	rm -rf htmlcov
	rm -rf dist

doc:
	make -C docs html
	@echo "open file://`pwd`/docs/_build/html/index.html"

.PHONY: all flake test vtest cov clean doc ci
