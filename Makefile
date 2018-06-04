SHELL := /bin/bash

MODULES=starfish examples

all:	lint mypy test

lint:
	flake8 $(MODULES)

test:
	pytest -v -n 8 --junitxml=test-reports/junit.xml --cov=starfish

mypy:
	mypy --ignore-missing-imports $(MODULES)

include notebooks/subdir.mk
