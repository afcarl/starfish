SHELL := /bin/bash

MODULES=starfish tests examples

all:	lint mypy pytest

lint:
	flake8 $(MODULES)

pytest:
	pytest -v -n 8 --junitxml=test-reports/junit.xml --cov=starfish

mypy:
	mypy --ignore-missing-imports $(MODULES)

test_srcs := $(wildcard tests/test_*.py)

test: STARFISH_COVERAGE := 1
test: $(test_srcs) lint
	coverage combine
	rm -f .coverage.*

$(test_srcs): %.py :
	if [ "$(STARFISH_COVERAGE)" == 1 ]; then \
		STARFISH_COVERAGE=1 coverage run -p --source=starfish -m unittest $(subst /,.,$*); \
	else \
		python -m unittest $(subst /,.,$*); \
	fi

.PHONY : $(test_srcs)

include notebooks/subdir.mk
