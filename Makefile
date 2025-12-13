.PHONY: install format lint test init

PROJECT_NAME = heart-failure-prediction
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

help:
	@grep -E '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'

install: ## install dependencies
	poetry install

format: ## auto format files
	ruff format .
	ruff check . --fix

lint: ## check formating
	ruff check .
	ruff format --check

test: ## run tests
	poetry run pytest tests/

init: ## create poetry environment
	poetry env use $(PYTHON_VERSION)
	@echo ">>> Poetry environment created."
