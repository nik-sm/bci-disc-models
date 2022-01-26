# shell used by make
SHELL=/bin/bash

# global variables
PYTHON_VERSION = 3.8
SRC_NAME = src
VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python
EGG = $(SRC_NAME).egg-info


all: setup lint

lint: setup
	$(VENV_NAME)/bin/pre-commit run --all-files

clean:
	find . -type d | grep "__pycache__" | xargs -n1 rm -rf
	find . -type f -path "*/*.pyc" -delete
	find . -type f -path "*/*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .pytype

# Project setup
.PHONY: setup
setup:  $(EGG) 

$(EGG): $(PYTHON) setup.py requirements.txt
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PYTHON) -m pip install -Ue .
	$(VENV_NAME)/bin/pre-commit install

$(PYTHON):
	python$(PYTHON_VERSION) -m pip install virtualenv
	python$(PYTHON_VERSION) -m virtualenv $(VENV_NAME)

# Careful!
# useful e.g. after modifying __init__.py files
destroy-setup: confirm-destroy
	rm -rf $(EGG)
	rm -rf $(VENV_NAME)

confirm-destroy:
	@( read -p "Destroy env? [y/n]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )
