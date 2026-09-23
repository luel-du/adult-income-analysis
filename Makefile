.PHONY: all install lint format format-check test check run docker-build docker-run docker-test clean

IMAGE_NAME := adult-income-analysis

# Install, lint, and test
all: install lint format-check test

# Install dependencies
install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

# Lint the code with ruff
lint:
	python -m ruff check src tests

# Auto-format the code with ruff
format:
	python -m ruff format src tests

# Verify formatting without changing files
format-check:
	python -m ruff format --check src tests

# Run tests
test:
	python -m pytest -q --cov=src --cov-report=term-missing

# Run all local checks: lint, formatting, and tests
check: lint format-check test

# Run the application
run:
	python src/main.py

# Build the Docker image
docker-build:
	docker build -t $(IMAGE_NAME) .

# Run the application inside Docker
docker-run:
	docker run -it --rm $(IMAGE_NAME)

# Run the test suite inside Docker
docker-test:
	docker run --rm $(IMAGE_NAME) python -m pytest -q

# Clean generated files
clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache