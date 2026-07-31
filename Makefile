.PHONY: install dev lint test app docker-build

install:
	python -m pip install -e ".[all]"

dev:
	python -m pip install -e ".[all,dev]"

lint:
	python -m ruff check .
	python -m ruff format --check .

test:
	python -m pytest

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t rlhf-chatbot .
