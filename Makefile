PY ?= python

.PHONY: setup data train api web test test-web demo bench docker

setup:            ## create venv, install backend + frontend deps
	$(PY) -m venv .venv
	.venv/bin/pip install -r backend/requirements.txt || .venv/Scripts/pip install -r backend/requirements.txt
	cd frontend && npm install

data:             ## regenerate the demo dataset deterministically (seed 42) and train all models
	$(PY) scripts/generate_demo_data.py --name demo_obs --size medium --scenario observational --seed 42 --train

api:              ## start the API on :8000
	cd backend && $(PY) -m uvicorn app.main:app --port 8000

web:              ## start the dashboard on :5173 (proxies /api to :8000; override with VITE_API_TARGET)
	cd frontend && npm run dev

test:             ## backend test-suite
	cd backend && $(PY) -m pytest

test-web:
	cd frontend && npm test && npm run build

demo:             ## scenario walk-through printed to the terminal
	$(PY) scripts/run_demo_pipeline.py --n 20000

bench:
	$(PY) scripts/benchmark.py --sizes 8000 40000

docker:           ## not verified in the build environment (no Docker available there)
	docker compose up --build
