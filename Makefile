PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: all setup build model assets api eval bench perf thinking-bench clean

all: setup build model

setup:
	bash scripts/00_install_deps.sh

build:
	bash scripts/01_build_llama_cpp.sh

model:
	bash scripts/02_download_model.sh

assets:
	bash scripts/05_fetch_assets.sh

api:
	bash scripts/03_run_api.sh

eval:
	bash scripts/04_eval.sh

bench:
	$(PYTHON) -m jev.cli bench --targets 32,128,512,1024,2048

perf:
	bash scripts/07_bench_perf.sh

thinking-bench:
	bash scripts/06_thinking_bench.sh

clean:
	rm -rf results logs/*.log
