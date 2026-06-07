# Convenience targets (requires GNU make). On this machine `python` is a dead Store stub, so PY
# defaults to the py313 interpreter; override with `make demo-up PY=python` if the env is active.
# No `make`? Run the scripts directly, e.g. `D:/conda/envs/py313/python.exe scripts/demo.py up`.
PY   ?= D:/conda/envs/py313/python.exe
BOT  ?= p7_crypto_book
PORT ?= 8501

.PHONY: demo-up demo-down demo-status install test lint

demo-up:     ; $(PY) scripts/demo.py up     --bot $(BOT) --port $(PORT)   # start mock bot + cockpit
demo-down:   ; $(PY) scripts/demo.py down   --bot $(BOT) --port $(PORT)   # stop them
demo-status: ; $(PY) scripts/demo.py status --bot $(BOT) --port $(PORT)   # quick health
install:     ; $(PY) -m pip install -e .
test:        ; $(PY) -m pytest -q
lint:        ; $(PY) -m ruff check src tests
