#!/usr/bin/env bash
set -euo pipefail
python -m partslab demo
python -m partslab audit --out /tmp/partslab-controls.json
