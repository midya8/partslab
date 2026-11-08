#!/usr/bin/env bash
set -euo pipefail
python -m partslab demo
python -m partslab audit --out /tmp/partslab-controls.json
python -m partslab face --problem examples/face.json --out /tmp/partslab-face.json
python -m partslab verify-face --witness /tmp/partslab-face.json
