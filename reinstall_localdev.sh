#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install -e "$SCRIPT_DIR"
pip install -e "$SCRIPT_DIR/../coconut-crab"
pip install -e "$SCRIPT_DIR/../labelstudio-tools"
