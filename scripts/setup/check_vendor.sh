#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENDOR="$ROOT/third_party/speaker_extraction_SpEx"
if [[ ! -f "$VENDOR/nnet/spex_plus.py" ]]; then
  echo "Clone vendor: git clone --depth 1 https://github.com/xuchenglin28/speaker_extraction_SpEx.git $VENDOR"
  exit 1
fi
PYTHONPATH="$ROOT/src:$VENDOR" python -c "from nnet.spex_plus import SpEx_Plus; print('OK')"
