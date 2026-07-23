#!/bin/sh
# Auto-detect GPU on host and bring up the stack in the matching mode.
# Card is only added/removed across reboots, so detection at boot is enough.
# Installed to run at boot via photo-lib.service (systemd).
set -e
cd "$(dirname "$0")"

if nvidia-smi -L >/dev/null 2>&1; then
    echo "[up.sh] GPU detected -> CUDA mode"
    exec docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
else
    echo "[up.sh] No GPU driver -> CPU mode"
    exec docker compose -f docker-compose.yml up -d
fi
