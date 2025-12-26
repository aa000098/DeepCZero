#!/usr/bin/env bash
#set -e

# 이 스크립트 위치 기준으로 프로젝트 루트 계산
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"

mkdir -p "$THIRD_PARTY_DIR"
cd "$ROOT_DIR"

echo "[submodule] syncing..."
git submodule sync --recursive

echo "[submodule] init/update..."
git submodule update --init --recursive

echo "[submodule] done."
