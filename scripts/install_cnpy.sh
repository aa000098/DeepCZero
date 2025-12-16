#!/usr/bin/env bash
#set -e

# 이 스크립트 위치 기준으로 프로젝트 루트 계산
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"

mkdir -p "$THIRD_PARTY_DIR"
cd "$ROOT_DIR"

# 이미 서브모듈이 있으면 스킵
if [ ! -d "$THIRD_PARTY_DIR/cnpy" ]; then
    git submodule add https://github.com/rogersce/cnpy.git third_party/cnpy
else
    echo "cnpy already exists at third_party/cnpy, skipping git submodule add"
fi

