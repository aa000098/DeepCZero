#!/usr/bin/env bash
# set -euo pipefail

echo "[0/4] Detecting distro..."
if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script supports apt-based distros (Ubuntu/Debian)."
  exit 1
fi

echo "[1/3] apt update"
sudo apt-get update -y

echo "[2/3] Installing libcurl dev headers (for <curl/curl.h>)"
# 보통 openssl variant가 가장 무난
sudo apt-get install -y libcurl4-openssl-dev libzip-dev  || true

echo "[3/3] Verifying curl header exists"
if ! pkg-config --exists libcurl; then
  echo "WARNING: pkg-config cannot find libcurl (but headers may still exist)."
fi

# curl.h 위치 확인(있으면 OK)
if ! dpkg -L libcurl4-openssl-dev 2>/dev/null | grep -q "curl/curl.h"; then
  if ! dpkg -L libcurl4-gnutls-dev 2>/dev/null | grep -q "curl/curl.h"; then
    echo "ERROR: curl/curl.h not found after install."
    exit 1
  fi
fi

echo
echo "Done. You can now rebuild:"
echo "  make clean && make -j\$(nproc)"

