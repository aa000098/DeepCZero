#!/usr/bin/env bash
#set -euo pipefail

echo "[0/3] Check apt-get..."
if ! command -v apt-get >/dev/null 2>&1; then
  echo "ERROR: This script supports apt-based distros (Ubuntu/Debian)."
  exit 1
fi

echo "[1/3] apt update"
sudo apt-get update -y

echo "[2/3] Install dependencies (valgrind/graphviz/curl-dev/zip-dev)"
# curl dev는 openssl variant 우선, 안되면 gnutls로 fallback
sudo apt-get install -y \
  valgrind \
  graphviz \
  libzip-dev \

if sudo apt-get install -y libcurl4-openssl-dev; then
  echo "Installed: libcurl4-openssl-dev"
else
  echo "openssl variant failed; trying gnutls variant..."
  sudo apt-get install -y libcurl4-gnutls-dev
  echo "Installed: libcurl4-gnutls-dev"
fi

echo "[3/3] Quick checks"
echo "- curl headers:"
(ls /usr/include/curl/curl.h >/dev/null 2>&1 && echo "  OK: /usr/include/curl/curl.h") || echo "  WARN: curl.h not found"
echo "- libzip headers:"
(ls /usr/include/zip.h >/dev/null 2>&1 && echo "  OK: /usr/include/zip.h") || echo "  WARN: zip.h not found"

echo
echo "Done."

