#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STB_DIR="${PROJECT_ROOT}/third_party/stb"

echo "=== Installing stb_image (header-only library) ==="
echo "Project root: ${PROJECT_ROOT}"
echo "stb dir: ${STB_DIR}"

# stb 폴더 생성
mkdir -p "${STB_DIR}"

# stb_image.h 다운로드
echo "[1/2] Downloading stb_image.h..."
curl -fsSL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h \
     -o "${STB_DIR}/stb_image.h"

# stb_image_resize2.h 다운로드 (리사이즈용)
echo "[2/2] Downloading stb_image_resize2.h..."
curl -fsSL https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h \
     -o "${STB_DIR}/stb_image_resize2.h"

echo
echo "✓ stb_image installed successfully!"
echo "  - ${STB_DIR}/stb_image.h"
echo "  - ${STB_DIR}/stb_image_resize2.h"
echo
echo "Usage in C++:"
echo "  #define STB_IMAGE_IMPLEMENTATION"
echo "  #include \"stb_image.h\""
echo "  #define STB_IMAGE_RESIZE_IMPLEMENTATION"
echo "  #include \"stb_image_resize2.h\""
