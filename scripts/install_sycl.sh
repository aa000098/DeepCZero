#!/bin/bash
# Intel oneAPI SYCL Toolkit 설치 스크립트
# DeepCZero GPU 백엔드 (USE_SYCL=1) 빌드에 필요
#
# 사용법:
#   sudo bash scripts/install_sycl.sh
#
# 설치 후:
#   source /opt/intel/oneapi/setvars.sh
#   make USE_SYCL=1

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ---- 1. 이미 설치되어 있는지 확인 ----
SETVARS="/opt/intel/oneapi/setvars.sh"

# icpx가 PATH에 없지만 설치는 되어있는 경우 → setvars 실행
if ! command -v icpx &>/dev/null && [ -f "$SETVARS" ]; then
    warn "oneAPI가 설치되어 있지만 환경변수가 설정되지 않았습니다."
    info "setvars.sh 실행 중..."
    source "$SETVARS" --force > /dev/null 2>&1
fi

GPU_MISSING=0
if ! dpkg -s intel-opencl-icd &>/dev/null || ! dpkg -s intel-level-zero-gpu &>/dev/null || ! dpkg -s level-zero &>/dev/null; then
    GPU_MISSING=1
fi

if command -v icpx &>/dev/null; then
    info "icpx가 이미 설치되어 있습니다: $(which icpx)"
    icpx --version
    echo ""
    if command -v sycl-ls &>/dev/null; then
        info "사용 가능한 SYCL 디바이스:"
        sycl-ls
    fi

    if [ "$GPU_MISSING" -eq 0 ]; then
        info "추가 설치가 필요하지 않습니다."
        exit 0
    fi

    warn "GPU 런타임 패키지가 누락되어 있습니다. 추가 설치를 진행합니다."
    if [ "$EUID" -ne 0 ]; then
        error "GPU 패키지 설치에 root 권한이 필요합니다. sudo bash $0 으로 실행해주세요."
    fi

    # Intel Graphics 저장소 추가 (level-zero 패키지 제공)
    if [ ! -f /etc/apt/sources.list.d/intel-graphics.list ]; then
        info "Intel Graphics APT repository 등록 중..."
        wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
            gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics-keyring.gpg 2>/dev/null
        . /etc/os-release
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics-keyring.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}/lts/2350 unified" | \
            tee /etc/apt/sources.list.d/intel-graphics.list > /dev/null
    fi

    # GPU 런타임만 설치하고 나머지 스킵
    apt-get update -qq
    GPU_PACKAGES=(intel-opencl-icd intel-level-zero-gpu level-zero)
    if apt-get install -y "${GPU_PACKAGES[@]}"; then
        info "GPU 런타임 패키지 설치 완료"
    else
        warn "GPU 런타임 패키지 일부를 설치할 수 없습니다."
    fi

    # render 그룹 권한 설정
    TARGET_USER="${SUDO_USER:-$USER}"
    if getent group render &>/dev/null; then
        if ! id -nG "$TARGET_USER" | grep -qw render; then
            info "render 그룹에 $TARGET_USER 추가 (GPU 디바이스 접근 권한)"
            usermod -aG render "$TARGET_USER"
        fi
    fi

    # WSL2 Level Zero 드라이버 환경변수 설정
    ZE_LIB="/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so"
    TARGET_HOME=$(eval echo "~$TARGET_USER")
    if [ -f "$ZE_LIB" ] && [ -f "$TARGET_HOME/.bashrc" ]; then
        if ! grep -q "ZE_ENABLE_ALT_DRIVERS" "$TARGET_HOME/.bashrc"; then
            info ".bashrc에 ZE_ENABLE_ALT_DRIVERS 환경변수 추가"
            echo "" >> "$TARGET_HOME/.bashrc"
            echo "# Intel GPU Level Zero driver for WSL2" >> "$TARGET_HOME/.bashrc"
            echo "export ZE_ENABLE_ALT_DRIVERS=$ZE_LIB" >> "$TARGET_HOME/.bashrc"
        fi
    fi

    # 설치 확인
    if command -v sycl-ls &>/dev/null; then
        echo ""
        info "사용 가능한 SYCL 디바이스:"
        sycl-ls
        if sycl-ls 2>/dev/null | grep -qi "gpu"; then
            info "GPU 디바이스가 감지되었습니다!"
        else
            warn "GPU 디바이스가 감지되지 않았습니다. (CPU 폴백으로 동작)"
            if [ -f /proc/sys/fs/binfmt_misc/WSLInterop ]; then
                warn "WSL2 GPU 사용을 위해 확인해주세요:"
                echo "  1. Windows에서 최신 Intel 그래픽 드라이버 설치"
                echo "  2. /dev/dxg 존재 확인: ls -la /dev/dxg"
                echo "  3. WSL 재시작: PowerShell에서 wsl --shutdown 후 재실행"
            fi
        fi
    fi
    exit 0
fi

# ---- 2. root 권한 확인 ----
if [ "$EUID" -ne 0 ]; then
    error "root 권한이 필요합니다. sudo bash $0 으로 실행해주세요."
fi

# ---- 3. OS 확인 ----
if [ ! -f /etc/os-release ]; then
    error "지원되지 않는 OS입니다. Ubuntu/Debian 계열만 지원합니다."
fi

. /etc/os-release
info "OS: $PRETTY_NAME"

if [[ "$ID" != "ubuntu" && "$ID" != "debian" && "$ID_LIKE" != *"debian"* ]]; then
    error "이 스크립트는 Ubuntu/Debian 계열만 지원합니다. 다른 OS는 수동 설치가 필요합니다."
fi

# ---- 4. Intel GPG key + APT repository 등록 ----
info "Intel oneAPI APT repository 등록 중..."

apt-get update -qq
apt-get install -y -qq gpg-agent wget > /dev/null

# GPG key
wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg

# APT source
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
    tee /etc/apt/sources.list.d/oneAPI.list > /dev/null

# Intel Graphics repository (GPU 런타임용)
info "Intel Graphics APT repository 등록 중..."
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics-keyring.gpg 2>/dev/null
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics-keyring.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}/lts/2350 unified" | \
    tee /etc/apt/sources.list.d/intel-graphics.list > /dev/null

apt-get update -qq

# ---- 5. Intel oneAPI Base Toolkit 설치 ----
# 필수 컴포넌트만 설치 (전체 toolkit 대신 필요한 패키지만)
info "Intel oneAPI 패키지 설치 중... (수 분 소요될 수 있습니다)"

PACKAGES=(
    intel-oneapi-compiler-dpcpp-cpp   # icpx 컴파일러 + SYCL runtime
    intel-oneapi-mkl-devel            # oneMKL (BLAS, VM 등)
)

apt-get install -y "${PACKAGES[@]}"

# ---- 5b. GPU 런타임 패키지 설치 ----
info "Intel GPU 런타임 패키지 설치 중..."

GPU_PACKAGES=(
    intel-opencl-icd                  # Intel OpenCL ICD (GPU 디바이스)
    intel-level-zero-gpu              # Level Zero GPU 드라이버
    level-zero                        # Level Zero 런타임
)

# GPU 패키지는 없을 수 있으므로 실패해도 계속 진행
if apt-get install -y "${GPU_PACKAGES[@]}" 2>/dev/null; then
    info "GPU 런타임 패키지 설치 완료"
else
    warn "GPU 런타임 패키지 일부를 설치할 수 없습니다."
    warn "GPU 없이 CPU 폴백으로 동작합니다."
fi

# ---- 5c. GPU 디바이스 권한 설정 ----
TARGET_USER="${SUDO_USER:-$USER}"
if getent group render &>/dev/null; then
    if ! id -nG "$TARGET_USER" | grep -qw render; then
        info "render 그룹에 $TARGET_USER 추가 (GPU 디바이스 접근 권한)"
        usermod -aG render "$TARGET_USER"
    fi
fi

# ---- 5d. WSL2 Level Zero 드라이버 환경변수 설정 ----
ZE_LIB="/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so"
TARGET_HOME=$(eval echo "~$TARGET_USER")
if [ -f "$ZE_LIB" ] && [ -f "$TARGET_HOME/.bashrc" ]; then
    if ! grep -q "ZE_ENABLE_ALT_DRIVERS" "$TARGET_HOME/.bashrc"; then
        info ".bashrc에 ZE_ENABLE_ALT_DRIVERS 환경변수 추가"
        echo "" >> "$TARGET_HOME/.bashrc"
        echo "# Intel GPU Level Zero driver for WSL2" >> "$TARGET_HOME/.bashrc"
        echo "export ZE_ENABLE_ALT_DRIVERS=$ZE_LIB" >> "$TARGET_HOME/.bashrc"
    fi
fi

# ---- 6. 환경변수 설정 ----
if [ -f "$SETVARS" ]; then
    info "설치 완료. 환경변수 설정 중..."
    source "$SETVARS" --force > /dev/null 2>&1
else
    warn "setvars.sh를 찾을 수 없습니다. 수동 설정이 필요합니다."
fi

# ---- 7. 설치 확인 ----
echo ""
info "=== 설치 확인 ==="

if command -v icpx &>/dev/null; then
    info "icpx 컴파일러: $(which icpx)"
    icpx --version 2>&1 | head -1
else
    warn "icpx를 찾을 수 없습니다. 환경변수를 설정해주세요:"
    echo "  source /opt/intel/oneapi/setvars.sh"
fi

if command -v sycl-ls &>/dev/null; then
    echo ""
    info "사용 가능한 SYCL 디바이스:"
    sycl-ls
    if sycl-ls 2>/dev/null | grep -qi "gpu"; then
        info "GPU 디바이스가 감지되었습니다!"
    else
        warn "GPU 디바이스가 감지되지 않았습니다. (CPU 폴백으로 동작)"
        if [ -f /proc/sys/fs/binfmt_misc/WSLInterop ]; then
            warn "WSL2 GPU 사용을 위해 확인해주세요:"
            echo "  1. Windows에서 최신 Intel 그래픽 드라이버 설치"
            echo "  2. /dev/dxg 존재 확인: ls -la /dev/dxg"
            echo "  3. WSL 재시작: PowerShell에서 wsl --shutdown 후 재실행"
        fi
    fi
else
    warn "sycl-ls를 찾을 수 없습니다."
fi

# ---- 8. 사용 안내 ----
echo ""
echo "=========================================="
info "설치가 완료되었습니다!"
echo ""
echo "  # 매 세션마다 환경변수 설정 (또는 .bashrc에 추가)"
echo "  source /opt/intel/oneapi/setvars.sh"
echo ""
echo "  # DeepCZero SYCL 빌드"
echo "  make clean && make USE_SYCL=1"
echo ""
echo "  # SYCL 테스트"
echo "  make USE_SYCL=1 bin/container/tensor/tensor_sycl_test"
echo "  LD_LIBRARY_PATH=bin ./bin/container/tensor/tensor_sycl_test"
echo "=========================================="
