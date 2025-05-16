# DeepCZero

**C++로 딥러닝 프레임워크를 밑바닥부터 직접 구현하는 프로젝트입니다.**  
Python 기반의 [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3)를 모티브로 하되,  
더 저수준에서 제어 가능한 구조와 고성능 구현을 목표로 합니다.  
  
## 개발 노트

프로젝트 진행 중의 기술 결정, 개선 사항, 설계 기록은 아래 Notion 페이지에서 확인하실 수 있습니다:

🔗 [DeepCZero 개발노트 (Notion)](https://www.notion.so/DeepCZero-1d2c4cd6527d80a69debd81ff4fb6f80)
  

## 프로젝트 구조
DeepCZero/  
├── src/  # 프레임워크 핵심 구현 (Variable, Function 등)  
├── include/  # 헤더 파일  
├── test/  # 테스트 코드 (자동화된 유닛 테스트들)  
├── build/  # (자동 생성) 오브젝트 파일 저장  
├── bin/  # (자동 생성) 테스트 실행 파일  
├── Makefile  # 빌드 및 테스트 자동화  
└── README.md   
  
## 빌드 및 실행 방법
### 전체 빌드
```bash
make
```   
### 전체 테스트
```bash
make test
```   
### 전체 메모리 누수 테스트
```bash
make memory
```   
  
## 주요 구현 클래스 목록
- Tensor: 행렬 저장 및 행렬 연산 지원 클래스
- Variable: Tensor 기반 값 및 gradient 저장, weak_ptr 기반 creator 추적
- Function: 다입력/다출력 지원, 자동 그래프 연결 및 shared_from_this 연동
- Graph: 계산 그래프 자동 구성 및 Topological 정렬 기반 backward 연산
- 연산자: Square, Exp, Add 등 기본 Function 구현
- 메모리 안정성 확보: Valgrind를 통한 누수 점검 완료 (cyclic reference 제거)

  
 ## 향후 계획
### 1단계: 지금 설계 기반에서 확장 가능한 핵심
- 자동 미분 엔진
- 기본 연산 (Square, Add, Mul 등)
- Tensor 지원 확장 (vector, matrix)
- Computational Graph
- 멀티스레딩 지원 (ThreadPool 등)  

### 2단계: 딥러닝 프레임워크로서의 최소 성능 확보
- GEMM 커널 (자체 구현 or BLAS 연동)
- Convolution 연산 지원
- Detect 모델/LLM용 연산 흐름 구성  

### 3단계: 최적화 + 하드웨어 적응
- Eigen / xtensor 최적화
- OpenBLAS / Intel MKL 연동
- GPU 커널 백엔드 (OpenCL, CUDA 등)  

### 4단계: 실용성 및 테스트/빌드 환경 강화
- CMake 전환
- gtest 기반 유닛 테스트
- 모델 저장/불러오기, 학습 루프, Optimizer  
  

## 라이선스
MIT License  
  
