# ------------------------------------------------------------
# 최상위 CMake 파일: pm25_model_compile/CMakeLists.txt
# ------------------------------------------------------------

cmake_minimum_required(VERSION 3.5)

# 프로젝트 이름과 언어 설정 (C 프로젝트)
project(PM25_model C)

# C 표준을 C99로 설정
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# PIC(Position Independent Code) 옵션 활성화
# → 공유 라이브러리로 만들 때 필수
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------
# 1) 소스 파일 목록 수집
#
# 현재 디렉토리 아래의 *.c 파일들과
# layers/ 아래의 *.c, utils/ 아래의 *.c 파일을
# 모두 SRC_LIST 변수에 담는다.
# ------------------------------------------------------------
file(GLOB SRC_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.c
)

# ------------------------------------------------------------
# 2) 공유 라이브러리 타깃 생성
#
# add_library(<타겟명> SHARED <소스파일들...>)
# 이 예시에서는 타겟명을 pm25_model 으로 정했다.
# 그러면 결과물은 libpm25_model.so 로 만들어진다.
# ------------------------------------------------------------
add_library(pm25_model SHARED ${SRC_LIST})

# ------------------------------------------------------------
# 3) 헤더(include) 디렉토리 지정
#
# forward.h 를 비롯해 layers/, utils/ 디렉토리에
# 있는 헤더 파일들을 참조할 수 있도록 include 경로를 추가.
# ------------------------------------------------------------
target_include_directories(pm25_model
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/layers
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# ------------------------------------------------------------
# 4) 필요하다면 추가 라이브러리(예: math) 링크
#
# 예를 들어, 코드 내부에서 math.h 의 expf(), sqrtf() 등을
# 사용할 경우 math 라이브러리를 링크해야 한다.
#    target_link_libraries(pm25_model m)
# 필요 없으면 주석 처리 또는 제거.
# ------------------------------------------------------------
target_link_libraries(pm25_model
    PUBLIC
    m    # math 라이브러리 (expf, fabsf 등 수학 함수가 필요하면)
)
