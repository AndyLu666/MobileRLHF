cmake_minimum_required(VERSION 3.22.1)
project(mobilefinetuner_android_sdk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MFT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../..")

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OPERATORS_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
option(MFT_ANDROID_USE_OPENBLAS "Enable OpenBLAS for Android native kernels" ON)
if(MFT_ANDROID_USE_OPENBLAS)
    set(USE_BLAS ON CACHE BOOL "Enable BLAS for Android native kernels" FORCE)
    if(NOT OPENBLAS_ROOT)
        set(
            OPENBLAS_ROOT
            "${MFT_ROOT}/third_party/openblas-android-arm64"
            CACHE PATH
            "Bundled Android arm64 OpenBLAS root"
            FORCE
        )
    endif()
else()
    set(USE_BLAS OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory("${MFT_ROOT}/operator" "${CMAKE_BINARY_DIR}/mobilefinetuner-operator")

add_library(mobilefinetuner_jni SHARED
    mobile_finetuner_jni.cpp
)

target_include_directories(mobilefinetuner_jni
    PRIVATE
        "${MFT_ROOT}/operator/include"
        "${MFT_ROOT}/operator"
        "${MFT_ROOT}/operator/finetune_ops"
)

find_library(log_lib log)

target_link_libraries(mobilefinetuner_jni
    PRIVATE
        operators
        ${log_lib}
)

if(ANDROID)
    foreach(target_name operators mobilefinetuner_jni)
        if(TARGET ${target_name})
            target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
        endif()
    endforeach()
endif()
