cmake_minimum_required(VERSION 3.16)
project(QwenLoRAUnified CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared operator library root
get_filename_component(OPERATORS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../operator" ABSOLUTE)
set(USE_BLAS ON CACHE BOOL "Enable BLAS acceleration for Qwen examples" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Disable operator tests in app builds" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable operator examples in app builds" FORCE)
add_subdirectory("${OPERATORS_ROOT}" operators_build)

# Executables: wikitext & mmlu
add_executable(train_wikitext
    src/wikitext_main.cpp
)
target_link_libraries(train_wikitext PRIVATE operators)

add_executable(train_mmlu
    src/mmlu_main.cpp
)
target_link_libraries(train_mmlu PRIVATE operators)

add_executable(infer_qwen
    src/infer_qwen.cpp
)
target_link_libraries(infer_qwen PRIVATE operators)

add_executable(hfrl_infer_sim
    src/hfrl_infer_sim.cpp
)
target_link_libraries(hfrl_infer_sim PRIVATE operators)

if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(train_wikitext PRIVATE ${ACCELERATE_LIBRARY})
        target_link_libraries(train_mmlu PRIVATE ${ACCELERATE_LIBRARY})
        target_link_libraries(infer_qwen PRIVATE ${ACCELERATE_LIBRARY})
        target_link_libraries(hfrl_infer_sim PRIVATE ${ACCELERATE_LIBRARY})
    endif()
endif()
