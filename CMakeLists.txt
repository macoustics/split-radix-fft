cmake_minimum_required(VERSION 3.10)
project(SplitRadixFft VERSION 1.0 LANGUAGES CXX)

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(BUILD_TESTS_SPLIT_RADIX_FFT "Build the tests for SplitRadixFFT" OFF)
if(BUILD_TESTS_SPLIT_RADIX_FFT)
    # Add CPM
    include(cmake/CPM.cmake)
    CPMAddPackage("gh:catchorg/Catch2@3.4.0")

    enable_testing()
    add_subdirectory(tests)
endif()

