cmake_minimum_required(VERSION 3.10)

# CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # 86, default is 3090
endif()

# Project language
project(StarHub VERSION 0.1 LANGUAGES CXX CUDA)

# Cmake path
LIST(APPEND CMAKE_PREFIX_PATH /usr/lib/x86_64-linux-gnu/cmake)

# Language options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# External libraries

# Eigen
set(EIGEN_ROOT "${CMAKE_SOURCE_DIR}/external/eigen")
set(EIGEN_INCLUDE_DIR ${EIGEN_ROOT})
include_directories(${EIGEN_INCLUDE_DIR})

# OpenCV
find_package(OpenCV REQUIRED core)
include_directories(${OpenCV_INCLUDE_DIR})

# Boost
find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

# FLANN
find_package(FLANN REQUIRED)
include_directories(${FLANN_INCLUDE_DIRS})

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

# Include
include_directories("${CMAKE_SOURCE_DIR}/libs")

# subdirectories
add_subdirectory(external)

# Libraries
add_subdirectory(libs/star)

# Examples
add_executable(rgbd_to_pcd examples/rgbd_to_pcd.cpp)
add_executable(compile_test examples/compile_test.cu)
target_link_libraries(compile_test star_common_lib)