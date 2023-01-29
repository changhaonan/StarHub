cmake_minimum_required(VERSION 3.24)

# CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # 86, default is 3090
endif()

# CUDA runtime
set(CUDA_RUNTIME_LIBRARY Shared)

# Project language
project(StarHub VERSION 0.1 LANGUAGES C CXX CUDA)

# Cmake path
LIST(APPEND CMAKE_PREFIX_PATH /usr/lib/x86_64-linux-gnu/cmake)
LIST(APPEND CMAKE_PREFIX_PATH /home/robot-learning/Software/libtorch/share/cmake)

# Language options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile choice
set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "Release")

# External libraries
# Eigen
set(EIGEN_ROOT "${CMAKE_SOURCE_DIR}/external/eigen")
set(EIGEN_INCLUDE_DIR ${EIGEN_ROOT})
include_directories(${EIGEN_INCLUDE_DIR})

# GLFW package
find_package(glfw3 REQUIRED)

# OpenGL package
# find_package(OpenGL REQUIRED)

# OpenCV
find_package(OpenCV 4.6 REQUIRED COMPONENTS core imgproc highgui features2d)
include_directories(${OpenCV_INCLUDE_DIR})

# Boost
find_package(Boost REQUIRED COMPONENTS system filesystem program_options)
include_directories(${Boost_INCLUDE_DIRS})

# FLANN
find_package(FLANN REQUIRED)
include_directories(${FLANN_INCLUDE_DIRS})

# PCL
find_package(PCL REQUIRED COMPONENTS io visualization)
include_directories(${PCL_INCLUDE_DIRS})

# Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Easy3DViewer
set(Easy3DViewer_ROOT "${CMAKE_SOURCE_DIR}/external/Easy3DViewer")
set(Easy3DViewer_INCLUDE_DIR "${Easy3DViewer_ROOT}/utils/cpp")
include_directories(${Easy3DViewer_INCLUDE_DIR})

# Include
include_directories("${CMAKE_SOURCE_DIR}/libs")
include_directories("${CMAKE_SOURCE_DIR}/examples")

# subdirectories
add_subdirectory(external)

# Libraries
add_subdirectory(libs/star)

# Examples
add_subdirectory(examples)