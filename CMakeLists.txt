cmake_minimum_required(VERSION 3.10)

project(StarHub VERSION 0.1 LANGUAGES CXX CUDA)

# Language options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# External libraries

# Eigen
set(EIGEN_ROOT "${CMAKE_SOURCE_DIR}/external/eigen")
set(EIGEN_INCLUDE_DIR ${EIGEN_ROOT})
include_directories(${EIGEN_INCLUDE_DIR})

# Boost
find_package(Boost)
include_directories(${Boost_INCLUDE_DIRS})

# subdirectories
add_subdirectory(external)

add_executable(rgbd_to_pcd examples/rgbd_to_pcd.cpp)