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

# Include
include_directories("${CMAKE_SOURCE_DIR}/libs")

# subdirectories
add_subdirectory(external)

# Libraries
add_subdirectory(libs/star)

add_executable(rgbd_to_pcd examples/rgbd_to_pcd.cpp)
add_executable(compile_test examples/compile_test.cu)
target_link_libraries(compile_test star_common_lib)