file(GLOB GEOMETRY_SRC 
    *.h
    geometry_map/*.h geometry_map/*.hpp geometry_map/*.cpp geometry_map/*.cu geometry_map/*.cuh
    surfel/*.h surfel/*.hpp surfel/*.cpp surfel/*.cu surfel/*.cuh
    tsdf/*.h tsdf/*.hpp tsdf/*.cpp tsdf/*.cu tsdf/*.cuh
    rgbd_image/*.h rgbd_image/*.hpp rgbd_imagel/*.cpp rgbd_image/*.cu rgbd_image/*.cuh
    sampling/*.h sampling/*.hpp sampling/*.cpp sampling/*.cu sampling/*.cuh
    node_graph/*.h node_graph/*.hpp node_graph/*.cpp node_graph/*.cu node_graph/*.cuh
    keypoint/*.h keypoint/*.hpp keypoint/*.cpp keypoint/*.cu keypoint/*.cuh)

file(GLOB RENDER_SRC render/*.h render/*.cu)
file(GLOB GLAD_SRC render/glad.c render/glad/glad.h)

source_group("src" FILES ${GEOMETRY_SRC})
source_group("render" FILES ${RENDER_SRC})
source_group("glad" FILES ${GLAD_SRC})

add_library(glad SHARED ${GLAD_SRC})

add_library(geometry_lib SHARED ${GEOMETRY_SRC} ${RENDER_SRC})
set_property(TARGET geometry_lib PROPERTY POSITION_INDEPENDENT_CODE ON)
# This library onlys works when it static
target_link_libraries(geometry_lib common_lib math_lib img_proc_lib glfw glad)