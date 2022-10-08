file(GLOB GEOMETRY_SRC 
    *.h
    geometry_map/*.h geometry_map/*.hpp geometry_map/*.cpp geometry_map/*.cu geometry_map/*.cuh
    surfel/*.h surfel/*.hpp surfel/*.cpp surfel/*.cu surfel/*.cuh
    tsdf/*.h tsdf/*.hpp tsdf/*.cpp tsdf/*.cu tsdf/*.cuh
    rgbd_image/*.h rgbd_image/*.hpp rgbd_imagel/*.cpp rgbd_image/*.cu rgbd_image/*.cuh
    sampling/*.h sampling/*.hpp sampling/*.cpp sampling/*.cu sampling/*.cuh)
source_group("src" FILES ${GEOMETRY_SRC})
add_library(star_geometry_lib SHARED ${GEOMETRY_SRC})
target_link_libraries(star_geometry_lib star_common_lib star_math_lib)