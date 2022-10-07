file(GLOB GEOMETRY_SRC 
    *.h
    surfel/*.h surfel/*.hpp surfel/*.cpp surfel/*.cu surfel/*.cuh
    sampling/*.h sampling/*.hpp sampling/*.cpp sampling/*.cu sampling/*.cuh)
source_group("src" FILES ${GEOMETRY_SRC})
add_library(star_geometry_lib SHARED ${GEOMETRY_SRC})
target_link_libraries(star_geometry_lib star_common_lib star_math_lib)