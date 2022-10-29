file(GLOB MONO_GEOMETRY_SRC *.h *.hpp *.cpp *.cu)
source_group("src" FILES ${MONO_GEOMETRY_SRC})
add_library(mono_star_geometry_lib ${MONO_GEOMETRY_SRC})  # static lib for opengl
set_target_properties(mono_star_geometry_lib PROPERTIES LINKER_LANGUAGE CUDA)
set_property(TARGET mono_star_geometry_lib PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(mono_star_geometry_lib mono_star_common_lib geometry_lib)