file(GLOB MONO_MEASURE_SRC *.h *.hpp *.cpp *.cu)
source_group("src" FILES ${MONO_MEASURE_SRC})
add_library(mono_star_measure_lib SHARED ${MONO_MEASURE_SRC})
set_target_properties(mono_star_measure_lib PROPERTIES LINKER_LANGUAGE CUDA)
# Link library (Commonly shared lib)
target_link_libraries(mono_star_measure_lib mono_star_common_lib common_lib geometry_lib torch_utils_lib)