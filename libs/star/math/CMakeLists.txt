file(GLOB MATH_SRC *.h *.hpp *.cuh *.cpp *.cu)
source_group("src" FILES ${MATH_SRC})
add_library(math_lib SHARED ${MATH_SRC})
set_target_properties(math_lib PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(math_lib common_lib)