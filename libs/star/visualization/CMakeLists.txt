file(GLOB VISUALIZATION_SRC *.h *.hpp *.cuh *.cpp *.cu)
source_group("src" FILES ${VISUALIZATION_SRC})
add_library(visualization_lib SHARED ${VISUALIZATION_SRC})
set_target_properties(visualization_lib PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(visualization_lib common_lib ${PCL_VISUALIZATION_LIBRARIES})