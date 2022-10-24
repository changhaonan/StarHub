file(GLOB DATA_PROC_SRC *.h *.hpp *.cpp *.cu *.cuh)
source_group("src" FILES ${DATA_PROC_SRC})
add_library(img_proc_lib SHARED ${DATA_PROC_SRC})
target_link_libraries(img_proc_lib common_lib math_lib)