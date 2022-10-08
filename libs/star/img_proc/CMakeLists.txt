file(GLOB DATA_PROC_SRC *.h *.hpp *.cpp *.cu *.cuh)
source_group("src" FILES ${DATA_PROC_SRC})
add_library(star_img_proc_lib SHARED ${DATA_PROC_SRC})
target_link_libraries(star_img_proc_lib star_common_lib star_math_lib)