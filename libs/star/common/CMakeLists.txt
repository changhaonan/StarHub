file(GLOB COMMON_SRC 
    *.h *.hpp *.cuh *.cpp *.cu 
    containers/*.hpp containers/*.cuh containers/*.cpp containers/*.cu
    types/*.h)
source_group("src" FILES ${COMMON_SRC})
add_library(common_lib SHARED ${COMMON_SRC})
set_target_properties(common_lib PROPERTIES LINKER_LANGUAGE CUDA)
# Link library (Commonly shared lib)
target_link_libraries(common_lib ${OpenCV_LIBRARIES})
target_link_libraries(common_lib ${PCL_COMMON_LIBRARIES})
target_link_libraries(common_lib ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries(common_lib cuda)