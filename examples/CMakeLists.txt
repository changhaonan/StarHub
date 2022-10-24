# Exampled relied on StarHub

# MonoStar
add_subdirectory(mono_star)

# Example: Read image
add_executable(rgbd_to_pcd rgbd_to_pcd.cu)
target_link_libraries(rgbd_to_pcd ${OpenCV_LIBRARIES})
target_link_libraries(rgbd_to_pcd io_lib geometry_lib visualization_lib)

# Example: Compile test
add_executable(compile_test compile_test.cu)
target_link_libraries(compile_test common_lib)