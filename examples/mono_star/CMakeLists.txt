# Libraries
add_subdirectory(common)
add_subdirectory(measure)
add_subdirectory(geometry)
add_subdirectory(opt)

# Executables
add_executable(mono_star app.cu)
target_link_libraries(mono_star 
    mono_star_common_lib 
    mono_star_measure_lib
    mono_star_geometry_lib
    mono_star_opt_lib
    common_lib 
    geometry_lib 
    io_lib
    visualization_lib)

add_executable(mono_star_wo_kp app_wo_kp.cu)
target_link_libraries(mono_star_wo_kp 
    mono_star_common_lib 
    mono_star_measure_lib
    mono_star_geometry_lib
    mono_star_opt_lib
    common_lib 
    geometry_lib 
    io_lib
    visualization_lib)

add_executable(test_kp test_kp.cu)
target_link_libraries(test_kp 
    mono_star_common_lib 
    mono_star_measure_lib
    mono_star_geometry_lib
    mono_star_opt_lib
    common_lib 
    geometry_lib 
    io_lib
    visualization_lib)