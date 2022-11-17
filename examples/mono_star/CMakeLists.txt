# Libraries
add_subdirectory(common)
add_subdirectory(measure)
add_subdirectory(geometry)
# add_subdirectory(opt)

# Executables
add_executable(mono_star app.cu)
target_link_libraries(mono_star 
    mono_star_common_lib 
    mono_star_measure_lib
    mono_star_geometry_lib
    common_lib 
    geometry_lib 
    io_lib
    visualization_lib)