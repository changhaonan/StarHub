add_subdirectory(common)
add_subdirectory(geometry)
add_subdirectory(measure)

# Executables
add_executable(super_reconstruct app.cu)
target_link_libraries(super_reconstruct 
    sr_common_lib 
    sr_measure_lib
    common_lib 
    geometry_lib 
    io_lib
    visualization_lib
    ${Boost_PROGRAM_OPTIONS_LIBRARIES})