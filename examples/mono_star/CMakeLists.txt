# Libraries
add_subdirectory(common)

# Executables
add_executable(mono_star app.cu)
target_link_libraries(mono_star mono_star_common_lib)