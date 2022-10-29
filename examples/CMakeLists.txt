# Exampled relied on StarHub

add_executable(test_gl test_gl.cpp)
target_link_libraries(test_gl PUBLIC glfw glad PRIVATE ${CMAKE_DL_LIBS})

# MonoStar
add_subdirectory(mono_star)