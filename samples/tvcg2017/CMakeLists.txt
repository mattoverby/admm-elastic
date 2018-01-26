
add_executable( signorini ${CMAKE_CURRENT_SOURCE_DIR}/signorini.cpp )
target_link_libraries( signorini admmelastic ${RENDER_LIBS} )

add_executable( torus ${CMAKE_CURRENT_SOURCE_DIR}/torus.cpp )
target_link_libraries( torus admmelastic ${RENDER_LIBS} )
