
add_executable( signorini ${CMAKE_CURRENT_SOURCE_DIR}/signorini.cpp )
target_link_libraries( signorini admmelastic ${RENDER_LIBS} )
