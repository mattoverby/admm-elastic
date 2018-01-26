
add_executable( bunnyexpand ${CMAKE_CURRENT_SOURCE_DIR}/bunnyexpand.cpp )
target_link_libraries( bunnyexpand admmelastic ${RENDER_LIBS} )

add_executable( trianglestrain ${CMAKE_CURRENT_SOURCE_DIR}/trianglestrain.cpp )
target_link_libraries( trianglestrain admmelastic ${RENDER_LIBS} )
