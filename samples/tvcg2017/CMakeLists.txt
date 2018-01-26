
add_executable( signorini ${CMAKE_CURRENT_SOURCE_DIR}/signorini.cpp )
target_link_libraries( signorini admmelastic ${RENDER_LIBS} )

add_executable( stretched_beams ${CMAKE_CURRENT_SOURCE_DIR}/stretched_beams.cpp )
target_link_libraries( stretched_beams admmelastic ${RENDER_LIBS} )
