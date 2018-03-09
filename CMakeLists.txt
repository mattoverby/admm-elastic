cmake_minimum_required(VERSION 2.6)
project(admm-elastic C CXX)

############################################################
#
#	Build Options
#
############################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output")
add_definitions(-DADMMELASTIC_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/output")
add_definitions(-DADMMELASTIC_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
option(MCL_BUILD_SAMPLES "Build mcl library samples" OFF)

# Lots of warnings
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-long-long")
endif()

############################################################
#
#	Libraries
#
############################################################

# Keep a variable of linked libs and link at end
set(LINKED_LIBS "")

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Intel MKL (for Pardiso)
set(MKL_THREADING "sequential")
find_package(MKL)
if(MKL_FOUND)
	include_directories(${MKL_INCLUDE_DIRS})
	set(LINKED_LIBS "${LINKED_LIBS};${MKL_LIBRARIES}")
	add_definitions(-DEIGEN_USE_MKL_ALL)
else()
	message("MKL not found, using Eigen solvers")
endif()

# Eigen3 (header only)
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/Eigen3")
#find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# mcloptlib (header only), submodule in deps/
set( MCLOPTLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/mcloptlib" )
include_directories( ${MCLOPTLIB_DIR}/include )

# mclscene (header only), submodule in deps/
set( MCLSCENE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/mclscene" )
add_definitions( -DMCL_USE_GLEW )
include_directories(SYSTEM ${MCLSCENE_DIR}/include )

# glfw3
find_package(GLFW 3 REQUIRED)
include_directories(SYSTEM ${GLFW_INCLUDE_DIR})

# glew
find_package(GLEW REQUIRED)
include_directories(SYSTEM ${GLEW_INCLUDE_DIR})

# opengl
find_package(OpenGL 3.2 REQUIRED)
include_directories(SYSTEM ${OPENGL_INCLUDE_DIR})

# Add project sources
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src )
set( ADMMELASTIC_SRCS
	src/Solver.cpp
	src/ExplicitForce.cpp
	src/TetEnergyTerm.cpp
	src/TriEnergyTerm.cpp
)

# Finally, create the library
add_library( admmelastic ${ADMMELASTIC_SRCS} )
link_libraries( admmelastic ${LINKED_LIBS} )

# TODO check for WHATEVER_FOUND and don't
# compile rendered samples if libs not found.
set( RENDER_LIBS 
	${OPENGL_LIBRARIES}
	${GLEW_LIBRARIES}
	${GLFW_LIBRARIES}
)

############################################################
#
#	Binaries
#
############################################################

enable_testing()
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/samples )


