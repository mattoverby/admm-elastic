#
#	The easiest way to include this project in your own is
#	to add it as a subdirectory, then use the ADMME_INCLUDE_DIRS
#	and ADMME_LIBRARIES cmake variables set (in parent scope) by this script.
#	See admm-elastic-samples for an example.
#

cmake_minimum_required(VERSION 2.6)
project(admm-elastic C CXX)

#
#	Build Mode
#
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
option(ADMME_VERIFY "Use admm-elastic verification checks" OFF) # Run additional verification steps for debugging
option(ADMME_BUILD_SAMPLES "Build admm-elastic samples" ON)
if( ADMME_VERIFY )
	add_definitions( -DPDADMM_VERIFY )
endif()


############################################################
#
#	Libraries
#
############################################################


# Eigen library (header only)
set( EIGEN_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/deps/Eigen3" )
include_directories( SYSTEM ${EIGEN_INCLUDE} ) # SYSTEM hushes warnings

# OpenMP
# Should add OMP nested, but will leave disabled (default) for now
find_package(OpenMP)
if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# CppOptimizationLibrary (header only)
set( CPPMIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/cppoptlib" )
include_directories( ${CPPMIN_DIR}/include )

# Set the include dirs
set( ADMME_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/src/system
	${CMAKE_CURRENT_SOURCE_DIR}/src/collision
)

# Create a variable containing all of the source code
set( ADMME_SRCS
	src/system/System.hpp			src/system/System.cpp
	src/system/Force.hpp			src/system/Force.cpp
	src/system/ExplicitForce.hpp		src/system/ExplicitForce.cpp
	src/system/TriangleForce.hpp		src/system/TriangleForce.cpp
	src/system/AnchorForce.hpp		src/system/AnchorForce.cpp
	src/system/TetForce.hpp			src/system/TetForce.cpp
	src/system/BendForce.hpp		src/system/BendForce.cpp
	src/system/CollisionForce.hpp		src/system/CollisionForce.cpp
	src/collision/CollisionShape.hpp
	src/collision/CollisionCylinder.hpp
	src/collision/CollisionSphere.hpp
	src/collision/CollisionFloor.hpp
)

# Finally, create the library
include_directories( ${ADMME_INCLUDE_DIRS} )
add_library( admmelastic ${ADMME_SRCS} )
set( ADMME_LIBRARIES admmelastic )

# Set the parent scope variables
get_directory_property(HasParent PARENT_DIRECTORY)
if(HasParent)
	set( ADMME_LIBRARIES ${ADMME_LIBRARIES} PARENT_SCOPE )
	set( ADMME_INCLUDE_DIRS ${ADMME_INCLUDE_DIRS} PARENT_SCOPE )
endif(HasParent)


############################################################
#
#	Binaries
#
############################################################

if( ADMME_BUILD_SAMPLES )

add_executable( singletet ${CMAKE_CURRENT_SOURCE_DIR}/samples/singletet.cpp )
target_link_libraries( singletet ${ADMME_LIBRARIES} )

add_executable( singlenode ${CMAKE_CURRENT_SOURCE_DIR}/samples/singlenode.cpp )
target_link_libraries( singlenode ${ADMME_LIBRARIES} )

endif()

