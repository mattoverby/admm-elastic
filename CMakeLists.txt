# Copyright Matt Overby 2025.
# Distributed under the MIT License.

cmake_minimum_required(VERSION 3.0)
project(admmpd LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(ADMMPD_DEBUG "compile in test mode" OFF)
option(ADMMPD_EXAMPLES "compile visual examples" ON)
option(ADMMPD_UNIT_TESTS "compile unit tests" ON)
option(ADMMPD_STATIC_LIBRARY "compile as static library" ON)

if (ADMMPD_DEBUG)
	set(CMAKE_BUILD_TYPE Debug)
	option(MCLGEOM_DISABLE_ASSERT "Disable mclAssert" OFF)
	if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wundef -Wextra -Wno-long-long -Wlogical-op -Wshadow -Wmaybe-uninitialized -Wno-sign-compare")
	endif()
else()
	if (NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE "Release")
		option(MCLGEOM_DISABLE_ASSERT "Disable mclAssert" ON)
	endif()
endif()

# if not static, is header-only
if (ADMMPD_STATIC_LIBRARY)
	add_definitions(-DADMMPD_STATIC_LIBRARY)
endif()

# thread building blocks for parallelization
find_package(TBB REQUIRED)

# mclgeom for geometry helpers
include(mclgeom)

# Create the library
set(ADMMPD_INCLUDE
	${CMAKE_CURRENT_SOURCE_DIR}/src/ADMMPD.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/Energies.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/MatrixTypes.hpp)
set(ADMMPD_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/src/ADMMPD.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/Energies.cpp)
add_library(${PROJECT_NAME} ${ADMMPD_INCLUDE} ${ADMMPD_SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC mcl::geom TBB::tbb)
target_include_directories(${PROJECT_NAME} INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
add_library(admmpd::core ALIAS ${PROJECT_NAME})

# build viewer tests
if(ADMMPD_EXAMPLES)
	include(libigl)
	igl_include(glfw)
	include(mclccd)

	# For mesh data
	add_definitions(-DADMMPD_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

	add_executable(bunny ${CMAKE_CURRENT_SOURCE_DIR}/test/bunny.cpp)
	target_link_libraries(bunny igl::glfw admmpd::core)

	add_executable(cloth ${CMAKE_CURRENT_SOURCE_DIR}/test/cloth.cpp)
	target_link_libraries(cloth igl::glfw admmpd::core)

	add_executable(selfcollision ${CMAKE_CURRENT_SOURCE_DIR}/test/selfcollision.cpp)
	target_link_libraries(selfcollision igl::glfw mcl::ccd admmpd::core)

	add_executable(dillo ${CMAKE_CURRENT_SOURCE_DIR}/test/dillo.cpp)
	target_link_libraries(dillo igl::glfw admmpd::core)
endif()

# unit testing (WIP)
if(ADMMPD_UNIT_TESTS)
    enable_testing()
    add_executable(unit_tests ${CMAKE_CURRENT_SOURCE_DIR}/test/unit_tests.cpp)
    target_link_libraries(unit_tests PUBLIC admmpd::core)
    add_test(NAME unit_testing COMMAND unit_tests)
endif()

# install
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES ${ADMMPD_INCLUDE} DESTINATION include/ADMMPD)