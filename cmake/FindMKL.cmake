#
#  Adapted from https://github.com/scivision/fortran2018-examples
#  MIT licence
#  Jozsef Bakosi, 2016, Los Alamos National Security, LLC.
#  Find the Math Kernel Library from Intel
#
# Output:
#  MKL_FOUND - System has MKL
#  MKL_INCLUDE_DIRS - MKL include files directories
#  MKL_LIBRARIES - The MKL libraries
#  MKL_INTERFACE_LIBRARY - MKL interface library
#  MKL_SEQUENTIAL_LAYER_LIBRARY - MKL sequential layer library
#  MKL_THREAD_LIBRARY - MKL thread library
#  MKL_CORE_LIBRARY - MKL core library
#
# Input:
#  MKL_ROOT - root directory for MKL
#  MKL_THREADING - Threading layer: "sequential" or "openmp"
#  
#
# Notes:
# - Just handles lp64 and linux for now
#

if(NOT MKL_ROOT)
	set(MKL_ROOT "/opt/intel/mkl")
endif()

if(NOT THREADING)
	set(MKL_THREADING "sequential")
endif()

# If already in cache, be silent
if (MKL_INCLUDE_DIRS AND MKL_LIBRARIES AND MKL_INTERFACE_LIBRARY AND MKL_THREAD_LIBRARY AND MKL_CORE_LIBRARY)
  set (MKL_FIND_QUIETLY TRUE)
endif()

if(NOT BUILD_SHARED_LIBS)
	set(INT_LIB "libmkl_intel_lp64.a")
	set(SEQ_LIB "libmkl_sequential.a")
	set(THR_LIB "libmkl_gnu_thread.a")
	set(COR_LIB "libmkl_core.a")
else()
	set(INT_LIB "mkl_intel_lp64")
	set(SEQ_LIB "mkl_sequential")
	set(THR_LIB "libmkl_gnu_thread")
	set(COR_LIB "mkl_core")
endif()

find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS ${MKL_ROOT}/include)

find_library(MKL_INTERFACE_LIBRARY
	NAMES ${INT_LIB}
	PATHS ${MKL_ROOT}/lib
		${MKL_ROOT}/lib/intel64
		$ENV{INTEL}/mkl/lib/intel64
	NO_DEFAULT_PATH)

find_library(MKL_SEQUENTIAL_LAYER_LIBRARY
	NAMES ${SEQ_LIB}
	PATHS ${MKL_ROOT}/lib
		${MKL_ROOT}/lib/intel64
		$ENV{INTEL}/mkl/lib/intel64
	NO_DEFAULT_PATH)

find_library(MKL_GNUTHREAD_LAYER_LIBRARY
	NAMES ${THR_LIB}
	PATHS ${MKL_ROOT}/lib
		${MKL_ROOT}/lib/intel64
		$ENV{INTEL}/mkl/lib/intel64
	NO_DEFAULT_PATH)

find_library(MKL_CORE_LIBRARY
	NAMES ${COR_LIB}
	PATHS ${MKL_ROOT}/lib
		${MKL_ROOT}/lib/intel64
		$ENV{INTEL}/mkl/lib/intel64
	NO_DEFAULT_PATH)

set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})

# Which threading layer library to use?
if(${MKL_THREADING} STREQUAL "openmp")
	set(MKL_THREAD_LIBRARY ${MKL_GNUTHREAD_LAYER_LIBRARY})
else()
	set(MKL_THREAD_LIBRARY ${MKL_SEQUENTIAL_LAYER_LIBRARY})
endif()

# Basically modeled after results from:
# https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
set(MKL_LIBRARIES "-Wl,--start-group")
list(APPEND MKL_LIBRARIES ${MKL_INTERFACE_LIBRARY} ${MKL_THREAD_LIBRARY} ${MKL_CORE_LIBRARY})
list(APPEND MKL_LIBRARIES "-Wl,--end-group;dl")

if (MKL_INCLUDE_DIR AND
	MKL_INTERFACE_LIBRARY AND
	MKL_THREAD_LIBRARY AND
	MKL_CORE_LIBRARY)
	set(MKL_FOUND 1)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
else()
	set(MKL_INCLUDE_DIRS "")
	set(MKL_LIBRARIES "")
	set(MKL_INTERFACE_LIBRARY "")
	set(MKL_THREAD_LIBRARY "")
	set(MKL_CORE_LIBRARY "")
	set(MKL_FOUND 0)
endif()

# Handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE if
# all listed variables are TRUE.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIRS MKL_INTERFACE_LIBRARY MKL_THREAD_LIBRARY MKL_CORE_LIBRARY)

MARK_AS_ADVANCED(MKL_INCLUDE_DIRS MKL_LIBRARIES MKL_INTERFACE_LIBRARY MKL_THREAD_LIBRARY MKL_CORE_LIBRARY)
