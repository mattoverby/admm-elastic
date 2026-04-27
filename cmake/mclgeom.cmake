if(TARGET mcl::geom)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    mclgeom
    GIT_REPOSITORY https://github.com/mattoverby/mclgeom.git
    GIT_TAG v0.3.0
)
FetchContent_MakeAvailable(mclgeom)