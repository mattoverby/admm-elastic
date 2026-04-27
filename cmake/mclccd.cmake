if(TARGET mcl::ccd)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    mclccd
    GIT_REPOSITORY https://github.com/mattoverby/mclccd.git
    GIT_TAG v0.2.0
)
FetchContent_MakeAvailable(mclccd)