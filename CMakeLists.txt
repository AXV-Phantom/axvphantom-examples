cmake_minimum_required(VERSION 3.28)

project(axvphantom_examples VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

option(AXVP_BUILD_EXAMPLE_CAMERA "Build the USB camera example" ON)

set(AXVP_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../axvphantom-sdk" CACHE PATH
    "Path to the local AXV Phantom SDK checkout")
set(AXVP_EXAMPLE_DEFAULT_MODEL_DIR "${AXVP_SDK_DIR}/data" CACHE PATH
    "Default SDK data directory used by the example application")

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(AXVP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AXVP_BUILD_BENCH OFF CACHE BOOL "" FORCE)

if(NOT EXISTS "${AXVP_SDK_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "AXVP_SDK_DIR does not point to a valid SDK checkout: ${AXVP_SDK_DIR}")
endif()

add_subdirectory("${AXVP_SDK_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/axvphantom-sdk"
                 EXCLUDE_FROM_ALL)

find_package(OpenCV REQUIRED COMPONENTS highgui videoio)

if(AXVP_BUILD_EXAMPLE_CAMERA)
    add_executable(axvp_camera_demo src/axvp_camera_demo.cpp)
    target_compile_features(axvp_camera_demo PRIVATE cxx_std_23)
    target_compile_definitions(axvp_camera_demo PRIVATE
        AXVP_EXAMPLE_DEFAULT_MODEL_DIR="${AXVP_EXAMPLE_DEFAULT_MODEL_DIR}"
    )
    target_link_libraries(axvp_camera_demo
        PRIVATE
            axvphantom
            ${OpenCV_LIBS}
    )
    target_include_directories(axvp_camera_demo PRIVATE
        ${OpenCV_INCLUDE_DIRS}
    )
endif()
