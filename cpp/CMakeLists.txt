cmake_minimum_required(VERSION 3.16)
project(make_training_data CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -----------------------------------------------------------------------------
# HIPO 4 library.
# Set HIPO_DIR via -DHIPO_DIR=/path/to/hipo, or rely on $HIPO env (default
# COATJAVA / CLAS12 install location).
# -----------------------------------------------------------------------------
if(NOT DEFINED HIPO_DIR)
    if(DEFINED ENV{HIPO_HOME})        # JLab module sets HIPO_HOME
        set(HIPO_DIR $ENV{HIPO_HOME})
    elseif(DEFINED ENV{HIPO})
        set(HIPO_DIR $ENV{HIPO})
    endif()
endif()
if(NOT DEFINED HIPO_DIR)
    message(FATAL_ERROR "HIPO_DIR not set. Pass -DHIPO_DIR=/path/to/hipo or export HIPO=/path/to/hipo")
endif()

message(STATUS "Using HIPO from: ${HIPO_DIR}")

find_library(HIPO4_LIB    NAMES hipo4 PATHS ${HIPO_DIR}/lib REQUIRED)
find_path   (HIPO4_INC    NAMES hipo4/reader.h PATHS ${HIPO_DIR}/include REQUIRED)

# Some HIPO builds also need lz4. Try to find it, fall back to system.
find_library(LZ4_LIB lz4 PATHS ${HIPO_DIR}/lib /usr/lib /usr/lib64)

add_executable(make_training_data make_training_data.cpp)
target_include_directories(make_training_data PRIVATE ${HIPO4_INC})
target_link_libraries(make_training_data PRIVATE ${HIPO4_LIB})
if(LZ4_LIB)
    target_link_libraries(make_training_data PRIVATE ${LZ4_LIB})
endif()
target_link_libraries(make_training_data PRIVATE stdc++fs)
