## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

message(CHECK_START "searching for GNU GCC + TBB")

# try finding Intel TBB
find_package(TBB QUIET)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND TBB_FOUND)
    # GNU GCC also needs TBB
    # GNU GCC + TBB needs either a CPU target
    if (NOT DEFINED PLSSVM_CPU_TARGET_ARCHS)
        if (PLSSVM_ENABLE_STDPAR_BACKEND MATCHES "ON")
            message(SEND_ERROR "Found requested stdpar backend using GNU GCC + TBB, but no \"cpu\" targets were specified!")
        else ()
            message(STATUS "Found stdpar backend using GNU GCC + TBB, but no \"cpu\" targets were specified!")
        endif ()
        message(CHECK_FAIL "skipped")
        return()
    endif ()
    # GNU GCC + TBB found
    message(CHECK_PASS "found")
    set_local_and_parent(PLSSVM_STDPAR_BACKEND "GNU_TBB")
else ()
    message(CHECK_FAIL "skipped")
    return()
endif ()

# add stdpar implementation specific source file to library
target_sources(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp)

## try finding Boost atomic
if (POLICY CMP0144)
    cmake_policy(SET CMP0144 OLD) # suppress cmake warning
endif ()
find_package(Boost 1.73.0 REQUIRED COMPONENTS atomic)
message(STATUS "Found Boost atomic version ${Boost_VERSION}.")
target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC Boost::atomic)

if (PLSSVM_ENABLE_FAST_MATH)
    set_local_and_parent(PLSSVM_STDPAR_BACKEND_COMPILER_FLAG -ffast-math)
endif ()

# GNU GCC must be linked against TBB
target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC TBB::tbb)

# set compile options
if (PLSSVM_STDPAR_BACKEND_COMPILER_FLAG)
    target_compile_options(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
    target_compile_options(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG})
endif ()

# set GNU GCC + TBB compile definition
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_STDPAR_BACKEND_HAS_GNU_TBB)
target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PRIVATE PLSSVM_STDPAR_BACKEND_HAS_GNU_TBB)