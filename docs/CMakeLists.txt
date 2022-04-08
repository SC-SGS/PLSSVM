## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

########################################################################################################################
##                                     setup documentation generation with doxygen                                    ##
########################################################################################################################
## use installed doxygen
find_package(Doxygen REQUIRED OPTIONAL_COMPONENTS dot)

## configure doxygen
set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/docs")
set(DOXYGEN_IMAGE_PATH "${PROJECT_SOURCE_DIR}/docs/resources")
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "README.md")
set(DOXYGEN_FILE_PATTERNS "*.hpp;*.cuh;*.cl;*.dox")
set(DOXYGEN_EXTENSION_MAPPING "cu=c++;cuh=c++;cl=c++")
set(DOXYGEN_STRIP_FROM_PATH "${PROJECT_SOURCE_DIR}")
set(DOXYGEN_EXCLUDE "${PROJECT_SOURCE_DIR}/src/main_train.cpp;${PROJECT_SOURCE_DIR}/src/main_predict.cpp")
set(DOXYGEN_ABBREVIATE_BRIEF "")
set(DOXYGEN_QUIET "YES")
set(DOXYGEN_HTML_TIMESTAMP "YES")
set(DOXYGEN_NUM_PROC_THREADS 0)
set(DOXYGEN_WARN_NO_PARAMDOC "YES")
set(DOXYGEN_SORT_MEMBER_DOCS "NO")
set(DOXYGEN_INLINE_INHERITED_MEMB "YES")

## enable processing of specific attributes and macros
set(DOXYGEN_ENABLE_PREPROCESSING "YES")
set(DOXYGEN_MACRO_EXPANSION "YES")
set(DOXYGEN_EXPAND_ONLY_PREDEF "YES")
set(DOXYGEN_PREDEFINED "__attribute__((x))=;__CUDA_ARCH__=0")

set(DOXYGEN_VERBATIM_VARS DOXYGEN_ALIASES)
set(DOXYGEN_ALIASES
        [[license="\par License^^\parblock^^"  ]]
        )

## add doxygen as target
doxygen_add_docs(
        doc
        "${PROJECT_SOURCE_DIR}/include;${PROJECT_SOURCE_DIR}/docs/resources;${PROJECT_SOURCE_DIR}/README.md"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Generating API documentation with Doxygen"
)

## create shortcut for index.html
add_custom_command(
        TARGET doc
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -P "${PROJECT_SOURCE_DIR}/cmake/create_documentation_shortcut.cmake"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)