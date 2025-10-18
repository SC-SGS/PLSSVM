# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "Documentation:  ")

########################################################################################################################
#                                      setup documentation generation with doxygen                                     #
########################################################################################################################
# use installed doxygen
find_package(Doxygen REQUIRED OPTIONAL_COMPONENTS dot)

# configure doxygen
set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/docs")
set(DOXYGEN_IMAGE_PATH "${PROJECT_SOURCE_DIR}/docs/resources")
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/README.md")
set(DOXYGEN_FILE_PATTERNS "*.hpp;*.cuh;*.cl;*.dox")
set(DOXYGEN_EXTENSION_MAPPING "cu=c++;cuh=c++;cl=c++")
set(DOXYGEN_STRIP_FROM_PATH "${PROJECT_SOURCE_DIR}")
set(DOXYGEN_EXCLUDE "${PROJECT_SOURCE_DIR}/src/main_train.cpp;${PROJECT_SOURCE_DIR}/src/main_predict.cpp;${PROJECT_SOURCE_DIR}/src/main_scale.cpp")
set(DOXYGEN_ABBREVIATE_BRIEF "")
set(DOXYGEN_QUIET "YES")
set(DOXYGEN_HTML_TIMESTAMP "YES")
set(DOXYGEN_NUM_PROC_THREADS 0)
set(DOXYGEN_WARN_NO_PARAMDOC "YES")
set(DOXYGEN_SORT_MEMBER_DOCS "NO")
set(DOXYGEN_INLINE_INHERITED_MEMB "YES")
set(DOXYGEN_USE_MATHJAX "YES")
set(DOXYGEN_PROJECT_LOGO "${PROJECT_SOURCE_DIR}/docs/resources/logo_90x55.png")
set(DOXYGEN_EXCLUDE_SYMBOLS "*_HPP_")
set(DOXYGEN_HTML_COLORSTYLE "TOGGLE")

set(DOXYGEN_DOT_IMAGE_FORMAT "svg")
set(DOXYGEN_DOT_GRAPH_MAX_NODES "100")
set(DOXYGEN_INTERACTIVE_SVG "YES")
set(DOXYGEN_INCLUDE_GRAPH "NO")
set(DOXYGEN_EXTRACT_PRIVATE "YES")
set(DOXYGEN_EXAMPLE_PATH "${PROJECT_SOURCE_DIR}/docs/resources/examples")

# enable processing of specific attributes and macros
set(DOXYGEN_ENABLE_PREPROCESSING "YES")
set(DOXYGEN_MACRO_EXPANSION "YES")
set(DOXYGEN_EXPAND_ONLY_PREDEF "YES")
set(DOXYGEN_EXPAND_AS_DEFINED "YES")
set(DOXYGEN_PREDEFINED "__attribute__((x))=;__CUDA_ARCH__=0")

set(DOXYGEN_VERBATIM_VARS DOXYGEN_ALIASES)
set(DOXYGEN_ALIASES [[license="\par License^^\parblock^^"  ]])

# sanitize README formulas for Doxygen if the sed utility is available
find_program(PLSSVM_HAS_SED "sed")
if (PLSSVM_HAS_SED)
    set(DOXYGEN_MD_FILTER
        # replace markdown math syntax with the corresponding Doxygen syntax, BUT ONLY in the first 350 lines
        " -e '1,350 s/\\$/\\\\f\\$/g'"
        # remove section between 'Table of Contents' and 'Introduction'
        " -e '/^## Table of Contents/,/^## Introduction to PLSSVM/{//!d}'"
        # convert Markdown header to Doxygen TOC
        " -e 's/^## Table of Contents/@tableofcontents/g'"
        # remove Python Bindings intro section
        " -e '/^# The Python3 Bindings/,/^We currently support two kinds of Python3 bindings/{//!d}'"
        # fix Doxygen problems when linking to other Markdown files
        " -e 's|\\[examples/python/interactive\\](examples/python/interactive/README.md)|\\\\ref examples_interactive|g'"
        " -e 's|<!-- DOXYGEN_INTERACTIVE_EXAMPLE_LINK -->|/* \\\\page examples_interactive Interactive Example|g'"
        " -e 's|\\[examples/python/sklearn\\](examples/python/sklearn/README.md)|\\\\ref examples_sklearn|g'"
        " -e 's|<!-- DOXYGEN_SKLEARN_EXAMPLE_LINK -->|/* \\\\page examples_sklearn Sklearn Examples|g'"
        " -e 's|\\[our API documentation\\](bindings/Python/README.md)|\\\\ref bindings_python|g'"
        " -e 's|<!-- DOXYGEN_PYTHON_BINDINGS_LINK -->|/* \\\\page bindings_python Python Bindings|g'"
        # replace emoji codes with unicode symbols
        " -e 's|:white_check_mark:|✓|g'"
        " -e 's|:x:|✗|g'"
        # convert Markdown header to Doxygen TOC (msut be done here)
        " -e 's/^# The Python3 Bindings/@tableofcontents{HTML:3}/g'"
    )
    string(JOIN " " DOXYGEN_MD_FILTER_CMD ${DOXYGEN_MD_FILTER})
    
    set(DOXYGEN_FILTER_PATTERNS
        "*.md=sed ${DOXYGEN_MD_FILTER_CMD}"
    )
    list(APPEND DOXYGEN_FILTER_PATTERNS
         "*.cl=sed -e 's/ PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST)/, Args... PLSSVM_OPENCL_KERNEL_FUNCTION_PARAMETER_LIST)/g'"
    )
endif ()

# add doxygen as target
doxygen_add_docs(
    doc
    "${PROJECT_SOURCE_DIR}/include;${PROJECT_SOURCE_DIR}/docs/resources;${PROJECT_SOURCE_DIR}/README.md;${PROJECT_SOURCE_DIR}/bindings/Python/README.md;${PROJECT_SOURCE_DIR}/examples/python/sklearn/README.md;${PROJECT_SOURCE_DIR}/examples/python/interactive/README.md"
    ALL # add to the default build target
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Generating API documentation with Doxygen"
)

# create shortcut for index.html
add_custom_command(
    TARGET doc
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -P "${PROJECT_SOURCE_DIR}/cmake/create_documentation_shortcut.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

# install targets for the documentation
include(GNUInstallDirs)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/docs/html" DESTINATION "${CMAKE_INSTALL_DOCDIR}")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/documentation.html" DESTINATION "${CMAKE_INSTALL_DOCDIR}")

list(POP_BACK CMAKE_MESSAGE_INDENT)
