# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# read the compile_commands.json file with ALL compiler flags including the ones that should be removed
file(READ "${PLSSVM_COMPILE_COMMANDS_JSON_FILE}" CONTENTS)

# create a list of all offensive flags (the order IS important!)
set(PLSSVM_OFFENSIVE_CLANG_TIDY_FLAGS
    # stdpar NVHPC
    "-gpu=(fastmath|unified)"
    "-stdpar=(gpu|multicore)"
    "-tp=native"
    "-mp"
    "(^|[ \t])-fast([ \t]|$)"
    # stdpar Intel LLVM (icpx)
    "-fiopenmp"
    "-fsycl-pstl-offload=(cpu|gpu)"
    "-fsycl-targets=[^ ]+"
    # (stdpar) AdaptiveCpp
    "--acpp-stdpar-unconditional-offload"
    "--acpp-stdpar-system-usm"
    "--acpp-stdpar"
    "--acpp-targets=[^ ]+"
    # Intel LLVM (icpx)
    "-fno-sycl-id-queries-fit-in-int"
    # Kokkos SYCL
    "-Xsycl-target-backend=[^ ]+"
    "-fno-sycl-rdc"
    "-fsycl-dead-args-optimization"
    "-fsycl-unnamed-lambda"
)

# remove problematic flags (regex-based)
foreach (flag_regex IN LISTS PLSSVM_OFFENSIVE_CLANG_TIDY_FLAGS)
    string(REGEX REPLACE "${flag_regex}" "" CONTENTS "${CONTENTS}")
endforeach ()

# write the new compile_commands.json file were the offending compiler flags are removed
file(WRITE "${PLSSVM_COMPILE_COMMANDS_JSON_FILE}" "${CONTENTS}")