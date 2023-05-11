#!/usr/bin/bash

POCL_VERSION=3.1
if [ -f pocl.tar.gz ]; then
    rm -rf pocl.tar.gz*
fi
if [ ! -d "$POCL_PATH" ]; then
    wget https://github.com/pocl/pocl/archive/refs/tags/v${POCL_VERSION}.tar.gz -O pocl.tar.gz
    tar xzf pocl.tar.gz
    cd pocl-${POCL_VERSION} || exit
    mkdir -p build
    cd build || exit
    cmake -DCMAKE_INSTALL_PREFIX="$POCL_PATH" -DCMAKE_C_COMPILER=/import/sgs.scratch/vancraar/spack/opt/spack/linux-ubuntu20.04-skylake/gcc-9.4.0/llvm-14.0.6-s7bbf6lgiqt47pxskcquoxj7fpttlyxs/bin/clang -DCMAKE_CXX_COMPILER=/import/sgs.scratch/vancraar/spack/opt/spack/linux-ubuntu20.04-skylake/gcc-9.4.0/llvm-14.0.6-s7bbf6lgiqt47pxskcquoxj7fpttlyxs/bin/clang++  -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_CUDA=OFF -DENABLE_TCE=OFF -DENABLE_HSA=OFF -DLLVM_PATH=/import/sgs.scratch/vancraar/spack/opt/spack/linux-ubuntu20.04-skylake/gcc-9.4.0/llvm-14.0.6-s7bbf6lgiqt47pxskcquoxj7fpttlyxs -DWITH_LLVM_CONFIG=/import/sgs.scratch/vancraar/spack/opt/spack/linux-ubuntu20.04-skylake/gcc-9.4.0/llvm-14.0.6-s7bbf6lgiqt47pxskcquoxj7fpttlyxs/bin/llvm-config -DINSTALL_OPENCL_HEADERS=True -DENABLE_LLVM=True -DSTATIC_LLVM=True -DENABLE_ICD=False ..
    make -j 4 install
    cd ../..
    rm pocl.tar.gz*
fi