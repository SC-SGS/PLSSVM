#!/usr/bin/bash

set -x
set -e

GCC_VERSION=12.2.0

if [ -n "$1" ] ; then
  GCC_ROOT_DIR=$1
else
  GCC_ROOT_DIR=gcc-${GCC_VERSION}_install
fi

if [ ! -d gcc-$GCC_VERSION ] ; then
  wget ftp://ftp.fu-berlin.de/unix/languages/gcc/releases/gcc-$GCC_VERSION/gcc-$GCC_VERSION.tar.xz
  tar xf gcc-$GCC_VERSION.tar.xz
  rm gcc-$GCC_VERSION.tar.xz
fi

cd gcc-$GCC_VERSION
./contrib/download_prerequisites
cd ..

mkdir -p build-gcc-$GCC_VERSION
cd build-gcc-$GCC_VERSION
../gcc-$GCC_VERSION/configure --prefix="$GCC_ROOT_DIR" --disable-nls --enable-languages=c,c++ --disable-multilib
make -j "$(nproc)"
make install
cd ..
rm -rf gcc-$GCC_VERSION
rm -rf build-gcc-$GCC_VERSION


