# Least-Squares Support-Vector Machine

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e780a63075ce40c29c49d3df4f57c2af)](https://www.codacy.com/gh/SC-SGS/PLSSVM/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=SC-SGS/PLSSVM&amp;utm_campaign=Badge_Grade) &ensp; [![Generate documentation](https://github.com/SC-SGS/PLSSVM/actions/workflows/documentation.yml/badge.svg)](https://sc-sgs.github.io/PLSSVM/) &ensp; [![Build Status Linux CPU + GPU](https://simsgs.informatik.uni-stuttgart.de/jenkins/buildStatus/icon?job=PLSSVM%2FMultibranch-Github%2Fmain&subject=Linux+CPU/GPU)](https://simsgs.informatik.uni-stuttgart.de/jenkins/view/PLSSVM/job/PLSSVM/job/Multibranch-Github/job/main/) &ensp; [![Windows CPU](https://github.com/SC-SGS/PLSSVM/actions/workflows/msvc_windows.yml/badge.svg)](https://github.com/SC-SGS/PLSSVM/actions/workflows/msvc_windows.yml) 

Implementation of a parallel [least-squares support-vector machine](https://en.wikipedia.org/wiki/Least-squares_support-vector_machine) using multiple different backends.
The currently available backends are:
  - [OpenMP](https://www.openmp.org/)
  - [CUDA](https://developer.nvidia.com/cuda-zone)
  - [OpenCL](https://www.khronos.org/opencl/)
  - [SYCL](https://www.khronos.org/sycl/)

## Getting Started

### Dependencies

General dependencies: 
  - a C++17 capable compiler (e.g. [`gcc`](https://gcc.gnu.org/) or [`clang`](https://clang.llvm.org/))
  - [CMake](https://cmake.org/) 3.18 or newer
  - [cxxopts](https://github.com/jarro2783/cxxopts), [fast_float](https://github.com/fastfloat/fast_float) and [{fmt}](https://github.com/fmtlib/fmt) (all three are automatically build during the CMake configuration if they couldn't be found using the respective `find_package` call)
  - [GoogleTest](https://github.com/google/googletest) if testing is enabled (automatically build during the CMake configuration if `find_package(GTest)` wasn't successful)
  - [doxygen](https://www.doxygen.nl/index.html) if documentation generation is enabled
  - [OpenMP](https://www.openmp.org/) 4.0 or newer (optional) to speed-up file parsing

Additional dependencies for the OpenMP backend:
  - compiler with OpenMP support

Additional dependencies for the CUDA backend:
  - CUDA SDK
  - either NVIDIA [`nvcc`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) or [`clang` with CUDA support enabled](https://llvm.org/docs/CompileCudaWithLLVM.html)

Additional dependencies for the OpenCL backend:
  - OpenCL runtime and header files

Additional dependencies for the SYCL backend:
  - the code must be compiled with a SYCL capable compiler; currently tested with [DPC++](https://github.com/intel/llvm) and [hipSYCL](https://github.com/illuhad/hipSYCL)

Additional dependencies if `PLSSVM_ENABLE_TESTING` and `PLSSVM_GENERATE_TEST_FILE` are both set to `ON`:
  - [Python3](https://www.python.org/) with the [`argparse`](https://docs.python.org/3/library/argparse.html) and [`sklearn`](https://scikit-learn.org/stable/) modules


### Building

Building the library can be done using the normal CMake approach:

```bash
> git clone git@gitlab-sim.informatik.uni-stuttgart.de:vancraar/Bachelor-Code.git SVM
> cd SVM/SVM
> mkdir build && cd build
> cmake -DPLSSVM_TARGET_PLATFORMS="..." [optional_options] ..
> cmake --build .
```

#### Target Platform Selection

The **required** CMake option `PLSSVM_TARGET_PLATFORMS` is used to determine for which targets the backends should be compiled.
Valid targets are:
  - `cpu`: compile for the CPU; **no** architectural specifications  is allowed
  - `nvidia`: compile for NVIDIA GPUs; **at least one** architectural specification is necessary, e.g. `nvidia:sm_86,sm_70`
  - `amd`: compile for AMD GPUs; **at least one** architectural specification is necessary, e.g. `amd:gfx906`
  - `intel`: compile for Intel GPUs; **no** architectural specification is allowed

At least one of the above targets must be present.

To retrieve the architectural specification, given an NVIDIA or AMD GPU name, a simple Python3 script `utility/gpu_name_to_arch.py` is provided
(requiring Python3 [`argparse`](https://docs.python.org/3/library/argparse.html) as dependency):

```bash
> python3 utility/gpu_name_to_arch.py --help
usage: gpu_name_to_arch.py [-h] [--name NAME]

optional arguments:
  -h, --help   show this help message and exit
  --name NAME  the full name of the GPU (e.g. GeForce RTX 3080)
```

Example invocations:

```bash
> python3 utility_scripts/gpu_name_to_arch.py --name "GeForce RTX 3080"
sm_86
> python3 utility_scripts/gpu_name_to_arch.py --name "Radeon VII"
gfx906
```

If no GPU name is provided, the script tries to automatically detect any NVIDIA or AMD GPU
(requires the Python3 dependencies [`GPUtil`](https://pypi.org/project/GPUtil/) and [`pyamdgpuinfo`](https://pypi.org/project/pyamdgpuinfo/)).

If the architectural information for the requested GPU could not be retrieved, one option would be to have a look at:
  - for NVIDIA GPUs:  [Your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
  - for AMD GPUs: [ROCm Documentation](https://github.com/RadeonOpenCompute/ROCm_Documentation/blob/master/ROCm_Compiler_SDK/ROCm-Native-ISA.rst)


#### Optional CMake Options

The `[optional_options]` can be one or multiple of:

  - `PLSSVM_ENABLE_OPENMP_BACKEND=ON|OFF|AUTO` (default: `AUTO`):
    - `ON`: check for the OpenMP backend and fail if not available
    - `AUTO`: check for the OpenMP backend but **do not** fail if not available
    - `OFF`: do not check for the OpenMP backend
  - `PLSSVM_ENABLE_CUDA_BACKEND=ON|OFF|AUTO` (default: `AUTO`):
    - `ON`: check for the CUDA backend and fail if not available
    - `AUTO`: check for the CUDA backend but **do not** fail if not available
    - `OFF`: do not check for the CUDA backend
  - `PLSSVM_ENABLE_OPENCL_BACKEND=ON|OFF|AUTO` (default: `AUTO`):
    - `ON`: check for the OpenCL backend and fail if not available
    - `AUTO`: check for the OpenCL backend but **do not** fail if not available
    - `OFF`: do not check for the OpenCL backend
  - `PLSSVM_ENABLE_SYCL_BACKEND=ON|OFF|AUTO` (default: `AUTO`):
    - `ON`: check for the SYCL backend and fail if not available
    - `AUTO`: check for the SYCL backend but **do not** fail if not available
    - `OFF`: do not check for the SYCL backend

**Attention:** at least one backend must be enabled and available!

  - `PLSSVM_ENABLE_ASSERTS=ON|OFF` (default: `OFF`): enables custom assertions regardless whether the `DEBUG` macro is defined or not
  - `PLSSVM_THREAD_BLOCK_SIZE` (default: `16`): set a specific thread block size used in the GPU kernels (for fine-tuning optimizations)
  - `PLSSVM_INTERNAL_BLOCK_SIZE` (default: `6`: set a specific internal block size used in the GPU kernels (for fine-tuning optimizations)
  - `PLSSVM_EXECUTABLES_USE_SINGLE_PRECISION` (default: `OFF`): enables single precision calculations instead of double precision for the `svm-train` and `svm-predict` executables
  - `PLSSVM_ENABLE_LTO=ON|OFF` (default: `ON`): enable interprocedural optimization (IPO/LTO) if supported by the compiler
  - `PLSSVM_ENABLE_DOCUMENTATION=ON|OFF` (default: `OFF`): enable the `doc` target using doxygen
  - `PLSSVM_ENABLE_TESTING=ON|OFF` (default: `ON`): enable testing using GoogleTest and ctest
  - `PLSSVM_GENERATE_TIMING_SCRIPT=ON|OFF` (default: `OFF`): configure a timing script usable for performance measurement

If `PLSSVM_ENABLE_TESTING` is set to `ON`, the following options can also be set:
  - `PLSSVM_GENERATE_TEST_FILE=ON|OFF` (default: `ON`): automatically generate test files
    - `PLSSVM_TEST_FILE_NUM_DATA_POINTS` (default: `5000`): the number of data points in the test file

If the SYCL backend is available and DPC++ is used, the option `PLSSVM_SYCL_DPCPP_USE_LEVEL_ZERO` can be used to select Level-Zero as the
DPC++ backend instead of OpenCL.
To use DPC++ as compiler simply set the `CMAKE_CXX_COMPILER` to the respective DPC++ clang path during CMake invocation.

### Running the tests

To run the tests after building the library (with `PLSSVM_ENABLE_TESTING` set to `ON`) use:

```bash
> ctest
```

### Generating test coverage results

To enable the generation of test coverage reports using `locv` the library must be compiled using the custom `Coverage` `CMAKE_BUILD_TYPE`.
Additionally, it's advisable to use smaller test files to shorten the `ctest` step.

```bash
> cmake -DCMAKE_BUILD_TYPE=Coverage -DPLSSVM_TARGET_PLATFORMS="..." \
        -DPLSSVM_TEST_FILE_NUM_DATA_POINTS=100 \
        -DPLSSVM_TEST_FILE_NUM_FEATURES=50 ..
> cmake --build . -- coverage
```

The resulting `html` coverage report is located in the `coverage` folder in the build directory.

### Creating the documentation

If doxygen is installed and `PLSSVM_ENABLE_DOCUMENTATION` is set to `ON` the documentation can be build using
```bash
> make doc
```
The documentation of the current state of the main branch can be found [here](https://sc-sgs.github.io/PLSSVM/).

## Installing

The library supports the `install` target:

```bash
> cmake --build . -- install
```

## Usage

### Generating data

The repository comes with a Python3 script (in the `data/` directory) to simply generate arbitrarily large data sets.

In order to use all functionality, the following Python3 modules must be installed:
[`argparse`](https://docs.python.org/3/library/argparse.html), [`numpy`](https://pypi.org/project/numpy/),
[`pandas`](https://pypi.org/project/pandas/), [`sklearn`](https://scikit-learn.org/stable/),
[`arff`](https://pypi.org/project/arff/), [`matplotlib`](https://pypi.org/project/matplotlib/) and
[`mpl_toolkits`](https://pypi.org/project/matplotlib/)

```bash
> python3 generate_data.py --help
usage: generate_data.py [-h] --output OUTPUT --format FORMAT [--problem PROBLEM] --samples SAMPLES [--test_samples TEST_SAMPLES] --features FEATURES [--plot]

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       the output file to write the samples to (without extension)
  --format FORMAT       the file format; either arff or libsvm
  --problem PROBLEM     the problem to solve; one of: blobs, blobs_merged, planes, planes_merged, ball
  --samples SAMPLES     the number of training samples to generate
  --test_samples TEST_SAMPLES
                        the number of test samples to generate; default: 0
  --features FEATURES   the number of features per data point
  --plot                plot training samples; only possible if 0 < samples <= 2000 and 1 < features <= 3
```

An example invocation generating a data set consisting of blobs with 1000 data points with 200 features each could look like:

```bash
> python3 generate_data.py --ouput data_file --format libsvm --problem blobs --samples 1000 --features 200
```

### Training

```bash
> ./svm-train --help
LS-SVM with multiple (GPU-)backends
Usage:
  ./svm-train [OPTION...] training_set_file [model_file]

  -t, --kernel_type arg         set type of kernel function.
                                         0 -- linear: u'*v
                                         1 -- polynomial: (gamma*u'*v + coef0)^degree
                                         2 -- radial basis function: exp(-gamma*|u-v|^2) (default: 0)
  -d, --degree arg              set degree in kernel function (default: 3)
  -g, --gamma arg               set gamma in kernel function (default: 1 / num_features)
  -r, --coef0 arg               set coef0 in kernel function (default: 0)
  -c, --cost arg                set the parameter C (default: 1)
  -e, --epsilon arg             set the tolerance of termination criterion (default: 0.001)
  -b, --backend arg             choose the backend: openmp|cuda|opencl|sycl (default: openmp)
  -p, --target_platform arg     choose the target platform: automatic|cpu|gpu_nvidia|gpu_amd|gpu_intel (default: automatic)
  -q, --quiet                   quiet mode (no outputs)
  -h, --help                    print this helper message
      --input training_set_file

      --model model_file
```

An example invocation using the CUDA backend could look like:

```bash
> ./svm-train --backend cuda --input /path/to/data_file
```

Another example targeting NVIDIA GPUs using the SYCL backend looks like:

```bash
> ./svm-train --backend sycl --target_platform gpu_nvidia --input /path/to/data_file
```

The `--target_platform=automatic` flags works for the different backends as follows:

- `OpenMP`: always selects a CPU
- `CUDA`: always selects an NVIDIA GPU (if no NVIDIA GPU is available, throws an exception)
- `OpenCL`: tries to find available devices in the following order: NVIDIA GPUs 🠦 AMD GPUs 🠦 Intel GPUs 🠦 CPU
- `SYCL`: tries to find available devices in the following order: NVIDIA GPUs 🠦 AMD GPUs 🠦 Intel GPUs 🠦 CPU

### Predicting

```bash
> ./svm-predict --help
LS-SVM with multiple (GPU-)backends
Usage:
  ./svm-predict [OPTION...] test_file model_file [output_file]

  -b, --backend arg          choose the backend: openmp|cuda|opencl|sycl (default: openmp)
  -p, --target_platform arg  choose the target platform: automatic|cpu|gpu_nvidia|gpu_amd|gpu_intel (default: automatic)
  -q, --quiet                quiet mode (no outputs)
  -h, --help                 print this helper message
      --test test_file
      --model model_file
      --output output_file
```

An example invocation could look like:

```bash
> ./svm-predict --backend cuda --test /path/to/test_file --model /path/to/model_file
```

Another example targeting NVIDIA GPUs using the SYCL backend looks like:

```bash
> ./svm-predict --backend sycl --target_platform gpu_nvidia --test /path/to/test_file --model /path/to/model_file
```

The `--target_platform=automatic` flags works like in the training (`./svm-train`) case.

## Example code for usage as library

A simple C++ program (`main.cpp`) using this library could look like:

```cpp
#include "plssvm/core.hpp"

#include <exception>
#include <iostream>
#include <vector>

int main(i) {
    try {
        // parse SVM parameter from command line
        plssvm::parameter<double> params;
        params.backend = plssvm::backend_type::cuda;

        params.parse_train_file("train_file.libsvm");

        // create C-SVM (based on selected backend)
        auto svm = plssvm::make_csvm(params);

        // learn
        svm->learn();

        // get accuracy
        std::cout << "accuracy: " << svm->accuracy() << std::endl;

        // predict
        std::vector<double> point = { ... };
        std::cout << "label: " << svm->predict(point) << std::endl;

        // write model file to disk
        svm->write_model("model_file.libsvm");
    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
```

With a corresponding minimal CMake file:

```cmake
cmake_minimum_required(VERSION 3.16)

project(LibraryUsageExample
        LANGUAGES CXX)

find_package(plssvm CONFIG REQUIRED)

add_executable(prog main.cpp)

target_compile_features(prog PUBLIC cxx_std_17)
target_link_libraries(prog PUBLIC plssvm::svm-all)
```


## License

The PLSSVM library is distributed under the MIT [license](https://github.com/SC-SGS/PLSSVM/blob/main/LICENSE.md).
