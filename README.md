# Least-Squares Support-Vector Machine

Implementation of a parallel [least-squares support-vector machine](https://en.wikipedia.org/wiki/Least-squares_support-vector_machine) using multiple different backends.
The currently available backends are:
- [OpenMP](https://www.openmp.org/)
- [CUDA](https://developer.nvidia.com/cuda-zone)
- [OpenCL](https://www.khronos.org/opencl/)

TODO: description, UPDATES

## Getting Started

### Dependencies

General dependencies:
- a C++17 capable compiler (e.g. [`gcc`](https://gcc.gnu.org/) or [`clang`](https://clang.llvm.org/))
- [CMake](https://cmake.org/) 3.18 or newer
- [cxxopts](https://github.com/jarro2783/cxxopts), [fast_float](https://github.com/fastfloat/fast_float) and [{fmt}](https://github.com/fmtlib/fmt) (all three are automatically build during the CMake configuration)
- [GoogleTest](https://github.com/google/googletest) if testing is enabled (automatically build during the CMake configuration)
- [doxygen](https://www.doxygen.nl/index.html) if documentation generation is enabled

Additional dependencies for the OpenMP backend:
- compiler with OpenMP support

Additional dependencies for the CUDA backend:
- CUDA SDK
- either NVIDIA [`nvcc`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) or [`clang` with CUDA support enabled](https://llvm.org/docs/CompileCudaWithLLVM.html)

Additional dependencies for the OpenCL backend:
- OpenCL runtime and header files

Additional dependencies if `PLSSVM_ENABLE_TESTING` and `PLSSVM_GENERATE_TEST_FILE` are both set to `ON`:
- [Python3](https://www.python.org/) with the following modules installed: <br>argparse, numpy, pandas, sklearn, arff, matplotlib, mpl_toolkits

### Installing

Building the library can be done using the normal CMake approach:

```bash
> git clone git@gitlab-sim.informatik.uni-stuttgart.de:vancraar/Bachelor-Code.git SVM
> cd SVM/SVM
> mkdir build && cd build
> cmake [options] ..
> cmake --build .
```

Where `[options]` can be one or multiple of:

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

**Attention:** at least one backend must be enabled and available!    

- `PLSSVM_ENABLE_LTO=ON|OFF` (default: `ON`): enable interprocedural optimization (IPO/LTO) if supported by the compiler
- `PLSSVM_ENABLE_DOCUMENTATION=ON|OFF` (default: `OFF`): enable the `doc` target using doxygen
- `PLSSVM_ENABLE_TESTING=ON|OFF` (default: ON): enable testing using GoogleTest and ctest

If `PLSSVM_ENABLE_TESTING` is set to `ON`, the following options can also be set:
- `PLSSVM_GENERATE_TEST_FILE=ON|OFF` (default: `ON`): automatically generate test files 
    - `PLSSVM_TEST_FILE_NUM_DATA_POINTS` (default: `5000`): the number of data points in the test file
    - `PLSSVM_TEST_FILE_NUM_FEATURES` (default: `2000`): the number of features per data point

If the CUDA backend is available, the option `PLSSVM_CUDA_ARCHITECTURES` can be used to select CUDA architectures to compile for:
- per default CMake tries to find all available NVIDIA GPUs and sets the architecture values accordingly
- additional architecture values can be set using `PLSSVM_CUDA_ARCHITECTURES` (e.g. `-DPLSSVM_CUDA_ARCHITECTURES=60;70`)
- if **no** architectures are set, the CUDA backend is compiled for a few default architectures: `52;60;61;70;75;86`

### Running the tests

To run the tests after building the library (with `PLSSVM_ENABLE_TESTING` set to `ON`) use:

```bash
> ctest
```

## Usage

### Generating data

The repository comes with a python3 script (in the `data/` directory) to simply generate arbitrarily large data sets:

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
  -b, --backend arg             chooses the backend openmp|cuda|opencl (default: openmp)
  -q, --quiet                   quiet mode (no outputs)
  -h, --help                    print this helper message
      --input training_set_file
                                
      --model model_file 
```

An example invocation using the cuda backend could look like:

```bash
> ./svm-train --backend cuda --input /path/to/data_file
```

### Predict

TODO: write


## Example code using this library

A simple C++ program using this library could look like:

```cpp
#include "plssvm/core.hpp"

int main(int argc, char *argv[]) {
    // parse SVM parameter from command line
    plssvm::parameter<double> params{ argc, argv };

    // create SVM (based on selected backend)
    auto svm = plssvm::make_SVM(params);
    
    // learn
    svm->learn(params.input_filename, params.model_filename);
    return 0;
}
```