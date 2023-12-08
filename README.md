![![PLSSVM](../resources/logo_245x150.png)](docs/resources/logo_245x150.png)

# PLSSVM - Parallel Least Squares Support Vector Machine

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e780a63075ce40c29c49d3df4f57c2af)](https://www.codacy.com/gh/SC-SGS/PLSSVM/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=SC-SGS/PLSSVM&amp;utm_campaign=Badge_Grade) &ensp; [![Generate documentation](https://github.com/SC-SGS/PLSSVM/actions/workflows/documentation.yml/badge.svg)](https://sc-sgs.github.io/PLSSVM/) 

## Table of Contents

- [Introduction To PLSSVM](#introduction-to-plssvm)
- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Building PLSSVM](#building-plssvm)
  - [Running the Tests](#running-the-tests)
  - [Generating Test Coverage Results](#generating-test-coverage-results)
  - [Creating the Documentation](#creating-the-documentation)
  - [Installing](#installing)
- [Usage](#usage)
  - [Generating Artificial Data](#generating-artificial-data)
  - [Training using `plssvm-train`](#training-using-plssvm-train)
  - [Predicting using `plssvm-predict`](#predicting-using-plssvm-predict)
  - [Data Scaling using `plssvm-scale`](#data-scaling-using-plssvm-scale)
  - [Example Code for PLSSVM Used as a Library](#example-code-for-plssvm-used-as-a-library)
  - [Example Using the Python Bindings Available For PLSSVM](#example-using-the-python-bindings-available-for-plssvm)
- [Citing PLSSVM](#citing-plssvm)
- [License](#license)

## Introduction to PLSSVM

A [Support Vector Machine (SVM)](https://en.wikipedia.org/wiki/Support-vector_machine) is a supervised machine learning model.
In its basic form SVMs are used for binary classification tasks.
Their fundamental idea is to learn a hyperplane which separates the two classes best, i.e., where the widest possible margin around its decision boundary is free of data.
This is also the reason, why SVMs are also called "large margin classifiers".
To predict to which class a new, unseen data point belongs, the SVM simply has to calculate on which side of the previously calculated hyperplane the data point lies.
This is very efficient since it only involves a single scalar product of the size corresponding to the numer of features of the data set.

<p align="center">
  <img alt="strong scaling CPU" src=".figures/support_vector_machine.png" width="50%">
</p>

However, normal SVMs suffer in their potential parallelizability.
Determining the hyperplane boils down to solving a convex quadratic problem.
For this, most SVM implementations use Sequential Minimal Optimization (SMO), an inherently sequential algorithm.
The basic idea of this algorithm is that it takes a pair of data points and calculates the hyperplane between them.
Afterward, two new data points are selected and the existing hyperplane is adjusted accordingly.
This procedure is repeat until a new adjustment would be smaller than some epsilon greater than zero.

Some SVM implementations try to harness some parallelization potential by not drawing point pairs but group of points.
In this case, the hyperplane calculation inside this group is parallelized.
However, even then modern highly parallel hardware can not be utilized efficiently.

Therefore, we implemented a version of the original proposed SVM called [Least Squares Support Vector Machine (LS-SVM)](https://en.wikipedia.org/wiki/Least-squares_support-vector_machine).
The LS-SVMs reformulated the original problem such that it boils down to solving a system of linear equations.
For this kind of problem many highly parallel algorithms and implementations are known.
We decided to use the [Conjugate Gradient (CG)](https://en.wikipedia.org/wiki/Conjugate_gradient_method) to solve the system of linear equations.

We support multi-class classification using the one vs. all (also one vs. rest or OAA).
Therefore, our CG algorithm solves multiple right-hand sides simultaneously. 
This has the implication that, except for binary classification, our model file isn't LIBSVM conform, since for each support vector #classes many weights must be saved instead of only #classes - 1.

Since one of our main goals was performance, we parallelized the implicit matrix-vector multiplication inside the CG algorithm.
To do so, we use multiple different frameworks to be able to target a broad variety of different hardware platforms.
The currently available frameworks (also called backends in our PLSSVM implementation) are:

- [OpenMP](https://www.openmp.org/)
- [CUDA](https://developer.nvidia.com/cuda-zone)
- [HIP](https://github.com/ROCm-Developer-Tools/HIP) (only tested on AMD GPUs)
- [OpenCL](https://www.khronos.org/opencl/)
- [SYCL](https://www.khronos.org/sycl/) (tested implementations are [DPC++](https://github.com/intel/llvm) and [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (formerly known as hipSYCL); specifically the versions [sycl-nightly/20231201](https://github.com/intel/llvm/tree/sycl-nightly/20230110) and AdaptiveCpp release [v23.10.0](https://github.com/AdaptiveCpp/AdaptiveCpp/releases/tag/v23.10.0))

## Getting Started

### Dependencies

General dependencies:

- a C++17 capable compiler (e.g. [`gcc`](https://gcc.gnu.org/) or [`clang`](https://clang.llvm.org/))
- [CMake](https://cmake.org/) 3.21 or newer
- [cxxopts ≥ v3.1.1](https://github.com/jarro2783/cxxopts), [fast_float ≥ v3.10.0](https://github.com/fastfloat/fast_float), [{fmt} ≥ v10.1.1](https://github.com/fmtlib/fmt), and [igor](https://github.com/bluescarni/igor) (all four are automatically build during the CMake configuration if they couldn't be found using the respective `find_package` call)
- [GoogleTest ≥ v1.14.0](https://github.com/google/googletest) if testing is enabled (automatically build during the CMake configuration if `find_package(GTest)` wasn't successful)
- [doxygen](https://www.doxygen.nl/index.html) if documentation generation is enabled
- [Pybind11 ≥ v2.11.1](https://github.com/pybind/pybind11) if Python bindings are enabled
- [OpenMP](https://www.openmp.org/) 4.0 or newer (optional) to speed-up library utilities (like file parsing)
- multiple Python modules used in the utility scripts, to install all modules use `pip install --user -r install/python_requirements.txt`

Additional dependencies for the OpenMP backend:

- compiler with OpenMP support

Additional dependencies for the CUDA backend:

- CUDA SDK
- either NVIDIA [`nvcc`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) or [`clang` with CUDA support enabled](https://llvm.org/docs/CompileCudaWithLLVM.html)

Additional dependencies for the HIP backend:

- working ROCm and HIP installation
- [clang with HIP support](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-FAQ.html)

Additional dependencies for the OpenCL backend:

- OpenCL runtime and header files

Additional dependencies for the SYCL backend:

- the code must be compiled with a SYCL capable compiler; currently tested with [DPC++](https://github.com/intel/llvm) and [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)

Additional dependencies if `PLSSVM_ENABLE_TESTING` and `PLSSVM_GENERATE_TEST_FILE` are both set to `ON`:

- [Python3](https://www.python.org/) with the [`argparse`](https://docs.python.org/3/library/argparse.html), [`timeit`](https://docs.python.org/3/library/timeit.html), [`sklearn`](https://scikit-learn.org/stable/), and [`humanize`](https://pypi.org/project/humanize/) modules

### Building PLSSVM

To download PLSSVM use:

```bash
git clone https://github.com/SC-SGS/PLSSVM.git
cd PLSSVM 
```

We provided a Python3 requirements file to install all *necessary* Python3 dependencies:

```bash
pip install -r install/python_requirements.txt
```

Building the library can be done using the normal CMake approach:

```bash
mkdir build && cd build 
cmake -DPLSSVM_TARGET_PLATFORMS="..." [optional_options] .. 
cmake --build . -j
```

#### (Automatic) Target Platform Selection

The CMake option `PLSSVM_TARGET_PLATFORMS` is used to determine for which targets the backends should be compiled.
Valid targets are:

- `cpu`: compile for the CPU; an **optional** architectural specifications is allowed but only used when compiling with DPC++, e.g., `cpu:avx2`
- `nvidia`: compile for NVIDIA GPUs; **at least one** architectural specification is necessary, e.g., `nvidia:sm_86,sm_70`
- `amd`: compile for AMD GPUs; **at least one** architectural specification is necessary, e.g., `amd:gfx906`
- `intel`: compile for Intel GPUs; **at least one** architectural specification is necessary, e.g., `intel:skl`

At least one of the above targets must be present. If the option `PLSSVM_TARGET_PLATFORMS` is not present, the targets 
are automatically determined using the Python3 `utility_scripts/plssvm_target_platforms.py` script (required Python3 dependencies:
[`argparse`](https://docs.python.org/3/library/argparse.html), [`py-cpuinfo`](https://pypi.org/project/py-cpuinfo/),
[`GPUtil`](https://pypi.org/project/GPUtil/), [`pyamdgpuinfo`](https://pypi.org/project/pyamdgpuinfo/), and
[`pylspci`](https://pypi.org/project/pylspci/)).

Note that when using DPC++ only a single architectural specification for `cpu`, `nvidia` or `amd` is allowed and that
automatically retrieving AMD GPU information on Windows is currently not supported due to `pyamdgpuinfo` limitations.


```bash
python3 utility_scripts/plssvm_target_platforms.py --help
```
```
usage: plssvm_target_platforms.py [-h] [--quiet]

optional arguments:
  -h, --help  show this help message and exit
  --quiet     only output the final PLSSVM_TARGET_PLATFORMS string
```

Example invocation:

```bash
python3 utility_scripts/plssvm_target_platforms.py
```
```
Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz: {'avx512': True, 'avx2': True, 'avx': True, 'sse4_2': True}

Found 1 NVIDIA GPU(s):
  1x NVIDIA GeForce RTX 3080: sm_86

Possible -DPLSSVM_TARGET_PLATFORMS entries:
cpu:avx512;nvidia:sm_86
```


or with the `--quiet` flag provided:


```bash
python3 utility_scripts/plssvm_target_platforms.py --quiet
```
```
cpu:avx512;intel:dg1
```

If the architectural information for the requested GPU could not be retrieved, one option would be to have a look at:

- for NVIDIA GPUs:  [Your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
- for AMD GPUs: [clang AMDGPU backend usage](https://llvm.org/docs/AMDGPUUsage.html)
- for Intel GPUs and CPUs: [Ahead of Time Compilation](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html) and [Intel graphics processor table](https://dgpu-docs.intel.com/devices/hardware-table.html)


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

- `PLSSVM_ENABLE_HIP_BACKEND=ON|OFF|AUTO` (default: `AUTO`):
  - `ON`: check for the HIP backend and fail if not available
  - `AUTO`: check for the HIP backend but **do not** fail if not available
  - `OFF`: do not check for the HIP backend

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
- `PLSSVM_USE_FLOAT_AS_REAL_TYPE=ON|OFF` (default: `OFF`): use `float` as real_type instead of `double`
- `PLSSVM_THREAD_BLOCK_SIZE` (default: `8`): set a specific thread block size used in the GPU kernels (for fine-tuning optimizations)
- `PLSSVM_INTERNAL_BLOCK_SIZE` (default: `4`: set a specific internal block size used in the GPU kernels (for fine-tuning optimizations)
- `PLSSVM_ENABLE_LTO=ON|OFF` (default: `ON`): enable interprocedural optimization (IPO/LTO) if supported by the compiler
- `PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE=ON|OFF` (default: `ON`): enforce the maximum (device) memory allocation size for the plssvm::solver_type::automatic solver
- `PLSSVM_ENABLE_DOCUMENTATION=ON|OFF` (default: `OFF`): enable the `doc` target using doxygen
- `PLSSVM_ENABLE_PERFORMANCE_TRACKING=ON|OFF` (default: `OFF`): enable gathering performance characteristics for the three executables using YAML files; example Python3 scripts to perform performance measurements and to process the resulting YAML files can be found in the `utility_scripts/` directory (requires the Python3 modules [wrapt-timeout-decorator](https://pypi.org/project/wrapt-timeout-decorator/), [`pyyaml`](https://pyyaml.org/), and [`pint`](https://pint.readthedocs.io/en/stable/))
- `PLSSVM_ENABLE_TESTING=ON|OFF` (default: `ON`): enable testing using GoogleTest and ctest
- `PLSSVM_ENABLE_LANGUAGE_BINDINGS=ON|OFF` (default: `OFF`): enable language bindings
- `PLSSVM_STL_DEBUG_MODE_FLAGS=ON|OFF` (default: `OFF`): enable STL debug modes (**note**: changes the resulting library's ABI!)

If `PLSSVM_ENABLE_TESTING` is set to `ON`, the following options can also be set:

- `PLSSVM_GENERATE_TEST_FILE=ON|OFF` (default: `ON`): automatically generate test files
  - `PLSSVM_TEST_FILE_NUM_DATA_POINTS` (default: `5000`): the number of data points in the test file
  - `PLSSVM_TEST_FILE_NUM_FEATURES` (default: `2000`): the number of features per data point in the test file

If `PLSSVM_ENABLE_LANGUAGE_BINDINGS` is set to `ON`, the following option can also be set:

- `PLSSVM_ENABLE_PYTHON_BINDINGS=ON|OFF` (default: `PLSSVM_ENABLE_LANGUAGE_BINDINGS`): enable Python bindings using Pybind11; **note:** `PLSSVM_ENABLE_LANGUAGE_BINDINGS` must be set that this option has any effect

If `PLSSVM_ENABLE_PYTHON_BINDINGS` is set to `ON`, the following options can also be set:

- `PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE` (default: `std::string`): the default `label_type` used if the generic `plssvm.Model` and `plssvm.DataSet` Python classes are used

If the SYCL backend is available additional options can be set.

- `PLSSVM_ENABLE_SYCL_ADAPITVECPP_BACKEND=ON|OFF|AUTO` (default: `AUTO`):
  - `ON`: check for AdaptiveCpp as implementation for the SYCL backend and fail if not available
  - `AUTO`: check for AdaptiveCpp as implementation for the SYCL backend but **do not** fail if not available
  - `OFF`: do not check for AdaptiveCpp as implementation for the SYCL backend

- `PLSSVM_ENABLE_SYCL_DPCPP_BACKEND=ON|OFF|AUTO` (default: `AUTO`):
  - `ON`: check for DPC++ as implementation for the SYCL backend and fail if not available
  - `AUTO`: check for DPC++ as implementation for the SYCL backend but **do not** fail if not available
  - `OFF`: do not check for DPC++ as implementation for the SYCL backend

To use DPC++ for SYCL, simply set the `CMAKE_CXX_COMPILER` to the respective DPC++ clang executable during CMake invocation.

If the SYCL implementation is DPC++ the following additional options are available:

- `PLSSVM_SYCL_BACKEND_DPCPP_ENABLE_AOT` (default: `ON`): enable Ahead-of-Time (AOT) compilation for the specified target platforms
- `PLSSVM_SYCL_BACKEND_DPCPP_USE_LEVEL_ZERO` (default: `ON`): use DPC++'s Level-Zero backend instead of its OpenCL backend **(only available if a CPU or Intel GPU is targeted)**
- `PLSSVM_SYCL_BACKEND_DPCPP_GPU_AMD_USE_HIP` (default: `ON`): use DPC++'s HIP backend instead of its OpenCL backend for AMD GPUs **(only available if an AMD GPU is targeted)**

If the SYCL implementation is AdaptiveCpp the following additional option is available:

- `PLSSVM_SYCL_BACKEND_ADAPTIVECPP_USE_GENERIC_SSCP` (default: `ON`): use AdaptiveCpp's new SSCP compilation flow

If more than one SYCL implementation is available the environment variables `PLSSVM_SYCL_ADAPTIVECPP_INCLUDE_DIR` and `PLSSVM_SYCL_DPCPP_INCLUDE_DIR`
**must** be set to the respective SYCL include paths. Note that those paths **must not** be present in the `CPLUS_INCLUDE_PATH` environment variable or compilation will fail.

- `PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION` (`dpcpp`|`adaptivecpp`): specify the preferred SYCL implementation if the `sycl_implementation_type` option is set to `automatic`; additional the specified SYCL implementation is used in the `plssvm::sycl` namespace, the other implementations are available in the `plssvm::dpcpp` and `plssvm::adaptivecpp` namespace respectively

### Running the Tests

To run the tests after building the library (with `PLSSVM_ENABLE_TESTING` set to `ON`) use:

```bash
ctest
```

### Generating Test Coverage Results

To enable the generation of test coverage reports using `locv` the library must be compiled using the custom `Coverage` `CMAKE_BUILD_TYPE`.
Additionally, it's advisable to use smaller test files to shorten the `ctest` step.

```bash
cmake -DCMAKE_BUILD_TYPE=Coverage -DPLSSVM_TARGET_PLATFORMS="..." \
      -DPLSSVM_TEST_FILE_NUM_DATA_POINTS=100 \
      -DPLSSVM_TEST_FILE_NUM_FEATURES=50 ..
cmake --build . -- coverage
```

The resulting `html` coverage report is located in the `coverage` folder in the build directory.

### Creating the Documentation

If doxygen is installed and `PLSSVM_ENABLE_DOCUMENTATION` is set to `ON` the documentation can be build using

```bash
cmake --build . -- doc
```

The documentation of the current state of the main branch can be found [here](https://sc-sgs.github.io/PLSSVM/).

### Installing

The library supports the `install` target:

```bash
cmake --build . -- install
```

Afterward, the necessary exports should be performed:

```bash
export CMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}/share/plssvm/cmake:${CMAKE_PREFIX_PATH}
export MANPATH=${CMAKE_INSTALL_PREFIX}/share/man:$MANPATH

export PATH=${CMAKE_INSTALL_PREFIX}/bin:${PATH}
export LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}
```

## Usage

PLSSVM provides three executables: `plssvm-train`, `plssvm-predict`, and `plssvm-scale`.
In addition, PLSSVM can also be used as a library in third-party code.
For more information, see the respective `man` pages which are installed via `cmake --build . -- install`.

### Generating Artificial Data

The repository comes with a Python3 script (in the `utility_scripts/` directory) to simply generate arbitrarily large data sets.

In order to use all functionality, the following Python3 modules must be installed:
[`argparse`](https://docs.python.org/3/library/argparse.html), [`timeit`](https://docs.python.org/3/library/timeit.html), 
[`numpy`](https://pypi.org/project/numpy/), [`pandas`](https://pypi.org/project/pandas/),
[`sklearn`](https://scikit-learn.org/stable/), [`arff`](https://pypi.org/project/arff/),
[`matplotlib`](https://pypi.org/project/matplotlib/), [`mpl_toolkits`](https://pypi.org/project/matplotlib/),
and [`humanize`](https://pypi.org/project/humanize/).

```
usage: generate_data.py [-h] [--output OUTPUT] [--format FORMAT] [--problem PROBLEM] --samples SAMPLES [--test_samples TEST_SAMPLES] --features FEATURES [--classes CLASSES] [--plot]

options:
  -h, --help            show this help message and exit
  --output OUTPUT       the output file to write the samples to (without extension)
  --format FORMAT       the file format; either arff, libsvm, or csv
  --problem PROBLEM     the problem to solve; one of: blobs, blobs_merged, planes, ball
  --samples SAMPLES     the number of training samples to generate
  --test_samples TEST_SAMPLES
                        the number of test samples to generate; default: 0
  --features FEATURES   the number of features per data point
  --classes CLASSES     the number of classes to generate; default: 2
  --plot                plot training samples; only possible if 0 < samples <= 2000 and 1 < features <= 3
```

An example invocation generating a data set consisting of blobs with 1000 data points with 200 features each and 
4 classes could look like:

```bash
python3 generate_data.py --output data_file --format libsvm --problem blobs --samples 1000 --features 200 --classes 4
```

### Training using `plssvm-train`

```bash
./plssvm-train --help
```
```
LS-SVM with multiple (GPU-)backends
Usage:
  ./plssvm-train [OPTION...] training_set_file [model_file]

  -t, --kernel_type arg         set type of kernel function. 
                                         0 -- linear: u'*v
                                         1 -- polynomial: (gamma*u'*v + coef0)^degree 
                                         2 -- radial basis function: exp(-gamma*|u-v|^2) (default: 0)
  -d, --degree arg              set degree in kernel function (default: 3)
  -g, --gamma arg               set gamma in kernel function (default: 1 / num_features)
  -r, --coef0 arg               set coef0 in kernel function (default: 0)
  -c, --cost arg                set the parameter C (default: 1)
  -e, --epsilon arg             set the tolerance of termination criterion (default: 0.001)
  -i, --max_iter arg            set the maximum number of CG iterations (default: num_features)
  -l, --solver arg              choose the solver: automatic|cg_explicit|cg_streaming|cg_implicit (default: automatic)
  -a, --classification arg      the classification strategy to use for multi-class classification: oaa|oao (default: oaa)
  -b, --backend arg             choose the backend: automatic|openmp|cuda|hip|opencl|sycl (default: automatic)
  -p, --target_platform arg     choose the target platform: automatic|cpu|gpu_nvidia|gpu_amd|gpu_intel (default: automatic)
      --sycl_kernel_invocation_type arg
                                choose the kernel invocation type when using SYCL as backend: automatic|nd_range (default: automatic)
      --sycl_implementation_type arg
                                choose the SYCL implementation to be used in the SYCL backend: automatic|dpcpp|adaptivecpp (default: automatic)
      --performance_tracking arg
                                the output YAML file where the performance tracking results are written to; if not provided, the results are dumped to stderr
      --use_strings_as_labels   use strings as labels instead of plane numbers
      --verbosity               choose the level of verbosity: full|timing|libsvm|quiet (default: full)
  -q, --quiet                   quiet mode (no outputs regardless the provided verbosity level!)
  -h, --help                    print this helper message
  -v, --version                 print version information
      --input training_set_file
                                
      --model model_file 
```

The help message only print options available based on the CMake invocation. 
For example, if CUDA was not available during the build step, it will not show up as possible backend in the description of the `--backend` option.

The most minimal example invocation is:

```bash
./plssvm-train /path/to/data_file
```

An example invocation using the CUDA backend could look like:

```bash
./plssvm-train --backend cuda --input /path/to/data_file
```

Another example targeting NVIDIA GPUs using the SYCL backend looks like:

```bash
./plssvm-train --backend sycl --target_platform gpu_nvidia --input /path/to/data_file
```

The `--backend=automatic` option works as follows:

- if the `gpu_nvidia` target is available, check for existing backends in order `cuda` 🠦 `hip` 🠦 `opencl` 🠦 `sycl`
- otherwise, if the `gpu_amd` target is available, check for existing backends in order `hip` 🠦 `opencl` 🠦 `sycl`
- otherwise, if the `gpu_intel` target is available, check for existing backends in order `sycl` 🠦 `opencl`
- otherwise, if the `cpu` target is available, check for existing backends in order `sycl` 🠦 `opencl` 🠦 `openmp`

Note that during CMake configuration it is guaranteed that at least one of the above combinations does exist.

The `--target_platform=automatic` option works for the different backends as follows:

- `OpenMP`: always selects a CPU
- `CUDA`: always selects an NVIDIA GPU (if no NVIDIA GPU is available, throws an exception)
- `HIP`: always selects an AMD GPU (if no AMD GPU is available, throws an exception)
- `OpenCL`: tries to find available devices in the following order: NVIDIA GPUs 🠦 AMD GPUs 🠦 Intel GPUs 🠦 CPU
- `SYCL`: tries to find available devices in the following order: NVIDIA GPUs 🠦 AMD GPUs 🠦 Intel GPUs 🠦 CPU

The `--sycl_kernel_invocation_type` and `--sycl_implementation_type` flags are only used if the `--backend` is `sycl`, otherwise a warning is emitted on `stderr`.
If the `--sycl_kernel_invocation_type` is `automatic`, the `nd_range` invocation type is currently always used.
If the `--sycl_implementation_type` is `automatic`, the used SYCL implementation is determined by the `PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION` CMake flag.

### Predicting using `plssvm-predict`

```bash
./plssvm-preidct --help
```
```
LS-SVM with multiple (GPU-)backends
Usage:
  ./plssvm-preidct [OPTION...] test_file model_file [output_file]

  -b, --backend arg             choose the backend: automatic|openmp|cuda|hip|opencl|sycl (default: automatic)
  -p, --target_platform arg     choose the target platform: automatic|cpu|gpu_nvidia|gpu_amd|gpu_intel (default: automatic)
      --sycl_implementation_type arg
                                choose the SYCL implementation to be used in the SYCL backend: automatic|dpcpp|adaptivecpp (default: automatic)
      --performance_tracking arg
                                the output YAML file where the performance tracking results are written to; if not provided, the results are dumped to stderr
      --use_strings_as_labels   use strings as labels instead of plane numbers
      --verbosity               choose the level of verbosity: full|timing|libsvm|quiet (default: full)
  -q, --quiet                   quiet mode (no outputs regardless the provided verbosity level!)
  -h, --help                    print this helper message
  -v, --version                 print version information
      --test test_file          
      --model model_file        
      --output output_file
```

An example invocation could look like:

```bash
./plssvm-preidct --backend cuda --test /path/to/test_file --model /path/to/model_file
```

Another example targeting NVIDIA GPUs using the SYCL backend looks like:

```bash
./plssvm-preidct --backend sycl --target_platform gpu_nvidia --test /path/to/test_file --model /path/to/model_file
```

The `--target_platform=automatic` and `--sycl_implementation_type` flags work like in the training (`./plssvm-train`) case.

### Data Scaling using `plssvm-scale`

```bash
./plssvm-scale --help
```
```
LS-SVM with multiple (GPU-)backends
Usage:
  ./plssvm-scale [OPTION...] input_file [scaled_file]

  -l, --lower arg               lower is the lowest (minimal) value allowed in each dimension (default: -1)
  -u, --upper arg               upper is the highest (maximal) value allowed in each dimension (default: 1)
  -f, --format arg              the file format to output the scaled data set to (default: libsvm)
  -s, --save_filename arg       the file to which the scaling factors should be saved
  -r, --restore_filename arg    the file from which previous scaling factors should be loaded
      --performance_tracking arg
                                the output YAML file where the performance tracking results are written to; if not provided, the results are dumped to stderr
      --use_strings_as_labels   use strings as labels instead of plane numbers
      --verbosity               choose the level of verbosity: full|timing|libsvm|quiet (default: full)
  -q, --quiet                   quiet mode (no outputs regardless the provided verbosity level!)
  -h, --help                    print this helper message
  -v, --version                 print version information
      --input input_file        
      --scaled scaled_file
```

An example invocation could look like:

```bash
./plssvm-scale -l -0.5 -u 1.5 --input /path/to/input_file --scaled /path/to/scaled_file
```

An example invocation to scale a train and test file in the same way looks like:

```bash
./plssvm-scale -l -1.0 -u 1.0 -s scaling_parameter.txt train_file.libsvm train_file_scaled.libsvm
./plssvm-scale -r scaling_parameter.txt test_file.libsvm test_file_scaled.libsvm
```

### Example Code for PLSSVM Used as a Library

A simple C++ program (`main.cpp`) using PLSSVM as library could look like:

```cpp
#include "plssvm/core.hpp"

#include <exception>
#include <iostream>
#include <vector>

int main() {
    try {
        // create a new C-SVM parameter set, explicitly overriding the default kernel function
        const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::polynomial };

        // create two data sets: one with the training data scaled to [-1, 1] 
        // and one with the test data scaled like the training data
        const plssvm::data_set train_data{ "train_file.libsvm", { -1.0, 1.0 } };
        const plssvm::data_set test_data{ "test_file.libsvm", train_data.scaling_factors()->get() };

        // create C-SVM using the default backend and the previously defined parameter
        const auto svm = plssvm::make_csvm(params);

        // fit using the training data, (optionally) set the termination criterion
        const plssvm::model model = svm->fit(train_data, plssvm::epsilon = 10e-6);

        // get accuracy of the trained model
        const double model_accuracy = svm->score(model);
        std::cout << "model accuracy: " << model_accuracy << std::endl;

        // predict the labels
        const std::vector<int> predicted_label = svm->predict(model, test_data);
        // output a more complete classification report
        const std::vector<int> &correct_label = test_data.labels().value();
        std::cout << plssvm::classification_report{ correct_label, predicted_label } << std::endl;

        // write model file to disk
        model.save("model_file.libsvm");
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
target_link_libraries(prog PUBLIC plssvm::plssvm-all)
```

### Example Using the Python Bindings Available For PLSSVM

Roughly the same can be achieved using our Python bindings with the following Python script (note: needs [`sklearn`](https://scikit-learn.org/stable/)):

```python
import plssvm
from sklearn.metrics import classification_report

try:
    # create a new C-SVM parameter set, explicitly overriding the default kernel function
    params = plssvm.Parameter(kernel_type=plssvm.KernelFunctionType.POLYNOMIAL)
  
    # create two data sets: one with the training data scaled to [-1, 1]
    # and one with the test data scaled like the training data
    train_data = plssvm.DataSet("train_data.libsvm", scaling=(-1.0, 1.0))
    test_data = plssvm.DataSet("test_data.libsvm", scaling=train_data.scaling_factors())
  
    # create C-SVM using the default backend and the previously defined parameter
    svm = plssvm.CSVM(params)
  
    # fit using the training data, (optionally) set the termination criterion
    model = svm.fit(train_data, epsilon=10e-6)
  
    # get accuracy of the trained model
    model_accuracy = svm.score(model)
    print("model accuracy: {}".format(model_accuracy))
  
    # predict labels
    predicted_label = svm.predict(model, test_data)
    # output a more complete classification report
    correct_label = test_data.labels()
    print(classification_report(correct_label, predicted_label))
  
    # write model file to disk
    model.save("model_file.libsvm")
except plssvm.PLSSVMError as e:
    print(e)
except RuntimeError as e:
    print(e)
```

**Note:** it may be necessary to set the environment variable `PYTHONPATH` to the `lib` folder in the PLSSVM install path.

We also provide Python bindings for a `plssvm.SVC` class that offers the same interface as the [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) class.
Note that currently not all functionality has been implemented in PLSSVM.
The respective functions will throw a Python `AttributeError` if called.
For a detailed overview of the functions that are currently implemented, see [our API documentation](bindings/Python/README.md).

## Citing PLSSVM

If you use PLSSVM in your research, we kindly request you to cite:

```text
@inproceedings{9835379,
  author={Van Craen, Alexander and Breyer, Marcel and Pfl\"{u}ger, Dirk},
  booktitle={2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)}, 
  title={PLSSVM: A (multi-)GPGPU-accelerated Least Squares Support Vector Machine}, 
  year={2022},
  volume={},
  number={},
  pages={818-827},
  doi={10.1109/IPDPSW55747.2022.00138}
}
```
For a full list of all publications involving PLSSVM see our [Wiki Page](https://github.com/SC-SGS/PLSSVM/wiki/List-of-Publications-involving-PLSSVM).

## License

The PLSSVM library is distributed under the [MIT license](https://github.com/SC-SGS/PLSSVM/blob/main/LICENSE.md).
