## CMAKE
getestet mit: `nvvc 9.0`, `g++ 6.3`, `cmake 3.10.1`

	1. mkdir Release
	2. cd Release
	3. cmake -DCMAKE_BUILD_TYPE=Release ..
	4. make
	5. cd ..

oder `./build.sh`

## RUN

    ./svm-train-gpu
    Usage: svm-train [options] training_set_file [model_file]
    options:
    -t kernel_type : set type of kernel function (default 0)
            0 -- linear: u'*v
            1 -- polynomial: (gamma*u'*v + coef0)^degree
            2 -- radial basis function: exp(-gamma*|u-v|^2)
    -d degree : set degree in kernel function (default 3)
    -g gamma : set gamma in kernel function (default 1/num_features)
    -r coef0 : set coef0 in kernel function (default 0)
    -c cost : set the parameter C (default 1)
    -e epsilon : set tolerance of termination criterion (default 0.001)
    -q : quiet mode (no outputs)