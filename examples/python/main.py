import plssvm

try:
    # create a new C-SVM parameter set, explicitly overriding the default kernel function
    params = plssvm.Parameter(kernel_type=plssvm.KernelFunctionType.POLYNOMIAL)

    # create two data sets: one with the training data scaled to [-1, 1]
    # and one with the test data scaled like the training data
    train_data = plssvm.DataSet("train_data.libsvm", scaling=plssvm.DataSetScaling(-1.0, 1.0))
    test_data = plssvm.DataSet("test_data.libsvm", scaling=train_data.scaling_factors())

    # create C-SVM using the default backend and the previously defined parameter
    svm = plssvm.CSVM(params)

    # fit using the training data, (optionally) set the termination criterion
    model = svm.fit(train_data, epsilon=10e-6)

    # get accuracy of the trained model
    model_accuracy = svm.score(model)
    print("model accuracy: {}".format(model_accuracy))

    # predict labels
    label = svm.predict(model, test_data)

    # write model file to disk
    model.save("model_file.libsvm")
except plssvm.PLSSVMError as e:
    print(e)
except RuntimeError as e:
    print(e)
