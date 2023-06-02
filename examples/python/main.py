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
