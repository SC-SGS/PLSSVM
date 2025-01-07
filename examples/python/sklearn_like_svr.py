from sklearn.datasets import make_regression
import plssvm

num_samples = 2**8
num_features = 2**6

samples, labels = make_regression(n_samples=num_samples, n_features=num_features, noise=30)

# create a C-SVR
svr = plssvm.SVR(kernel='linear', C=1.0, tol=1e-3, verbose=False)

# fit the model
svr.fit(samples, labels)

# score the data set
model_accuracy = svr.score(samples, labels)
print("model accuracy (R^2): {0:.2f}".format(model_accuracy * 100))
