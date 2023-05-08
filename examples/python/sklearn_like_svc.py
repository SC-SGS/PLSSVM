from sklearn.datasets import make_classification
import plssvm

num_samples = 2**8
num_features = 2**6

samples, labels = make_classification(n_samples=num_samples, n_features=num_features, n_redundant=0,
                                      n_informative=2, n_clusters_per_class=1)

# create C-SVM
svc = plssvm.SVC(kernel='linear', C=1.0, tol=10e-3, verbose=False)

# fit the model
svc.fit(samples, labels)

# score the data set
model_accuracy = svc.score(samples, labels)
print("model accuracy: {0:.2f}".format(model_accuracy * 100))
