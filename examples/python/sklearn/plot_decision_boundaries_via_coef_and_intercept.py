import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# generate a toy dataset with 3 classes
X, y = make_classification(n_samples=300, n_features=2, n_classes=3, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=1)

# create the plot
def test(model, svc_name, axis):
    # fit an SVC model with a linear kernel
    model.fit(X, y)
    print(f"{model} score is {model.score(X, y):.2f}")

    # retrieve coefficients and intercepts
    coef = model.coef_  # Shape: (n_classes, n_features)
    intercept = model.intercept_  # Shape: (n_classes,)

    # plot the dataset
    for class_value in range(len(model.classes_)):
        axis.scatter(X[y == class_value, 0], X[y == class_value, 1], label=f"class {class_value}")

    # plot decision boundaries for each ovr classifier
    for i in range(len(model.classes_)):
        # using the coef_ and intercept_ to compute the boundary line
        w = coef[i]  # coefficients for the ith class
        b = intercept[i]  # intercept for the ith class
        # decision boundary equation: w[0]*x + w[1]*y + b = 0 => y = -(w[0]/w[1])*x - b/w[1]
        if w[1] != 0:  # avoid division by zero
            x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500)
            y_vals = -(w[0] / w[1]) * x_vals - b / w[1]
            axis.plot(x_vals, y_vals, linestyle="--", label=f"boundary for class {i} vs rest")

    axis.set_title(f"{svc_name} Decision Boundaries (ovr with 3 classes)")
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.legend(loc='upper right')


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# train using sklean
from sklearn.svm import SVC

sklearn_model = SVC(kernel="linear", decision_function_shape="ovr")
test(sklearn_model, "sklearn.svm.SVC", ax[0])

# train using PLSSVM
from plssvm.svm import SVC

plssvm_model = SVC(kernel="linear", decision_function_shape="ovr")
test(plssvm_model, "plssvm.SVC", ax[1])

fig.tight_layout()
plt.show()
