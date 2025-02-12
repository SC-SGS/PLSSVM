import matplotlib.pyplot as plt

import plssvm as svm
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay

# import some data to play with
iris = datasets.load_iris()
# take the first two features
X = iris.data[:, :2]
y = iris.target

# we create an instance of SVM and fit out data
models = [
    ("linear kernel (ovr)", svm.SVC(kernel="linear", decision_function_shape="ovr")),
    ("RBF kernel (ovr)", svm.SVC(kernel="rbf", gamma=0.7, decision_function_shape="ovr")),
    ("polynomial (d=3) kernel (ovr)", svm.SVC(kernel="poly", degree=3, gamma="auto", decision_function_shape="ovr")),
    ("sigmoid kernel (ovr)", svm.SVC(kernel="sigmoid", gamma="scale", decision_function_shape="ovr")),
    ("laplacian kernel (ovr)", svm.SVC(kernel="laplacian", gamma="auto", decision_function_shape="ovr")),
    ("chi_squared kernel (ovr)", svm.SVC(kernel="chi_squared", gamma="auto", decision_function_shape="ovr")),
    ("linear kernel (ovo)", svm.SVC(kernel="linear", decision_function_shape="ovo")),
    ("RBF kernel (ovo)", svm.SVC(kernel="rbf", gamma=0.7, decision_function_shape="ovo")),
    ("polynomial (d=3) kernel (ovo)", svm.SVC(kernel="poly", degree=3, gamma="auto", decision_function_shape="ovo")),
    ("sigmoid kernel (ovo)", svm.SVC(kernel="sigmoid", gamma="auto", decision_function_shape="ovo")),
    ("laplacian kernel (ovo)", svm.SVC(kernel="laplacian", gamma="auto", decision_function_shape="ovo")),
    ("chi_squared kernel (ovo)", svm.SVC(kernel="chi_squared", gamma="auto", decision_function_shape="ovo")),
]
models = [(title, clf.fit(X, y)) for (title, clf) in models]

fig, sub = plt.subplots(2, 6, figsize=(27, 9))
fig.suptitle("SVC classifiers on 2D iris dataset")

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# plot the data
for (title, clf), ax in zip(models, sub.flatten()):
    # plot the decision boundary
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.7,
        ax=ax,
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
    )
    # plot the dataset
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")

    score = clf.score(X, y)
    print(f"{clf}: {score:.2f}")
    # add the model accuracy as textbox
    ax.text(x_max, y_min, ("%.2f" % score).lstrip("0"), size=15,
            bbox=dict(boxstyle="round", alpha=0.8, facecolor="white"),
            horizontalalignment="right")

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

fig.tight_layout()
plt.show()