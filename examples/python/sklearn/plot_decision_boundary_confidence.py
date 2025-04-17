import numpy as np
import matplotlib.pyplot as plt
import sklearn
import plssvm
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# generate synthetic dataset with 4 classes
X, y = make_classification(n_classes=4, n_clusters_per_class=1, n_features=2,
                           n_informative=2, n_redundant=0, random_state=42, n_samples=300)

# scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# fit SVC with an RBF kernel and "ovo" decision_function_shape for multi-class classification
def create_plot(clf, axis):
    ax_idx = 0
    for dfs in ["ovr", "ovo"]:
        clf.set_params(decision_function_shape=dfs)
        clf.fit(X, y)
        print(f"Training score {clf}: {clf.score(X, y):.2f}")

        # create grid for visualization
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))

        # get decision function values for grid points
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # Shape (num_points, num_class_pairs)

        # handle one-vs-one and one-vs-rest
        if dfs == "ovo":
            # aggregate the pairwise decision function outputs to determine the predicted class
            n_classes = len(np.unique(y))
            votes = np.zeros((Z.shape[0], n_classes))  # Initialize votes for each class

            # fill votes by iterating through each class pair
            idx = 0
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    # class i vs Class j
                    votes[:, i] += (Z[:, idx]>0).astype(int)  # increment vote for class i
                    votes[:, j] += (Z[:, idx]<=0).astype(int)  # increment vote for class j
                    idx += 1
        else:
            votes = Z

        # determine predicted class by majority vote
        Z_pred = np.argmax(votes, axis=1)  # class with the highest vote
        Z_confidence = np.max(Z, axis=1)  # confidence based on raw decision values

        # reshape for plotting
        Z_pred = Z_pred.reshape(xx.shape)
        Z_confidence = Z_confidence.reshape(xx.shape)

        # plot the decision boundaries (class regions)
        axis[ax_idx].pcolormesh(xx, yy, Z_pred, alpha=0.3)

        # overlay confidence as a grayscale gradient
        cb_values = axis[ax_idx].pcolormesh(xx, yy, Z_confidence, cmap="Greys", alpha=0.45)
        fig.colorbar(cb_values, ax=axis[ax_idx], label="confidence")

        # plot training data points
        scatter = axis[ax_idx].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        axis[ax_idx].legend(handles=scatter.legend_elements()[0], labels=["Class 0", "Class 1", "Class 2", "Class 3"])
        axis[ax_idx].set_title(f"{clf} decision boundary {dfs} voting")
        axis[ax_idx].set_xlabel("Feature 1")
        axis[ax_idx].set_ylabel("Feature 2")
        ax_idx = ax_idx + 1

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(26, 9))

# fit sklearn
sklearn_svc = sklearn.svm.SVC(kernel='rbf', C=10)
create_plot(sklearn_svc, ax[0, :])

# fit PLSSVM
plssvm_svc = plssvm.SVC(kernel='rbf', C=10)
create_plot(plssvm_svc, ax[1, :])

plt.tight_layout()
plt.show()
