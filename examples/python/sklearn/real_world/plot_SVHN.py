import matplotlib.pyplot as plt
from plssvm.svm import SVC
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_svmlight_file
import time

# load the dataset, note: must be locally available
X_train, y_train = load_svmlight_file("SVHN")
X_test, y_test = load_svmlight_file("SVHN.t")

# convert label 10 to 0 (as SVHN labels digits 1-10, where 10 represents '0')
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

# reshape image arrays to (num_samples, 32, 32, 3) -> (num_samples, 3072)
X_train = X_train.reshape(-1, 32*32*3)
X_test = X_test.reshape(-1, 32*32*3)

# output number of samples
print(f"training dataset size: {X_train.shape}")
print(f"test dataset size: {X_test.shape}")

# normalize the pixel values
X_train = X_train.astype("float64") / 255.0
X_test = X_test.astype("float64") / 255.0

# train SVC with an RBF kernel
svm_clf = SVC(kernel="rbf", tol=1e-7)
fit_start_time = time.time()
svm_clf.fit(X_train, y_train)
fit_end_time = time.time()
print(f"fit model in {fit_end_time - fit_start_time:.2f}s")

# predict on test set
predict_start_time = time.time()
y_pred = svm_clf.predict(X_test)
predict_end_time = time.time()
print(f"predict labels in {predict_end_time - predict_start_time:.2f}s")

# evaluate the model
print(f"accuracy: {accuracy_score(y_test, y_pred)}")
print(f"\nclassification report:\n{classification_report(y_test, y_pred)}")

# confusion Matrix
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('svhn_confusion_matrix.png')
# plt.show()

# visualize some predictions
fig, axes = plt.subplots(3, 7, figsize=(27, 9))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i].toarray().ravel().reshape(32, 32, 3))
    ax.set_title(f"Pred: {y_pred[i]} | True: {y_test[i]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig('svhn.png')
# plt.show()
