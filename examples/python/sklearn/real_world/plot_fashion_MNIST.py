import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plssvm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# combine train and test sets for expansion
X_all = np.vstack((X_train, X_test))
y_all = np.hstack((y_train, y_test))

# flatten images (28x28 -> 784 features per sample)
X_all = X_all.reshape(len(X_all), -1).astype(np.float32) / 255.0  # Normalize pixel values

# train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# output number of samples
print(f"training dataset size: {X_train.shape}")
print(f"test dataset size: {X_test.shape}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train SVM Classifier with RBF kernel
svm_clf = SVC(kernel="rbf", C=10, gamma="scale", tol=1e-6)
fit_start_time = time.time()
svm_clf.fit(X_train_scaled, y_train)
fit_end_time = time.time()
print(f"fit model in {fit_end_time - fit_start_time:.2f}s")

# predict on test data
predict_start_time = time.time()
y_pred = svm_clf.predict(X_test_scaled)
predict_end_time = time.time()
print(f"predict labels in {predict_end_time - predict_start_time:.2f}s")

# evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.4f}")
print("\nclassification report:\n", classification_report(y_test, y_pred))

# confusion Matrix
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# visualize some predictions
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Pred: {y_pred[i]} | True: {y_test[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()
