import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plssvm import SVR
import plssvm
from sklearn.metrics import mean_squared_error, r2_score
import time

# load California Housing dataset
data = fetch_california_housing()

# features and target
X = data.data
y = data.target

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# output number of samples
print(f"training dataset size: {X_train.shape}")
print(f"test dataset size: {X_test.shape}")

# feature scaling for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train Support Vector Regression (SVR)
svr = SVR(kernel='rbf', C=100, gamma='auto', tol=1e-8)
fit_start_time = time.time()
svr.fit(X_train_scaled, y_train)
fit_end_time = time.time()
print(f"fit model in {fit_end_time - fit_start_time:.2f}s")

# Predict on test data
predict_start_time = time.time()
y_pred = svr.predict(X_test_scaled)
predict_end_time = time.time()
print(f"predict labels in {predict_end_time - predict_start_time:.2f}s")

# calculate metrics
print(f"\nregression report:\n{plssvm.regression_report(y_test, y_pred)}")

# visualize the true vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (SVR)')

plt.tight_layout()
plt.show()
