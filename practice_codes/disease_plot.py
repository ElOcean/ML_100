import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Load training and testing data
X_train = np.loadtxt('data/disease_X_train.txt')
y_train = np.loadtxt('data/disease_y_train.txt')
X_test = np.loadtxt('data/disease_X_test.txt')
y_test = np.loadtxt('data/disease_y_test.txt')

# Baseline Prediction
y_pred_baseline = np.full(y_test.shape, y_train.mean())

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Plotting
plt.figure(figsize=(14, 6))

# Baseline Prediction Plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_baseline, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.title('Baseline Prediction: Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Linear Regression Prediction Plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.title('Linear Regression: Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()
