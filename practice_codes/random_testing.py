from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load the training and test data
X_train = np.loadtxt('online2_data/X_train.txt', usecols=(0, 1))
y_train = np.loadtxt('online2_data/y_train.txt')
X_test = np.loadtxt('online2_data/X_test.txt', usecols=(0, 1))
y_test = np.loadtxt('online2_data/y_test.txt')

# Step 1: Train the model
#X_train, _, y_train, _ = train_test_split(values_file, target_file, test_size=0, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression(random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Step 2: Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o', s=50, label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='x', s=50, label='Test Data')

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("Classification Data")
plt.legend()

plt.show()

# Step 3: Compute the accuracy of the classification
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.3f}".format(accuracy))
