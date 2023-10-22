# Classifier 
#the purpose is to classify the different type of breast cancer
import sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt

raw_data = load_breast_cancer()
#print(raw_data)

#set labels and features after printing the data
labels = raw_data['target'] ## so called Y value
label_names = raw_data['target_names']
features = raw_data['data'] ## so called X value
feature_names = raw_data['feature_names']

# Print two classes to get the real names
# ['malignant' 'benign'] (index 0  &  1) works as labels
print(label_names) 

# Split the data 
train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

# Naive Bayes is used to train | binary classification -case
model = GaussianNB()
trained_model = model.fit(train_data, train_labels)

# Predict 
# Prediction is made for the test data, 
prediction = trained_model.predict(test_data)
#print( prediction)

accuracy = accuracy_score(test_labels, prediction)
print(f'True val: {test_labels[10]} predicted val: {prediction[10]}')
print(f"Accuracy: {accuracy:.2f}")

### PLOT GRAPH
#plt.figure(figsize=(8, 6))
#plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap='coolwarm', marker='o', s=50)
#plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap='coolwarm', marker='x', s=50)
#
#plt.xlabel("Feature 0")
#plt.ylabel("Feature 1")
#plt.title("Breast Cancer Data")
#plt.show()

# Separate train data by labels
train_data_malignant = train_data[train_labels == 0]
train_data_benign = train_data[train_labels == 1]

# Separate test data by labels
test_data_malignant = test_data[test_labels == 0]
test_data_benign = test_data[test_labels == 1]

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(train_data_malignant, np.zeros_like(train_data_malignant),
            c='red', marker='o', label='Train Malignant', s=50)
plt.scatter(train_data_benign, np.zeros_like(train_data_benign),
            c='blue', marker='o', label='Train Benign', s=50)
plt.scatter(test_data_malignant, np.zeros_like(test_data_malignant),
            c='red', marker='x', label='Test Malignant', s=50)
plt.scatter(test_data_benign, np.zeros_like(test_data_benign),
            c='blue', marker='x', label='Test Benign', s=50)

plt.xlabel("Feature 0")
plt.title("Breast Cancer Data (First Feature)")
plt.legend(loc='upper right')
plt.show()

