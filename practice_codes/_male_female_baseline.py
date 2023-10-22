import numpy as np

# Load training and testing data
X_train = np.loadtxt('data/male_female_X_train.txt', usecols=(0, 1))
y_train = np.loadtxt('data/male_female_y_train.txt')
X_test = np.loadtxt('data/male_female_X_test.txt', usecols=(0, 1))
y_test = np.loadtxt('data/male_female_y_test.txt')


# random class labels
random_predict = np.random.randint(2, size=len(X_test))

# random classifier accuracy
correct_predictions_random = sum(random_predict == y_test)
accuracy_random = correct_predictions_random / len(y_test)

# find the class for train
class_counts = [0, 0]
for label in y_train:
    class_counts[int(label)] += 1
#print (class_counts)
if class_counts[0] >= class_counts[1]:
    most_likely_class = 0 
else:
    most_likely_class = 1


# Calculate accuracy for the most likely classifier
correct_predictions = sum([most_likely_class] * len(y_test) == y_test)
accuracy_most_likely = correct_predictions / len(y_test)

# Print the accuracy of each baseline classifier
print("random classifier: {:.3f}".format(accuracy_random))
print("highest priori classifier: {:.3f}".format(accuracy_most_likely))
