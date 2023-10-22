import numpy as np


X_train = np.loadtxt('data/male_female_X_train.txt', usecols=(0, 1))
y_train = np.loadtxt('data/male_female_y_train.txt')
X_test = np.loadtxt('data/male_female_X_test.txt', usecols=(0, 1))
y_test = np.loadtxt('data/male_female_y_test.txt')

# prior probabilities
prior_male = np.mean(y_train == 0)
prior_female = np.mean(y_train == 1)

print("Male Prior probability: {:.5f}".format(prior_male))
print("Female Prior probability: {:.5f}".format(prior_female))


#  histogram() returns tuple.
height_hist, height_bins = np.histogram(X_train[:, 0], bins=10)
weight_hist, weight_bins = np.histogram(X_train[:, 1], bins=10)

# test histograms
#print(height_hist)
#print(weight_hist)

# likelihood 
def likelihood(hist, bins, sample):
    for i in range(len(bins) - 1):
        if bins[i] <= sample < bins[i + 1]:
            bin_index = i
            break
    else:
        bin_index = len(bins) - 2  

    return hist[bin_index] / len(X_train)

# height
# define array with zeros
height_likelihood_male = np.zeros(len(X_test))
height_likelihood_female = np.zeros(len(X_test))
for i, sample in enumerate(X_test[:, 0]):
    height_likelihood_male[i] = likelihood(height_hist, height_bins, sample)
    height_likelihood_female[i] = likelihood(height_hist, height_bins, sample)

# weight
weight_likelihood_male = np.zeros(len(X_test))
weight_likelihood_female = np.zeros(len(X_test))
for i, sample in enumerate(X_test[:, 1]):
    weight_likelihood_male[i] = likelihood(weight_hist, weight_bins, sample)
    weight_likelihood_female[i] = likelihood(weight_hist, weight_bins, sample)


# print("\nHeight Likelihood (Female):")
# print(height_likelihood_female)


# Combine likelihoods by multiplying
com_likelihood_male = height_likelihood_male * weight_likelihood_male
com_likelihood_female = height_likelihood_female * weight_likelihood_female

# print("\nCombined Likelihood (Male):")
# print(com_likelihood_male)

#store predictions
predicted_male_height = []
predicted_male_weight = []
predicted_combined = []

for i in range(len(height_likelihood_male)):
    #  height likelihood
    if height_likelihood_male[i] > height_likelihood_female[i]:
        predicted_male_height.append(1)
    else:
        predicted_male_height.append(0)

    # weight likelihood
    if weight_likelihood_male[i] > weight_likelihood_female[i]:
        predicted_male_weight.append(1)
    else:
        predicted_male_weight.append(0)

    # combined likelihood
    if com_likelihood_male[i] > com_likelihood_female[i]:
        predicted_combined.append(1)
    else:
        predicted_combined.append(0)

predicted_male_height = np.array(predicted_male_height)
predicted_male_weight = np.array(predicted_male_weight)
predicted_male_combined = np.array(predicted_combined)

# accuracy for each method
accuracy_height = np.mean(predicted_male_height == y_test)
accuracy_weight = np.mean(predicted_male_weight == y_test)
accuracy_combined = np.mean(predicted_male_combined == y_test)

# Print accuracies
print("\nAccuracy for height: {:.3f}".format(accuracy_height))
print("Accuracy for weight: {:.3f}".format(accuracy_weight))
print("Accuracy for height and weight Combined: {:.3f}".format(accuracy_combined))
