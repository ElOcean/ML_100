from matplotlib import pyplot as plt
import numpy as np


#heights (the first) and weights (the second column).
X_train = np.loadtxt('online2_data/X_train.txt', usecols=(0, 1))
y_train = np.loadtxt('online2_data/y_train.txt')
X_test = np.loadtxt('online2_data/X_test.txt', usecols=(0, 1))
y_test = np.loadtxt('online2_data/y_test.txt')

# data based on class labels (male = 0, female = 1)
males_data = X_train[y_train == 0]  
females_data = X_train[y_train == 1]
#print(males_data)

# Separate data
males_height = males_data[:, 0]
females_height = females_data[:, 0]


height_bins = np.linspace(80, 220, 11)  # 10 bins between 80 and 220

# Compute histograms for height and weight
males_height_hist, _ = np.histogram(males_height, bins=height_bins)
females_height_hist, _ = np.histogram(females_height, bins=height_bins)

plt.figure(figsize=(6, 6))
plt.hist([males_height, females_height], bins=height_bins, alpha=0.5, label=['Male', 'Female'], color=['blue', 'red'])
plt.xlabel('Height (cm)')
plt.legend()
plt.title('Height Histogram')
plt.show()


y_pred_male = np.full_like(y_test, 0)
y_pred_female = np.full_like(y_test, 1)
total_samples = len(y_test)

correct_male = np.sum(y_test == y_pred_male)
accuracy_male = correct_male / total_samples

correct_female = np.sum(y_test == y_pred_female)
accuracy_female = correct_female / total_samples
print("Males: {:.2f}%".format(accuracy_male * 100))
print("Females: {:.2f}%".format(accuracy_female * 100))

### separate the data to get pdf separately
mu_male = np.mean(males_data)
sigma_male = np.std(males_data)

mu_female = np.mean(females_data)
sigma_female = np.std(females_data)

males_pdf = []
females_pdf = []

for x in males_data:
    expo = -((x - mu_male) ** 2) / (2 * (sigma_male ** 2))
    pdf = (1 / (sigma_male * np.sqrt(2 * np.pi))) * np.exp(expo)
    males_pdf.append(pdf)

for x in females_data:
    expo = -((x - mu_female) ** 2) / (2 * (sigma_female ** 2))
    pdf = (1 / (sigma_female * np.sqrt(2 * np.pi))) * np.exp(expo)
    females_pdf.append(pdf)

plt.plot(males_data, males_pdf, 'ko', label="males")
plt.plot(females_data, females_pdf, 'ro', label="females")

plt.xlabel('height')
plt.ylabel('likelihood')
plt.legend()
plt.show()


prior_male = len(males_data) / len(X_train)
prior_female = len(females_data) / len(X_train)

#Compare the test samples and prediction
# 
# Accuracy true_y_values(test) vs predicted 

plt.show()



