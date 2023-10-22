from matplotlib import pyplot as plt
import numpy as np


#heights (the first) and weights (the second column).
values_file = np.loadtxt('data/male_female_X_train.txt',usecols=(0,1))
target_file = np.loadtxt('data/smale_female_y_train.txt')

# data based on class labels
males_data = values_file[target_file == 0]
females_data = values_file[target_file == 1]
#print(males_data)

# Separate data
males_height = males_data[:, 0]
females_height = females_data[:, 0]
males_weight = males_data[:, 1]
females_weight = females_data[:, 1]

# bin edges 
height_bins = np.linspace(80, 220, 11)  # 10 bins between 80 and 220
weight_bins = np.linspace(30, 180, 11)  # 10 bins between 30 and 180

# Compute histograms for height and weight
males_height_hist, _ = np.histogram(males_height, bins=height_bins)
females_height_hist, _ = np.histogram(females_height, bins=height_bins)
males_weight_hist, _ = np.histogram(males_weight, bins=weight_bins)
females_weight_hist, _ = np.histogram(females_weight, bins=weight_bins)

plt.figure(figsize=(4, 6))
plt.hist([males_height, females_height], bins=height_bins, alpha=0.5, label=['Male', 'Female'], color=['blue', 'red'])
plt.xlabel('Height (cm)')
plt.legend()
plt.title('Height Histogram')

plt.figure(figsize=(4, 6))
plt.hist([males_weight, females_weight], bins=weight_bins, alpha=0.5, label=['Male', 'Female'], color=['blue', 'red'])
plt.xlabel('Weight')
plt.legend()
plt.title('Weight Histogram')

plt.show()
