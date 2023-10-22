from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


#heights (the first) and weights (the second column).
values_file = np.loadtxt('data/male_female_X_train.txt',usecols=(0,1))
target_file = np.loadtxt('data/male_female_y_train.txt')
X_test = np.loadtxt('data/male_female_X_test.txt', usecols=(0, 1))
y_test = np.loadtxt('data/male_female_y_test.txt')

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

# Models
model = LinearRegression()
# train 
males_model = model.fit(males_height.reshape(-1,1), males_weight )
females_model = model.fit(females_height.reshape(-1,1), females_weight)

# Get the coefficients (slope and intercept) of the fitted lines
slope_male = males_model.coef_
intercept_male = males_model.intercept_
slope_female = females_model.coef_
intercept_female = females_model.intercept_
print(slope_female)
print(slope_male)


plt.figure(figsize=(6, 6))
plt.plot(males_height, males_weight, 'co', label='male')
plt.plot(females_height, females_weight, 'rx', label='Female')

# Plot regression lines for males and females
plt.plot(males_height, slope_male * males_height + intercept_male, 'b-', label='Male Regression Line')
plt.plot(females_height, slope_female * females_height + intercept_female, 'g-', label='Female Regression Line')


plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()

plt.show()
