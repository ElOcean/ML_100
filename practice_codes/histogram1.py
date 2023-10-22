from matplotlib import pyplot as plt
import numpy as np


#heights (the first) and weights (the second column).
values_file = np.loadtxt('data/male_female_X_train.txt',usecols=(0,1))

print(values_file)
#labels 0: male 1: female
target_file = np.loadtxt('data/male_female_y_train.txt')


males = values_file[target_file == 0]
females = values_file[target_file == 1]

plt.hist(males[:, 0], label='Male', color='blue')
plt.hist(females[:, 0], label='Female',color='red')

plt.xlabel('Height (cm)')

plt.legend()
plt.show()







