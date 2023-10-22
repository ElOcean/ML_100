import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


X_train = np.loadtxt('data/disease_X_train.txt')
y_train = np.loadtxt('data/disease_y_train.txt')
X_test = np.loadtxt('data/disease_X_test.txt')
y_test = np.loadtxt('data/disease_y_test.txt')


# baseline 
y_pred_baseline = np.full(y_test.shape, y_train.mean())
mse = mean_squared_error(y_test, y_pred_baseline)
print(f"Baseline MSE: {mse}")


# LinearReg
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f"Linear Model MSE: {mse_linear}")


# DecisionTree
tree_regressor = DecisionTreeRegressor()
tree_regressor.fit(X_train, y_train)

y_pred_tree = tree_regressor.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print(f"Decision Tree MSE: {mse_tree}")


# RandomForest
forest_regressor = RandomForestRegressor()
forest_regressor.fit(X_train, y_train)
y_pred_forest = forest_regressor.predict(X_test)
mse_forest = mean_squared_error(y_test, y_pred_forest)
print(f"Random Forest MSE: {mse_forest}")
