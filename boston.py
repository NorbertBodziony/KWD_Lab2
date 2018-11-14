from sklearn.datasets import load_boston
import pandas as pd
boston_market_data = load_boston()

from sklearn.model_selection import train_test_split

boston_data = pd.DataFrame(boston_market_data.data, columns=[boston_market_data.feature_names])
boston_data.head()

print(boston_data.describe())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
boston_norm_data=load_boston()
boston_norm_data['data'] = scaler.fit_transform(boston_market_data['data'])

boston_data = pd.DataFrame(boston_norm_data.data, columns=[boston_market_data.feature_names])
boston_data.head()

print(boston_data.describe())

boston_train_data, boston_test_data, \
boston_train_target, boston_test_target = \
train_test_split(boston_norm_data['data'],boston_market_data['target'], test_size=0.1)

print("Training dataset:")
print("boston_train_data:", boston_train_data.shape)
print("boston_train_target:", boston_train_target.shape)

print("Testing dataset:")
print("boston_test_data:", boston_test_data.shape)
print("boston_test_target:", boston_test_target.shape)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(boston_train_data, boston_train_target)

id=5
linear_regression_prediction = linear_regression.predict(boston_test_data[id,:].reshape(1,-1))

print(boston_test_data[id,:].shape)
print(boston_test_data[id,:].reshape(1,-1).shape)

from sklearn.metrics import mean_squared_error
print("Mean squared error of a learned model: %.2f" %
mean_squared_error(boston_test_target, linear_regression.predict(boston_test_data)))

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(boston_test_target, linear_regression.predict(boston_test_data)))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(LinearRegression(), boston_norm_data['data'], boston_market_data['target'], cv=4)
print('Cross-validation')
print(scores)