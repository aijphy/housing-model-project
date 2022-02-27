import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

# retrieve data and get rid of the unused columns:
data = pd.read_csv('./data/bostonhousing.csv')
data = data.drop('CRIM', axis=1)
data = data.drop('ZN', axis=1)
data = data.drop('INDUS', axis=1)
data = data.drop('CHAS', axis=1)
data = data.drop('NOX', axis=1)
data = data.drop('AGE', axis=1)
data = data.drop('DIS', axis=1)
data = data.drop('RAD', axis=1)
data = data.drop('TAX', axis=1)
data = data.drop('B', axis=1)
data['MEDV'] = data['MEDV']*4.31 #purchasing power since 1978

prices = data['MEDV']
features = data.drop('MEDV',axis=1)
print(data)

print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))