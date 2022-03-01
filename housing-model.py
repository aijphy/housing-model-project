import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import vpython as vs
import matplotlib.pyplot as plt
import seaborn as sns

# retrieve data and get rid of the unused columns:
# could preprocess with SQL, but with so few modifications I did this by hand
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
data['MEDV'] = data['MEDV']*1000*21 #purchasing power since 1978

prices = data['MEDV']
features = data.drop('MEDV',axis=1)

print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))


print("Statistics for Boston housing dataset:")
print(f"Minimum price: ${np.amin(prices):.2f}")
print(f"Maximum price: ${np.amax(prices):.2f}")
print(f"Mean price: ${np.mean(prices):.2f}")
print(f"Median price: ${np.median(prices):.2f}")
print(f"Standard deviation of prices: ${np.std(prices):.2f}")

# plot pairplot:
#sns.pairplot(data,height=2.5)
#plt.tight_layout()

# plot correlation matrix:
#cm = np.corrcoef(data.values.T)
#sns.set(font_scale=1.5)
#hm = sns.heatmap(cm,
#                cbar=True,
#                annot=True,
#                square=True,
#                fmt='.2f',
#                annot_kws={'size': 15},
#                yticklabels=data.columns,xticklabels=data.columns)



