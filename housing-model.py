# suppress warnings
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "sklearn")
#

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
import visuals as vs
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)

#print("Training and testing split was successful.")

# analyze the performance vs tree depth
#vs.ModelLearning(features, prices)
#vs.ModelComplexity(X_train, y_train)
# depth of 4 yields best results


# perform grid search over max depth of a tree trained on X, y
def fit_model(X, y):
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

    regressor = DecisionTreeRegressor()

    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    grid = grid.fit(X,y)
    
    return grid.best_estimator_

reg = fit_model(X_train, y_train)

# pickle the sklearn model:
joblib.dump(reg, 'model.pkl')

#print(f"Parameter 'max_depth' is {reg.get_params()['max_depth']}")

# test the model with some example houses:
# # of rooms, student/pupil, poverty level (%)
#client_data =  [[5, 15, 17],
#                [4, 22, 32],
#                [8,  12, 3]]

#for i, price in enumerate (reg.predict(client_data)):
#    print(f"Predicted selling price for Client {i+1}'s home: ${price:.2f}")

# perform trials to get a sense of deviation:
#vs.PredictTrials(features, prices, fit_model, client_data)
# range of 70k in prices ^
