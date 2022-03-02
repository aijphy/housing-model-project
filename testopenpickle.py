import pickle
#import numpy as np
#import pandas as pd
from sklearn.model_selection import ShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
#import visuals as vs
#import matplotlib.pyplot as plt
#import seaborn as sns
#import joblib

with open('model.pkl','rb') as f:
    reg = pickle.load(f)
    cdata = [[4,  12, 3]]

    print(reg.predict(cdata)[0])