import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from random import randint as rnd


data = load_breast_cancer()
data = load_breast_cancer()
X, y = data.data, data.target
scaler = RobustScaler()
for i in range(X.shape[1]):
    X[:,i] = scaler.fit_transform(X[:,i].reshape(-1,1)).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)

import pandas as pd
from pandasgui import show

df = pd.DataFrame(X, columns=data.feature_names)
df['class']=y
show(df)
