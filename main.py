import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from scipy.sparse import hstack
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

date = pd.read_excel('C:\licenta\pacienti.xlsx', sheet_name= 'b')
X= date.iloc[:, 0:1].values; #input
Y= date.iloc[:, 1:].values; #output

#encoding months from first imagine study

transformer = make_column_transformer((OneHotEncoder(drop='if_binary'), [0]))
X_transformed = transformer.fit_transform(X[:, :1])
X = hstack([X_transformed, X[:, 1:]])

#training test and test
X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)
Y_train, Y_test = train_test_split(Y, test_size=0.3, random_state=0)

# Standardizarea caracteristicilor de intrare direct pe matricea rară X
sc_X = StandardScaler(with_mean=False)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#standardizare pe outpuut
scaleY = StandardScaler()
Y_train = scaleY.fit_transform(Y_train)
Y_test = scaleY.transform(Y_test)

#ANN
