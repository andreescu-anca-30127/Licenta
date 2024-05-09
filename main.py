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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo

# fetch dataset
breast_cancer_wisconsin_prognostic = fetch_ucirepo(id=16)
# data (as pandas dataframes)
X = breast_cancer_wisconsin_prognostic.data.features
y = breast_cancer_wisconsin_prognostic.data.targets
print(X)
print(y)
# metadata
print(breast_cancer_wisconsin_prognostic.metadata)

# variable information
print(breast_cancer_wisconsin_prognostic.variables)

 # Citirea datelor
input =X.iloc[:, 1:31].values # input
print(input)
output=X.iloc[:,31:32].values
print(output)
#print(X.iloc[:,0].values)
# Separarea datelor în seturi de antrenare și de test
X_train, X_test, Y_train, Y_test = train_test_split(input, output, test_size=0.3, random_state=0)
#reshap
Y_train = Y_train.reshape(-1,1)
# realizare matrice 2d pentru outptu deoarece am outptu numa pe o coloana
Y_test= Y_test.reshape(-1,1)
# # Standardizarea caracteristicilor de intrare
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# Normalizarea caracteristicilor de intrare
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#normalizare pe  iesire
sc_Y= MinMaxScaler()
Y_train = sc_Y.fit_transform(Y_train)
Y_test = sc_Y.transform(Y_test)

# Crearea rețelei neuronale
def ANN(Y_train, output, batch, epochs, error,num_layers,num_neurons):
    classifier = Sequential()

    # Adăugarea straturilor ascunse
    for i in range(num_layers):
        # Primul strat ascuns trebuie să aibă dimensiunea de intrare specificată
        if i == 0:
            classifier.add(Dense(input_dim= 30, units=num_neurons, activation="relu", kernel_initializer="uniform"))

        else:
            classifier.add(Dense(units=num_neurons, activation="relu", kernel_initializer="uniform"))




    classifier.add(Dense(output))
    classifier.summary()
    create_optimizer = Adam(learning_rate=0.04e-4, epsilon = 1e-08)
    classifier.compile(optimizer= create_optimizer, loss=error, metrics=['accuracy'])
    classifier.fit(X_train, Y_train, batch_size=batch, epochs=epochs,validation_split=0.1)

    # The prediction
    yhat = classifier.predict(X_test)
    bias = classifier.layers[0].get_weights()[1]
    weights = classifier.layers[0].get_weights()[0]
    return yhat, bias, weights
yhat, bias, weights = ANN(Y_train, output=1, batch=5, epochs=150, error='mse',num_layers=1,num_neurons=23)

# # Definirea listelor de valori posibile pentru parametrii căutați
# best_mse = float('inf')
# best_params = {}
#
# num_neurons_values = [23, 25, 24]  # Numărul de neuroni
# batch_size_values = [3, 4, 5]   # Dimensiunea batch-ului
# epochs_values = [100, 150, 200, 250]  # Numărul de epoci
#
#
# for num_neurons in num_neurons_values:
#     for batch_size in batch_size_values:
#         for epochs in epochs_values:
#
#                 # Antrenarea modelului cu parametrii actuali
#                 yhat, _, _ = ANN(Y_train, output=1, batch=batch_size, epochs=epochs, error='mse', num_layers=1,num_neurons=num_neurons)
#
#                 # Calculul erorii pe setul de testare
#                 mse = np.square(np.subtract(Y_test, yhat)).mean()
#
#                 # Actualizarea celor mai buni parametri și a celei mai mici erori
#                 if mse < best_mse:
#                     best_mse = mse
#                     best_params = {'num_neurons': num_neurons, 'batch_size': batch_size, 'epochs': epochs,}
#
#
# print("Best parameters:", best_params)
# print("Best MSE:", best_mse)
# Evaluarea vizuală
plt.plot(Y_test, 'red', label='Real Output')
plt.plot(yhat, 'green', label='Predicted Output')
plt.title('Model Evaluation')
plt.xlabel('Number of samples')
plt.ylabel('Measured value')
plt.legend()
plt.show()

# Calcularea erorilor
mse = np.square(np.subtract(Y_test, yhat)).mean()
me = np.square(np.subtract(Y_test, yhat)).min()
# Afisarea erorilor
print("Mean Squared Error (MSE):", mse)
print("Minimal Error (MAE):", me)
