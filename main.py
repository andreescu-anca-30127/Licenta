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
# Citirea datelor
date = pd.read_excel('C:\licenta\pacienti.xlsx', sheet_name='b')
X = date.iloc[:, 0:1].values.reshape(-1,1)  # input
Y = date.iloc[:, 1:].values  # output

# Encodarea lunilor din prima imagine de studiu
transformer = make_column_transformer((OneHotEncoder(drop='if_binary'), [0]))
X_transformed = transformer.fit_transform(X[:, :1])
X = hstack([X_transformed, X[:, 1:]])

# Separarea datelor în seturi de antrenare și de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Transformarea matricei rare X_train într-o matrice densă
X_train_dense = X_train.toarray()

# Standardizarea caracteristicilor de intrare
sc_X = StandardScaler(with_mean=False)
X_train_dense = sc_X.fit_transform(X_train_dense)
X_test = sc_X.transform(X_test)


# Crearea rețelei neuronale
def ANN(Y_train, Y_test, output, batch, epochs, error, num_layers, num_neurons):
    classifier = Sequential()
    # Adăugarea straturilor ascunse
    for i in range(num_layers):
        # Primul strat ascuns trebuie să aibă dimensiunea de intrare specificată
        if i == 0:
            classifier.add(
                Dense(input_dim=X_train_dense.shape[1], units=num_neurons, activation="relu", kernel_initializer="uniform"))
        else:
            classifier.add(Dense(units=num_neurons, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dense(output))
    classifier.compile(optimizer='adam', loss=error, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    classifier.fit(X_train_dense, Y_train, batch_size=batch, epochs=epochs, validation_split=0.3, callbacks=[early_stopping])

    # Prezicerea rezultatelor pe setul de test
    yhat = classifier.predict(X_test)
    bias = classifier.layers[0].get_weights()[1]
    weights = classifier.layers[0].get_weights()[0]
    return yhat, bias, weights


# Apelarea funcției ANN pentru antrenarea rețelei
yhat, bias, weights = ANN(Y_train, Y_test, output=5, batch=5, epochs=50, error='mse',  num_layers=2, num_neurons=3)

# Plot pentru coloana 1
plt.plot(Y_test[:, 0], 'red', label='Real output')
plt.plot(yhat[:, 0], 'green', label='Predicted output')
plt.title('Column 1')
plt.xlabel('Month from first imaging study')
plt.ylabel('Tumor volume [cm^3]')
plt.legend()
plt.show()

# Calcul și afișare eroare medie pătratică pentru coloana 1
print('The mean squared error for column 1 is:')
print(np.square(np.subtract(Y_test[:, 0], yhat[:, 0])).mean())
print('The minimum error is:')
print(np.square(np.subtract(Y_test[:, 0], yhat[:, 0])).min())

# Plot pentru coloana 2
plt.plot(Y_test[:, 1], 'red', label='Real output')
plt.plot(yhat[:, 1], 'blue', label='Predicted output')
plt.title('Column 2')
plt.xlabel('Month from first imaging study')
plt.ylabel('Tumor volume [cm^3]')
plt.legend()
plt.show()

# Calcul și afișare eroare medie pătratică pentru coloana 2
print('The mean squared error for column 2 is:')
print(np.square(np.subtract(Y_test[:, 1], yhat[:, 1])).mean())
print('The minimum error is:')
print(np.square(np.subtract(Y_test[:, 1], yhat[:, 1])).min())

# Plot pentru coloana 3
plt.plot(Y_test[:, 2], 'red', label='Real output')
plt.plot(yhat[:, 2], 'pink', label='Predicted output')
plt.title('Column 3')
plt.xlabel('Month from first imaging study')
plt.ylabel('Tumor volume [cm^3]')
plt.legend()
plt.show()

# Calcul și afișare eroare medie pătratică pentru coloana 3
print('The mean squared error for column 3 is:')
print(np.square(np.subtract(Y_test[:, 2], yhat[:, 2])).mean())
print('The minimum error is:')
print(np.square(np.subtract(Y_test[:, 2], yhat[:, 2])).min())

# Plot pentru coloana 4
plt.plot(Y_test[:, 3], 'red', label='Real output')
plt.plot(yhat[:, 3], 'purple', label='Predicted output')
plt.title('Column 4')
plt.xlabel('Month from first imaging study')
plt.ylabel('Tumor volume [cm^3]')
plt.legend()
plt.show()

# Calcul și afișare eroare medie pătratică pentru coloana 4
print('The mean squared error for column 4 is:')
print(np.square(np.subtract(Y_test[:, 3], yhat[:, 3])).mean())
print('The minimum error is:')
print(np.square(np.subtract(Y_test[:, 3], yhat[:, 3])).min())

# Plot pentru coloana 5
plt.plot(Y_test[:, 4], 'red', label='Real output')
plt.plot(yhat[:, 4], 'pink', label='Predicted output')
plt.title('Column 5')
plt.xlabel('Month from first imaging study')
plt.ylabel('Tumor volume [cm^3]')
plt.legend()
plt.show()

# Calcul și afișare eroare medie pătratică pentru coloana 5
print('The mean squared error for column 5 is:')
print(np.square(np.subtract(Y_test[:, 4], yhat[:, 4])).mean())
print('The minimum error is:')
print(np.square(np.subtract(Y_test[:, 4], yhat[:, 4])).min())
