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
print(X_test)
# Standardizarea caracteristicilor de intrare direct pe matricea rară X
sc_X = StandardScaler(with_mean=False)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#standardizare pe outpuut
#scaleY = StandardScaler()
#Y_train = scaleY.fit_transform(Y_train)
#Y_test = scaleY.transform(Y_test)

def ANN(Y_train, output, batch, epochs, error):
    classifier = Sequential()
    classifier.add(Dense(input_dim=X_train.shape[1], units=6, activation="relu", kernel_initializer='uniform'))
    classifier.add(Dense(output))
    classifier.compile(optimizer='adam', loss=error, metrics=['accuracy'])
    classifier.fit(X_train, Y_train, batch_size=batch, epochs=epochs)

    # The prediction
    yhat = classifier.predict(X_test)
    bias = classifier.layers[0].get_weights()[1]
    weights = classifier.layers[0].get_weights()[0]
    return yhat, bias, weights





yhat, bias, weights = ANN(Y_train, output=5, batch=5, epochs=400, error='mse')

#Calcularea erorilor
mse=np.mean(np.square(Y_test -yhat)) #eroarea medie patratica
mae=np.mean(np.abs(Y_test - yhat)) # eroarea medie absoluta

#afisarea erorilor
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

#plt.plot(Y_test, 'blue', label = 'real output')
#plt.plot(yhat,'pink', label ='my output')
#plt.title('Low grade liomas ')
#plt.xlabel('month from first imaging study')
#plt.ylabel('tumor volume[cm^3]')
#plt.show()

plt.plot(Y_test[:,0],'red', label = 'real output')
plt.plot(yhat[:,0],'green', label = 'predicted output')
plt.title('cred ca primul?')
plt.xlabel('month from first imaging study')
plt.ylabel('tumor volume[cm^3]')
plt.legend()
plt.show()
print('The mean squared error for 1 is:')
print(np.square(np.subtract(Y_test[:,0],yhat[:,0])).mean())
print('The minimum error  is:')
print(np.square(np.subtract(Y_test[:,0],yhat[:,0])).min())


plt.plot(Y_test[:,1],'red', label = 'real output')
plt.plot(yhat[:,1],'blue', label = 'predicted output')
plt.title('cred ca 2?')
plt.xlabel('month from first imaging study')
plt.ylabel('tumor volume[cm^3]')
plt.legend()
plt.show()
print('The mean squared error for is:')
print(np.square(np.subtract(Y_test[:,1],yhat[:,1])).mean())
print('The minimum error for is:')
print(np.square(np.subtract(Y_test[:,1],yhat[:,1])).min())

plt.plot(Y_test[:,2],'red',label='real output')
plt.plot(yhat[:,2],'pink', label = 'predicted output')
plt.title('coloana  3 cred?')
plt.xlabel('month from first imaging study')
plt.ylabel('tumor volume[cm^3]')
plt.legend()
plt.show()
print('The mean squared error is:')
print(np.square(np.subtract(Y_test[:,2],yhat[:,2])).mean())
print('The minimum error for is:')
print(np.square(np.subtract(Y_test[:,2],yhat[:,2])).min())

plt.plot(Y_test[:,3],'red',label='real output')
plt.plot(yhat[:,3],'purple', label = 'predicted output')
plt.title('coloana  4 cred?')
plt.xlabel('month from first imaging study')
plt.ylabel('tumor volume[cm^3]')
plt.legend()
plt.show()
print('The mean squared error is:')
print(np.square(np.subtract(Y_test[:,3],yhat[:,3])).mean())
print('The minimum error for is:')
print(np.square(np.subtract(Y_test[:,3],yhat[:,3])).min())

plt.plot(Y_test[:,4],'red',label='real output')
plt.plot(yhat[:,4],'pink', label = 'predicted output')
plt.title('coloana  5 cred?')
plt.xlabel('month from first imaging study')
plt.ylabel('tumor volume[cm^3]')
plt.legend()
plt.show()
print('The mean squared error is:')
print(np.square(np.subtract(Y_test[:,4],yhat[:,4])).mean())
print('The minimum error for is:')
print(np.square(np.subtract(Y_test[:,4],yhat[:,4])).min())