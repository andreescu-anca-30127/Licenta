import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import r2_score
from ucimlrepo import fetch_ucirepo
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
# # # model cu mai multe date pentru clasificare
from sklearn.preprocessing import StandardScaler

#
# # Citirea datelor
dataset = pd.read_csv("C:\\Users\\Admin\\Desktop\\skolika\\data.csv")
# Afișarea primelor 5 rânduri ale dataset-ului
print(dataset.head())
print(dataset.describe())
# ARIA TUMORII
output = dataset[['area_mean']].values
#
# Eliminarea coloanelor 'area_mean', 'id', și 'diagnosis' din input
input = dataset.drop(columns=['area_mean', 'id', 'diagnosis']).values

# Afișarea descrierii pentru input
input_dataset = pd.DataFrame(input)
print(input_dataset.info())
print(input_dataset.describe())
#
# Separarea datelor în seturi de antrenare și de test
X_train, X_test, Y_train, Y_test = train_test_split(input, output, test_size=0.2, random_state=42)
#
# Reshape output
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# # Normalizarea caracteristicilor de intrare
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Normalizarea caracteristicilor de ieșire
sc_Y = MinMaxScaler()
Y_train = sc_Y.fit_transform(Y_train)
Y_test = sc_Y.transform(Y_test)
#
# Standarzidare
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# # Normalizarea caracteristicilor de ieșire
# sc_Y = StandardScaler()
# Y_train = sc_Y.fit_transform(Y_train)
# Y_test = sc_Y.transform(Y_test)

#robust scaller
# sc_X = RobustScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# # Normalizarea caracteristicilor de ieșire
# sc_Y = RobustScaler()
# Y_train = sc_Y.fit_transform(Y_train)
# Y_test = sc_Y.transform(Y_test)

#  crearea modelului pentru predictia ariei
# Definirea modelului
model = Sequential()
# model.add(Dense(15, activation='relu', input_dim=29))
model.add(Dense(15, input_dim=29))
model.add(LeakyReLU(alpha=0.1))
# model.add(Dropout(0.001))
# model.add(Dense(8, activation='relu'))
model.add(Dense(8))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1, activation='linear'))#strat iesire
# model.add(Dense(1))#strat iesire
# model.add(LeakyReLU(alpha=0.1))

# Creare optimizator cu o rată de învățare specificată
optimizer_arie = Adam(learning_rate=0.001)
# Compilarea modelului cu optimizatorul definit
model.compile(optimizer=optimizer_arie, loss='mse')
model.summary()
history = model.fit(X_train, Y_train, epochs=300, validation_split=0.2)

# Predicții
yhat = model.predict(X_test)

# Plotarea istoricului pierderii
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoci')
plt.legend()
plt.show()

# Evaluarea vizuală a predicțiilor
plt.plot(Y_test, 'red', label='Realitatea')
plt.plot(yhat, 'green', label='Predictia')
plt.title('Evaluarea modelului pentru aria tumorii')
plt.xlabel('Number of samples')
plt.ylabel('Measured value')
plt.legend()
plt.show()

# Calcularea erorilor
mse_value = np.mean(np.square(np.subtract(Y_test, yhat)))
mae_value = np.min(np.square(np.subtract(Y_test, yhat)))
r2 = r2_score(Y_test, yhat)
print("Eroarea medie patratica (MSE):", mse_value)
print("Eroarea minima (MAE):", mae_value)
print("Coeficinetul de determinare (R²):", r2)

#CLASIFICARE


# # Distribuția clasei 'diagnosis'
# counts = dataset['diagnosis'].value_counts()
# print(counts)
#
# # Codificarea etichetelor
# etichete = LabelEncoder()
# dataset["diagnosis"] = etichete.fit_transform(dataset["diagnosis"])
#
# # Vizualizarea distribuției claselor
# explode = (0, 0.05)
#
#
# # roz si portocaliu
# colors = ['#FF69B4', '#FFA500']
# # Crearea plotului
# plt.figure(figsize=(8,8))
# counts.plot(kind='pie', fontsize=12, explode=explode, autopct='%.1f%%', colors=colors)
# plt.title('Diagramă pentru clasificarea tipului tumorii')
# plt.xlabel('Diagnostic', fontsize=10)
# plt.ylabel('Cazuri', fontsize=10)
# # Adăugarea informației în legenda
# plt.legend(labels=[f'{label} ({count})' for label, count in counts.items()], loc="best")
# plt.show()
#
# # Definirea intrarilor si iesirilor
# #din intrare am taiat coloana cu diagnostic și de id
# x2 = dataset.drop(columns=["diagnosis", 'id']).values
# y2 = dataset["diagnosis"]
#
# #preprocesare
# # Normalizarea datelor de intrare
# scalernormalizare = MinMaxScaler()
# x2 = scalernormalizare.fit_transform(x2)
#
# # Standardizarea datelor de intrare
# # scaler = StandardScaler()
# # x2= scaler.fit_transform(x2)
#
#
# # Împărțirea datelor în seturi de antrenament și de testare
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=42)
#
#
# def clasiificare(x2_train, y2_train, input_dim=30, epochs=200, validation_split=0.2):
#     model2 = Sequential()
#     model2.add(Dense(15, activation='relu', input_dim=input_dim))
#     model2.add(Dense(8, activation='relu'))
#     model2.add(Dense(1, activation='sigmoid'))
#     optimizer_clasificare = Adam(learning_rate=0.001)
#
#     # Compilarea modelului
#     model2.compile(optimizer=optimizer_clasificare, loss='binary_crossentropy', metrics=['accuracy'])
#     model2.summary()
#     # Antrenarea modelului
#     istorie = model2.fit(x2_train, y2_train, epochs=epochs, validation_split=validation_split)
#     return model2, istorie
#
# model2, istorie = clasiificare(x2_train, y2_train, input_dim=30, epochs=200, validation_split=0.2)
#
# # Afișarea istoriei pierderii și acurateței
# tr_loss2 = istorie.history['loss']
# val_loss2 = istorie.history['val_loss']
# tr_acc2 = istorie.history['accuracy']
# val_acc2 = istorie.history['val_accuracy']
#
# # Crearea listei cu epoci
# Epochs = [i + 1 for i in range(len(tr_loss2))]
#
# # Plotarea istoricului pierderii
# plt.figure()
# plt.plot(Epochs, tr_loss2, 'blue', label='Training loss')
# plt.plot(Epochs, val_loss2, 'pink', label='Validation loss')
# plt.title('Loss pe antrenare vs validare')
# plt.xlabel('Epoci')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(False)  # Elimină gridul
# plt.show()
#
# # Plotarea istoricului acurateței
# plt.figure()
# plt.plot(Epochs, tr_acc2, 'blue', label='Training accuracy')
# plt.plot(Epochs, val_acc2, 'red', label='Validation accuracy')
# plt.title('Acuratețe pe validare vs antrenare')
# plt.xlabel('Epoci')
# plt.ylabel('Acuratete')
# plt.legend()
# plt.grid(False)  # Elimină gridul
# plt.show()
#
# # Predicții pe setul de testare
# y_pred2 = model2.predict(x2_test)
#
# # Afișarea graficului valorilor prezise vs. valorilor reale
# plt.figure()
#
# plt.scatter(range(len(y2_test)), y2_test, color='red', label='Realitate', alpha=0.5)
# plt.scatter(range(len(y_pred2)), y_pred2, color='blue', label='Predicție', alpha=0.5)
# plt.xlabel('Sample')
# plt.ylabel('Rezultat')
# plt.title('Realitate vs Predicție')
# plt.legend()
# plt.show()
#
# #metrici utilizate
# # Calcularea scorului R2
# R2_clasificare = r2_score(y2_test, y_pred2)
# print("R2 Score pentru clasificare =", R2_clasificare)
#
# # Evaluarea acuratetii modelului pe setul de testare
# test_loss, test_accuracy = model2.evaluate(x2_test, y2_test)
# print("Acuratetea pe testare penrtu clasificare:", test_accuracy)
