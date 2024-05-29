import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.layers import Dense, Dropout
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
# Citirea datelor
df = pd.read_csv("C:\\Users\\Admin\\Desktop\\skolika\\data.csv")

# Afișarea primelor 5 rânduri ale dataset-ului
print(df.head())
print(df.info())
print(df.describe())
output = df[['area_mean']].values

# Eliminarea coloanelor 'area_mean', 'id', și 'diagnosis' din input
input = df.drop(columns=['area_mean', 'id', 'diagnosis']).values

# Afișarea descrierii pentru input
input_df = pd.DataFrame(input)
print(input_df.info())
print(input_df.describe())

# Separarea datelor în seturi de antrenare și de test
X_train, X_test, Y_train, Y_test = train_test_split(input, output, test_size=0.2, random_state=42)

# Reshape output
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

# Normalizarea caracteristicilor de intrare
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Normalizarea caracteristicilor de ieșire
sc_Y = MinMaxScaler()
Y_train = sc_Y.fit_transform(Y_train)
Y_test = sc_Y.transform(Y_test)

# Definirea modelului
model = Sequential()

# Adăugarea straturilor ascunse cu regularizare l1 și Dropout
model.add(Dense(15, activation='relu', input_dim=29))  # actualizat la numărul de coloane de intrare
# model.add(Dropout(0.001))
model.add(Dense(8, activation='relu'))

# Adăugarea stratului de ieșire
model.add(Dense(1, activation='linear'))

# Creare optimizator cu o rată de învățare specificată
optimizer = Adam(learning_rate=0.001)

# Compilarea modelului cu optimizatorul definit
model.compile(optimizer=optimizer, loss='mse')

# Sumarul modelului
model.summary()

# Antrenarea modelului
history = model.fit(X_train, Y_train, epochs=200, validation_split=0.2)

# Predicții și evaluare
yhat = model.predict(X_test)

# Calcularea și afișarea MSE
mse = mean_squared_error(Y_test, yhat)
print(f"Mean Squared Error (MSE): {mse}")

# Plotarea istoricului pierderii
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Evaluarea vizuală a predicțiilor
plt.plot(Y_test, 'red', label='Real Output')
plt.plot(yhat, 'green', label='Predicted Output')
plt.title('Model Evaluation')
plt.xlabel('Number of samples')
plt.ylabel('Measured value')
plt.legend()
plt.show()

# Calcularea erorilor
mse_value = np.mean(np.square(np.subtract(Y_test, yhat)))
mae_value = np.min(np.square(np.subtract(Y_test, yhat)))

print("Mean Squared Error (MSE):", mse_value)
print("Minimal Error (MAE):", mae_value)
mse = mean_squared_error(Y_test, yhat)
r2 = r2_score(Y_test, yhat)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")



# # Distribuția clasei 'diagnosis'
# counts = df['diagnosis'].value_counts()
# print(counts)
#
# # Codificarea etichetelor
# le = LabelEncoder()
# df["diagnosis"] = le.fit_transform(df["diagnosis"])
#
# # Vizualizarea distribuției claselor
# plt.figure(figsize=(20, 6))
# explode = (0, 0.05)
# counts.plot(kind='pie', fontsize=12, explode=explode, autopct='%.1f%%')
# plt.title('Diagnosis')
# plt.xlabel('Diagnosis', weight="bold", color="#000000", fontsize=14, labelpad=20)
# plt.ylabel('Counts', weight="bold", color="#000000", fontsize=14, labelpad=20)
# plt.legend(labels=counts.index, loc="best")
# plt.show()
#
# # Eliminarea coloanei 'id'
# df.drop(columns=["id"], inplace=True)
#
# # Separarea datelor de intrare și ieșire
# x2 = df.drop(columns=["diagnosis"])
# y2 = df["diagnosis"]
#
# # Normalizarea datelor de intrare
# scaler = MinMaxScaler()
# x2 = scaler.fit_transform(x2)
#
# # Împărțirea datelor în seturi de antrenament și de testare
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=42)
#
# # Definirea modelului Sequential
# model2 = Sequential()
# model2.add(Dense(15, activation='relu', input_dim=30))
# model2.add(Dense(8, activation='relu'))
# model2.add(Dense(1, activation='sigmoid'))
#
# # Compilarea modelului
# model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Afișarea sumarului modelului
# model2.summary()
#
# # Antrenarea modelului
# istorie = model2.fit(x2_train, y2_train, epochs=200, validation_split=0.2)
#
# # Afișarea istoriei pierderii și acurateței
# tr_loss2 = istorie.history['loss']
# val_loss2 = istorie.history['val_loss']
# tr_acc2 = istorie.history['accuracy']
# val_acc2 = istorie.history['val_accuracy']
#
# # Determinarea epocii cu cea mai mică pierdere de validare
# index_loss2 = np.argmin(val_loss2)
# val_lowest2 = val_loss2[index_loss2]
#
# # Crearea listei cu epoci
# Epochs = [i + 1 for i in range(len(tr_loss2))]
# loss_label = f'best epoch= {str(index_loss2 + 1)}'
#
# # Plotarea istoricului pierderii
# plt.figure(figsize=(20, 8))
# plt.style.use('fivethirtyeight')
# plt.plot(Epochs, tr_loss2, 'r', label='Training loss')
# plt.plot(Epochs, val_loss2, 'g', label='Validation loss')
# plt.scatter(index_loss2 + 1, val_lowest2, s=150, c='blue', label=loss_label)
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # Plotarea istoricului acurateței
# plt.figure(figsize=(20, 8))
# plt.style.use('fivethirtyeight')
# plt.plot(Epochs, tr_acc2, 'r', label='Training accuracy')
# plt.plot(Epochs, val_acc2, 'g', label='Validation accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # Predicții pe setul de testare
# y_pred2 = model2.predict(x2_test)
#
# # Afișarea graficului valorilor prezise vs. valorilor reale
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(y2_test)), y2_test, color='red', label='Actual', alpha=0.5)
# plt.scatter(range(len(y_pred2)), y_pred2, color='blue', label='Predicted', alpha=0.5)
# plt.xlabel('Sample')
# plt.ylabel('Outcome')
# plt.title('Actual vs Predicted')
# plt.legend()
# plt.show()
# # Calcularea scorului R2
# R2 = r2_score(y2_test, y_pred2)
# print("R2 Score =", R2)