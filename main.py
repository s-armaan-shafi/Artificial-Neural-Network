
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv("Churn_Modelling (AI).csv")


df = df.dropna()
df = df.drop_duplicates()


df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)


label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])


df = pd.get_dummies(df, columns=['Geography'], drop_first=True)


X = df.drop('Exited', axis=1).values
y = df['Exited'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(units=16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))  # Binary Classification


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50, batch_size=10)


y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ ANN Model Test Accuracy: {accuracy:.2f}")
