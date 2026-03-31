import pandas as pd    #бібліотека для роботи з csv електронними таблицями
import numpy as np
import tensorflow as tf    #бібліотека нейронних мереж від гугла
from tensorflow import keras    #робота з aпі
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder    #функція яка перетворює текстові мітки в числа
import matplotlib.pyplot as plt    #функція яка працбє з графіками

df = pd.read_csv('data/figures.csv')
print(df.head())

encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

X = df[['area', 'perimeter', 'corners']] #ознаки
y = df['label_enc'] #мітки

model = keras.Sequential([layers.Dense(8, activation='relu', input_shape=(3, )),
                          layers.Dense(8, activation = 'relu'), #прихований шар
                          layers.Dense(3, activation = 'softmax')
                          ])
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']) #ваги

history = model.fit(X, y, epochs = 200, verbose= 0) #навчання нейронки

plt.plot(history.history['loss'], label = 'Втрати')
plt.plot(history.history['accuracy'], label = 'Точність')
plt.xlabel('Епохи')
plt.ylabel('Значення')
plt.title('Процес навчання моделі')
plt.legend()
plt.show()

test = np.array([16, 16, 0])

pred = model.predict(test)
print(f'Ймовірність кожного класу {pred}')
print(f'Результат {encoder.inverse_transform(np.argmax(pred))}')