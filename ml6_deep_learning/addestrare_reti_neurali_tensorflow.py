
# perc_diabetes_tensorflow.py

import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import hard_sigmoid


if __name__ == "__main__":
    # Carica il set di dati sul diabete da CSV
    # e converte in una matrice NumPy adatta per
    # l'estrazione nel formato X, y necessaria per TensorFlow
    diabetes = pd.read_csv('diabetes.csv').values

    # Estrae la matrice delle features e il vettore della risposta
    # in specifiche variabili
    X = diabetes[:, 0:8]
    y = diabetes[:, 8]

    # Crea il 'Perceptron' tramite le API Keras
    model = Sequential()
    model.add(Dense(1, input_shape=(8,), activation=hard_sigmoid, kernel_initializer='glorot_uniform'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Addestrare il  perceptron tramite la discesa stocastica del gradiente
    # con un set di validazione del 20%
    model.fit(X, y, epochs=225, batch_size=25, verbose=1, validation_split=0.2)

    # Valutazione dell'accuratezza del modello
    _, accuracy = model.evaluate(X, y)
    print("%0.3f" % accuracy)