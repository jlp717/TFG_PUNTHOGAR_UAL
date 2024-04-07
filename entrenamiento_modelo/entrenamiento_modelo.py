import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Cargar y procesar los datos del archivo
def cargar_y_procesar_datos(archivo):
    df = pd.read_csv(archivo, sep='#')
    preguntas = df['pregunta'].tolist()
    categorias = df['categoria'].tolist()

    # Convertir categorías a números
    label_encoder = LabelEncoder()
    categorias_numeros = label_encoder.fit_transform(categorias)

    # Crear y adaptar el tokenizador
    tokenizer = TextVectorization(output_mode='int')
    tokenizer.adapt(preguntas)

    return tokenizer(preguntas), categorias_numeros, tokenizer, label_encoder

# Construir el modelo LSTM
def construir_modelo(tokenizer, num_clases):
    model = Sequential([
        Embedding(input_dim=len(tokenizer.get_vocabulary()), output_dim=64, mask_zero=True),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(num_clases, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Entrenar el modelo
def entrenar_modelo(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)



# Probar el modelo y guardar
def probar_y_guardar_modelo(model, tokenizer, label_encoder, archivo):
    while True:
        pregunta = input("Introduce una pregunta (o 'exit' para salir): ")
        if pregunta.lower() == 'exit':
            model.save('modelo_entrenado.h5')
            print(f"Modelo guardado")
            break

        pregunta_vectorizada = tokenizer([pregunta])
        prediccion = model.predict(pregunta_vectorizada)
        categoria_predicha = label_encoder.inverse_transform([np.argmax(prediccion)])[0]
        print("Categoría predicha:", categoria_predicha)

        correcto = input("¿Es correcta la predicción? (s/n): ")
        categoria_usada = categoria_predicha if correcto.lower() == 's' else input("Indica la categoría correcta: ")

        # Guardar en el archivo usando codificación UTF-8
        with open(archivo, 'a+', encoding='utf-8') as f:
            f.seek(0, 2)  # Moverse al final del archivo
            if f.tell() > 0:  # Si el archivo no está vacío, añadir un salto de línea al inicio
                f.write("\n")
            f.write(f"{pregunta}#{categoria_usada}")
            print(f"Registro guardado: {pregunta} -> {categoria_usada}")

if __name__ == "__main__":
    archivo_entrenamiento = "C:/Users/Javier/Documents/JAVIER-UAL/4º AÑO/TFG/punthogar_flask/TFG_PUNTHOGAR_UAL/entrenamiento_modelo/dataset/preguntas_ia.txt"
    x_train, y_train, tokenizer, label_encoder = cargar_y_procesar_datos(archivo_entrenamiento)
    model = construir_modelo(tokenizer, len(set(y_train)))
    entrenar_modelo(model, x_train, y_train)
    probar_y_guardar_modelo(model, tokenizer, label_encoder, archivo_entrenamiento)
