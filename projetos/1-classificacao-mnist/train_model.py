# Entrega do desafio técnico - Matheus Souza da Silva Leite

import os
from pathlib import Path

# O desafio exige treinamento apenas em CPU.
# Esta variável precisa ser configurada antes de importar o TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf


SEED = 42
VALIDATION_SIZE = 6000
EPOCHS = 12
BATCH_SIZE = 128


def load_data():
    """Carrega, normaliza e divide o MNIST em treino, validação e teste."""

    (x_train_full, y_train_full), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Converte os pixels de 0-255 para valores entre 0 e 1.
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Adiciona o canal grayscale:
    # (quantidade, 28, 28) -> (quantidade, 28, 28, 1)
    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Split explícito:
    # 54.000 imagens para treino e 6.000 para validação.
    x_train = x_train_full[:-VALIDATION_SIZE]
    y_train = y_train_full[:-VALIDATION_SIZE]

    x_validation = x_train_full[-VALIDATION_SIZE:]
    y_validation = y_train_full[-VALIDATION_SIZE:]

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    )


def build_model():
    """Constrói uma CNN pequena e adequada para execução em Edge AI."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),

            # Primeiro bloco convolucional
            tf.keras.layers.Conv2D(
                16,
                kernel_size=(3, 3),
                padding="same",
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Segundo bloco convolucional
            tf.keras.layers.Conv2D(
                32,
                kernel_size=(3, 3),
                padding="same",
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Terceiro bloco convolucional
            tf.keras.layers.Conv2D(
                64,
                kernel_size=(3, 3),
                padding="same",
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Camadas finais de classificação
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="cnn_mnist_edge",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    # Torna os resultados mais reproduzíveis.
    tf.keras.utils.set_random_seed(SEED)

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) = load_data()

    print("Formato do conjunto de treino:", x_train.shape)
    print("Formato do conjunto de validação:", x_validation.shape)
    print("Formato do conjunto de teste:", x_test.shape)

    model = build_model()
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        shuffle=True,
        verbose=2,
    )

    validation_loss, validation_accuracy = model.evaluate(
        x_validation,
        y_validation,
        verbose=0,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    print("\nResultados finais")
    print("-" * 40)
    print(f"Épocas executadas: {len(history.history['loss'])}")
    print(f"Perda de validação: {validation_loss:.4f}")
    print(
        f"Acurácia de validação: "
        f"{validation_accuracy:.4f} ({validation_accuracy:.2%})"
    )
    print(f"Acurácia de teste: {test_accuracy:.4f} ({test_accuracy:.2%})")

    script_directory = Path(__file__).resolve().parent
    model_path = script_directory / "model.h5"

    model.save(model_path)

    print(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()
