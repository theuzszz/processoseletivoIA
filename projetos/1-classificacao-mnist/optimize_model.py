# Entrega do desafio técnico - Matheus Souza da Silva Leite

import os
from pathlib import Path

# Mantém a execução em CPU e reduz mensagens informativas do TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


def file_size_kb(path):
    """Retorna o tamanho de um arquivo em kilobytes."""

    return path.stat().st_size / 1024


def main():
    script_directory = Path(__file__).resolve().parent
    keras_model_path = script_directory / "model.h5"
    tflite_model_path = script_directory / "model.tflite"

    if not keras_model_path.is_file():
        raise FileNotFoundError(
            "O arquivo model.h5 não foi encontrado. "
            "Execute train_model.py antes da otimização."
        )

    print(f"Carregando modelo: {keras_model_path}")

    # compile=False é suficiente porque não faremos novo treinamento.
    model = tf.keras.models.load_model(
        keras_model_path,
        compile=False,
    )

    print("Convertendo o modelo para TensorFlow Lite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Dynamic Range Quantization:
    # quantiza principalmente os pesos do modelo, reduzindo seu tamanho.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    tflite_model_path.write_bytes(tflite_model)

    keras_size = file_size_kb(keras_model_path)
    tflite_size = file_size_kb(tflite_model_path)
    reduction = (1 - tflite_size / keras_size) * 100

    print("\nResultados da otimização")
    print("-" * 40)
    print(f"Técnica: Dynamic Range Quantization")
    print(f"Tamanho do model.h5: {keras_size:.2f} KB")
    print(f"Tamanho do model.tflite: {tflite_size:.2f} KB")
    print(f"Redução de tamanho: {reduction:.2f}%")

    # Teste básico para confirmar que o arquivo pode ser aberto.
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_model_path)
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nVerificação do modelo TFLite")
    print("-" * 40)
    print(f"Entrada: {input_details[0]['shape']}")
    print(f"Tipo da entrada: {input_details[0]['dtype']}")
    print(f"Saída: {output_details[0]['shape']}")
    print(f"Modelo otimizado salvo em: {tflite_model_path}")


if __name__ == "__main__":
    main()
