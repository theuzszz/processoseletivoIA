# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Matheus Souza da Silva Leite

### 1️⃣ Resumo da Arquitetura do Modelo

O modelo implementado é uma rede neural convolucional sequencial para classificação das dez classes do MNIST. As imagens foram normalizadas para o intervalo [0, 1] e tiveram seu formato ajustado de `(28, 28)` para `(28, 28, 1)`, acrescentando explicitamente o canal de escala de cinza.

Foi realizado um split explícito do conjunto originalmente destinado ao treinamento: 54.000 imagens foram usadas para treino e 6.000 para validação. O conjunto oficial de teste, com 10.000 imagens, foi mantido separado e utilizado somente para a avaliação final.

A CNN possui três blocos convolucionais. O primeiro utiliza 16 filtros, o segundo 32 e o terceiro 64, todos com kernels 3x3 e `padding="same"`. Cada bloco é composto por `Conv2D`, `BatchNormalization`, ativação ReLU e `MaxPooling2D` 2x2. Depois dos blocos, a saída é achatada com `Flatten`, processada por uma camada densa de 64 neurônios com ReLU e regularizada por um `Dropout` de 40%. A camada de saída possui 10 neurônios com ativação softmax, um para cada dígito de 0 a 9.

Foi utilizado o otimizador Adam com taxa de aprendizado de 0,001, batch size de 128 e limite de 12 épocas. O treinamento empregou `EarlyStopping` monitorando `val_loss`, com paciência de duas épocas e `restore_best_weights=True`.

A quantidade progressiva de 16, 32 e 64 filtros foi escolhida para permitir que as primeiras camadas identificassem características simples, como bordas, enquanto as camadas seguintes combinassem essas informações em padrões mais complexos. O batch size de 128 foi escolhido como equilíbrio entre estabilidade do gradiente, consumo de memória e velocidade de treinamento em CPU. A paciência de duas épocas evita interromper o treinamento por causa de uma única oscilação da perda de validação, mas também impede a execução desnecessária de todas as 12 épocas quando não há melhora.

O treinamento foi configurado exclusivamente para CPU e utilizou seed 42 para aumentar a reprodutibilidade.

### 2️⃣ Bibliotecas Utilizadas

As principais tecnologias e versões utilizadas foram:

- Python 3.11.15;
- TensorFlow 2.21.0;
- Keras 3.12.3, utilizado pela interface `tf.keras`;
- NumPy 2.2.6;
- `pathlib`, da biblioteca padrão do Python, para manipulação segura dos caminhos dos artefatos;
- TensorFlow Lite, disponibilizado pelo próprio TensorFlow, para conversão, otimização e inferência do modelo de Edge AI.

### 3️⃣ Técnica de Otimização do Modelo

A técnica utilizada foi a **Dynamic Range Quantization**, aplicada durante a conversão do modelo Keras para TensorFlow Lite.

O modelo `model.h5` foi carregado com `tf.keras.models.load_model()` e convertido por meio de `tf.lite.TFLiteConverter.from_keras_model()`. A otimização foi ativada explicitamente com:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

### 4️⃣ Métricas e Tamanhos dos Modelos

O treinamento utilizou um split explícito de 54.000 imagens para treino e 6.000 imagens para validação. A métrica utilizada foi a acurácia, adequada para a classificação multiclasse balanceada do MNIST.

- Melhor acurácia de validação observada durante o treinamento: **99,00%**;
- Acurácia no conjunto completo de teste: **98,85%**;
- Acurácia do `model.h5` em 300 amostras no validador oficial: **99,33%**;
- Acurácia do `model.tflite` em 300 amostras no validador oficial: **99,33%**;
- Tamanho real do `model.h5`: **788,91 KB**;
- Tamanho real do `model.tflite`: **69,76 KB**;
- Redução de tamanho após a otimização: **91,16%**, equivalente a aproximadamente **719,15 KB**.

A igualdade de acurácia entre o modelo Keras e o modelo TensorFlow Lite nas 300 amostras indica que a quantização dinâmica preservou o desempenho nessa avaliação.

### 5️⃣ Dificuldades, Decisões e Limitações

A principal dificuldade encontrada foi a compatibilidade de serialização do arquivo H5 entre versões diferentes do Keras. O modelo gerado inicialmente não era carregado pelo ambiente Python 3.10 do GitHub Actions porque havia sido salvo com uma versão mais recente do Keras. Para tornar a execução reproduzível, as versões foram fixadas no `requirements.txt` em TensorFlow 2.21.0, Keras 3.12.3 e NumPy 2.2.6, e o modelo foi gerado novamente nesse ambiente compatível.

A arquitetura foi mantida com três blocos convolucionais e filtros progressivos de 16, 32 e 64 para equilibrar capacidade de aprendizado, tempo de treinamento em CPU e tamanho do artefato para Edge AI. O `Dropout` de 0,4 e o `EarlyStopping` com paciência de duas épocas foram escolhidos para reduzir sobreajuste e evitar épocas desnecessárias.

Como limitação, o treinamento foi executado somente em CPU e o MNIST possui imagens simples em tons de cinza. Portanto, os resultados não devem ser generalizados diretamente para bases coloridas ou cenários visuais mais complexos.

### 6️⃣ Inferência com o Modelo Otimizado

O script `run_inference.py` carregou especificamente o arquivo `model.tflite` e executou cinco amostras do conjunto de teste. A saída observada foi:

```text
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

Nas cinco amostras observadas, todas as previsões coincidiram com os rótulos reais. O caso do dígito 7, por exemplo, foi classificado corretamente mesmo após a conversão e a quantização dinâmica. Essa pequena amostra confirma o funcionamento da inferência, mas a acurácia calculada no conjunto de validação continua sendo uma evidência mais representativa do desempenho geral.
