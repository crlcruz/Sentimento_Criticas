"""
Análise de sentimento de críticas de cinema em português com deep learning
Trabalho da disciplina Redes Neurais e Deep Learning (MBA)
Autores: Samir Saraiva, Claudio Cruz, Vinicius Ferreira e João Felipe

ETAPA 2 — Modelo 1: Bi-LSTM treinada do zero (Keras)

Arquitetura: TextVectorization (20 mil palavras, 300 tokens) → Embedding 64 (com
máscara de padding) → Bi-LSTM 64 → Dropout 0,4 → Dense 32 ReLU → saída sigmoide.
~1,35 milhão de parâmetros. Adam, lote 32, até 15 épocas com parada antecipada
(paciência 3, restaura os melhores pesos pela perda de validação).
Treinada localmente numa RTX 3050 Laptop (cerca de 20 s por época no IMDB).

Resultados (F1 macro):          IMDB   AdoroCinema   Papo de Cinema   Ouro
  treino IMDB PT-BR              0,886     0,834          0,735        0,632
  treino nativo                  0,578     0,628          0,577        0,374
  treino IMDB + nativo           0,877     0,872          0,700        0,667
Com só 3.192 críticas nativas a rede decora o treino (98% no treino, 57% na
validação): sem pré-treinamento, poucos dados não bastam. Os resultados também se
mostraram sensíveis a hiperparâmetros (lote 128 → 32 mudou 9 pontos no Papo de Cinema).

Uso:
  python treinar_bilstm.py --cenario imdb
  python treinar_bilstm.py --cenario nativo
  python treinar_bilstm.py --cenario imdb+nativo
"""

import argparse
import os
import time

import numpy as np

import comum

VOCAB_SIZE = 20000
MAX_LEN = 300
BATCH = 32   # lote menor: garante passos de gradiente suficientes também nos cenários com poucos dados


def como_tensor(textos):
    import tensorflow as tf
    return tf.constant([[comum.inicio_e_fim(t, MAX_LEN)] for t in textos], dtype=tf.string)


def construir(textos_treino):
    from tensorflow import keras
    from tensorflow.keras import layers
    # Vocabulário aprendido do próprio treino (as 20 mil palavras mais frequentes).
    # Índice 0 = padding, 1 = palavra fora do vocabulário.
    vet = layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN,
                                   standardize="lower_and_strip_punctuation")
    vet.adapt(como_tensor(textos_treino))
    modelo = keras.Sequential([
        keras.Input(shape=(1,), dtype="string"),
        vet,
        # Embeddings aprendidos do zero; mask_zero faz a LSTM ignorar o padding.
        # (Por isso o comum.py descarta textos sem nenhuma palavra: sequência 100% padding
        #  derruba o LSTM do cuDNN na GPU.)
        layers.Embedding(VOCAB_SIZE, 64, mask_zero=True),
        # LSTM bidirecional: lê a crítica nos dois sentidos e junta os dois estados finais (2 x 64).
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.4),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return modelo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cenario", choices=list(comum.CENARIOS), default="imdb")
    ap.add_argument("--amostra-imdb", type=int, default=None, help="limita os exemplos de treino do IMDB")
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--paciencia", type=int, default=3, help="épocas sem melhora na validação antes de parar")
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(comum.SEED)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    treino, valid, testes = comum.montar_dados(args.cenario, args.amostra_imdb)
    modelo = construir(treino["texto"])
    modelo.summary()

    inicio = time.time()
    # Parada antecipada pela perda de validação; restaura os melhores pesos.
    # Lote 32 (e não 128) garante passos de gradiente suficientes no cenário nativo (3.192 textos):
    # com lote 128 e paciência 2, a rede parava antes de aprender (previa tudo negativo).
    hist = modelo.fit(
        como_tensor(treino["texto"]), treino["rotulo"].values,
        validation_data=(como_tensor(valid["texto"]), valid["rotulo"].values),
        epochs=args.epocas, batch_size=BATCH, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.paciencia, restore_best_weights=True)],
    )
    minutos = (time.time() - inicio) / 60

    os.makedirs(comum.PASTA_MODELOS, exist_ok=True)
    modelo.save(os.path.join(comum.PASTA_MODELOS, f"bilstm__{args.cenario}.keras"))

    print("\n[AVALIAÇÃO]")
    resultados = []
    for nome, df in testes.items():
        prob = modelo.predict(como_tensor(df["texto"]), batch_size=256, verbose=0).ravel()
        resultados.append(comum.avaliar(nome, df["rotulo"].values, prob))
        comum.salvar_previsoes("bilstm", args.cenario, nome, df, prob)

    comum.salvar_resultados("bilstm", args.cenario, resultados, info={
        "n_treino": len(treino), "epocas_rodadas": len(hist.history["loss"]),
        "melhor_val_loss": float(np.min(hist.history["val_loss"])), "minutos_treino": round(minutos, 1),
        "parametros": int(modelo.count_params()), "batch": BATCH, "paciencia": args.paciencia,
    })


if __name__ == "__main__":
    main()
