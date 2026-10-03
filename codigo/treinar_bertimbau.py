"""
Análise de sentimento de críticas de cinema em português com deep learning
Trabalho da disciplina Redes Neurais e Deep Learning (MBA)
Autores: Samir Saraiva, Claudio Cruz, Vinicius Ferreira e João Felipe

ETAPA 2 — Modelo 2: BERTimbau com fine-tuning completo (Keras / KerasHub)

BERTimbau (SOUZA; NOGUEIRA; LOTUFO, 2020): BERT-base pré-treinado em português
do Brasil (brWaC, ~2,7 bilhões de tokens). Pesos:
https://huggingface.co/neuralmind/bert-base-portuguese-cased
Na primeira execução, o script baixa ~440 MB para dados/bertimbau/.

Modelo: BertBackbone (12 camadas, 768 dimensões, ~108 milhões de parâmetros, todos
treináveis) → token [CLS] → Dropout 0,1 → saída sigmoide. Entrada: 200 palavras
(início + fim da crítica), 256 tokens. AdamW, lr 2e-5 com aquecimento de 10% e
decaimento cosseno, lote 16, 2 épocas, precisão mista (float16).

Onde rodou: Google Colab, GPU NVIDIA T4 (16 GB), ~525 ms por lote; cada treino
levou de 5 a 21 minutos. Comando usado no Colab:
  python -u treinar_bertimbau.py --cenario <imdb|nativo|imdb+nativo> --batch 16

Resultados (F1 macro):          IMDB   AdoroCinema   Papo de Cinema   Ouro
  treino IMDB PT-BR              0,911     0,947          0,838        0,587
  treino nativo                  0,877     0,880          0,870        0,799
  treino IMDB + nativo           0,917     0,954          0,892        0,807   ← melhor modelo
No conjunto ouro, o melhor modelo acertou 82% (IC 95%: 69%–90%; p < 0,001 contra
a linha de base de 56%). Treinado só com o IMDB traduzido, reconhecia 6 de 22
críticas negativas; com críticas nativas, até 18 de 22.

GPU com pouca memória (ex.: RTX 3050 Laptop com ~2 GB livres): o fine-tuning
completo precisa de ~3 GB. Alternativas: --congelar 8 (ajusta só as 4 camadas
finais), --batch 4, --max-len 192, ou liberar a GPU (modo gráfico híbrido).
"""

import argparse
import os
import time

import numpy as np

import comum

PASTA_BERT = os.path.join(comum.PASTA_DADOS, "bertimbau")
HF = "https://huggingface.co/neuralmind/bert-base-portuguese-cased/resolve"
ARQUIVOS = {
    "config.json": f"{HF}/main/config.json",
    "tokenizer_config.json": f"{HF}/main/tokenizer_config.json",
    "vocab.txt": f"{HF}/main/vocab.txt",
    # O repositório principal não tem safetensors; a versão convertida está na PR nº 5
    "model.safetensors": f"{HF}/refs%2Fpr%2F5/model.safetensors",
}


def baixar_bertimbau():
    import requests
    os.makedirs(PASTA_BERT, exist_ok=True)
    for nome, url in ARQUIVOS.items():
        destino = os.path.join(PASTA_BERT, nome)
        if os.path.exists(destino) and os.path.getsize(destino) > 0:
            continue
        print(f"   Baixando {nome}...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            feito = 0
            with open(destino + ".parcial", "wb") as f:
                for bloco in r.iter_content(1 << 20):
                    f.write(bloco)
                    feito += len(bloco)
                    if total > 5e6:
                        print(f"\r      {feito / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
            if total > 5e6:
                print()
        os.replace(destino + ".parcial", destino)


def carregar_backbone(pasta):
    """Monta o BertBackbone do KerasHub e copia os pesos do BERTimbau (safetensors).
    A cópia é feita aqui mesmo, sem depender do conversor interno do KerasHub, cujo
    comportamento muda entre versões (ex.: Colab x PC). Aceita as duas convenções de
    nomes (LayerNorm.gamma/beta ou weight/bias, com ou sem prefixo 'bert.').
    O 'pooler' não é usado (classificamos pelo token [CLS])."""
    import json
    import keras_hub
    from safetensors import safe_open

    with open(os.path.join(pasta, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    backbone = keras_hub.models.BertBackbone(
        vocabulary_size=cfg["vocab_size"], num_layers=cfg["num_hidden_layers"],
        num_heads=cfg["num_attention_heads"], hidden_dim=cfg["hidden_size"],
        intermediate_dim=cfg["intermediate_size"], max_sequence_length=cfg["max_position_embeddings"])

    with safe_open(os.path.join(pasta, "model.safetensors"), framework="np") as st:
        chaves = set(st.keys())

        def achar(nome):
            variantes = {nome, nome.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias"),
                         nome.replace("LayerNorm.weight", "LayerNorm.gamma").replace("LayerNorm.bias", "LayerNorm.beta")}
            for v in variantes:
                for c in (v, "bert." + v):
                    if c in chaves:
                        return c
            raise KeyError(f"'{nome}' não encontrado no checkpoint. Exemplos de chaves: {sorted(chaves)[:8]}")

        def copiar(variavel, nome, transf=None):
            t = st.get_tensor(achar(nome))
            if transf is not None:
                t = transf(t, tuple(variavel.shape))
            if tuple(t.shape) != tuple(variavel.shape):
                raise ValueError(f"Formato incompatível em {nome}: {t.shape} vs {variavel.shape}")
            variavel.assign(t)

        T = lambda t, forma: np.reshape(np.transpose(t), forma)    # Linear do PyTorch -> EinsumDense do Keras
        R = lambda t, forma: np.reshape(t, forma)
        TT = lambda t, forma: np.transpose(t)                       # Linear do PyTorch -> Dense do Keras

        copiar(backbone.get_layer("token_embedding").embeddings, "embeddings.word_embeddings.weight")
        copiar(backbone.get_layer("position_embedding").position_embeddings, "embeddings.position_embeddings.weight")
        copiar(backbone.get_layer("segment_embedding").embeddings, "embeddings.token_type_embeddings.weight")
        copiar(backbone.get_layer("embeddings_layer_norm").gamma, "embeddings.LayerNorm.weight")
        copiar(backbone.get_layer("embeddings_layer_norm").beta, "embeddings.LayerNorm.bias")
        for i in range(backbone.num_layers):
            bloco = backbone.get_layer(f"transformer_layer_{i}")
            att = bloco._self_attention_layer
            p = f"encoder.layer.{i}."
            for nome_keras, nome_hf in [("query_dense", "query"), ("key_dense", "key"), ("value_dense", "value")]:
                camada = getattr(att, nome_keras)
                copiar(camada.kernel, f"{p}attention.self.{nome_hf}.weight", T)
                copiar(camada.bias, f"{p}attention.self.{nome_hf}.bias", R)
            copiar(att.output_dense.kernel, f"{p}attention.output.dense.weight", T)
            copiar(att.output_dense.bias, f"{p}attention.output.dense.bias", R)
            copiar(bloco._self_attention_layer_norm.gamma, f"{p}attention.output.LayerNorm.weight")
            copiar(bloco._self_attention_layer_norm.beta, f"{p}attention.output.LayerNorm.bias")
            copiar(bloco._feedforward_intermediate_dense.kernel, f"{p}intermediate.dense.weight", TT)
            copiar(bloco._feedforward_intermediate_dense.bias, f"{p}intermediate.dense.bias")
            copiar(bloco._feedforward_output_dense.kernel, f"{p}output.dense.weight", TT)
            copiar(bloco._feedforward_output_dense.bias, f"{p}output.dense.bias")
            copiar(bloco._feedforward_layer_norm.gamma, f"{p}output.LayerNorm.weight")
            copiar(bloco._feedforward_layer_norm.beta, f"{p}output.LayerNorm.bias")
    print(f"   Pesos do BERTimbau carregados ({backbone.num_layers} camadas).")
    return backbone


def construir(args):
    import keras
    import keras_hub

    # Tokenizador WordPiece do próprio BERTimbau (29.794 subpalavras). O modelo é "cased":
    # maiúsculas são preservadas, por isso lowercase=False.
    tokenizer = keras_hub.models.BertTokenizer(vocabulary=os.path.join(PASTA_BERT, "vocab.txt"), lowercase=False)
    # O pré-processador acrescenta [CLS] no início e [SEP] no fim, trunca/completa até max_len
    # e gera as três entradas do BERT: token_ids, segment_ids e padding_mask.
    preproc = keras_hub.models.BertTextClassifierPreprocessor(tokenizer, sequence_length=args.max_len)
    backbone = carregar_backbone(PASTA_BERT)
    backbone.get_layer("pooled_dense").trainable = False   # não usado (classificamos pelo [CLS])

    # Congelamento opcional das camadas iniciais (só para GPUs com pouca memória).
    # Na versão final (Colab, T4 16 GB) usamos --congelar 0: fine-tuning completo.
    if args.congelar:
        backbone.get_layer("token_embedding").trainable = False
        for i in range(args.congelar):
            backbone.get_layer(f"transformer_layer_{i}").trainable = False

    # Cabeça de classificação: o vetor do token [CLS] da última camada resume a crítica inteira
    # (graças à atenção, ele "vê" todos os outros tokens). Dropout + uma saída sigmoide = P(positivo).
    # A saída fica em float32 mesmo com precisão mista, para a perda ser calculada com estabilidade.
    entradas = backbone.input
    seq = backbone(entradas)["sequence_output"]
    cls = seq[:, 0, :]                               # representação do token [CLS]
    x = keras.layers.Dropout(0.1)(cls)
    saida = keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="classificador")(x)
    modelo = keras.Model(entradas, saida)
    return modelo, preproc


def tokenizar(preproc, textos, max_palavras, lote=512):
    """Tokeniza tudo de uma vez (em lotes) e devolve arrays numpy."""
    partes = {"token_ids": [], "segment_ids": [], "padding_mask": []}
    textos = [comum.inicio_e_fim(t, max_palavras) for t in textos]
    for i in range(0, len(textos), lote):
        x = preproc(textos[i:i + lote])
        for k in partes:
            partes[k].append(np.asarray(x[k]))
    return {k: np.concatenate(v) for k, v in partes.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cenario", choices=list(comum.CENARIOS), default="imdb")
    ap.add_argument("--amostra-imdb", type=int, default=15000, help="exemplos de treino do IMDB (BERT converge com poucos)")
    ap.add_argument("--max-len", type=int, default=256, help="tokens por texto (máx. 512)")
    ap.add_argument("--palavras", type=int, default=200, help="palavras mantidas (início + fim) antes de tokenizar")
    ap.add_argument("--batch", type=int, default=8, help="lote (usamos 16 no Colab/T4; 8 ou menos em GPUs de 4 GB)")
    ap.add_argument("--acumular", type=int, default=1, help="acumulação de gradiente (lote efetivo = batch x acumular)")
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--congelar", type=int, default=0, help="nº de camadas iniciais do BERT a congelar (0-12)")
    ap.add_argument("--sem-fp16", action="store_true", help="desativa precisão mista")
    args = ap.parse_args()

    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")  # reduz fragmentação de memória
    import tensorflow as tf
    import keras
    keras.utils.set_random_seed(comum.SEED)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Precisão mista: cálculos em float16 (mais rápidos e com metade da memória) e pesos em float32.
    # O Keras aplica escala automática da perda para evitar underflow dos gradientes.
    if gpus and not args.sem_fp16:
        keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"   GPU: {'sim' if gpus else 'NÃO (vai ser muito lento em CPU)'} | precisão: {keras.mixed_precision.global_policy().name}")

    baixar_bertimbau()
    treino, valid, testes = comum.montar_dados(args.cenario, args.amostra_imdb)

    modelo, preproc = construir(args)
    print("\n   Tokenizando...")
    x_tr = tokenizar(preproc, treino["texto"].tolist(), args.palavras)
    x_va = tokenizar(preproc, valid["texto"].tolist(), args.palavras)
    y_tr = treino["rotulo"].values.astype("float32")
    y_va = valid["rotulo"].values.astype("float32")

    # Receita padrão de fine-tuning do BERT (DEVLIN et al., 2019): AdamW com taxa pequena (2e-5),
    # aquecimento linear nos primeiros 10% dos passos e depois decaimento (aqui, cosseno).
    # Taxas altas "apagam" o que o modelo aprendeu no pré-treinamento (esquecimento catastrófico).
    passos = int(np.ceil(len(y_tr) / args.batch)) * args.epocas
    aquecimento = max(1, int(0.1 * passos))
    agenda = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0, warmup_target=args.lr, warmup_steps=aquecimento,
        decay_steps=max(1, passos - aquecimento))
    otimizador = keras.optimizers.AdamW(learning_rate=agenda, weight_decay=0.01,
                                        gradient_accumulation_steps=args.acumular if args.acumular > 1 else None)
    otimizador.exclude_from_weight_decay(var_names=["bias", "gamma", "beta"])   # sem decaimento em vieses e LayerNorm
    modelo.compile(optimizer=otimizador, loss="binary_crossentropy", metrics=["accuracy"],
                   jit_compile=False)   # XLA reserva buffers grandes; desligado para caber em GPUs de 4 GB
    treinaveis = sum(int(np.prod(v.shape)) for v in modelo.trainable_weights)
    print(f"   Parâmetros: {modelo.count_params():,} (treináveis: {treinaveis:,})")
    print(f"   Lote {args.batch} x acumulação {args.acumular} = lote efetivo {args.batch * args.acumular}; "
          f"{args.epocas} épocas; {len(y_tr):,} exemplos")

    # 2 épocas bastam: o modelo já chega pré-treinado. A parada antecipada (paciência 1) restaura
    # os pesos da época com menor perda de validação — nos três cenários, foi a época 1.
    inicio = time.time()
    try:
        hist = modelo.fit(
            x_tr, y_tr, validation_data=(x_va, y_va), epochs=args.epocas, batch_size=args.batch, verbose=1,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)])
    except tf.errors.ResourceExhaustedError:
        print("\n   FALTOU MEMÓRIA NA GPU. Tente, nesta ordem: --congelar 8 ; depois --batch 4 ; "
              "depois --max-len 192. Ou libere a GPU (LEIAME) ou use o Google Colab.")
        raise SystemExit(1)
    minutos = (time.time() - inicio) / 60

    os.makedirs(comum.PASTA_MODELOS, exist_ok=True)
    modelo.save_weights(os.path.join(comum.PASTA_MODELOS, f"bertimbau__{args.cenario}.weights.h5"))

    # Avaliação nos 4 conjuntos de teste (os mesmos da Bi-LSTM; ver comum.montar_dados)
    print("\n[AVALIAÇÃO]")
    resultados = []
    for nome, df in testes.items():
        x_te = tokenizar(preproc, df["texto"].tolist(), args.palavras)
        prob = modelo.predict(x_te, batch_size=args.batch * 4, verbose=0).ravel().astype(float)
        resultados.append(comum.avaliar(nome, df["rotulo"].values, prob))
        comum.salvar_previsoes("bertimbau", args.cenario, nome, df, prob)

    comum.salvar_resultados("bertimbau", args.cenario, resultados, info={
        "n_treino": len(treino), "epocas_rodadas": len(hist.history["loss"]),
        "melhor_val_loss": float(np.min(hist.history["val_loss"])), "minutos_treino": round(minutos, 1),
        "parametros": int(modelo.count_params()), "parametros_treinaveis": treinaveis,
        "hiperparametros": vars(args),
    })


if __name__ == "__main__":
    main()
