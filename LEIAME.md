# Análise de sentimento de críticas de cinema em português com deep learning

Trabalho da disciplina **Redes Neurais e Deep Learning** (MBA).
**Autores:** Samir Saraiva, Claudio Cruz, Vinicius Ferreira e João Felipe.

O projeto compara uma **Bi-LSTM treinada do zero** e o **BERTimbau com fine-tuning**,
ambos em Keras. Cada modelo foi treinado em três fontes de dados e avaliado em críticas
reais de cinema em português. O relatório completo está em `relatorio.pdf`.

## Resultado principal

O **BERTimbau treinado com IMDB PT-BR + críticas nativas** foi o melhor em todos os conjuntos de teste.

| F1 macro | IMDB (3.000) | AdoroCinema (169) | Papo de Cinema (916) | Conjunto ouro (50) |
|---|---|---|---|---|
| Bi-LSTM · IMDB | 0,886 | 0,834 | 0,735 | 0,632 |
| Bi-LSTM · nativo | 0,578 | 0,628 | 0,577 | 0,374 |
| Bi-LSTM · IMDB + nativo | 0,877 | 0,872 | 0,700 | 0,667 |
| BERTimbau · IMDB | 0,911 | 0,947 | 0,838 | 0,587 |
| BERTimbau · nativo | 0,877 | 0,880 | 0,870 | 0,799 |
| **BERTimbau · IMDB + nativo** | **0,917** | **0,954** | **0,892** | **0,807** |

- **Acurácia no teste engana.** O primeiro modelo, uma Bi-LSTM treinada no B2W, teve 95,4% no teste e só 46% nas críticas reais.
- **O domínio dos dados importa mais que o volume.** 3 mil críticas nativas superaram 15 mil traduzidas no conjunto ouro. O reconhecimento de críticas negativas foi de 27% para 82%.
- **O pré-treinamento faz diferença com poucos dados.** Só com as críticas nativas, a Bi-LSTM não aprendeu, enquanto o BERTimbau chegou a F1 de 0,870 no Papo de Cinema.
- **O melhor modelo acertou 82% do conjunto ouro** (IC 95%: 69%–90%), com p < 0,001 em relação à linha de base de 56%.

## Estrutura

```
codigo/
  sentimento_cinema_ptbr.py   Etapa 1: B2W vs IMDB PT-BR avaliados em críticas reais
  comum.py                    Etapa 2: dados, divisão, balanceamento e métricas (compartilhado)
  coletar_criticas.py         Etapa 2: coleta do Papo de Cinema e do AdoroCinema
  treinar_bilstm.py           Etapa 2: modelo 1 (Bi-LSTM)
  treinar_bertimbau.py        Etapa 2: modelo 2 (BERTimbau)
  comparar_resultados.py      Etapa 2: tabela e figura comparativas
  bertimbau_colab.ipynb       Execução do BERTimbau no Google Colab (com as saídas)
dados/
  corpus_criticas.csv         10.030 críticas coletadas (7.031 com rótulo)
  teste_ouro.csv              50 críticas do conjunto ouro (Rapadura + Omelete)
  rotulos_manuais.csv         rótulos manuais das críticas do Cinema com Rapadura
resultados/
  comparacao.md / .csv        tabela completa (acurácia, IC 95%, F1, recall por classe)
  comparacao_f1.png           Figura 1 do relatório
  bilstm__*.json, bertimbau__*.json, log_*.txt
  etapa1/                     resultados da etapa 1 (B2W e IMDB)
environment.yml               ambiente conda
```

Ficaram fora da entrega, por serem públicos ou grandes e reproduzíveis: o IMDB PT-BR
([Kaggle](https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr)), o
[B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01), os pesos do BERTimbau
(baixados automaticamente) e o cache HTML da coleta.

## Como reproduzir

1. **Ambiente** (Linux, conda). O `tensorflow-text` do pip é incompatível com o TensorFlow do
   conda-forge, e o KerasHub funciona sem ele:
   ```bash
   conda env create -f environment.yml && conda activate sentimento
   pip install keras-hub safetensors matplotlib && pip uninstall -y tensorflow-text
   ```
2. **Dados:** baixe o IMDB PT-BR do Kaggle e salve em `dados/imdb-reviews-pt-br.csv`.
   Para refazer a coleta, que leva cerca de 4h30:
   ```bash
   python coletar_criticas.py --max-paginas 400 --max-paginas-filmes 20 --paginas-por-filme 10 --pausa 1.0
   ```
3. **Bi-LSTM** (qualquer GPU; na RTX 3050 cada época leva cerca de 20 s):
   ```bash
   for c in imdb nativo imdb+nativo; do python treinar_bilstm.py --cenario $c; done
   ```
4. **BERTimbau** (GPU com 16 GB; usamos a T4 do Google Colab, com 5 a 21 minutos por cenário):
   ```bash
   for c in imdb nativo imdb+nativo; do python -u treinar_bertimbau.py --cenario $c --batch 16; done
   ```
   Em GPUs de 4 GB, use `--congelar 8 --batch 8`. Nesse caso, os resultados não são comparáveis aos do relatório.
5. **Comparação:** `python comparar_resultados.py`.

## Desenho experimental

- **Rótulos:** nota normalizada ≥ 0,7 vira positivo e ≤ 0,4 vira negativo. As notas intermediárias ficam de fora.
- **Divisão por filme** (70/15/15 por hash): todas as críticas de um filme ficam na mesma partição.
- **Treino balanceado** 50/50 por subamostragem. Os testes mantêm a distribuição natural.
- **Conjunto ouro:** fica isolado, nunca entra no treino, e o coletor exclui essas URLs.
- **Métricas:** acurácia com IC 95% de Wilson, linha de base, F1 macro e recall por classe.
- **Os dois modelos veem exatamente os mesmos conjuntos de treino, validação e teste.**
- **Coleta ética:** respeita o robots.txt, espera 1 s entre requisições e não guarda dados pessoais.
