"""
Análise de sentimento de críticas de cinema em português com deep learning
Trabalho da disciplina Redes Neurais e Deep Learning (MBA)
Autores: Samir Saraiva, Claudio Cruz, Vinicius Ferreira e João Felipe

ETAPA 1 (exploratória) — B2W vs IMDB PT-BR, avaliados em críticas reais
========================================================================

Primeiro experimento do projeto. Treina uma Bi-LSTM em avaliações rotuladas em
português e mede o desempenho em críticas de cinema reais, coletadas da web e
rotuladas pela nota publicada (Omelete) ou manualmente pela equipe (Cinema com
Rapadura). Foi este experimento que revelou o problema central do trabalho:
acurácia alta no teste não garante bom desempenho fora do domínio de treino.

  FASE 1 - TREINO      Bi-LSTM treinada do zero. Usa o IMDB PT-BR (Kaggle) se o
                       arquivo dados/imdb-reviews-pt-br.csv existir; senão, baixa
                       o B2W-Reviews01 (avaliações de produtos) do GitHub.
  FASE 2 - COLETA      Coleta críticas do Cinema com Rapadura e do Omelete e aplica
                       os rótulos manuais de rotulos_manuais.csv (colunas url,rotulo).
  FASE 3 - AVALIAÇÃO   Acurácia, precisão, recall e matriz de confusão no teste e nas
                       críticas coletadas.

Resultados obtidos (50 críticas rotuladas; linha de base = 56%):
  - Treino no B2W:      95,4% no teste  →  46% nas críticas reais (49 de 50 previstas como negativas)
  - Treino no IMDB PT-BR: 88,8% no teste →  62% nas críticas reais (só 6 de 22 negativas reconhecidas)
As 50 críticas rotuladas aqui foram congeladas em dados/teste_ouro.csv e viraram o
"conjunto ouro" da etapa 2 (ver comum.py).

Uso:
    python sentimento_cinema_ptbr.py                  # roda as 3 fases
    python sentimento_cinema_ptbr.py --fase treino | coleta | avaliacao
    python sentimento_cinema_ptbr.py --amostra 20000  # treino rápido com amostra
"""

import argparse
import json
import os
import re
import sys
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
PASTA_DADOS = "dados"
ARQ_MODELO = os.path.join(PASTA_DADOS, "modelo_sentimento_ptbr.keras")
ARQ_IMDB_PTBR = os.path.join(PASTA_DADOS, "imdb-reviews-pt-br.csv")
ARQ_B2W = os.path.join(PASTA_DADOS, "B2W-Reviews01.csv")
URL_B2W = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv"

ARQ_COLETA = os.path.join(PASTA_DADOS, "criticas_coletadas.csv")
ARQ_URLS_MANUAIS = "urls_criticas.txt"        # opcional: uma URL de crítica por linha
ARQ_ROTULOS_MANUAIS = "rotulos_manuais.csv"   # opcional: colunas url,rotulo (pos/neg)
ARQ_PREVISOES = os.path.join(PASTA_DADOS, "previsoes_criticas.csv")

VOCAB_SIZE = 20000     # tamanho do vocabulário
MAX_LEN = 300          # tokens por texto
EPOCAS_MAX = 8         # EarlyStopping interrompe antes se a validação parar de melhorar
BATCH_SIZE = 128
SEED = 42

# Coleta: páginas de listagem de críticas. O script procura nelas links para
# críticas individuais. Se o layout de algum site mudar, liste URLs em urls_criticas.txt.
FONTES = {
    "Cinema com Rapadura": "https://cinemacomrapadura.com.br/criticas/",
    "Omelete": "https://www.omelete.com.br/criticas",
}
MAX_CRITICAS_POR_FONTE = 60
PAUSA_ENTRE_REQUISICOES = 2.0   # segundos; seja educado com os servidores
USER_AGENT = "ProjetoAcademicoSentimentoPTBR/1.0 (trabalho de disciplina de MBA)"

# Nota normalizada (0 a 1) -> rótulo. Notas intermediárias ficam como "neutro"
# e não entram na avaliação binária (mas aparecem no relatório).
LIMIAR_POSITIVO = 0.7   # ex.: 3,5/5 ou 7/10 para cima
LIMIAR_NEGATIVO = 0.4   # ex.: 2/5 ou 4/10 para baixo


# =============================================================================
# UTILITÁRIOS
# =============================================================================
def inicio_e_fim(texto, max_palavras=MAX_LEN):
    """Críticas longas costumam dar o veredito no final. Em vez de cortar o fim,
    mantém a primeira e a última metade do limite de palavras."""
    palavras = str(texto).split()
    if len(palavras) <= max_palavras:
        return " ".join(palavras)
    metade = max_palavras // 2
    return " ".join(palavras[:metade] + palavras[-metade:])


def como_tensor(textos):
    """Converte uma lista/coluna de textos no formato que o modelo espera: (n, 1) string."""
    import tensorflow as tf
    return tf.constant([[str(t)] for t in textos], dtype=tf.string)


def rotulo_da_nota(nota_normalizada):
    if nota_normalizada is None or pd.isna(nota_normalizada):
        return None
    if nota_normalizada >= LIMIAR_POSITIVO:
        return 1
    if nota_normalizada <= LIMIAR_NEGATIVO:
        return 0
    return None


# =============================================================================
# FASE 1 - DADOS DE TREINO E MODELO
# =============================================================================
def carregar_imdb_ptbr():
    df = pd.read_csv(ARQ_IMDB_PTBR)
    col_texto = "text_pt" if "text_pt" in df.columns else next(c for c in df.columns if "pt" in c.lower())
    col_rotulo = "sentiment" if "sentiment" in df.columns else df.columns[-1]
    rotulos = df[col_rotulo].astype(str).str.lower().map({"pos": 1, "neg": 0, "1": 1, "0": 0})
    df = pd.DataFrame({"texto": df[col_texto], "rotulo": rotulos}).dropna()
    print(f"   IMDB PT-BR: {len(df):,} críticas de cinema (tradução automática do IMDB original)")
    return df


def carregar_b2w():
    if not os.path.exists(ARQ_B2W):
        import requests
        print("   Baixando B2W-Reviews01 do GitHub (≈40 MB, só na primeira vez)...")
        r = requests.get(URL_B2W, timeout=300)
        r.raise_for_status()
        with open(ARQ_B2W, "wb") as f:
            f.write(r.content)
    df = pd.read_csv(ARQ_B2W, usecols=["review_title", "review_text", "overall_rating"], low_memory=False)
    df = df.dropna(subset=["review_text", "overall_rating"])
    df = df[df["overall_rating"] != 3]                      # nota 3 é ambígua
    df["rotulo"] = (df["overall_rating"] >= 4).astype(int)  # 1-2 negativo, 4-5 positivo
    df["texto"] = df["review_title"].fillna("") + ". " + df["review_text"]
    # O B2W tem muito mais avaliações positivas: balanceia por subamostragem
    n = df["rotulo"].value_counts().min()
    df = pd.concat([g.sample(n, random_state=SEED) for _, g in df.groupby("rotulo")])
    print(f"   B2W-Reviews01: {len(df):,} avaliações (balanceadas, {n:,} por classe)")
    return df[["texto", "rotulo"]]


def carregar_dados_treino(amostra=None):
    os.makedirs(PASTA_DADOS, exist_ok=True)
    print("\n[FASE 1] Carregando dados de treino em português...")
    df = carregar_imdb_ptbr() if os.path.exists(ARQ_IMDB_PTBR) else carregar_b2w()
    # Descarta textos sem nenhuma letra/número (só emoji ou pontuação): viram uma
    # sequência 100% padding, que o cuDNN (LSTM na GPU) não aceita.
    tem_palavra = df["texto"].astype(str).str.contains(r"[^\W_]", regex=True)
    if (~tem_palavra).sum():
        print(f"   Descartados {(~tem_palavra).sum()} textos sem palavras (só emoji/pontuação)")
    df = df[tem_palavra]
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    if amostra and amostra < len(df):
        df = df.iloc[:amostra]
        print(f"   Usando amostra de {amostra:,} exemplos")
    df["texto"] = df["texto"].map(inicio_e_fim)
    return df


def construir_modelo(textos_treino):
    from tensorflow import keras
    from tensorflow.keras import layers

    # A camada TextVectorization fica DENTRO do modelo: ele recebe texto cru e
    # o vocabulário é aprendido do próprio português (nada de word_index do IMDB).
    vetorizador = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LEN,
        standardize="lower_and_strip_punctuation",
    )
    vetorizador.adapt(como_tensor(textos_treino))

    modelo = keras.Sequential([
        keras.Input(shape=(1,), dtype="string"),
        vetorizador,
        # mask_zero=True faz a LSTM ignorar o padding (resolve o problema do padding='post')
        layers.Embedding(VOCAB_SIZE, 64, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.4),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return modelo


def fase_treino(amostra=None):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split

    tf.keras.utils.set_random_seed(SEED)
    df = carregar_dados_treino(amostra)

    # Divisão 70% treino / 15% validação / 15% teste, estratificada
    treino, resto = train_test_split(df, test_size=0.30, stratify=df["rotulo"], random_state=SEED)
    valid, teste = train_test_split(resto, test_size=0.50, stratify=resto["rotulo"], random_state=SEED)
    print(f"   Treino: {len(treino):,} | Validação: {len(valid):,} | Teste: {len(teste):,}")

    modelo = construir_modelo(treino["texto"])
    modelo.summary()

    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    historico = modelo.fit(
        como_tensor(treino["texto"]), treino["rotulo"].values,
        validation_data=(como_tensor(valid["texto"]), valid["rotulo"].values),
        epochs=EPOCAS_MAX, batch_size=BATCH_SIZE, callbacks=[parada], verbose=1,
    )

    modelo.save(ARQ_MODELO)
    teste.to_csv(os.path.join(PASTA_DADOS, "conjunto_teste.csv"), index=False)
    pd.DataFrame(historico.history).to_csv(os.path.join(PASTA_DADOS, "historico_treino.csv"), index=False)
    print(f"\n   Modelo salvo em {ARQ_MODELO}")
    return modelo


# =============================================================================
# FASE 2 - COLETA DE CRÍTICAS COM NOTA
# =============================================================================
class Coletor:
    def __init__(self):
        import requests
        self.sessao = requests.Session()
        self.sessao.headers["User-Agent"] = USER_AGENT
        self.robots = {}
        self.estatisticas = {}

    def permitido(self, url):
        """Respeita o robots.txt de cada site."""
        dominio = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if dominio not in self.robots:
            rp = RobotFileParser()
            try:
                r = self.sessao.get(dominio + "/robots.txt", timeout=15)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:
                rp.parse([])
            self.robots[dominio] = rp
        return self.robots[dominio].can_fetch(USER_AGENT, url)

    def baixar(self, url):
        if not self.permitido(url):
            print(f"      bloqueado pelo robots.txt: {url}")
            return None
        time.sleep(PAUSA_ENTRE_REQUISICOES)
        try:
            r = self.sessao.get(url, timeout=20)
        except Exception as e:
            print(f"      erro de conexão: {e}")
            return None
        if r.status_code != 200:
            print(f"      HTTP {r.status_code}: {url}")
            return None
        r.encoding = r.apparent_encoding or r.encoding
        return r.text

    def descobrir_links(self, url_listagem):
        """Procura na página de listagem links que pareçam críticas individuais."""
        from bs4 import BeautifulSoup
        html = self.baixar(url_listagem)
        if not html:
            return []
        dominio = urlparse(url_listagem).netloc
        links = []
        for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
            url = urljoin(url_listagem, a["href"]).split("#")[0].split("?")[0]
            if (urlparse(url).netloc == dominio and "critica" in url.lower()
                    and url.rstrip("/") != url_listagem.rstrip("/")
                    and not re.search(r"/(page|pagina|tag|categoria|autor)/", url)
                    and url not in links):
                links.append(url)
        return links[:MAX_CRITICAS_POR_FONTE]


def _objetos_jsonld(soup):
    """Percorre todos os blocos JSON-LD (schema.org) da página."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            dados = json.loads(tag.string or tag.get_text() or "")
        except (json.JSONDecodeError, TypeError):
            continue
        pilha = [dados]
        while pilha:
            item = pilha.pop()
            if isinstance(item, list):
                pilha.extend(item)
            elif isinstance(item, dict):
                yield item
                pilha.extend(v for v in item.values() if isinstance(v, (dict, list)))


def _numero(valor):
    try:
        return float(str(valor).replace(",", "."))
    except (TypeError, ValueError):
        return None


def extrair_critica(html, url):
    """Extrai título, texto e nota de uma página de crítica.
    Ordem de tentativa para a nota:
      1. JSON-LD schema.org (reviewRating) — usado por muitos sites para o Google
      2. Padrões no texto como 'Nota: 4/5' ou 'nota 8 de 10'
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    nota = nota_max = None
    origem_nota = None
    corpo_jsonld = None
    for obj in _objetos_jsonld(soup):
        rating = obj.get("reviewRating")
        if isinstance(rating, dict) and nota is None:
            nota = _numero(rating.get("ratingValue"))
            nota_max = _numero(rating.get("bestRating")) or 5.0
            if nota is not None:
                origem_nota = "json-ld"
        corpo_jsonld = corpo_jsonld or obj.get("reviewBody") or obj.get("articleBody")

    # Texto da crítica: parágrafos dentro de <article>, senão parágrafos longos da página
    container = soup.find("article") or soup
    paragrafos = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    paragrafos = [p for p in paragrafos if len(p) > 60]
    texto = " ".join(paragrafos) or (corpo_jsonld or "")
    if len(texto) < 200 and corpo_jsonld:
        texto = corpo_jsonld

    if nota is None:
        m = re.search(r"\bnota\b[^0-9]{0,15}(\d+(?:[.,]\d+)?)\s*(?:/|de)\s*(\d+)",
                      soup.get_text(" ", strip=True), flags=re.IGNORECASE)
        if m:
            nota, nota_max = _numero(m.group(1)), _numero(m.group(2))
            origem_nota = "texto"

    titulo = soup.find("h1")
    titulo = titulo.get_text(" ", strip=True) if titulo else ""
    if not titulo:
        og = soup.find("meta", property="og:title")
        titulo = (og.get("content", "").strip() if og else "") or (
            soup.title.string.strip() if soup.title and soup.title.string else "")

    nota_norm = nota / nota_max if nota is not None and nota_max and 0 <= nota <= nota_max else None
    return {
        "url": url, "titulo": titulo, "texto": texto,
        "nota": nota, "nota_max": nota_max, "nota_normalizada": nota_norm,
        "origem_nota": origem_nota, "n_palavras": len(texto.split()),
    }


def fase_coleta():
    print("\n[FASE 2] Coletando críticas com nota...")
    os.makedirs(PASTA_DADOS, exist_ok=True)
    coletor = Coletor()

    alvos = []  # (fonte, url)
    for fonte, listagem in FONTES.items():
        print(f"   -> {fonte}: procurando críticas em {listagem}")
        links = coletor.descobrir_links(listagem)
        print(f"      {len(links)} links de críticas encontrados")
        alvos += [(fonte, u) for u in links]
    if os.path.exists(ARQ_URLS_MANUAIS):
        with open(ARQ_URLS_MANUAIS, encoding="utf-8") as f:
            manuais = [l.strip() for l in f if l.strip().startswith("http")]
        print(f"   -> {len(manuais)} URLs de {ARQ_URLS_MANUAIS}")
        alvos += [(urlparse(u).netloc, u) for u in manuais]

    registros = []
    for i, (fonte, url) in enumerate(alvos, 1):
        print(f"   [{i}/{len(alvos)}] {url}")
        html = coletor.baixar(url)
        if not html:
            continue
        reg = extrair_critica(html, url)
        reg["fonte"] = fonte
        if reg["n_palavras"] < 80:
            print("      texto curto demais, descartado (provavelmente não é uma crítica)")
            continue
        registros.append(reg)

    df = pd.DataFrame(registros, columns=["fonte", "url", "titulo", "texto", "nota", "nota_max",
                                          "nota_normalizada", "origem_nota", "n_palavras"])

    # Rótulos manuais (para críticas sem nota): arquivo url,rotulo com pos/neg
    df["rotulo"] = df["nota_normalizada"].map(rotulo_da_nota)
    df["origem_rotulo"] = np.where(df["rotulo"].notna(), "nota", None)
    if os.path.exists(ARQ_ROTULOS_MANUAIS) and len(df):
        manuais = pd.read_csv(ARQ_ROTULOS_MANUAIS)
        mapa = dict(zip(manuais["url"], manuais["rotulo"].astype(str).str.lower().map({"pos": 1, "neg": 0})))
        tem_manual = df["url"].map(mapa).notna()
        df.loc[tem_manual, "rotulo"] = df.loc[tem_manual, "url"].map(mapa)
        df.loc[tem_manual, "origem_rotulo"] = "manual"

    df.to_csv(ARQ_COLETA, index=False)

    # Relatório de quanto cada fonte realmente entregou (nada de falhar em silêncio)
    print("\n   RESUMO DA COLETA")
    if df.empty:
        print("   Nenhuma crítica coletada. Verifique as mensagens acima (HTTP 403, robots.txt,")
        print(f"   layout alterado). Alternativa: liste URLs de críticas em {ARQ_URLS_MANUAIS}.")
        return df
    resumo = df.groupby("fonte").agg(
        criticas=("url", "count"),
        com_nota=("nota", lambda s: s.notna().sum()),
        positivas=("rotulo", lambda s: (s == 1).sum()),
        negativas=("rotulo", lambda s: (s == 0).sum()),
    )
    resumo["neutras_ou_sem_rotulo"] = resumo["criticas"] - resumo["positivas"] - resumo["negativas"]
    print(resumo.to_string())
    sem_rotulo = df["rotulo"].isna().sum()
    if sem_rotulo:
        print(f"\n   {sem_rotulo} críticas sem rótulo. Para incluí-las na avaliação, crie {ARQ_ROTULOS_MANUAIS}")
        print("   com as colunas url,rotulo (pos ou neg) e rode a fase de coleta de novo.")
    print(f"   Dados salvos em {ARQ_COLETA}")
    return df


# =============================================================================
# FASE 3 - AVALIAÇÃO
# =============================================================================
def relatorio(nome, y_real, prob):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    y_prev = (prob >= 0.5).astype(int)
    acc = accuracy_score(y_real, y_prev)
    base = max(np.mean(y_real), 1 - np.mean(y_real))  # acurácia de "chutar sempre a classe maioritária"
    print(f"\n   === {nome} ({len(y_real)} textos) ===")
    print(f"   Acurácia: {acc:.3f}   (linha de base, sempre a classe mais comum: {base:.3f})")
    print(classification_report(y_real, y_prev, labels=[0, 1],
                                target_names=["negativo", "positivo"], digits=3, zero_division=0))
    cm = confusion_matrix(y_real, y_prev, labels=[0, 1])
    print("   Matriz de confusão (linhas = real, colunas = previsto)")
    print(pd.DataFrame(cm, index=["real neg", "real pos"], columns=["prev neg", "prev pos"]).to_string())
    return acc


def fase_avaliacao(modelo=None):
    from tensorflow import keras
    print("\n[FASE 3] Avaliação")
    if modelo is None:
        if not os.path.exists(ARQ_MODELO):
            sys.exit(f"   Modelo não encontrado em {ARQ_MODELO}. Rode antes: --fase treino")
        modelo = keras.models.load_model(ARQ_MODELO)

    # 3a. Conjunto de teste (mesma distribuição do treino)
    teste = pd.read_csv(os.path.join(PASTA_DADOS, "conjunto_teste.csv"))
    prob_teste = modelo.predict(como_tensor(teste["texto"]), batch_size=256, verbose=0).ravel()
    acc_teste = relatorio("Conjunto de teste", teste["rotulo"].values, prob_teste)

    # 3b. Críticas reais coletadas (distribuição diferente: é aqui que se vê a generalização)
    if not os.path.exists(ARQ_COLETA):
        print(f"\n   Sem críticas coletadas ({ARQ_COLETA}). Rode --fase coleta para a avaliação no mundo real.")
        return
    criticas = pd.read_csv(ARQ_COLETA)
    if criticas.empty:
        print("\n   O arquivo de coleta está vazio.")
        return
    criticas["prob_positivo"] = modelo.predict(
        como_tensor(criticas["texto"].astype(str).map(inicio_e_fim)), batch_size=64, verbose=0).ravel()
    criticas["previsao"] = np.where(criticas["prob_positivo"] >= 0.5, "positivo", "negativo")
    criticas.drop(columns=["texto"]).to_csv(ARQ_PREVISOES, index=False)

    rotuladas = criticas[criticas["rotulo"].notna()]
    if len(rotuladas) == 0:
        print("\n   Nenhuma crítica com rótulo; só há previsões (sem como medir acerto).")
    else:
        if len(rotuladas) < 50:
            print(f"\n   ATENÇÃO: só {len(rotuladas)} críticas rotuladas. Com menos de ~50, a acurácia")
            print("   varia muito de uma coleta para outra; trate o resultado como indicativo.")
        acc_real = relatorio("Críticas de cinema coletadas", rotuladas["rotulo"].astype(int).values,
                             rotuladas["prob_positivo"].values)
        print(f"\n   Queda de desempenho do teste para o mundo real: {acc_teste - acc_real:+.3f}")
        print("   (uma queda grande indica diferença de domínio entre os dados de treino e as críticas)")

        erros = rotuladas[(rotuladas["prob_positivo"] >= 0.5) != (rotuladas["rotulo"] == 1)]
        if len(erros):
            print("\n   Exemplos de erros (útil para a discussão do trabalho):")
            for _, r in erros.head(8).iterrows():
                nome = r["titulo"] if isinstance(r["titulo"], str) and r["titulo"].strip() else r["url"]
                real = "positivo" if r["rotulo"] == 1 else "negativo"
                origem = (f"nota {r['nota']:g}/{r['nota_max']:g}" if pd.notna(r["nota"])
                          else "rótulo manual")
                print(f"   - {str(nome)[:90]}\n     real: {real} ({origem}) | P(positivo)={r['prob_positivo']:.2f}")

    print(f"\n   Previsões de todas as críticas salvas em {ARQ_PREVISOES}")


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fase", choices=["treino", "coleta", "avaliacao", "tudo"], default="tudo")
    parser.add_argument("--amostra", type=int, default=None, help="limita o nº de exemplos de treino")
    args = parser.parse_args()

    modelo = None
    if args.fase in ("treino", "tudo"):
        modelo = fase_treino(args.amostra)
    if args.fase in ("coleta", "tudo"):
        fase_coleta()
    if args.fase in ("avaliacao", "tudo"):
        fase_avaliacao(modelo)
