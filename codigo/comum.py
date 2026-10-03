"""
Análise de sentimento de críticas de cinema em português com deep learning
Trabalho da disciplina Redes Neurais e Deep Learning (MBA)
Autores: Samir Saraiva, Claudio Cruz, Vinicius Ferreira e João Felipe

ETAPA 2 — módulo compartilhado pelos dois modelos (Bi-LSTM e BERTimbau)

Garante que os dois modelos sejam treinados e avaliados EXATAMENTE nos mesmos
conjuntos, para que a comparação entre eles seja justa.

Fontes de dados:
  - IMDB PT-BR (Kaggle): ~49 mil críticas de usuários do IMDB, traduzidas
    automaticamente do inglês. Partição fixa 70/15/15 por sorteio com semente.
  - Corpus nativo (coletar_criticas.py): 7.031 críticas escritas em português com
    nota fora da faixa neutra — Papo de Cinema (5.762, críticos) e AdoroCinema
    (1.269, espectadores). Partição 70/15/15 por hash do FILME (sem vazamento).
  - Conjunto ouro (dados/teste_ouro.csv): 50 críticas do Cinema com Rapadura
    (rótulo manual da equipe) e do Omelete (nota). Nunca entra no treino.

Cenários de treino: "imdb", "nativo" e "imdb+nativo". O treino é balanceado 50/50
por subamostragem (cada fonte nativa separadamente); os testes mantêm a
distribuição natural, por isso reportamos F1 macro e a linha de base.

Métricas: acurácia com IC 95% de Wilson, linha de base (classe majoritária),
F1 macro, precisão/recall por classe e matriz de confusão. Cada treino grava um
JSON em dados/resultados/, lido depois por comparar_resultados.py.
"""

import hashlib
import json
import math
import os

import numpy as np
import pandas as pd

PASTA_DADOS = "dados"
PASTA_MODELOS = os.path.join(PASTA_DADOS, "modelos")
PASTA_RESULTADOS = os.path.join(PASTA_DADOS, "resultados")
ARQ_IMDB_PTBR = os.path.join(PASTA_DADOS, "imdb-reviews-pt-br.csv")
ARQ_CORPUS = os.path.join(PASTA_DADOS, "corpus_criticas.csv")
ARQ_COLETA_ANTIGA = os.path.join(PASTA_DADOS, "criticas_coletadas.csv")  # gerado pelo script da etapa 1
ARQ_OURO = os.path.join(PASTA_DADOS, "teste_ouro.csv")                    # cópia congelada do conjunto ouro

SEED = 42
LIMIAR_POSITIVO = 0.7   # nota normalizada >= 0,7 -> positivo (ex.: 7/10, 3,5/5)
LIMIAR_NEGATIVO = 0.4   # nota normalizada <= 0,4 -> negativo (ex.: 4/10, 2/5)
MAX_TESTE_IMDB = 3000   # tamanho fixo do teste IMDB (mesmo para os dois modelos)

CENARIOS = {
    "imdb": "Treino só no IMDB PT-BR (traduzido)",
    "nativo": "Treino só no corpus nativo (Papo de Cinema + AdoroCinema)",
    "imdb+nativo": "Treino no IMDB PT-BR + corpus nativo",
}


# -----------------------------------------------------------------------------
# Utilitários de texto e rótulo
# -----------------------------------------------------------------------------
def inicio_e_fim(texto, max_palavras):
    """Críticas longas costumam dar o veredito no fim. Mantém a primeira e a
    última metade do limite de palavras, em vez de simplesmente cortar o fim."""
    palavras = str(texto).split()
    if len(palavras) <= max_palavras:
        return " ".join(palavras)
    metade = max_palavras // 2
    return " ".join(palavras[:metade] + palavras[-metade:])


def rotulo_da_nota(nota_normalizada):
    if nota_normalizada is None or pd.isna(nota_normalizada):
        return None
    if nota_normalizada >= LIMIAR_POSITIVO:
        return 1
    if nota_normalizada <= LIMIAR_NEGATIVO:
        return 0
    return None


def particao_por_grupo(chave_grupo):
    """Divisão treino/validação/teste determinística pelo FILME (grupo).
    Todas as críticas de um mesmo filme caem na mesma partição, o que evita
    que o modelo 'decore' o filme no treino e seja testado nele. Por usar hash,
    a partição de um filme não muda quando novos dados são coletados."""
    h = int(hashlib.md5(str(chave_grupo).encode("utf-8")).hexdigest(), 16) % 100
    if h < 70:
        return "treino"
    if h < 85:
        return "valid"
    return "teste"


def _tem_palavra(serie):
    return serie.astype(str).str.contains(r"[^\W_]", regex=True)


# -----------------------------------------------------------------------------
# Carregamento das fontes
# -----------------------------------------------------------------------------
def carregar_imdb():
    """IMDB PT-BR com partição fixa (independe de amostragem)."""
    if not os.path.exists(ARQ_IMDB_PTBR):
        raise FileNotFoundError(
            f"{ARQ_IMDB_PTBR} não encontrado. Baixe em "
            "https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr e descompacte em dados/")
    df = pd.read_csv(ARQ_IMDB_PTBR)
    col_texto = "text_pt" if "text_pt" in df.columns else next(c for c in df.columns if "pt" in c.lower())
    col_rotulo = "sentiment" if "sentiment" in df.columns else df.columns[-1]
    rot = df[col_rotulo].astype(str).str.lower().map({"pos": 1, "neg": 0, "1": 1, "0": 0})
    df = pd.DataFrame({"texto": df[col_texto], "rotulo": rot}).dropna()
    df = df[_tem_palavra(df["texto"])].reset_index(drop=True)
    df["rotulo"] = df["rotulo"].astype(int)
    df["fonte"] = "IMDB PT-BR"
    rng = np.random.RandomState(SEED)
    sorteio = rng.rand(len(df))
    df["particao"] = np.where(sorteio < 0.70, "treino", np.where(sorteio < 0.85, "valid", "teste"))
    return df


def carregar_corpus():
    """Corpus nativo coletado por coletar_criticas.py (só linhas com rótulo)."""
    if not os.path.exists(ARQ_CORPUS):
        return pd.DataFrame(columns=["texto", "rotulo", "fonte", "particao", "tipo"])
    df = pd.read_csv(ARQ_CORPUS)
    df = df[df["rotulo"].notna() & _tem_palavra(df["texto"])].copy()
    df["rotulo"] = df["rotulo"].astype(int)
    return df.reset_index(drop=True)


def carregar_ouro():
    """Conjunto ouro (Rapadura com rótulo manual + Omelete com nota).
    Na primeira execução, congela uma cópia em dados/teste_ouro.csv para que o
    conjunto não mude se a coleta antiga for rodada de novo."""
    if not os.path.exists(ARQ_OURO):
        if not os.path.exists(ARQ_COLETA_ANTIGA):
            return pd.DataFrame(columns=["texto", "rotulo", "fonte", "url"])
        antigo = pd.read_csv(ARQ_COLETA_ANTIGA)
        antigo = antigo[antigo["rotulo"].notna()]
        antigo.to_csv(ARQ_OURO, index=False)
        print(f"   Conjunto ouro congelado em {ARQ_OURO} ({len(antigo)} críticas)")
    df = pd.read_csv(ARQ_OURO)
    df = df[df["rotulo"].notna()].copy()
    df["rotulo"] = df["rotulo"].astype(int)
    return df.reset_index(drop=True)


def urls_do_ouro():
    try:
        return set(carregar_ouro()["url"].astype(str))
    except Exception:
        return set()


def balancear(df, seed=SEED):
    """Subamostra a classe majoritária para 50/50 (usado só no treino)."""
    if df.empty or df["rotulo"].nunique() < 2:
        return df
    n = df["rotulo"].value_counts().min()
    return pd.concat([g.sample(n, random_state=seed) for _, g in df.groupby("rotulo")]) \
             .sample(frac=1, random_state=seed).reset_index(drop=True)


def montar_dados(cenario, amostra_imdb=None, verbose=True):
    """Retorna (treino, valid, testes) para o cenário.
    testes = dict nome -> DataFrame; os mesmos para qualquer modelo/cenário."""
    if cenario not in CENARIOS:
        raise ValueError(f"Cenário inválido: {cenario}. Opções: {list(CENARIOS)}")

    imdb = carregar_imdb()
    corpus = carregar_corpus()
    ouro = carregar_ouro()

    partes_treino, partes_valid = [], []
    if cenario in ("imdb", "imdb+nativo"):
        tr = imdb[imdb["particao"] == "treino"]
        if amostra_imdb and amostra_imdb < len(tr):
            tr = tr.sample(amostra_imdb, random_state=SEED)
        partes_treino.append(tr)
        va = imdb[imdb["particao"] == "valid"]
        partes_valid.append(va.sample(min(len(va), 3000), random_state=SEED))
    if cenario in ("nativo", "imdb+nativo"):
        if corpus.empty:
            raise RuntimeError("Corpus nativo vazio. Rode antes: python coletar_criticas.py")
        # Balanceia cada fonte nativa separadamente (AdoroCinema tende a ser muito positivo)
        for _, g in corpus[corpus["particao"] == "treino"].groupby("fonte"):
            partes_treino.append(balancear(g))
        partes_valid.append(corpus[corpus["particao"] == "valid"])

    treino = balancear(pd.concat(partes_treino, ignore_index=True))
    valid = pd.concat(partes_valid, ignore_index=True).sample(frac=1, random_state=SEED)

    testes = {}
    t_imdb = imdb[imdb["particao"] == "teste"]
    testes["IMDB PT-BR (teste)"] = t_imdb.sample(min(len(t_imdb), MAX_TESTE_IMDB), random_state=SEED)
    for fonte, g in corpus[corpus["particao"] == "teste"].groupby("fonte"):
        testes[f"{fonte} (teste)"] = g
    if not ouro.empty:
        testes["Ouro: Rapadura + Omelete"] = ouro

    if verbose:
        print(f"\n   Cenário: {cenario} — {CENARIOS[cenario]}")
        print(f"   Treino: {len(treino):,} textos | Validação: {len(valid):,}")
        print("   Composição do treino por fonte:")
        for fonte, n in treino["fonte"].value_counts().items():
            print(f"      {fonte}: {n:,}")
        print("   Conjuntos de teste:")
        for nome, t in testes.items():
            print(f"      {nome}: {len(t):,} textos ({t['rotulo'].mean():.0%} positivos)")
    return treino, valid, testes


# -----------------------------------------------------------------------------
# Avaliação e registro de resultados
# -----------------------------------------------------------------------------
def intervalo_wilson(acertos, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = acertos / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    margem = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (centro - margem, centro + margem)


def avaliar(nome, y_real, prob, verbose=True):
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
    y_real = np.asarray(y_real).astype(int)
    y_prev = (np.asarray(prob) >= 0.5).astype(int)
    n = len(y_real)
    acertos = int((y_real == y_prev).sum())
    acc = acertos / n if n else float("nan")
    base = max(y_real.mean(), 1 - y_real.mean()) if n else float("nan")
    f1_macro = f1_score(y_real, y_prev, average="macro", labels=[0, 1], zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(y_real, y_prev, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_real, y_prev, labels=[0, 1])
    ic = intervalo_wilson(acertos, n)
    res = {
        "conjunto": nome, "n": n, "acuracia": acc, "ic95": ic, "linha_base": base,
        "f1_macro": f1_macro,
        "precisao_neg": prec[0], "recall_neg": rec[0], "precisao_pos": prec[1], "recall_pos": rec[1],
        "matriz_confusao": cm.tolist(),
    }
    if verbose:
        print(f"\n   === {nome} ({n} textos) ===")
        print(f"   Acurácia: {acc:.3f}  IC95% [{ic[0]:.3f}, {ic[1]:.3f}]  | linha de base: {base:.3f}")
        print(f"   F1 macro: {f1_macro:.3f}  (média do F1 das duas classes; não é inflado por desbalanceamento)")
        print(f"   Negativo: precisão {prec[0]:.3f}  recall {rec[0]:.3f}  ({sup[0]} textos)")
        print(f"   Positivo: precisão {prec[1]:.3f}  recall {rec[1]:.3f}  ({sup[1]} textos)")
        print("   Matriz de confusão (linhas = real, colunas = previsto)")
        print(pd.DataFrame(cm, index=["real neg", "real pos"], columns=["prev neg", "prev pos"]).to_string())
    return res


def _json_padrao(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def salvar_resultados(modelo, cenario, resultados, info=None):
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    caminho = os.path.join(PASTA_RESULTADOS, f"{modelo}__{cenario}.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump({"modelo": modelo, "cenario": cenario, "info": info or {}, "resultados": resultados},
                  f, ensure_ascii=False, indent=2, default=_json_padrao)
    print(f"\n   Resultados salvos em {caminho}")


def salvar_previsoes(modelo, cenario, nome_conjunto, df, prob):
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    seguro = "".join(c if c.isalnum() else "_" for c in nome_conjunto)
    saida = df.drop(columns=[c for c in ["texto"] if c in df.columns]).copy()
    saida["prob_positivo"] = prob
    saida["previsao"] = (np.asarray(prob) >= 0.5).astype(int)
    saida.to_csv(os.path.join(PASTA_RESULTADOS, f"prev__{modelo}__{cenario}__{seguro}.csv"), index=False)
