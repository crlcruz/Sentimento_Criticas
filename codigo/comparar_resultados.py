"""
Análise de sentimento de críticas de cinema em português com deep learning
Trabalho da disciplina Redes Neurais e Deep Learning (MBA)
Autores: Samir Saraiva, Claudio Cruz, Vinicius Ferreira e João Felipe

ETAPA 2 — junta os resultados de todos os treinos (dados/resultados/*.json) numa
tabela comparativa e num gráfico para o relatório.

Espera os 6 arquivos: bilstm__{imdb,nativo,imdb+nativo}.json (treinados no PC) e
bertimbau__{imdb,nativo,imdb+nativo}.json (treinados no Colab e copiados para cá).

Uso:
  python comparar_resultados.py

Saídas:
  dados/resultados/comparacao.csv        tabela completa
  dados/resultados/comparacao.md         tabela pronta para colar no relatório
  dados/resultados/comparacao_f1.png     Figura 1 do relatório (F1 macro por conjunto de teste)
"""

import glob
import json
import os

import pandas as pd

import comum

NOME_MODELO = {"bilstm": "Bi-LSTM", "bertimbau": "BERTimbau"}
ORDEM_CENARIO = ["imdb", "nativo", "imdb+nativo"]
ROTULO_CENARIO = {"imdb": "IMDB PT-BR", "nativo": "Nativo", "imdb+nativo": "IMDB + nativo"}

# Paleta categórica de referência (slots 1 e 2) e tokens de texto/superfície
COR_MODELO = {"Bi-LSTM": "#2a78d6", "BERTimbau": "#eb6834"}
SUPERFICIE, TXT1, TXT2, GRADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def carregar():
    linhas = []
    for arq in sorted(glob.glob(os.path.join(comum.PASTA_RESULTADOS, "*__*.json"))):
        with open(arq, encoding="utf-8") as f:
            d = json.load(f)
        for r in d["resultados"]:
            linhas.append({
                "modelo": NOME_MODELO.get(d["modelo"], d["modelo"]),
                "cenario": d["cenario"],
                "conjunto": r["conjunto"],
                "n": r["n"],
                "acuracia": r["acuracia"],
                "ic95_inf": r["ic95"][0], "ic95_sup": r["ic95"][1],
                "linha_base": r["linha_base"],
                "f1_macro": r["f1_macro"],
                "recall_neg": r["recall_neg"], "recall_pos": r["recall_pos"],
                "minutos_treino": d["info"].get("minutos_treino"),
                "n_treino": d["info"].get("n_treino"),
            })
    return pd.DataFrame(linhas)


def tabela_markdown(df):
    df = df.copy()
    df["cenario"] = pd.Categorical(df["cenario"], ORDEM_CENARIO, ordered=True)
    df = df.sort_values(["conjunto", "cenario", "modelo"])
    out = ["| Conjunto de teste | Treino | Modelo | n | Acurácia (IC 95%) | Linha de base | F1 macro | Recall neg | Recall pos |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        out.append(f"| {r.conjunto} | {ROTULO_CENARIO.get(r.cenario, r.cenario)} | {r.modelo} | {r.n} | "
                   f"{r.acuracia:.1%} ({r.ic95_inf:.0%}–{r.ic95_sup:.0%}) | {r.linha_base:.1%} | "
                   f"{r.f1_macro:.3f} | {r.recall_neg:.2f} | {r.recall_pos:.2f} |")
    return "\n".join(out)


def grafico(df, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    conjuntos = list(dict.fromkeys(df["conjunto"]))
    cenarios = [c for c in ORDEM_CENARIO if c in set(df["cenario"])]
    modelos = [m for m in ["Bi-LSTM", "BERTimbau"] if m in set(df["modelo"])]

    ncol = min(2, len(conjuntos))
    nlin = (len(conjuntos) + ncol - 1) // ncol
    fig, eixos = plt.subplots(nlin, ncol, figsize=(6.2 * ncol, 3.6 * nlin), squeeze=False,
                              facecolor=SUPERFICIE, sharey=True)
    largura = 0.36
    for idx, conj in enumerate(conjuntos):
        ax = eixos[idx // ncol][idx % ncol]
        ax.set_facecolor(SUPERFICIE)
        sub = df[df["conjunto"] == conj]
        for j, mod in enumerate(modelos):
            for i, cen in enumerate(cenarios):
                linha = sub[(sub["modelo"] == mod) & (sub["cenario"] == cen)]
                if linha.empty:
                    continue
                v = float(linha["f1_macro"].iloc[0])
                x = i + (j - (len(modelos) - 1) / 2) * (largura + 0.02)
                # barra fina com ponta arredondada, ancorada na base
                ax.add_patch(FancyBboxPatch((x - largura / 2, 0), largura, v,
                                            boxstyle="round,pad=0,rounding_size=0.015",
                                            mutation_aspect=1 / 3, linewidth=0, facecolor=COR_MODELO[mod]))
                ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=TXT1)
        n = int(sub["n"].iloc[0])
        ax.set_title(f"{conj}  (n = {n})", fontsize=10.5, color=TXT1, loc="left", pad=8)
        ax.set_xticks(range(len(cenarios)))
        ax.set_xticklabels([f"Treino: {ROTULO_CENARIO[c]}" for c in cenarios], fontsize=8.5, color=TXT2)
        ax.set_xlim(-0.6, len(cenarios) - 0.4)
        ax.set_ylim(0, 1.08)
        ax.axhline(0.5, color=TXT2, linewidth=1, linestyle=(0, (3, 3)))
        ax.yaxis.grid(True, color=GRADE, linewidth=0.8)
        ax.set_axisbelow(True)
        for lado in ["top", "right", "left"]:
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(GRADE)
        ax.tick_params(axis="y", colors=TXT2, labelsize=8.5, length=0)
        ax.tick_params(axis="x", length=0)
        if idx % ncol == 0:
            ax.set_ylabel("F1 macro", color=TXT2, fontsize=9)
    for idx in range(len(conjuntos), nlin * ncol):
        eixos[idx // ncol][idx % ncol].axis("off")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legenda = [Patch(facecolor=COR_MODELO[m], label=m) for m in modelos]
    legenda.append(Line2D([0], [0], color=TXT2, linestyle=(0, (3, 3)), label="0,5 (nível do acaso)"))
    fig.legend(handles=legenda, loc="upper left", ncol=len(legenda), frameon=False, fontsize=9,
               bbox_to_anchor=(0.01, 0.985), labelcolor=TXT1)
    fig.suptitle("F1 macro por conjunto de teste, modelo e dados de treino", x=0.01, y=1.025,
                 ha="left", fontsize=12, color=TXT1)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(caminho, dpi=200, bbox_inches="tight", facecolor=SUPERFICIE)
    plt.close(fig)


def main():
    df = carregar()
    if df.empty:
        print("Nenhum resultado em dados/resultados/. Rode antes treinar_bilstm.py / treinar_bertimbau.py.")
        return
    df.to_csv(os.path.join(comum.PASTA_RESULTADOS, "comparacao.csv"), index=False)
    md = tabela_markdown(df)
    with open(os.path.join(comum.PASTA_RESULTADOS, "comparacao.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)

    print("\nF1 macro (linhas: conjunto de teste; colunas: modelo / treino)")
    piv = df.pivot_table(index="conjunto", columns=["modelo", "cenario"], values="f1_macro")
    print(piv.round(3).to_string())

    try:
        caminho = os.path.join(comum.PASTA_RESULTADOS, "comparacao_f1.png")
        grafico(df, caminho)
        print(f"\nGráfico salvo em {caminho}")
    except ImportError:
        print("\n(matplotlib não instalado: conda install -c conda-forge matplotlib)")


if __name__ == "__main__":
    main()
