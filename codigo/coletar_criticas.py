"""
Análise de sentimento de críticas de cinema em português com deep learning
Trabalho da disciplina Redes Neurais e Deep Learning (MBA)
Autores: Samir Saraiva, Claudio Cruz, Vinicius Ferreira e João Felipe

ETAPA 2 — coleta de críticas de cinema escritas em PORTUGUÊS, com nota (rótulo automático)

Fontes:
  - Papo de Cinema  (críticos profissionais; nota de 0 a 10 do autor da crítica,
                     lida da tabela "Grade crítica")
  - AdoroCinema     (críticas de espectadores; nota de 0,5 a 5 estrelas)

Rótulo: nota normalizada >= 0,7 → positivo; <= 0,4 → negativo; entre as duas → neutro
(fica fora do treino e do teste).

Coleta realizada (26/09/2026, ~4h35, 7.196 páginas, 0 erros):
  - Papo de Cinema: 8.579 críticas → 5.762 rotuladas (2.111 neg. / 3.651 pos.), mediana de 751 palavras
  - AdoroCinema:    1.451 críticas → 1.269 rotuladas (236 neg. / 1.033 pos.), 78 filmes, mediana de 44 palavras
  - 257 páginas antigas do Papo de Cinema sem nota extraível ficaram de fora.

Boas práticas adotadas:
  - Respeita o robots.txt de cada site e se identifica com um User-Agent honesto.
  - Pausa entre requisições (usamos --pausa 1.0) para não sobrecarregar os servidores.
  - Não guarda dados pessoais (o nome do usuário não é salvo).
  - Cache local do HTML: interrompida com Ctrl+C, a coleta continua de onde parou.
  - Nunca coleta URLs que pertencem ao conjunto ouro.
  - Divisão treino/validação/teste por FILME, para evitar vazamento de informação.

Comando usado na coleta final (com systemd-inhibit para o notebook não suspender):
  systemd-inhibit --what=idle:sleep:handle-lid-switch \\
    python coletar_criticas.py --max-paginas 400 --max-paginas-filmes 20 --paginas-por-filme 10 --pausa 1.0

Outros usos:
  python coletar_criticas.py --resumo                 # estatísticas do que já foi coletado
  python coletar_criticas.py --diagnostico URL        # inspeciona uma página (se o layout mudar)

Saída: dados/corpus_criticas.csv
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd

from comum import ARQ_CORPUS, PASTA_DADOS, particao_por_grupo, rotulo_da_nota, urls_do_ouro

USER_AGENT = "ProjetoAcademicoSentimentoPTBR/2.0 (trabalho de MBA; coleta educada, respeita robots.txt)"
PASTA_CACHE = os.path.join(PASTA_DADOS, "cache_html")

PAPO_LISTAGEM = "https://www.papodecinema.com.br/categoria/filmes/criticas/"
PAPO_REGEX_CRITICA = re.compile(r"^https://www\.papodecinema\.com\.br/filmes/[^/]+/$")

ADORO_BASE = "https://www.adorocinema.com"
ADORO_LISTAGENS = ["/filmes/numero-cinemas/"]
ADORO_REGEX_FILME = re.compile(r"/filmes/filme-(\d+)/?$")

COLUNAS = ["fonte", "tipo", "grupo", "url", "titulo", "texto", "nota", "nota_max",
           "nota_normalizada", "rotulo", "particao", "n_palavras", "chave"]


# =============================================================================
# HTTP educado com cache
# =============================================================================
class Http:
    def __init__(self, pausa=1.5, usar_cache=True):
        import requests
        self.sessao = requests.Session()
        self.sessao.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt;q=0.9"})
        self.pausa = pausa
        self.usar_cache = usar_cache
        self.robots = {}
        self.n_rede = self.n_cache = self.n_erros = 0
        os.makedirs(PASTA_CACHE, exist_ok=True)

    def _arq_cache(self, url):
        return os.path.join(PASTA_CACHE, hashlib.sha1(url.encode()).hexdigest() + ".html.gz")

    def permitido(self, url):
        p = urlparse(url)
        dominio = f"{p.scheme}://{p.netloc}"
        if dominio not in self.robots:
            rp = RobotFileParser()
            try:
                r = self.sessao.get(dominio + "/robots.txt", timeout=20)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:
                rp.parse([])
            self.robots[dominio] = rp
        return self.robots[dominio].can_fetch(USER_AGENT, url)

    def get(self, url, cache=True):
        arq = self._arq_cache(url)
        if cache and self.usar_cache and os.path.exists(arq):
            self.n_cache += 1
            with gzip.open(arq, "rt", encoding="utf-8") as f:
                return f.read()
        if not self.permitido(url):
            print(f"      [robots.txt não permite] {url}")
            return None
        time.sleep(self.pausa)
        for tentativa in range(3):
            try:
                r = self.sessao.get(url, timeout=30)
            except Exception as e:
                print(f"      erro de conexão ({e.__class__.__name__}), tentativa {tentativa + 1}/3")
                time.sleep(5 * (tentativa + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                espera = 30 * (tentativa + 1)
                print(f"      HTTP {r.status_code}; aguardando {espera}s")
                time.sleep(espera)
                continue
            if r.status_code != 200:
                self.n_erros += 1
                print(f"      HTTP {r.status_code}: {url}")
                return None
            r.encoding = r.apparent_encoding if not r.encoding or r.encoding.lower() == "iso-8859-1" else r.encoding
            self.n_rede += 1
            if cache:
                with gzip.open(arq, "wt", encoding="utf-8") as f:
                    f.write(r.text)
            return r.text
        self.n_erros += 1
        return None


# =============================================================================
# Utilitários de extração
# =============================================================================
def _sopa(html):
    from bs4 import BeautifulSoup
    return BeautifulSoup(html, "html.parser")


def _num(txt):
    try:
        return float(str(txt).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _normalizar_nome(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


def _jsonld(soup):
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


def _titulo(soup):
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    return soup.title.get_text(strip=True) if soup.title else ""


def _limpar(texto):
    return re.sub(r"\s+", " ", texto or "").strip()


# =============================================================================
# PAPO DE CINEMA
# =============================================================================
def papo_links_listagem(html, url_base):
    links = []
    for a in _sopa(html).find_all("a", href=True):
        u = urljoin(url_base, a["href"]).split("#")[0].split("?")[0]
        if not u.endswith("/"):
            u += "/"
        if PAPO_REGEX_CRITICA.match(u) and u not in links:
            links.append(u)
    return links


def _papo_autor(soup):
    meta = soup.find("meta", attrs={"name": "author"})
    if meta and meta.get("content"):
        return meta["content"]
    for obj in _jsonld(soup):
        autor = obj.get("author")
        if isinstance(autor, dict) and autor.get("name"):
            return autor["name"]
        if isinstance(autor, list) and autor and isinstance(autor[0], dict):
            return autor[0].get("name")
    el = soup.find(class_=re.compile(r"author|autor|byline", re.I))
    return el.get_text(" ", strip=True) if el else None


def _papo_grade(soup):
    """Lê a tabela 'Grade crítica' como lista de (nome, nota)."""
    pares = []
    # 1) Tabelas HTML de duas colunas com números
    for tabela in soup.find_all("table"):
        for tr in tabela.find_all("tr"):
            cel = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if len(cel) >= 2 and _num(cel[-1]) is not None and not re.search(r"m[ée]dia", cel[0], re.I):
                pares.append((cel[0], _num(cel[-1])))
        if pares:
            return pares
    # 2) Texto corrido depois de "Grade crítica": linhas alternando nome / número
    linhas = [l.strip() for l in soup.get_text("\n").split("\n") if l.strip()]
    for i, l in enumerate(linhas):
        if re.search(r"grade\s+cr[ií]tica", l, re.I):
            anterior = None
            for l2 in linhas[i + 1:i + 60]:
                if re.search(r"m[ée]dia", l2, re.I):
                    break
                n = _num(l2)
                if n is not None and anterior and _num(anterior) is None:
                    pares.append((anterior, n))
                else:
                    m = re.match(r"^(.+?)\s+(\d{1,2}(?:[.,]\d)?)$", l2)
                    if m and _num(m.group(1)) is None:
                        pares.append((m.group(1), _num(m.group(2))))
                anterior = l2
            break
    return [(nome, n) for nome, n in pares if n is not None and 0 <= n <= 10]


def papo_extrair(html, url):
    soup = _sopa(html)
    nota, nota_max = None, 10.0
    for obj in _jsonld(soup):                      # se houver schema.org Review, usa
        r = obj.get("reviewRating")
        if isinstance(r, dict) and _num(r.get("ratingValue")) is not None:
            nota = _num(r.get("ratingValue"))
            nota_max = _num(r.get("bestRating")) or 10.0
            break
    if nota is None:
        grade = _papo_grade(soup)
        if grade:
            autor = _normalizar_nome(_papo_autor(soup) or "")
            escolhida = next((n for nome, n in grade if autor and _normalizar_nome(nome) in autor
                              or autor and autor in _normalizar_nome(nome)), None)
            nota = escolhida if escolhida is not None else grade[0][1]  # 1ª linha = autor da crítica

    container = (soup.find(class_=re.compile(r"entry-content|post-content|single-content|td-post-content", re.I))
                 or soup.find("article") or soup)
    paragrafos = []
    for p in container.find_all("p"):
        t = _limpar(p.get_text(" ", strip=True))
        if len(t) > 60 and "abas seguintes" not in t.lower():
            paragrafos.append(t)
    texto = " ".join(paragrafos)
    slug = urlparse(url).path.strip("/").split("/")[-1]
    return {"fonte": "Papo de Cinema", "tipo": "crítico", "grupo": f"papo:{slug}", "url": url,
            "titulo": _titulo(soup), "texto": texto, "nota": nota, "nota_max": nota_max}


def coletar_papo(http, max_paginas, salvar, ja_tem, excluir):
    print(f"\n=== PAPO DE CINEMA (até {max_paginas} páginas de listagem) ===")
    novos = 0
    for pagina in range(1, max_paginas + 1):
        url_lista = PAPO_LISTAGEM if pagina == 1 else f"{PAPO_LISTAGEM}page/{pagina}/"
        html = http.get(url_lista, cache=pagina > 3)   # primeiras páginas mudam: sempre atualiza
        if not html:
            print(f"   Página {pagina}: sem resposta; encerrando a listagem.")
            break
        links = [u for u in papo_links_listagem(html, url_lista) if u not in ja_tem and u not in excluir]
        print(f"   Página {pagina}/{max_paginas}: {len(links)} críticas novas")
        for url in links:
            h = http.get(url)
            if not h:
                continue
            reg = papo_extrair(h, url)
            ja_tem.add(url)
            if len(reg["texto"].split()) < 80:
                continue
            salvar(reg)
            novos += 1
    return novos


# =============================================================================
# ADOROCINEMA (críticas de espectadores)
# =============================================================================
def adoro_ids_filmes(html):
    ids = []
    for a in _sopa(html).find_all("a", href=True):
        m = ADORO_REGEX_FILME.search(urlparse(urljoin(ADORO_BASE, a["href"])).path)
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


def adoro_extrair(html):
    """Extrai (nota, texto) de cada crítica de espectador na página."""
    soup = _sopa(html)
    titulo = _titulo(soup)
    criticas = []
    cartoes = soup.find_all(class_="review-card") or soup.find_all(class_=re.compile(r"review-card$|hred review", re.I))
    for c in cartoes:
        nota = None
        for el_nota in c.find_all(class_=re.compile(r"stareval-note|rating-note|\bnote\b", re.I)):
            nota = _num(el_nota.get_text(strip=True))
            if nota is not None and 0 <= nota <= 5:
                break
            nota = None
        el_txt = c.find(class_=re.compile(r"content-txt|review-card-content|review-content", re.I))
        texto = _limpar(el_txt.get_text(" ", strip=True)) if el_txt else ""
        if nota is not None and texto:
            criticas.append((nota, texto))
    return titulo, criticas


def coletar_adoro(http, max_paginas_filmes, paginas_por_filme, salvar, ja_tem):
    print(f"\n=== ADOROCINEMA (espectadores; até {max_paginas_filmes} páginas de filmes, "
          f"{paginas_por_filme} páginas de críticas por filme) ===")
    ids = []
    for base in ADORO_LISTAGENS:
        for pagina in range(1, max_paginas_filmes + 1):
            url = ADORO_BASE + base + (f"?page={pagina}" if pagina > 1 else "")
            html = http.get(url, cache=pagina > 3)
            if not html:
                break
            novos_ids = [i for i in adoro_ids_filmes(html) if i not in ids]
            if not novos_ids:
                break
            ids += novos_ids
    print(f"   {len(ids)} filmes encontrados")

    novos = 0
    for k, fid in enumerate(ids, 1):
        base = f"{ADORO_BASE}/filmes/filme-{fid}/criticas/espectadores/"
        vistos_filme = set()
        n_filme = 0
        for pg in range(1, paginas_por_filme + 1):
            url = base + (f"?page={pg}" if pg > 1 else "")
            html = http.get(url)
            if not html:
                break
            titulo, criticas = adoro_extrair(html)
            chaves = {hashlib.md5(t.encode()).hexdigest() for _, t in criticas}
            if not criticas or chaves <= vistos_filme:   # página vazia ou repetida: acabou
                break
            vistos_filme |= chaves
            for nota, texto in criticas:
                chave = hashlib.md5(texto.encode()).hexdigest()
                if chave in ja_tem or len(texto.split()) < 20:
                    continue
                ja_tem.add(chave)
                salvar({"fonte": "AdoroCinema", "tipo": "espectador", "grupo": f"adoro:{fid}",
                        "url": url, "titulo": titulo, "texto": texto, "nota": nota, "nota_max": 5.0,
                        "chave": chave})
                novos += 1
                n_filme += 1
        print(f"   [{k}/{len(ids)}] filme {fid}: {n_filme} críticas novas")
        if k == 3 and novos == 0:
            print("\n   ATENÇÃO: nenhuma crítica extraída nos 3 primeiros filmes. O layout do site pode")
            print("   ser diferente do esperado. Rode e me envie a saída de:")
            print(f"   python coletar_criticas.py --diagnostico {base}")
            break
    return novos


# =============================================================================
# Persistência e resumo
# =============================================================================
class Corpus:
    def __init__(self):
        if os.path.exists(ARQ_CORPUS):
            self.df = pd.read_csv(ARQ_CORPUS)
        else:
            self.df = pd.DataFrame(columns=COLUNAS)
        self.pendentes = []

    def chaves(self):
        return set(self.df["chave"].astype(str)) | set(self.df["url"].astype(str))

    def adicionar(self, reg):
        nota, nota_max = reg.get("nota"), reg.get("nota_max")
        norm = nota / nota_max if nota is not None and nota_max and 0 <= nota <= nota_max else None
        reg["nota_normalizada"] = norm
        reg["rotulo"] = rotulo_da_nota(norm)
        reg["particao"] = particao_por_grupo(reg["grupo"])
        reg["n_palavras"] = len(reg["texto"].split())
        reg.setdefault("chave", reg["url"])
        self.pendentes.append(reg)
        if len(self.pendentes) >= 25:
            self.gravar()

    def gravar(self):
        if self.pendentes:
            self.df = pd.concat([self.df, pd.DataFrame(self.pendentes)], ignore_index=True)
            self.df = self.df.drop_duplicates(subset="chave", keep="last")
            self.pendentes = []
        os.makedirs(PASTA_DADOS, exist_ok=True)
        self.df[COLUNAS].to_csv(ARQ_CORPUS, index=False)


def resumo():
    if not os.path.exists(ARQ_CORPUS):
        print("Nenhum corpus coletado ainda.")
        return
    df = pd.read_csv(ARQ_CORPUS)
    print(f"\n=== RESUMO DO CORPUS ({ARQ_CORPUS}) ===")
    df["classe"] = df["rotulo"].map({1.0: "positivo", 0.0: "negativo"}).fillna("neutro (fora)")
    tab = pd.crosstab(df["fonte"], df["classe"], margins=True, margins_name="TOTAL")
    print(tab.to_string())
    print("\nCríticas rotuladas por partição (dividida por filme):")
    rot = df[df["rotulo"].notna()]
    print(pd.crosstab(rot["fonte"], rot["particao"], margins=True, margins_name="TOTAL").to_string())
    print("\nFilmes distintos por fonte:", rot.groupby("fonte")["grupo"].nunique().to_dict())
    print("Palavras por crítica (mediana):", rot.groupby("fonte")["n_palavras"].median().to_dict())
    sem_nota = df["nota"].isna().sum()
    if sem_nota:
        print(f"\nATENÇÃO: {sem_nota} críticas sem nota extraída (ficam fora do treino/teste).")


def diagnostico(url):
    http = Http(pausa=0, usar_cache=False)
    html = http.get(url, cache=False)
    if not html:
        print("Não foi possível baixar a página.")
        return
    soup = _sopa(html)
    print(f"Título: {_titulo(soup)}")
    print(f"Tamanho do HTML: {len(html):,} caracteres")
    print("Tipos JSON-LD:", [o.get("@type") for o in _jsonld(soup)][:15])
    print("Tabelas:", len(soup.find_all("table")))
    classes = {}
    for el in soup.find_all(class_=True):
        for c in el.get("class", []):
            if re.search(r"review|critic|rating|note|star|grade|nota|autor|author|content", c, re.I):
                classes[c] = classes.get(c, 0) + 1
    print("Classes relevantes (classe: ocorrências):")
    for c, n in sorted(classes.items(), key=lambda x: -x[1])[:40]:
        print(f"   {c}: {n}")
    for marcador in ["Enviada em", "Grade crítica", "Grade Crítica"]:
        el = soup.find(string=re.compile(marcador))
        if el:
            cadeia = []
            p = el.parent
            for _ in range(6):
                if p is None:
                    break
                cadeia.append(f"{p.name}.{'.'.join(p.get('class', []))}")
                p = p.parent
            print(f"Ancestrais de '{marcador}': {' < '.join(cadeia)}")
    pag = [a["href"] for a in soup.find_all("a", href=True) if re.search(r"page[=/]|pagina", a["href"])]
    print("Links de paginação:", pag[:8])
    if "papodecinema" in url:
        print("\nExtração Papo:", {k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                                   for k, v in papo_extrair(html, url).items()})
        print("Grade lida:", _papo_grade(soup), "| Autor:", _papo_autor(soup))
    if "adorocinema" in url:
        titulo, criticas = adoro_extrair(html)
        print(f"\nExtração AdoroCinema: {len(criticas)} críticas")
        for nota, texto in criticas[:3]:
            print(f"   nota {nota}: {texto[:100]}...")


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonte", choices=["papo", "adoro", "todas"], default="todas")
    ap.add_argument("--max-paginas", type=int, default=100, help="Papo de Cinema: páginas de listagem (~10 críticas cada; o site tem ~400)")
    ap.add_argument("--max-paginas-filmes", type=int, default=15, help="AdoroCinema: páginas da lista de filmes")
    ap.add_argument("--paginas-por-filme", type=int, default=8, help="AdoroCinema: páginas de críticas por filme")
    ap.add_argument("--pausa", type=float, default=1.5, help="segundos entre requisições")
    ap.add_argument("--resumo", action="store_true")
    ap.add_argument("--diagnostico", metavar="URL")
    args = ap.parse_args()

    if args.diagnostico:
        diagnostico(args.diagnostico)
        sys.exit()
    if args.resumo:
        resumo()
        sys.exit()

    http = Http(pausa=args.pausa)
    corpus = Corpus()
    ja_tem = corpus.chaves()
    excluir = urls_do_ouro()          # nunca coletar críticas que estão no conjunto ouro
    inicio = time.time()
    try:
        if args.fonte in ("papo", "todas"):
            n = coletar_papo(http, args.max_paginas, corpus.adicionar, ja_tem, excluir)
            print(f"   Papo de Cinema: {n} críticas novas")
        if args.fonte in ("adoro", "todas"):
            n = coletar_adoro(http, args.max_paginas_filmes, args.paginas_por_filme, corpus.adicionar, ja_tem)
            print(f"   AdoroCinema: {n} críticas novas")
    except KeyboardInterrupt:
        print("\n   Interrompido. O que já foi coletado está salvo; rode de novo para continuar.")
    finally:
        corpus.gravar()
        minutos = (time.time() - inicio) / 60
        print(f"\n   {http.n_rede} páginas baixadas, {http.n_cache} do cache, {http.n_erros} erros, {minutos:.1f} min")
        resumo()
