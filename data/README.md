# Dados

## Corpus versionado

O repositório versiona o corpus FolhaSP/UOL diretamente em
`data/raw/folhasp.parquet` (via **git-lfs**). Isso garante que o projeto
funcione sem depender do Kaggle estar no ar.

| Campo | Valor |
|---|---|
| Origem | Kaggle — `marlesson/news-of-the-site-folhauol` |
| Versão observada | CSV com mtime `2019-09-21` |
| Linhas | 167.053 |
| Colunas | `title`, `text`, `date`, `category`, `link` |
| Janela temporal | 2015-01-01 → 2017-10-01 |
| Classe positiva | `category == "mercado"` — 20.970 linhas (12,55%) |
| Compressão | Parquet + zstd (level 3) |
| Tamanho | ~177 MB |

### Integridade

```
csv_sha256     = 33e338a6e5e1ec0d40368028a1cc7c197b26d7c6eb25ba40bddc62cc1632355c
parquet_sha256 = ddeddbb6c7d5f783d68dd641dc9763a32cc1b29839d7d7035fbeedcf930c979c
```

O `parquet_sha256` é estável quando o parquet é gerado pela mesma versão de
`pyarrow` (atualmente 24.0.0). Em outras versões o conteúdo lógico é idêntico
mas os bytes podem diferir. A verificação autoritativa é o `csv_sha256`.

## Regeneração

Caso o parquet precise ser reconstruído (ex.: nova versão do `pyarrow`),
obter novamente o CSV original do Kaggle e rodar:

```bash
uv run python scripts/convert_csv_to_parquet.py <caminho-para>/articles.csv
```

O script:

1. Lê o CSV com `pandas.read_csv`.
2. Converte `date` para `datetime64`.
3. Remove a coluna `subcategory` (94% nula, irrelevante para classificação binária).
4. Ordena por `(date, link)` com `mergesort` (estável, determinístico).
5. Escreve `data/raw/folhasp.parquet` em zstd level 3.

A conversão é _one-shot_ — não faz parte do pipeline de reprodução dos
resultados do artigo.

## Versionamento com git-lfs

Em um _clone_ novo, antes do primeiro `git pull`:

```bash
sudo apt install git-lfs      # uma vez por máquina
git lfs install               # uma vez por repositório
```

O filtro LFS está configurado em `.gitattributes` para:

- `data/raw/*.parquet` (~177 MB, corpus bruto).
- `artifacts/splits/*.parquet` — `train.parquet` fica em torno de 137 MB
  (acima do limite de 100 MB do GitHub), então os três splits passam por LFS
  por uniformidade.

## Recursos externos (não versionados)

### Vetores fastText pré-treinados (`cc.pt.300`)

Necessários para a Geração 1 (representação fastText-avg). Baixar em
`data/raw/` na primeira execução:

```bash
curl -L -o data/raw/cc.pt.300.vec.gz \
    https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz
gunzip data/raw/cc.pt.300.vec.gz
```

| Campo | Valor |
|---|---|
| Origem | Facebook Research — fastText Common Crawl |
| Dimensão | 300 |
| Tamanho (descomprimido) | ~4,5 GB |
| Licença | CC BY-SA 3.0 |

Ficam fora do LFS e do git (ver `.gitignore`): são baixáveis livremente e o
corpus é pequeno o suficiente para que o custo de versionar não se justifique.

### Modelo SpaCy `pt_core_news_lg`

Usado pelo pipeline agressivo de pré-processamento (Geração 1) para
lematização. Instalar uma vez por ambiente:

```bash
uv run python -m spacy download pt_core_news_lg
```

~560 MB. Fornece lematizador e vetores 300d embutidos (estes últimos não
são usados — fastText é a representação densa oficial do projeto).
