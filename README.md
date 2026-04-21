# ptbr-market-classification

Pipeline de pesquisa para **classificação binária de notícias em português
brasileiro** como *mercado/finanças* vs. *outras*, com foco em um artigo
BRACIS. O corpus-alvo é o Folha de São Paulo / UOL (~167 mil artigos,
~12,6% positivos — fortemente desbalanceado).

`plano_base.md` é a fonte da verdade sobre escopo, modelos e metodologia.
Este README cobre apenas o setup e o fluxo de reprodução de alto nível.

## Setup

### 1. Pré-requisitos de sistema

- Python 3.12 (ver `.python-version`).
- [`uv`](https://github.com/astral-sh/uv) para gestão de dependências.
- `git-lfs` — o corpus bruto e os splits são versionados via LFS.

```bash
sudo apt install git-lfs      # Linux; em macOS: brew install git-lfs
git lfs install               # uma vez por clone
```

### 2. Dependências Python

```bash
uv sync
```

### 3. Modelo SpaCy (necessário para a Geração 1)

```bash
uv run python -m spacy download pt_core_news_lg
```

### 4. Vetores fastText pré-treinados (necessário para a Geração 1)

```bash
curl -L -o data/raw/cc.pt.300.vec.gz \
    https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz
gunzip data/raw/cc.pt.300.vec.gz
```

Detalhes e integridade em `data/README.md`.

## Fluxo de reprodução

Os splits Out-of-Time 80/10/10 já vêm versionados em `artifacts/splits/`.
Para regenerá-los a partir do corpus bruto:

```bash
uv run python scripts/build_splits.py
```

Scripts de treino e avaliação (`run_gen1.py`, `run_ensemble.py`, etc.) são
adicionados conforme cada geração é implementada. Ver
`docs/estrutura_repositorio.md` para o layout completo e o fluxo ponta a
ponta previsto.

## Testes

```bash
uv run pytest                      # suíte completa (inclui integração)
uv run pytest -m "not integration" # rápido, sem corpus real
```

## Onde está o quê

- `plano_base.md` — plano científico (escopo, modelos, métricas).
- `CLAUDE.md` — instruções e restrições metodológicas não-negociáveis.
- `docs/estrutura_repositorio.md` — estrutura do repositório e convenções.
- `data/README.md` — corpus, recursos externos e integridade.
