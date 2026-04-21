# Estrutura do Repositório

Documento de referência da organização proposta para `ptbr-market-classification`.
A meta é um repositório enxuto e reprodutível, alinhado ao escopo descrito em
`plano_base.md` e às restrições metodológicas não-negociáveis registradas em
`CLAUDE.md`.

## Princípios norteadores

1. **Simples antes de sofisticado.** Cada arquivo existe por uma única razão.
   Sem _registries_ prematuros, sem hierarquias especulativas, sem utilitários
   que "podem vir a ser úteis". Três scripts diretos são preferíveis a uma
   abstração genérica.
2. **Reprodutibilidade de ponta a ponta.** Toda tabela, figura ou métrica do
   artigo deve ser regenerável a partir de um comando documentado, a partir
   de dados brutos + `uv sync`.
3. **Fonte de dados única: FolhaSP/UOL.** Não há múltiplas fontes, nem
   _scraping_, nem manifestos de integração. Se o trabalho parecer exigir
   outra fonte, isso é um sinal para revisar o escopo, não para expandir o
   repositório.
4. **O split temporal é sagrado.** O _fit_ de qualquer componente que aprende
   (vocabulário TF-IDF, _threshold_ de decisão, meta-classificador do
   _stacking_) acontece em treino e/ou validação. O teste é visto uma única
   vez, no final, e nunca influencia nenhum ajuste.

## Árvore de diretórios

```
ptbr-market-classification/
├── README.md                  # visão geral + setup
├── CLAUDE.md                  # instruções para o Claude Code
├── plano_base.md              # plano científico (escopo, modelos, métricas)
├── pyproject.toml             # dependências + config de ruff/pytest
├── uv.lock                    # lockfile
├── .python-version            # versão do Python (3.12)
├── .gitignore
│
├── data/
│   ├── raw/                   # corpus FolhaSP (não versionado)
│   └── README.md              # instruções para obter o dado
│
├── src/
│   └── ptbr_market/
│       ├── __init__.py
│       ├── data.py            # carregamento + split OOT 80/10/10
│       ├── preprocessing.py   # pipeline bimodal + máscara de entidades
│       ├── representations.py # BoW, TF-IDF, fastText
│       ├── gen1_classical.py  # modelos da Geração 1
│       ├── gen2_bert.py       # fine-tuning dos encoders
│       ├── gen3_llm.py        # cliente Ollama e prompts
│       ├── ensemble.py        # Soft Voting + Stacking
│       ├── evaluation.py      # F1, PR-AUC, ROC-AUC, McNemar
│       ├── threshold.py       # tuner congelado após validação
│       ├── profiling.py       # latência e VRAM
│       └── runs.py            # diretórios de run + metadata JSON
│
├── scripts/
│   ├── build_splits.py        # materializa splits OOT em disco
│   ├── run_gen1.py            # treina e avalia um modelo clássico
│   ├── run_gen2.py            # fine-tuning de um encoder
│   ├── run_gen3.py            # inferência zero-shot via Ollama
│   ├── run_ensemble.py        # Soft Voting + Stacking
│   └── build_paper_tables.py  # consolida artifacts nas tabelas do artigo
│
├── notebooks/
│   ├── 00_eda.ipynb           # análise exploratória do corpus
│   ├── 99_figures.ipynb       # figuras finais do artigo
│   └── colab/
│       ├── bootstrap_gen2.ipynb   # wrapper Colab para run_gen2.py
│       └── bootstrap_gen3.ipynb   # wrapper Colab para run_gen3.py
│
├── tests/
│   ├── conftest.py            # fixtures + mini corpus sintético com datas
│   ├── test_data.py           # disjunção temporal do split
│   ├── test_preprocessing.py  # bimodal + máscara de entidades
│   ├── test_threshold.py      # fit em val, freeze, apply em test
│   ├── test_evaluation.py     # F1, PR-AUC, McNemar
│   └── test_ensemble.py       # ausência de leakage no stacking
│
├── artifacts/
│   ├── splits/                # parquet dos splits OOT (gerado uma vez)
│   └── runs/                  # um diretório por experimento (ver convenções)
│
└── docs/
    └── estrutura_repositorio.md   # este documento
```

## Responsabilidades por diretório

### `data/`
Abriga **apenas** o corpus bruto da FolhaSP/UOL, não versionado. O `README.md`
deste diretório traz o passo-a-passo de download (Kaggle) e a validação de
integridade esperada (contagem de linhas, _hash_, janela temporal observada).
Nenhum arquivo derivado vive aqui.

### `src/ptbr_market/`
Módulos planos (sem subpacotes) organizados por responsabilidade única.
A fronteira entre módulos reflete a separação que importa para o artigo:

- `data.py` implementa o split _Out-of-Time_ 80/10/10 ordenado por data de
  publicação. É o único lugar que toca a ordem cronológica.
- `preprocessing.py` expõe o paradigma bimodal: pipeline agressivo com
  lematização (SpaCy) para a Geração 1 e passagem quase-crua para as
  Gerações 2 e 3. A máscara de entidades financeiras
  (`[VALOR_MONETARIO]`, `[PERCENTUAL]`) é uma configuração explícita, logada.
- `gen1_classical.py`, `gen2_bert.py`, `gen3_llm.py` expõem cada um uma
  interface uniforme (`fit`, `predict_proba`/`predict`, `save`, `load`) para
  que `ensemble.py` consuma seus _scores_ sem conhecer detalhes.
- `threshold.py` encapsula o ajuste de limiar em validação e impede, por
  construção, sua reaplicação em teste.
- `evaluation.py` concentra F1 (macro e da classe minoritária), PR-AUC,
  ROC-AUC e a tabela NxN de McNemar.
- `profiling.py` mede latência (ms por 1000 notícias) e VRAM de pico,
  reportadas junto às métricas de desempenho.
- `runs.py` define o diretório de run, gera `metadata.json` e persiste
  `predictions.csv` + `metrics.json`.

### `scripts/`
Entrypoints reprodutíveis, um por etapa. Cada script aceita argumentos
mínimos (modelo, variação, seed) e escreve exatamente um diretório em
`artifacts/runs/`. Scripts não importam uns aos outros — toda a lógica vive
em `src/`.

### `notebooks/`
Apenas **exploração** e **figuras finais**. Notebooks nunca são etapa do
pipeline de reprodução. Resultados numéricos do artigo vêm sempre de
`scripts/`; notebooks apenas leem `artifacts/` e produzem gráficos.

### `tests/`
Cobertura mínima mas estratégica. Prioriza invariantes metodológicas
(disjunção temporal do split, _threshold_ não vazando para teste, meta-
classificador treinado só em validação) e métricas manualmente verificáveis.
Usa _markers_ `slow` e `integration` para permitir `pytest -m "not slow"`
no ciclo rápido.

### `artifacts/`
Tudo o que é gerado pelo pipeline. `splits/` é construído uma única vez por
`build_splits.py`. `runs/` recebe um diretório por experimento.

### `docs/`
Documentos curtos de apoio. Nada que duplique o que já está em `plano_base.md`
ou em `CLAUDE.md`.

## Convenções

### Nomenclatura de _runs_

Um _run_ é um diretório em `artifacts/runs/` com o padrão:

```
{YYYYMMDD-HHMMSS}__{geracao}__{modelo}__{variante}/
```

Exemplos:

```
20260421-143005__gen1__linearsvc__tfidf-lemmatized
20260421-150210__gen2__bertimbau-base__raw
20260421-161940__gen3__qwen2.5-7b__zeroshot-json
20260421-170055__ensemble__stacking__linearsvc+bertimbau
```

### Esquema do `metadata.json`

Cada _run_ grava um `metadata.json` com campos obrigatórios:

```jsonc
{
  "run_id": "20260421-143005__gen1__linearsvc__tfidf-lemmatized",
  "git_commit": "abc1234",
  "started_at": "2026-04-21T14:30:05-03:00",
  "finished_at": "2026-04-21T14:31:47-03:00",
  "generation": "gen1",
  "model": "linearsvc",
  "variant": "tfidf-lemmatized",
  "config": { /* hiperparâmetros + flags de preprocessing */ },
  "split": {
    "train_window": ["2015-01-01", "2018-12-31"],
    "val_window":   ["2019-01-01", "2019-12-31"],
    "test_window":  ["2020-01-01", "2020-12-31"],
    "n_train": 133000, "n_val": 16600, "n_test": 16600
  },
  "threshold": { "fitted_on": "val", "value": 0.33 },
  "metrics": {
    "f1_macro": 0.00, "f1_minority": 0.00,
    "pr_auc": 0.00,   "roc_auc": 0.00
  },
  "efficiency": {
    "latency_ms_per_1k": 0.0,
    "vram_peak_mb": 0
  }
}
```

Além do `metadata.json`, cada _run_ persiste `predictions.csv` (colunas
`index, y_true, y_pred, y_score`) e um `metrics.json` redundante para leitura
rápida sem parser.

### Nomenclatura de modelos

Usa-se `snake-kebab` curto e previsível, sem sufixos de versão HuggingFace:
`bertimbau-base`, `bertimbau-large`, `finbert-ptbr`, `deb3rta`, `xlmr-base`,
`xlmr-large`, `albertina-900m`, `distilbertimbau`, `llama3.1-8b`, `qwen2.5-7b`,
`qwen2.5-14b`, `gemma2-9b`, `tucano-7b`, `bode-7b`. O caminho completo do
_checkpoint_ fica registrado em `config` dentro do `metadata.json`.

## Execução em Google Colab

Geração 2 (_fine-tuning_ de _encoders_, em especial BERTimbau Large, Albertina
900M e XLM-RoBERTa Large) e Geração 3 (LLMs via Ollama) são planejadas para
rodar em Colab (T4 16GB ou L4 24GB). Geração 1 e o _ensemble_ rodam
confortavelmente em máquina local e não demandam Colab.

A regra é **uma só base de código**: o notebook de Colab não contém lógica
científica, apenas _bootstrap_. Toda a lógica de treino, inferência e
avaliação vive em `src/ptbr_market/` e é invocada pelos mesmos scripts
(`scripts/run_gen2.py`, `scripts/run_gen3.py`) que rodam localmente.

### Layout no Google Drive

```
MyDrive/ptbr-market-classification/
├── data/raw/            # corpus FolhaSP (upload único)
├── artifacts/splits/    # parquets dos splits (upload único após build local)
└── artifacts/runs/      # um diretório por run, mesmo padrão do local
```

O corpus bruto e os _splits_ são enviados uma única vez ao Drive. Os _runs_
são gravados diretamente no Drive para sobreviver à efemeridade da sessão
Colab.

### Contrato entre código e ambiente

O código em `src/ptbr_market/runs.py` lê dois caminhos de raiz via variáveis
de ambiente:

- `PTBR_DATA_ROOT` (_default_: `./data`)
- `PTBR_ARTIFACTS_ROOT` (_default_: `./artifacts`)

Em local, os _defaults_ bastam. Em Colab, o notebook de _bootstrap_ define
ambos para caminhos dentro do Drive montado, antes de invocar o script.

### Anatomia de um notebook de _bootstrap_ Colab

Cada notebook em `notebooks/colab/` faz apenas estas quatro coisas, em ordem:

1. **Montar o Drive** (`google.colab.drive.mount`).
2. **Clonar o repositório** numa pasta efêmera (`/content/ptbr-market-classification`).
3. **Instalar dependências** via `uv sync` (ou `pip install -e .` como
   _fallback_, já que `uv` não é nativo no Colab).
4. **Exportar `PTBR_DATA_ROOT` e `PTBR_ARTIFACTS_ROOT`** apontando para o
   Drive e invocar o script canônico (`!python scripts/run_gen2.py --model ...`).

Gen 3 tem um passo extra antes de (4): subir o servidor Ollama em
_background_ (`ollama serve &`) e fazer `ollama pull <modelo>` ou importar
um `.GGUF` via `Modelfile` para modelos não-oficiais (Tucano, Bode).

Qualquer lógica que tente escapar desse recorte (processar dados, definir
métricas, fazer análise) deve ser movida para `src/` — o notebook não é o
lugar.

## Fluxo de reprodução

O caminho crítico para regenerar os resultados do artigo, partindo de um
_clone_ limpo:

1. `uv sync` — instala dependências a partir do `uv.lock`.
2. Baixar o corpus FolhaSP do Kaggle para `data/raw/` conforme `data/README.md`.
3. `python scripts/build_splits.py` — gera `artifacts/splits/{train,val,test}.parquet`.
4. (Uma vez) _Upload_ de `data/raw/` e `artifacts/splits/` para o Drive,
   seguindo o layout da seção _Execução em Google Colab_.
5. Para cada modelo da Geração 1: `python scripts/run_gen1.py --model <id> --variant <id>` (local).
6. Para cada _encoder_ da Geração 2: abrir `notebooks/colab/bootstrap_gen2.ipynb`,
   selecionar `<id>` e executar (Colab).
7. Para cada LLM da Geração 3: abrir `notebooks/colab/bootstrap_gen3.ipynb`,
   selecionar `<id>` e executar (Colab, com Ollama).
8. Sincronizar `MyDrive/.../artifacts/runs/` para `./artifacts/runs/` local
   (ex.: `rclone` ou download manual) antes do _ensemble_.
9. `python scripts/run_ensemble.py --base <run_gen1> --base <run_gen2>` (local).
10. `python scripts/build_paper_tables.py` — consolida `artifacts/runs/` em
    tabelas e CSVs consumidos pelo notebook `99_figures.ipynb`.

Cada etapa é idempotente: rodar novamente gera um novo _run_ (novo
_timestamp_), sem sobrescrever histórico.

## Anti-padrões explícitos

Decisões a recusar, mesmo quando parecerem convenientes:

- **Nunca** usar `train_test_split` com `shuffle=True` sobre o corpus. O split
  é estritamente cronológico.
- **Nunca** ajustar _threshold_, vocabulário TF-IDF, ou meta-classificador de
  _stacking_ com dados de teste. Estes sempre se ajustam em treino e/ou
  validação e são _congelados_ antes da avaliação final.
- **Nunca** incluir um modelo na tabela final sem registrar latência e VRAM.
  F1 sozinho não justifica inclusão (Green AI).
- **Nunca** adicionar fontes de dados além da FolhaSP/UOL. O escopo é
  deliberadamente estreito.
- **Nunca** portar módulos do projeto predecessor `economy-classifier`.
  Lições conceituais, sim; código, não.
- **Nunca** fazer do notebook parte do pipeline oficial de resultados.
  Notebooks são para exploração, figuras, e _bootstrap_ de ambiente Colab —
  nunca contêm lógica científica. Números do artigo saem de scripts.
- **Nunca** adicionar o _trailer_ `Co-Authored-By: Claude` em commits.
