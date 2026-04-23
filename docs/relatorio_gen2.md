# Relatório Gen 2 — Encoders BERT (parcial)

> **Status: parcial — 2 de 6 modelos do registry concluídos** (`bertimbau-base`,
> `finbert-ptbr`). Pendentes: `distilbertimbau`, `bertimbau-large`, `xlmr-base`,
> `deb3rta-base`. Esta versão será reescrita conforme novos ZIPs do Colab cheguem.

## Origem dos artefatos

| Pacote (Colab) | Coleta (UTC) | Commit | Runs trazidos |
|---|---|---|---|
| `artifacts-20260422T234816Z-3-001.zip` | 2026-04-22 23:48 | `faff651` (bertimbau) / `c3292ea` (finbert) | 4 (`bertimbau-base` × {bin,mc8}, `finbert-ptbr` × {bin,mc8}) |

`finbert-ptbr` foi treinado depois do fix `c3292ea` (`force problem_type=single_label` para
evitar BCELoss em `num_labels=2`); `bertimbau-base` rodou em `faff651` e não foi afetado
pelo bug. Splits versionados localmente (`artifacts/splits/*.parquet`) bateram em md5
com os duplicados do ZIP — confirmação de que os 4 runs viram exatamente os mesmos
133 725 / 16 701 / 16 627 artigos das janelas Out-of-Time do Gen 1.

## Objetivo

Verificar se encoders BERT pré-treinados em PT-BR superam o campeão Gen 1
(`tfidf + linearsvc + mc8`, F1-min = **0,8536**, PR-AUC = **0,9320**) na mesma
superfície binária e nas mesmas janelas Out-of-Time, e replicar a pergunta-mãe
do `plano_base.md`: a decomposição da classe negativa em 8 categorias
(`top7_plus_other`) também ajuda quando o modelo já tem representação contextual?

## Setup

- **Splits idênticos aos de Gen 1.** Mesmas janelas: train 2015-01-01 …
  2017-01-07 (n = 133 725); val 2017-01-08 … 2017-05-23 (n = 16 701); test
  2017-05-24 … 2017-10-01 (n = 16 627, positivos = 2 177).
- **Pré-processamento `raw`.** BERT recebe texto bruto, sem lematização nem
  máscara de entidades — decisão D-Gen2 (a máscara `[VALOR_MONETARIO]` etc. é só
  Gen 1, ver `memory/project_gen1_decisions.md`). Tokenização com cada
  tokenizer nativo, `max_length = 256`, `padding="max_length"`, truncamento à
  direita.
- **Encoders avaliados (2/6 do registry).**
  | Slug | HF id | Bucket |
  |---|---|---|
  | `bertimbau-base` | `neuralmind/bert-base-portuguese-cased` | PT-BR |
  | `finbert-ptbr` | `lucas-leme/FinBERT-PT-BR` | Domain |
- **Treino.** `epochs = 3`, `learning_rate = 2e-5`, `batch_size = 16`,
  `grad_accum = 1`, `fp16 = True`, `seed = 1`. Rodado em Colab T4 (16 GB)
  via `scripts/run_gen2.py` — bootstrap local, treino remoto.
- **Threshold calibrado no val, congelado no test.** Mesma grade do Gen 1
  (`[0.05, 0.95]` passo 0.01, objetivo `f1_minority`). Em multiclasse o score
  binário é `softmax(logits)[:, idx("mercado")]` — threshold continua binário.
- **Métricas.** PR-AUC (principal pelo desbalanceamento ≈ 12,3 %), F1-min,
  precisão/recall da minoritária, ROC-AUC, F1-macro e latência em ms/1k
  artigos (incluindo tokenização — flag `latency_includes_tokenization=true`).
  VRAM de pico é registrada no `metadata.json` para diagnóstico operacional
  mas **não entra em comparações cross-geração** (ver PA-003 em
  `pontos_de_atencao.md`): o eixo de eficiência do paper é unicamente
  latência, métrica medida de forma homogênea em todas as 3 gerações.
- **Dois modos por modelo.** `bin` (`label` ∈ {0,1}) e `mc8` (`category`
  colapsada em 8 classes `top7_plus_other`). 2 pares para McNemar
  intra-Gen 2.

## Resultados pareados (test split, n = 16 627) — *parcial*

| Bucket | Modelo | F1-min bin | F1-min mc8 | Δ F1 | PR-AUC bin | PR-AUC mc8 | Lat. bin | Lat. mc8 | Thr. bin | Thr. mc8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PT-BR  | `bertimbau-base` | 0,8754 | **0,8830** | +0,0076 | 0,9423 | **0,9464** | 4 116 | 4 111 | 0,05 ⚠ | 0,16 |
| Domain | `finbert-ptbr`   | 0,8654 | **0,8814** | +0,0159 | 0,9402 | **0,9462** | 3 956 | 4 034 | 0,21    | 0,30 |

Latência em ms / 1 000 artigos (Colab T4, fp16, inclui tokenização).
Thr. = threshold ótimo no val. VRAM ≈ 3,1 GB nos 4 runs (folgada no T4 de
16 GB) — registrada em `metadata.json` mas omitida da tabela porque não
entra em comparações cross-geração (PA-003). n = 2 modelos concluídos de 6;
tabela será reordenada quando o registry estiver completo.

**Leitura preliminar.** Os 2 encoders disponíveis cruzam o teto do Gen 1 nas
quatro variantes (mínimo: `finbert-ptbr` bin com F1-min = 0,8654, +1,2 pp sobre
o campeão; máximo: `bertimbau-base` mc8 com F1-min = 0,8830, +2,9 pp). PR-AUC
sobe de 0,9320 (Gen 1 campeão) para 0,9402–0,9464 — ganho consistente no
ranking probabilístico, não só em F1. Surpreendentemente, `bertimbau-base`
(genérico PT-BR) está empatado com o `finbert-ptbr` (especializado em
finanças/notícias) — possível indicação de que o pré-treino financeiro do
FinBERT-PT-BR não traz vantagem clara para a fronteira `mercado` vs. `outros`
neste corpus, ou que a janela de teste tem distribuição de assuntos
diferente da que o FinBERT viu. **Aguardar `deb3rta-base` para fechar a
leitura do bucket Domain.**

## McNemar intra-Gen 2 (binary vs mc8)

Mesmo teste do Gen 1: χ² com correção de continuidade; `b` = bin acerta e mc8
erra, `c` = bin erra e mc8 acerta.

| Modelo | b | c | p-valor | Vencedor |
|---|---:|---:|---:|:---|
| `bertimbau-base` | 174 | 201 | 0,179 | mc8 (n.s.) |
| `finbert-ptbr`   | 172 | 229 | 5,2 × 10⁻³ | **mc8** |

**Leitura.** No `finbert-ptbr` o ganho de mc8 sobre bin é estatisticamente
significativo, replicando o padrão visto em 12 dos 13 pares Gen 1. Em
`bertimbau-base` a direção é a mesma (mc8 > bin) mas o p-valor não cruza 0,05 —
o encoder genérico já parece extrair sinal contextual suficiente para que a
decomposição da classe negativa adicione menos. Δ F1-min em Gen 2 (+0,008 e
+0,016) é menor que o Δ médio do Gen 1 (+0,025 no campeão), consistente com a
hipótese de saturação parcial. **Conclusão sobre mc8 fica em suspenso até os
4 modelos restantes**: se `xlmr-base` e `bertimbau-large` mostrarem padrão
similar a `bertimbau-base` (n.s.) e os Domain (`deb3rta-base`) a `finbert-ptbr`
(significativo), abre-se a leitura de que decomposição beneficia mais
encoders já especializados.

## McNemar vs campeão Gen 1

Cada Gen 2 vs `linearsvc + tfidf-aggr-mc8` (auto-selecionado por maior F1-min
entre runs Gen 1 `*-aggr-*`, conforme `scripts/gen2_report.py`).

| Modelo Gen 2 | Variant | Δ F1-min | b (camp.) | c (gen2) | p-valor | Vencedor |
|---|---|---:|---:|---:|---:|:---|
| `bertimbau-base` | bin | +0,0219 | 307 | 389 | 2,1 × 10⁻³ | **gen2** |
| `bertimbau-base` | mc8 | +0,0295 | 302 | 411 | 5,2 × 10⁻⁵ | **gen2** |
| `finbert-ptbr`   | bin | +0,0119 | 326 | 373 | 8,2 × 10⁻² | gen2 (n.s.) |
| `finbert-ptbr`   | mc8 | +0,0279 | 305 | 409 | 1,2 × 10⁻⁴ | **gen2** |

**Leitura.** 3 das 4 variantes batem o campeão Gen 1 com significância
estatística. A exceção é `finbert-ptbr` em modo binary (p = 0,082) — a
direção é favorável ao Gen 2 mas não cruza 0,05; em mc8 o mesmo modelo já
ganha com folga (p ≈ 10⁻⁴). Isso reforça que **a decomposição mc8 é o
ingrediente que torna o ganho BERT vs Gen 1 estatisticamente sólido em
todos os modelos disponíveis até aqui**.

## Custo computacional

Latência ~4 s / 1 000 artigos no T4 com fp16 vs. **16 ms / 1 000 artigos** do
campeão Gen 1 (TF-IDF + LinearSVC + mc8) em CPU — ratio ≈ **250×**. Para a
narrativa Green-AI do paper: o ganho de +2,9 pp de F1-min no melhor caso
(`bertimbau-base` mc8 vs campeão Gen 1) custa duas ordens e meia de
magnitude em latência e move o problema de CPU comum para GPU. Tabela
completa de custo será reordenada quando os 4 modelos pendentes
(especialmente `distilbertimbau`, do bucket Efficiency) chegarem.

A comparação cross-geração de eficiência usa **somente latência**
(ms / 1 000 artigos), métrica medida de forma homogênea em Gen 1, Gen 2 e
Gen 3 sobre o mesmo cenário (inferência sobre o val). VRAM não é
reportada cross-geração — Gen 1 é CPU-only e instrumentar RAM
retroativamente foi descartado (PA-003 em `pontos_de_atencao.md`).

## Limitações conhecidas

- **Threshold no limite inferior da grade.** `bertimbau-base` bin congelou
  em `0,05`, o piso da grade `[0,05, 0,95]` usada em todo o experimento
  (Gen 1 + Gen 2). O ótimo real do val pode estar abaixo — encoders
  fine-tunados costumam concentrar scores perto de 0/1 e a classe
  minoritária (≈ 12,3 %) empurra o threshold ótimo para baixo. **Decisão:**
  manter a grade fixa para preservar a comparabilidade pareada entre
  modelos e gerações; tratar como limitação documentada no paper, não como
  ação corretiva. Detalhes e justificativa em
  [`pontos_de_atencao.md` § PA-001](pontos_de_atencao.md#pa-001--grade-de-threshold-fixa-em-005-095).
- **`finbert-ptbr` bin com p = 0,082.** Direção favorável ao Gen 2 mas não
  cruza 0,05. Não atacar via grade de threshold (PA-001); avaliar se
  ablação de `epochs` ou `max_length` muda o quadro depois que o registry
  estiver completo.

## Pendências do registry

| Slug | HF id | Bucket | Papel esperado na narrativa |
|---|---|---|---|
| `distilbertimbau`  | `adalbertojunior/distilbert-portuguese-cased` | Efficiency  | Trade-off F1 × latência: meio-caminho entre Gen 1 e BERTimbau-base |
| `bertimbau-large`  | `neuralmind/bert-large-portuguese-cased`      | PT-BR       | Limite superior de capacidade dentro da família; baseline de "investimento de parâmetros" |
| `xlmr-base`        | `FacebookAI/xlm-roberta-base`                 | Multilingual | Pré-treino multilíngue ajuda em PT-BR sem dado específico? |
| `deb3rta-base`     | `higopires/DeB3RTa-base`                      | Domain      | Encoder Domain mais recente — segunda observação do bucket, completa o pareamento com `finbert-ptbr` |

A tabela do registry está em `src/ptbr_market/gen2_bert.py:72-110`.

## Reprodução

```bash
# 1. Bootstrap local: gerar splits OOT (já cacheados em artifacts/splits/).
uv run python scripts/eda_splits.py   # se splits ainda não existirem

# 2. Treinar cada modelo Gen 2 nos dois modos (bin + mc8).
#    Em Colab T4/L4; localmente só faz dispatch.
for slug in bertimbau-base finbert-ptbr; do
  uv run python scripts/run_gen2.py --model $slug --target-mode binary
  uv run python scripts/run_gen2.py --model $slug --target-mode multiclass \
    --collapse-scheme top7_plus_other
done

# 3. Integrar artefatos vindos do Colab (ZIP em ~/Downloads/).
unzip -n ~/Downloads/artifacts-*.zip 'artifacts/runs/*__gen2__*' \
  -x 'artifacts/tmp/*' 'artifacts/splits/*'

# 4. Agregar e rodar McNemar (auto-seleciona campeão Gen 1).
uv run python scripts/gen2_report.py
# → artifacts/reports/{gen2_summary,gen2_paired,gen2_intra_mcnemar,gen2_vs_gen1_champion}.csv
```

## Próximos passos

1. **Aguardar os 4 modelos pendentes** (bin + mc8 cada → 8 runs adicionais),
   na mesma grade de threshold `[0,05, 0,95]` (ver PA-001). Quando chegarem,
   rerrodar `gen2_report.py` (incremental — apenas adiciona linhas) e
   reescrever este relatório removendo a marca "parcial".
2. **Não congelar campeão Gen 2 ainda.** Com 2/6 modelos a tabela está
   incompleta — `bertimbau-large` e `deb3rta-base` podem deslocar o teto.
3. **Adiar ensemble Gen 1 + Gen 2** (uma Soft Voting + uma Stacking, conforme
   `plano_base.md`) para depois de fechar o registry. LLMs (Gen 3) seguem
   excluídos do ensemble por decisão do plano.
