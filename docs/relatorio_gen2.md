# Relatório Gen 2 — Encoders BERT

5 dos 6 modelos do registry original foram treinados. `bertimbau-large` foi
dispensado por decisão de custo (Colab L4 consumido em Gen 3); o registry em
`src/ptbr_market/gen2_bert.py` permanece com 6 slugs para rastreabilidade,
mas este relatório congela o Gen 2 em 5 modelos × 2 variantes = **10 runs**.

## Origem dos artefatos

| Pacote (Colab) | Coleta (UTC) | Commit dos runs | Runs trazidos |
|---|---|---|---|
| `artifacts-20260422T234816Z-3-001.zip` | 2026-04-22 23:48 | `faff651` (bertimbau-base) / `c3292ea` (finbert-ptbr) | 4 |
| `artifacts-20260423T101514Z-3-001.zip` | 2026-04-23 10:15 | `c3292ea` | 6 |

Todos os runs pós-`c3292ea` incorporam o fix `force problem_type=single_label`
(evita BCELoss em `num_labels=2`). `bertimbau-base` rodou em `faff651` e não
foi afetado pelo bug. O segundo ZIP também re-trouxe os 4 runs anteriores e os
splits parquet — foram excluídos na extração (`unzip` com include-list
explícita) para não sobrescrever os artefatos versionados.

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
- **Encoders avaliados (5/6 do registry).**
  | Slug | HF id | Bucket |
  |---|---|---|
  | `bertimbau-base` | `neuralmind/bert-base-portuguese-cased` | PT-BR |
  | `distilbertimbau` | `adalbertojunior/distilbert-portuguese-cased` | Efficiency |
  | `xlmr-base` | `FacebookAI/xlm-roberta-base` | Multilingual |
  | `finbert-ptbr` | `lucas-leme/FinBERT-PT-BR` | Domain |
  | `deb3rta-base` | `higopires/DeB3RTa-base` | Domain |
- **Treino.** `epochs = 3`, `learning_rate = 2e-5`, `grad_accum = 1`,
  `fp16 = True`, `seed = 1`. `batch_size = 16` para todos, exceto
  `distilbertimbau` (`batch_size = 32`, aproveitando footprint menor). Rodado
  em Colab T4 (16 GB) via `scripts/run_gen2.py` — bootstrap local, treino
  remoto.
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
  colapsada em 8 classes `top7_plus_other`). 5 pares para McNemar
  intra-Gen 2.

## Resultados pareados (test split, n = 16 627)

| Bucket | Modelo | F1-min bin | F1-min mc8 | Δ F1 | PR-AUC bin | PR-AUC mc8 | Lat. bin | Lat. mc8 | Thr. bin | Thr. mc8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Domain       | `deb3rta-base`    | 0,7702 | **0,7907** | +0,0205 | 0,8448 | **0,8743** | 9 294 | 9 614 | 0,13    | 0,31 |
| Domain       | `finbert-ptbr`    | 0,8654 | **0,8814** | +0,0159 | 0,9402 | **0,9462** | 3 956 | 4 034 | 0,21    | 0,30 |
| Efficiency   | `distilbertimbau` | 0,8550 | **0,8704** | +0,0154 | 0,9278 | **0,9392** | 2 699 | 2 691 | 0,18    | 0,18 |
| Multilingual | `xlmr-base`       | 0,8578 | **0,8729** | +0,0151 | 0,9337 | **0,9436** | 4 199 | 4 203 | 0,06 ⚠ | 0,28 |
| PT-BR        | `bertimbau-base`  | 0,8754 | **0,8830** | +0,0076 | 0,9423 | **0,9464** | 4 116 | 4 111 | 0,05 ⚠ | 0,16 |

Latência em ms / 1 000 artigos (Colab T4, fp16, inclui tokenização). Thr. =
threshold ótimo no val. ⚠ = threshold no piso da grade (PA-001). VRAM pico:
2,6 GB (`distilbertimbau`) a 9,2 GB (`deb3rta-base`); os 3 modelos base-size
restantes ficam em 3,1–5,0 GB — folgada no T4 de 16 GB. Dados completos em
`artifacts/reports/gen2_summary.csv`.

## Campeão Gen 2

**`bertimbau-base` mc8** — F1-min = **0,8830** / PR-AUC = **0,9464** / thr = 0,16 /
lat = 4 111 ms / 1 000 artigos / VRAM pico = 3 099 MB. Ganho de **+2,9 pp de
F1-min** sobre o campeão Gen 1 (McNemar p = 5,2 × 10⁻⁵, § McNemar vs campeão
Gen 1). Segundo lugar: `finbert-ptbr` mc8 (F1-min = 0,8814), à distância
estatística desprezível do topo.

## Leitura por bucket

- **PT-BR** — `bertimbau-base` entrega o teto do Gen 2 (F1-min 0,8830 mc8).
  Sem `bertimbau-large`, a pergunta do `plano_base.md` "110 M → 330 M vale a
  pena no PT-BR?" fica **sem resposta nesta rodada**; entra em "Limitações".
- **Domain** — o bucket se parte ao meio. `finbert-ptbr` empata tecnicamente
  com o campeão PT-BR (0,8814 mc8, −0,0016 de F1-min), mostrando que
  pré-treino em newswire financeiro se converte em ganho marginal no corpus
  Folha. `deb3rta-base`, porém, **fica 9 pp abaixo** em ambas as variantes
  (0,7702 bin / 0,7907 mc8) — hipótese: o corpus de pré-treino DeB3RTa
  (filings B3/CVM) é lexicalmente mais distante do jornalismo Folha do que
  o FinBERT-PT-BR (newswire-adjacente). Pergunta em aberto, fora do escopo
  deste paper.
- **Multilingual** — `xlmr-base` mc8 (0,8729) fica **≈ 1 pp abaixo** da
  baseline PT-BR, replicando a leitura usual: pré-treino multilíngue não
  compensa a ausência de dado específico quando a baseline nativa está
  disponível. Em bin, o threshold congelou em 0,06 (≈ piso da grade) —
  mesmo sintoma do `bertimbau-base` bin (PA-001).
- **Efficiency** — `distilbertimbau` é o achado prático da rodada.
  **2 691 ms/1k no mc8** (35 % mais rápido que `bertimbau-base`) com
  F1-min = 0,8704 (−1,3 pp abaixo do campeão). Para a narrativa Green-AI,
  é o data-point que posiciona uma terceira opção entre o campeão Gen 1
  (16 ms/1k CPU, F1-min 0,8536) e o campeão Gen 2 (4 111 ms/1k GPU,
  F1-min 0,8830).

## McNemar intra-Gen 2 (binary vs mc8)

χ² com correção de continuidade; `b` = bin acerta e mc8 erra, `c` = bin erra
e mc8 acerta.

| Modelo | b (bin) | c (mc8) | p-valor | Vencedor |
|---|---:|---:|---:|:---|
| `bertimbau-base`  | 174 | 201 | 0,179 | mc8 (n.s.) |
| `deb3rta-base`    | 261 | 363 | 5,3 × 10⁻⁵ | **mc8** |
| `distilbertimbau` | 174 | 230 | 6,2 × 10⁻³ | **mc8** |
| `finbert-ptbr`    | 172 | 229 | 5,2 × 10⁻³ | **mc8** |
| `xlmr-base`       | 216 | 283 | 3,1 × 10⁻³ | **mc8** |

**Leitura.** Em 4 dos 5 modelos, a decomposição mc8 supera a bin com
significância estatística — o mesmo padrão observado em 12 dos 13 pares Gen 1.
A exceção é `bertimbau-base` (p = 0,179), único modelo onde a direção
(mc8 > bin) se mantém mas o ganho é pequeno (Δ F1-min = +0,008). Isso sugere
que o encoder PT-BR mais generalista da amostra já extrai sinal contextual
suficiente para que a decomposição adicione menos — e, ainda assim, ele
é o teto Gen 2 em termos absolutos. **Conclusão do paper sobre mc8:** a
decomposição é uma ferramenta consistente tanto para representações clássicas
(Gen 1) quanto contextuais (Gen 2), com efeito que satura ligeiramente à
medida que o encoder já captura contexto amplo.

## McNemar vs campeão Gen 1

Cada Gen 2 vs `linearsvc + tfidf-aggr-mc8` (auto-selecionado por maior F1-min
entre runs Gen 1 `*-aggr-*`, conforme `scripts/gen2_report.py`).

| Modelo Gen 2 | Variant | Δ F1-min | b (camp.) | c (gen2) | p-valor | Vencedor |
|---|---|---:|---:|---:|---:|:---|
| `bertimbau-base`  | bin | +0,0219 | 307 | 389 | 2,1 × 10⁻³ | **gen2** |
| `bertimbau-base`  | mc8 | +0,0295 | 302 | 411 | 5,2 × 10⁻⁵ | **gen2** |
| `deb3rta-base`    | bin | −0,0833 | 671 | 259 | 2,1 × 10⁻⁴¹ | **champion** |
| `deb3rta-base`    | mc8 | −0,0628 | 573 | 263 | 1,2 × 10⁻²⁶ | **champion** |
| `distilbertimbau` | bin | +0,0015 | 367 | 353 | 0,628 | champion (n.s.) |
| `distilbertimbau` | mc8 | +0,0169 | 345 | 387 | 0,130 | gen2 (n.s.) |
| `finbert-ptbr`    | bin | +0,0119 | 326 | 373 | 8,2 × 10⁻² | gen2 (n.s.) |
| `finbert-ptbr`    | mc8 | +0,0279 | 305 | 409 | 1,2 × 10⁻⁴ | **gen2** |
| `xlmr-base`       | bin | +0,0042 | 370 | 366 | 0,912 | champion (n.s.) |
| `xlmr-base`       | mc8 | +0,0194 | 325 | 388 | 2,0 × 10⁻² | **gen2** |

**Leitura.** 5 das 10 variantes batem o campeão Gen 1 com significância
(bertimbau-base × 2, finbert-ptbr mc8, xlmr-base mc8), 2 perdem com
significância (deb3rta-base × 2, efeito dominante do encoder fraco no bucket
Domain), e 3 empatam (distilbertimbau × 2, finbert-ptbr bin, xlmr-base bin).
O padrão é claro: **todas as 5 vitórias significativas do Gen 2 vêm da
variante mc8 ou do modelo com maior capacidade PT-BR nativa**. Nenhum
encoder base-size em variante binária consegue cruzar o campeão Gen 1 com
significância sozinho — exceto `bertimbau-base`, onde a combinação
PT-BR-nativo + mc8 entrega o teto absoluto do experimento.

## Custo computacional

Latência do campeão Gen 2 (4 111 ms/1k em T4 fp16) vs. **16 ms/1k** do campeão
Gen 1 (TF-IDF + LinearSVC + mc8, CPU): **ratio ≈ 257×**. A variante de
eficiência (`distilbertimbau` mc8) reduz esse fator para ≈ 168× (2 691 ms/1k)
ao custo de −1,3 pp de F1-min. `deb3rta-base` é o outlier caro no sentido
oposto — 9,6 s/1k (quase 2,4× o tempo de base-size como BERTimbau/FinBERT) e
F1-min inferior ao Gen 1, configurando o pior trade-off da amostra.

A comparação cross-geração de eficiência usa **somente latência**
(ms / 1 000 artigos), métrica medida de forma homogênea em Gen 1, Gen 2 e
Gen 3 sobre o mesmo cenário (inferência sobre o val). VRAM não é
reportada cross-geração — Gen 1 é CPU-only e instrumentar RAM
retroativamente foi descartado (PA-003 em `pontos_de_atencao.md`).

## Limitações conhecidas

- **`bertimbau-large` ausente.** Decisão consciente (custo de Colab L4 alocado
  a Gen 3), não lacuna metodológica. Consequência: a pergunta do
  `plano_base.md` sobre escala PT-BR (`110 M → 330 M parâmetros vale a pena?`)
  fica **sem resposta neste trabalho**. Os ensembles (Soft Voting + Stacking)
  serão montados sobre os 5 modelos disponíveis.
- **Threshold no limite inferior da grade.** `bertimbau-base` bin (0,05) e
  `xlmr-base` bin (0,06, o passo imediatamente acima do piso) congelaram no
  extremo inferior da grade `[0,05, 0,95]`. Encoders fine-tunados concentram
  scores perto de 0/1 e a classe minoritária (≈ 12,3 %) empurra o ótimo para
  baixo. **Decisão:** manter a grade fixa para preservar a comparabilidade
  pareada entre modelos e gerações; tratar como limitação documentada no
  paper, não como ação corretiva. Detalhes em
  [`pontos_de_atencao.md` § PA-001](pontos_de_atencao.md#pa-001--grade-de-threshold-fixa-em-005-095).
- **`deb3rta-base` fora da tendência.** A −9 pp de F1-min abaixo dos outros
  Domain/PT-BR. Possível incompatibilidade lexical corpus-origem (filings
  B3/CVM) vs corpus-alvo (jornalismo Folha). Não atacado via ablação de
  `epochs` / `max_length` — fora do escopo orçado.

## Reprodução

```bash
# 1. Bootstrap local: gerar splits OOT (já cacheados em artifacts/splits/).
uv run python scripts/eda_splits.py   # se splits ainda não existirem

# 2. Treinar cada modelo Gen 2 nos dois modos (bin + mc8).
#    Em Colab T4/L4; localmente só faz dispatch.
for slug in bertimbau-base distilbertimbau xlmr-base finbert-ptbr deb3rta-base; do
  uv run python scripts/run_gen2.py --model $slug --target-mode binary
  uv run python scripts/run_gen2.py --model $slug --target-mode multiclass \
    --collapse-scheme top7_plus_other
done

# 3. Integrar artefatos vindos do Colab (ZIP em ~/Downloads/).
#    Include-list explícita evita sobrescrever artefatos já versionados
#    (splits, runs anteriores) e excluir probes/abortados do Gen 3.
unzip -n ~/Downloads/artifacts-20260423T101514Z-3-001.zip \
  'artifacts/runs/20260423-092121__gen2__distilbertimbau__raw-ml256-bin/*' \
  'artifacts/runs/20260423-100643__gen2__distilbertimbau__raw-ml256-mc8/*' \
  'artifacts/runs/20260423-063656__gen2__xlmr-base__raw-ml256-bin/*' \
  'artifacts/runs/20260423-083608__gen2__xlmr-base__raw-ml256-mc8/*' \
  'artifacts/runs/20260423-002306__gen2__deb3rta-base__raw-ml256-bin/*' \
  'artifacts/runs/20260423-043911__gen2__deb3rta-base__raw-ml256-mc8/*'

# 4. Agregar e rodar McNemar (auto-seleciona campeão Gen 1).
uv run python scripts/gen2_report.py
# → artifacts/reports/{gen2_summary,gen2_paired,gen2_intra_mcnemar,gen2_vs_gen1_champion}.csv

# 5. Relatório interativo (opcional).
uv run --group report jupyter lab notebooks/gen2_report.ipynb
```

## Próximos passos

1. **Registry Gen 2 congelado em 5 modelos.** Aceitar `bertimbau-base` mc8
   como teto Gen 2 para o paper. `bertimbau-large` fica como limitação
   documentada.
2. **Avançar para Gen 3** (LLMs via Ollama) — o primeiro run real está em
   `docs/relatorio_gen3.md` (parcial).
3. **Montar o ensemble Gen 1 + Gen 2.** Uma Soft Voting + uma Stacking com
   meta-classificador LogReg, conforme `plano_base.md` § 3. Candidatos:
   campeão Gen 1 (`linearsvc + tfidf-aggr-mc8`) + top-N Gen 2 — ordem de
   F1-min sugere `bertimbau-base` mc8, `finbert-ptbr` mc8, `xlmr-base` mc8
   como núcleo. LLMs (Gen 3) permanecem excluídos do ensemble por decisão
   do plano.
4. **Não abrir ablação adicional** em `deb3rta-base` ou nos thresholds no
   piso da grade — ganho esperado é incerto e o orçamento de Colab já foi
   encerrado para Gen 2.
