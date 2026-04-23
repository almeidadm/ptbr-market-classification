# Relatório Gen 3 — LLMs zero-shot via Ollama (parcial)

> **Status: parcial — 1 run concluído de ≥ 12 planejados** (6 LLMs × 2 variantes
> {bin, mc8}). Único run disponível: `llama3.1-8b` zs-v1-bin, com comportamento
> degenerado no threshold estático (ver § Resultados). Relatório será reescrito
> conforme novos runs chegarem.

## Origem dos artefatos

| Pacote (Colab) | Coleta (UTC) | Commit do run | Runs trazidos |
|---|---|---|---|
| `artifacts-20260423T101514Z-3-001.zip` | 2026-04-23 10:15 | `90c3619` | 1 (`llama3.1-8b` × bin) |

O ZIP também continha três artefatos descartados na extração — **não foram
trazidos para `artifacts/runs/`** para não poluir o registry:

- `20260423-024047__gen3__llama3.1-8b__zs-v1-bin` — `predictions.csv` de
  29 bytes (apenas cabeçalho) e sem `metadata.json`. Falha antes da primeira
  amostra.
- `20260423-025543__gen3__llama3.1-8b__zs-v1-bin` — probe de 100 amostras
  (F1-min = 0, PR-AUC ≈ 0,86 sobre n = 100). Útil na depuração do pipeline,
  ruído para o sumário de 16 627 amostras.
- `20260423-082548__gen3__llama3.1-8b__zs-v1-mc8/` — diretório vazio no ZIP
  (sem arquivos internos). Sugere falha de empacotamento no Colab; o run mc8
  do `llama3.1-8b` precisa ser reexecutado.

## Objetivo

Replicar a narrativa de Gen 2 no regime zero-shot: um LLM generalista
(sem fine-tuning, só prompt) atinge o teto de Gen 2 (`bertimbau-base` mc8,
F1-min = **0,8830** / PR-AUC = **0,9464**) na mesma janela Out-of-Time de
test? Replicar a pergunta-mãe mc8 vs bin é **impossível nesta rodada**
porque o único run disponível é binário.

## Setup

- **Splits idênticos aos de Gen 1 / Gen 2.** Mesmas janelas: train
  2015-01-01 … 2017-01-07 (n = 133 725); val 2017-01-08 … 2017-05-23
  (n = 16 701); test 2017-05-24 … 2017-10-01 (n = 16 627, positivos = 2 177).
- **Pré-processamento `raw`.** `mask_entities = false`, `text_max_chars =
  4000`. Sem máscara de entidades (é decisão exclusiva de Gen 1).
- **Modelo e inferência.** Ollama tag `llama3.1:8b-instruct-q4_K_M`,
  `num_ctx = 2048`, `num_predict = 1`, `temperature = 0.0`, `seed = 1`,
  `top_logprobs = 20`. Bucket `global-native` (Ollama oficial, sem importação
  via GGUF).
- **Prompt versionado.** `v1`, hash `73c158511918fc39`, template em
  `prompts/gen3_v1_bin.txt` (conteúdo não duplicado aqui para manter o hash
  como única fonte de verdade).
- **Scoring: `logprobs_with_hard_fallback`.** Softmax restrito aos tokens-alvo
  dentro do top-20 logprobs do primeiro token (`mercado` vs `outros`).
  Fallback hard label (`y_score ∈ {0,0; 1,0}`) quando o token-alvo positivo
  não aparece no top-20. Dos 16 627 exemplos, **16 328 foram scored via
  logprobs** e **299 caíram em `parse_failure`** (1,8 %) — nenhum hard
  fallback propriamente dito. Consequências sobre PR-AUC em
  [`pontos_de_atencao.md` § PA-004](pontos_de_atencao.md#pa-004--comparabilidade-cross-geração-latência-val-vs-test-e-pr-auc-restrito-a-logprobs-gen-3).
- **Threshold: não aplicável.** `threshold.applicable = false`. Diferente de
  Gen 1 e Gen 2, o score do Gen 3 não é calibrado em grade no val — o
  corte usado para materializar `y_pred` é **0,5 estático**, decisão da
  arquitetura de scoring em `src/ptbr_market/gen3_llm.py`, não do relatório.
- **Métricas medidas sobre `test`.** Não sobre `val` (PA-004, item 1) —
  a assimetria `test` vs `val` da latência de Gen 3 foi aceita como
  diferença incidental.

## Resultados (test split, n = 16 627) — parcial

| Modelo | Variant | F1-min | PR-AUC | ROC-AUC | Precisão-min | Recall-min | Lat. (ms/1k) | VRAM pico | Parse fail. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama3.1-8b` | zs-v1-bin | 0,0000 | 0,6782 | 0,9052 | 0,0000 | 0,0000 | 1 160 911 | 5 371 MB | 299 |

Matriz de confusão: `[[14450, 0], [2177, 0]]` — o modelo prediz `outros` para
**todos os 16 627 exemplos** no threshold 0,5, incluindo os 2 177 positivos
verdadeiros.

## Leitura

- **F1-min = 0 é artefato do threshold, não do ranqueamento.** A ROC-AUC de
  **0,9052** mostra que o score softmax restrito separa positivos e
  negativos de forma comparável aos encoders base-size de Gen 2 — o sinal
  discriminativo existe no ranking. A PR-AUC de 0,6782 já é o regime em que o
  Llama fica 27 pp abaixo do campeão Gen 2 (0,9464). Mas nenhum exemplo
  cruza `y_score > 0,5` no regime atual, então o classificador hard entrega
  recall 0 para a classe minoritária.
- **Origem da degeneração.** O Llama 3.1 8B em q4_K_M, na fronteira
  `mercado` vs `outros` deste corpus, tende a concentrar a maior massa do
  primeiro token no lado negativo mesmo quando a semântica do artigo é
  claramente financeira. A renormalização softmax sobre 2 tokens no top-20
  ainda acaba abaixo de 0,5 na maioria dos casos positivos — o cut de 0,5
  é o errado, não o modelo.
- **Consequência prática.** Comparação justa com Gen 1/Gen 2 exige calibrar o
  threshold do Gen 3 (entra como pendência). No regime atual, qualquer
  McNemar contra os campeões se resume a "Gen 3 está dizendo sempre outros"
  — o teste confirma a direção mas a magnitude de `b` (1833 / 1963 em Gen 1 /
  Gen 2) é dominada pelos 2 177 positivos verdadeiros perdidos.
- **Latência ~1,16 s por amostra no T4.** Inclui RTT HTTP local + fila do
  daemon Ollama + cold start diluído (PA-004, item 1). Ordens de grandeza
  acima de Gen 2 (≈ 4 ms/amostra no campeão) e Gen 1 (≈ 0,016 ms/amostra no
  campeão). VRAM pico 5,4 GB — dentro do orçamento T4 (16 GB) com folga
  grande, compatível com `q4_K_M`.

## McNemar vs campeão Gen 1

`linearsvc + tfidf-aggr-mc8` (F1-min 0,8536).

| Modelo Gen 3 | Variant | Δ F1-min | b (gen1) | c (gen3) | p-valor | Vencedor |
|---|---|---:|---:|---:|---:|:---|
| `llama3.1-8b` | bin | −0,8536 | 1 833 | 285 | 1,0 × 10⁻²⁴⁷ | **gen1** |

## McNemar vs campeão Gen 2

`bertimbau-base + raw-ml256-mc8` (F1-min 0,8830).

| Modelo Gen 3 | Variant | Δ F1-min | b (gen2) | c (gen3) | p-valor | Vencedor |
|---|---|---:|---:|---:|---:|:---|
| `llama3.1-8b` | bin | −0,8830 | 1 963 | 306 | 8,2 × 10⁻²⁶⁵ | **gen2** |

`b` (champion acerta e gen3 erra) é dominado pelos 2 177 positivos
verdadeiros que Gen 3 atribuiu a `outros`; `c` (gen3 acerta e champion erra)
vem dos falsos positivos do campeão sobre `outros` (≈ 300 em ambos).
Resultado estatisticamente significativo por uma diferença enorme, mas
**o teste mede o threshold, não o modelo** — ver § Leitura.

## Custo computacional

`llama3.1-8b` zs-v1-bin a **1 160 911 ms / 1 000 artigos** no Colab T4
(q4_K_M, num_ctx 2048) vs **4 111 ms / 1 000 artigos** do campeão Gen 2 e
**16 ms / 1 000 artigos** do campeão Gen 1. Ratios aproximados:
**282× Gen 2** e **73 000× Gen 1** — ainda antes de entrar nos modelos de 14B
do bucket global-native. A comparação cross-geração usa somente latência
(PA-003) e carrega a assimetria `test` (Gen 3) vs `val` (Gen 1/Gen 2) como
diferença incidental documentada (PA-004, item 1).

## Limitações conhecidas

- **Threshold estático 0,5 no único run disponível.** F1-min = 0 apesar de
  ROC-AUC ≈ 0,91 — um sinal forte de que o corte precisa de ablação. Decisão
  metodológica a tomar antes de fechar o relatório final:
  - (a) aplicar a grade `[0,05, 0,95]` sobre o `y_score` softmax restrito e
    calibrar no val (paralelo a Gen 1/Gen 2 — quebra a nota
    `threshold.applicable = false` do pipeline);
  - (b) reportar Gen 3 apenas via hard-label puro (sem softmax, sem grade),
    abrindo mão de PR-AUC para Gen 3 e simetria de colunas no paper.
  Decisão fica em aberto até o segundo run completo.
- **PR-AUC de Gen 3 com otimismo estrutural.** Softmax restrito aos
  tokens-alvo no top-20 descarta massa fora dos alvos e renormaliza; em mc8
  a distorção é maior que em bin. [PA-004 item 2](pontos_de_atencao.md#pa-004--comparabilidade-cross-geração-latência-val-vs-test-e-pr-auc-restrito-a-logprobs-gen-3).
- **Latência medida sobre `test`**, não `val` — diferente de Gen 1/Gen 2.
  Magnitude da assimetria é pequena (partições de tamanho próximo, mesmo
  regime de textos), mas inclui RTT Ollama + warm-up que Gen 1/Gen 2 não
  carregam. PA-004 item 1.
- **299 `parse_failure` em 16 627 (1,8 %).** Tratados como `y_score = 0` na
  fallback, não afetaram latência nem métricas materialmente. Possivelmente
  ligados a textos que truncam antes da fronteira de resposta.
- **Sem pareamento bin↔mc8 disponível.** A pasta mc8 do `llama3.1-8b` veio
  vazia no ZIP; sem isso, `gen3_intra_mcnemar.csv` é empty e a pergunta
  "decomposição mc8 ajuda o LLM?" fica em aberto até o próximo run.

## Pendências do registry

| Slug | Ollama tag / origem | Bucket | Variantes faltantes |
|---|---|---|:---:|
| `llama3.1-8b`  | `llama3.1:8b-instruct-q4_K_M`  | global-native | mc8 |
| `qwen2.5-7b`   | `qwen2.5:7b-instruct-q4_K_M`   | global-native | bin + mc8 |
| `qwen2.5-14b`  | `qwen2.5:14b-instruct-q4_K_M`  | global-native | bin + mc8 |
| `gemma2-9b`    | `gemma2:9b-instruct-q4_K_M`    | global-native | bin + mc8 |
| `tucano`       | GGUF via Modelfile (commit `12f179a` fechou a conversão FP16→Q4_K_M local) | national | bin + mc8 |
| `bode`         | GGUF via Modelfile                                             | national | bin + mc8 |

Tabela alinhada ao `plano_base.md` § Gen 3. Registry e Modelfiles vivem em
`src/ptbr_market/gen3_llm.py`.

## Reprodução

```bash
# 1. Bootstrap local: gerar splits OOT (já cacheados em artifacts/splits/).
uv run python scripts/eda_splits.py   # se splits ainda não existirem

# 2. Rodar o LLM zero-shot nos dois modos (bin + mc8).
#    Em Colab T4/L4 com Ollama daemon rodando; localmente só faz dispatch.
#    Warmup explícito antes do loop (commit 90c3619) absorve cold-start.
uv run python scripts/run_gen3.py --model llama3.1-8b --target-mode binary \
  --prompt-version v1 --num-ctx 2048 --top-logprobs 20
uv run python scripts/run_gen3.py --model llama3.1-8b --target-mode multiclass \
  --collapse-scheme top7_plus_other --prompt-version v1 --num-ctx 2048 \
  --top-logprobs 20

# 3. Integrar artefatos vindos do Colab (ZIP em ~/Downloads/).
#    Include-list explícita evita runs abortados / probes / pastas vazias.
unzip -n ~/Downloads/artifacts-20260423T101514Z-3-001.zip \
  'artifacts/runs/20260423-030348__gen3__llama3.1-8b__zs-v1-bin/*'

# 4. Agregar e rodar McNemar (auto-seleciona campeões Gen 1 e Gen 2).
uv run python scripts/gen3_report.py
# → artifacts/reports/{gen3_summary,gen3_paired,gen3_intra_mcnemar,
#                      gen3_vs_gen1_champion,gen3_vs_gen2_champion}.csv
```

## Próximos passos

1. **Destravar a decisão de threshold para Gen 3** (opção a vs b em
   Limitações). A F1-min = 0 com ROC-AUC = 0,91 torna essa decisão a mais
   urgente do relatório — sem ela, qualquer run adicional do Gen 3 entrega
   o mesmo padrão degenerado nas tabelas.
2. **Reexecutar `llama3.1-8b` × mc8** — o ZIP trouxe pasta vazia, provável
   falha de empacotamento; o treinamento em si não chegou a ser confirmado
   como concluído no Colab.
3. **Ordem sugerida dos próximos modelos:** `qwen2.5-7b` → `gemma2-9b` →
   `qwen2.5-14b` → `tucano` → `bode`. Prioriza global-native antes dos
   importados via GGUF (tucano/bode via Modelfile, pipeline mais frágil);
   tamanho crescente dentro de cada grupo para falhar rápido se 7B já
   degenerar.
4. **Gen 3 permanece excluído dos ensembles** (decisão do `plano_base.md`):
   o ensemble Gen 1 + Gen 2 (Soft Voting + Stacking) segue a timeline
   acordada em `docs/relatorio_gen2.md` § Próximos passos.
