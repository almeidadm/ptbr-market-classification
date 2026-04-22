# Relatório Gen 1 — Classificação binária de notícias FolhaSP

## Objetivo

Primeira rodada de baselines clássicos (Gen 1) para `mercado` vs. `outros`
sobre o corpus FolhaSP/UOL (≈ 167 k artigos, positivos ≈ 12,3 %). O
objetivo é fixar um ponto de comparação reprodutível para os BERTs (Gen
2) e LLMs (Gen 3), e avaliar se decompor a classe negativa em 8
categorias (`top7_plus_other`) ajuda a superfície binária — pergunta
aberta do `plano_base.md`.

## Setup

- **Split OOT 80/10/10 por data de publicação.** Train = 2015-01-01 …
  2017-01-07 (n = 133 725); Val = 2017-01-08 … 2017-05-23 (n = 16 701);
  Test = 2017-05-24 … 2017-10-01 (n = 16 627, positivos = 2 177).
- **Pré-processamento `aggressive`.** Lematização com SpaCy
  `pt_core_news_lg` + máscara de entidades financeiras
  (`[VALOR_MONETARIO]`, `[PERCENTUAL]`, etc.), decisão D-Gen1 do
  `plano_base.md`. Cache em `artifacts/preprocessed/{split}__aggr.parquet`.
- **Representações ajustadas apenas no train.** TF-IDF
  (1–2 gramas, `sublinear_tf`), BoW (contagens inteiras em `float32`),
  fastText-avg (média 300-d com vetores `cc.pt.300`).
- **Classificadores.** LinearSVC (calibrado sigmoid, cv=3), Logistic
  Regression (`lbfgs`), MultinomialNB, ComplementNB, LightGBM
  (`deterministic=True`, `force_col_wise=True`). XGBoost foi retirado
  da matriz depois de estourar os 15 GB do host em multiclass × TF-IDF —
  `plano_base.md` permite "LightGBM/XGBoost" como intercambiáveis; fica
  só LightGBM como representante da família GBDT.
- **`seed = 1`** em todos os pontos que aceitam `random_state`
  (decisão D3).
- **Threshold calibrado no val, congelado no test.** Grade
  `[0.05, 0.95]` passo 0.01, objetivo `f1_minority`. Fit em val →
  aplicado literalmente em test.
- **Métricas.** PR-AUC (principal, por conta do desbalanceamento), F1-min,
  precisão/recall da minoritária, ROC-AUC, F1-macro, matriz de confusão,
  latência em ms/1k artigos sobre o val.
- **Dois modos comparados em paralelo** (todos os outros hiperparâmetros
  iguais): `bin` (treino com `label` ∈ {0,1}) e `mc8` (treino com
  `category` colapsada em 8 classes `top7_plus_other`; score binário =
  `predict_proba[:, idx("mercado")]`; threshold e métricas continuam
  binárias). 13 pares para McNemar.

## Resultados pareados (test split, n = 16 627)

| Representação | Modelo         | F1-min bin | F1-min mc8 |   Δ F1 | PR-AUC bin | PR-AUC mc8 | Lat. bin | Lat. mc8 |
|---------------|----------------|-----------:|-----------:|-------:|-----------:|-----------:|---------:|---------:|
| tfidf         | linearsvc      |     0,8285 | **0,8536** | +0,0251 |     0,9176 | **0,9320** |     4,0  |    16,1  |
| tfidf         | lightgbm       |     0,7992 | **0,8334** | +0,0343 |     0,8855 | **0,9132** |    89,0  |   152,1  |
| tfidf         | logreg         |     0,7966 | **0,8303** | +0,0337 |     0,8850 | **0,9161** |     1,2  |     6,2  |
| tfidf         | multinomialnb  |     0,6245 | **0,7294** | +0,1049 |     0,7804 | **0,7859** |     2,7  |     7,4  |
| tfidf         | complementnb   |     0,7137 | **0,7460** | +0,0323 |     0,7804 | **0,7862** |     2,7  |     7,4  |
| bow           | lightgbm       |     0,7827 | **0,8197** | +0,0370 |     0,8770 | **0,9092** |     6,8  |    25,5  |
| bow           | logreg         |     0,7859 | **0,8067** | +0,0208 |     0,8587 | **0,8692** |     0,3  |     0,9  |
| bow           | linearsvc      |     0,7900 | **0,8061** | +0,0161 |     0,8581 | **0,8786** |     1,1  |     3,0  |
| bow           | multinomialnb  | **0,7217** |     0,7152 | −0,0065 |     0,6586 | **0,6889** |     0,6  |     1,0  |
| bow           | complementnb   | **0,7182** |     0,7059 | −0,0123 |     0,6546 | **0,6801** |     0,6  |     1,0  |
| fasttext      | lightgbm       |     0,7643 | **0,7806** | +0,0164 |     0,8475 | **0,8613** |     1,5  |    13,5  |
| fasttext      | linearsvc      |     0,7343 | **0,7454** | +0,0111 |     0,7880 | **0,8064** |     3,0  |     4,0  |
| fasttext      | logreg         |     0,7039 | **0,7194** | +0,0155 |     0,7448 | **0,7818** |     0,8  |     1,1  |

Latência em ms por 1 000 artigos (inferência sobre o val). F1-min é o F1
da classe positiva (`mercado`, rótulo 1).

**Melhor combinação:** `tfidf + linearsvc + mc8` — F1-min = **0,8536**,
PR-AUC = **0,9320**, 16,1 ms / 1 k artigos. Sub-line é logreg (F1 ≈ 0,83)
a 6,2 ms / 1 k; LightGBM também cruza 0,83 mas custa 10× mais em
latência. Naive Bayes fica bem atrás no TF-IDF e no BoW — consistente
com a literatura quando as features são densas demais para as hipóteses
independentes.

## McNemar pareado (binary vs mc8)

Teste χ² com correção de continuidade; `b` = binary acerta e mc8 erra,
`c` = binary erra e mc8 acerta (ver `src/ptbr_market/evaluation.py`).

| Representação | Modelo        |     b |     c |     p-valor | Vencedor |
|---------------|---------------|------:|------:|------------:|:---------|
| tfidf         | linearsvc     |  102  |  256  | 6,2 × 10⁻¹⁶ | **mc8**  |
| tfidf         | logreg        |  107  |  256  | 8,0 × 10⁻¹⁵ | **mc8**  |
| tfidf         | lightgbm      |  141  |  298  | 9,7 × 10⁻¹⁴ | **mc8**  |
| tfidf         | multinomialnb |  467  |  560  | 4,1 × 10⁻³  | **mc8**  |
| tfidf         | complementnb  |  370  |  352  | 0,53        | empate (n.s.) |
| bow           | lightgbm      |  162  |  313  | 5,9 × 10⁻¹² | **mc8**  |
| bow           | logreg        |  212  |  366  | 2,0 × 10⁻¹⁰ | **mc8**  |
| bow           | multinomialnb |  293  |  468  | 2,8 × 10⁻¹⁰ | **mc8**  |
| bow           | complementnb  |  343  |  486  | 8,1 × 10⁻⁷  | **mc8**  |
| bow           | linearsvc     |  157  |  228  | 3,6 × 10⁻⁴  | **mc8**  |
| fasttext      | lightgbm      |  156  |  285  | 1,1 × 10⁻⁹  | **mc8**  |
| fasttext      | logreg        |  140  |  235  | 1,2 × 10⁻⁶  | **mc8**  |
| fasttext      | linearsvc     |  105  |  163  | 5,0 × 10⁻⁴  | **mc8**  |

**Leitura.** 12 de 13 pares dão vitória estatisticamente significativa
(p < 0,05) para mc8. O único empate é `tfidf + complementnb` (p ≈ 0,53),
onde as duas versões acertam e erram aproximadamente as mesmas
instâncias. Notar que `bow + multinomialnb` e `bow + complementnb` têm
Δ F1-min negativo **e** McNemar pró-mc8: o threshold ótimo muda entre os
dois regimes e a troca de precisão por recall desloca o F1 — mc8 acerta
mais amostras no total mas com um equilíbrio que favorece menos o F1
específico. PR-AUC (que é threshold-independente) é uniformemente maior
em mc8 nos 13 pares, o que reforça que a decomposição melhora o
_ranking_ das probabilidades.

**Por que mc8 ajuda.** Treinar com a decomposição força o modelo a
separar `mercado` de categorias vizinhas específicas (`colunas`,
`poder`, `mundo`) em vez de despejá-las todas num "negativo" homogêneo.
Classes como `colunas` têm vocabulário financeiro parcial e
contaminavam a fronteira binária; dar-lhes lugar próprio no softmax
afasta essa massa do lado positivo.

## Reprodução

```bash
# 1. Preprocessar (train/val/test, modo aggressive; cacheado).
uv run python scripts/run_gen1.py --representation tfidf   # aquece o cache

# 2. Matriz binária (5 modelos × 3 representações = 13 runs).
for rep in tfidf bow; do
  uv run python scripts/run_gen1.py --representation $rep --target-mode binary
done
uv run python scripts/run_gen1.py --representation fasttext --target-mode binary \
  --models linearsvc,logreg,lightgbm

# 3. Matriz mc8 (mesmos modelos × representações).
for rep in tfidf bow; do
  uv run python scripts/run_gen1.py --representation $rep --target-mode multiclass \
    --collapse-scheme top7_plus_other
done
uv run python scripts/run_gen1.py --representation fasttext --target-mode multiclass \
  --collapse-scheme top7_plus_other --models linearsvc,logreg,lightgbm

# 4. Agregar e rodar McNemar.
uv run python scripts/gen1_report.py
# → artifacts/reports/{gen1_summary,gen1_paired,gen1_mcnemar}.csv

# 5. Reproduzir plots e tabelas.
uv run jupyter nbconvert --execute notebooks/gen1_report.ipynb --inplace
```

## Próximos passos

1. **Congelar Gen 1.** Aceitar `tfidf + linearsvc + mc8` como o baseline
   oficial para o paper (F1-min 0,8536, PR-AUC 0,9320). Manter o resto da
   matriz como evidência da análise de ablação.
2. **Avançar para Gen 2 (BERT).** Rodar BERTimbau base, DeB3RTa e
   FinBERT-PT-BR em Colab T4 sobre os **mesmos splits** (copiar
   `artifacts/splits/*.parquet`). Usar o mesmo protocolo de threshold
   no val e McNemar contra o campeão Gen 1.
3. **Não tentar ablação mc48.** A ganho marginal de mc8 sobre binary
   (+2,5 pp de F1-min no campeão) já é o upper-bound plausível da
   decomposição; granular demais (48 classes) dilui o sinal sem retorno
   esperado e foi explicitamente descartado no `plano_base.md`.
