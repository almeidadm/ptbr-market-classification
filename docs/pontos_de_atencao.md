# Pontos de atenção metodológicos

Limitações conhecidas, decisões em aberto e armadilhas que afetam a leitura
dos relatórios. Cada ponto é registrado aqui *uma vez* — relatórios
específicos (`relatorio_gen1.md`, `relatorio_gen2.md`, …) referenciam este
documento em vez de duplicar o texto. A ordem é cronológica de descoberta,
não de severidade.

---

## PA-001 — Grade de threshold fixa em `[0,05, 0,95]`

**Descoberto em:** Gen 2, 2026-04-22, ao integrar
`artifacts-20260422T234816Z-3-001.zip`.

**Onde no código:** `src/ptbr_market/threshold.py:23-25`
(`GRID_START = 0.05`, `GRID_END = 0.95`, `GRID_N = 91` → passo 0,01).

**Observação.** Os 13 runs Gen 1 e os 4 runs Gen 2 disponíveis usam a mesma
grade. O `bertimbau-base` em modo binary (`raw-ml256-bin`) congelou em
**0,05 — exatamente o piso da grade.** O ótimo real do val pode estar abaixo
disso: encoders fine-tunados costumam produzir distribuições de score
concentradas perto de 0/1, e a classe minoritária (≈ 12,3 %) empurra o
threshold ótimo para baixo.

**Por que NÃO ampliar a grade no meio do experimento.**

1. **Comparação intra-Gen 2 fica enviesada.** McNemar pareado bin↔mc8 e
   modelo↔modelo só faz sentido se os dois lados foram calibrados na mesma
   superfície de hiperparâmetros. Se `distilbertimbau` puder explorar
   `[0,01, 0,04]` e `bertimbau-base` (já feito) não, qualquer Δ vira ruído
   de calibração.
2. **Comparação Gen 2 vs Gen 1 também trava.** O campeão Gen 1
   (`linearsvc + tfidf-aggr-mc8`) já está congelado na grade atual; um Gen 2
   com piso menor ganharia headroom de calibração que o Gen 1 não teve
   oportunidade de explorar.
3. **Re-tuning offline não é viável.** `predictions.csv` guarda scores do
   **test**, não do val (ver `gen1_classical.py` / `gen2_bert.py`).
   Para retunar com grade nova é preciso re-inferir no val — em Gen 2 isso
   exige os checkpoints, que **não vêm nos ZIPs do Colab** (só métricas e
   predições). Re-tuning vira "retreinar do zero", o que custa Colab.

**Decisão (2026-04-22, confirmada pelo usuário).** **Manter a grade
`[0,05, 0,95]` para todo o experimento (Gen 1, Gen 2 já feitos, Gen 2
pendentes, Gen 3).** O ganho qualitativo Gen 2 sobre Gen 1 já é claro
mesmo com `bertimbau-base bin` pinçado no piso (3 das 4 variantes batem o
campeão Gen 1 com p < 0,05; ver `relatorio_gen2.md`). Mudar a grade no meio
do experimento polui o log metodológico — exatamente o tipo de
inconsistência que `CLAUDE.md` cita como pitfall do projeto antigo.

**Como mencionar no paper.** Tratar como limitação documentada na seção de
metodologia: "thresholds calibrados em grade densa `[0,05, 0,95]` passo
0,01; um modelo (`bertimbau-base bin`) atingiu o limite inferior, indicando
que a calibração ótima poderia estar abaixo. A grade foi mantida fixa para
todo o experimento de modo a preservar a comparabilidade pareada
(McNemar) entre modelos e gerações." Não tratar como falha do modelo nem
como ponto a "consertar em trabalho futuro" — é uma escolha de design
metodológico assumida.

**Quando revisitar.** Se em algum momento for necessário **retreinar tudo
do zero** (ex.: trocar splits, mudar `max_length`, novo corpus), aí sim
considerar grade `[0,01, 0,95]` desde o início. Antes disso, não.

---

## PA-002 — `min_df=5` e `max_df=0,95` hardcoded em TF-IDF/BoW (Gen 1)

**Descoberto em:** Gen 1, 2026-04-22, na auditoria metodológica que seguiu
o registro da PA-001.

**Onde no código:** `src/ptbr_market/representations.py:34-48`
(`TFIDF_DEFAULTS`, `BOW_DEFAULTS`).

**Observação.** As duas representações cortam tokens em `< 5` documentos
(`min_df = 5`) e tokens em `> 95 %` dos documentos (`max_df = 0,95`). O
`relatorio_gen1.md` cita "1–2 gramas, `sublinear_tf`" mas omite essas duas
constantes. Em corpus financeiro de notícias, **tokens raros tendem a ser
discriminativos** — nomes próprios de empresas, tickers, ações específicas
("Americanas", "Petrobras", "JBS"). `min_df = 5` em 133 725 documentos de
train representa frequência mínima de ~3,7 × 10⁻⁵, critério agressivo de
descarte que pode estar cortando justamente o sinal mais específico da
classe minoritária `mercado`.

**Por que NÃO mudar agora.**

1. **Mesma lógica da PA-001.** Toda a matriz Gen 1 (13 runs × 3
   representações × 5 modelos) já está calibrada com essas constantes.
   Mudá-las refaz vocabulário → refaz TF-IDF/BoW → refit dos
   classificadores → refit do threshold → invalida o McNemar pareado.
2. **As constantes parecem ter vindo do projeto antigo
   (`economy-classifier`)** sem revalidação para FolhaSP. `CLAUDE.md`
   alerta explicitamente para não portar código — esse é o tipo de
   portabilidade silenciosa que o aviso cobre.
3. **Sem ablação prévia, não há base empírica para escolher valores
   melhores.** Trocar para "intuitivamente melhor" (ex.: `min_df = 2`) é
   tão arbitrário quanto manter o atual, mas com custo de retrain.

**Decisão (2026-04-22, confirmada pelo usuário).** Manter `min_df = 5` e
`max_df = 0,95` como estão. Documentar no paper que foram herdadas como
default e não ablacionadas — entra na seção de limitações. Mesma lógica
da PA-001: preservar comparabilidade pareada vale mais que tentar otimizar
constantes no meio do experimento.

**Como mencionar no paper.** Adicionar à descrição de TF-IDF/BoW:
"`min_df = 5` (tokens presentes em ≥ 5 documentos do train), `max_df =
0,95` (excluindo tokens em > 95 % dos documentos)". Em limitações:
"constantes herdadas do trabalho preliminar e fixadas a priori; em corpus
financeiro tokens raros (entidades nomeadas) podem carregar sinal
discriminativo, portanto o resultado Gen 1 deve ser interpretado como
lower bound do potencial real da família TF-IDF/BoW".

**Quando revisitar.** Se for preciso retreinar Gen 1 do zero (gatilhos da
PA-001) ou se sobrar budget para ablação dedicada (sweep `min_df ∈
{2, 5, 10} × max_df ∈ {0,90; 0,95; 0,99}` no modelo campeão — 9 runs em
CPU, viável em algumas horas locais). Antes disso, não.

---

## PA-003 — `vram_peak_mb = 0` em Gen 1 *(REJEITADA — 2026-04-22)*

**Status:** rejeitada após avaliação. **Não será corrigida.** Mantida no
documento como registro histórico para evitar re-descoberta em auditorias
futuras. **Ação prática:** comparações de eficiência entre gerações no
paper usam **apenas latência (`latency_ms_per_1k`)**.

**Resumo do que se observou.** Todos os 13 runs Gen 1 reportam
`vram_peak_mb = 0` no `metadata.json`, hardcoded em
`src/ptbr_market/gen1_pipeline.py:196`. Tecnicamente correto (Gen 1 roda
em CPU, não usa VRAM), mas a coluna existe nas tabelas de eficiência
porque Gen 2 (`gen2_bert.py:409`) e Gen 3 (`gen3_llm.py:740`) a preenchem
com valores reais.

**Decisão (2026-04-22, confirmada pelo usuário).** **Não reexecutar Gen 1
para instrumentar RAM.** O esforço de re-rodar os 13 runs Gen 1 não se
justifica para fechar essa simetria; em vez disso, o eixo de eficiência
do paper é tratado **só por latência** (ms / 1 000 artigos), que é
medida em todas as três gerações no mesmo cenário (inferência sobre o val)
e portanto comparável de forma honesta sem ambiguidade de unidade.

**Como mencionar no paper.** Tabela de eficiência cross-geração contém
uma única coluna de custo: **latência em ms / 1 000 artigos**. A coluna
VRAM aparece apenas em comparações **intra-Gen 2** e **intra-Gen 3**
(onde todos os modelos rodam em GPU e a métrica é homogênea), nunca
cruzando para Gen 1.

**Quando revisitar.** Não revisitar. Se um revisor pedir explicitamente
custo de memória cross-geração na rodada de revisão, retomar a discussão
da opção A (instrumentar `psutil` RSS) ainda é viável — mas não é prazo
proativo deste projeto.

---

## PA-004 — Comparabilidade cross-geração: latência (`val` vs `test`) e PR-AUC restrito a logprobs (Gen 3)

**Descoberto em:** Gen 3, 2026-04-22, na auditoria de prontidão de
execução (antes do primeiro run real do Ollama).

**Onde no código.**

- Latência:
  - Gen 1 mede sobre **`val`**: `src/ptbr_market/gen1_pipeline.py:166`
    (passada extra de `predict_proba` cronometrada).
  - Gen 2 mede sobre **`val`**: `src/ptbr_market/gen2_bert.py:362`
    (passada extra com `time.perf_counter`, inclui tokenização).
  - Gen 3 mede sobre **`test`**: `src/ptbr_market/gen3_llm.py:401-407,
    759-764` — soma de `time.perf_counter()` por requisição **dentro da
    própria inferência de avaliação**, inclui RTT HTTP local + fila do
    daemon Ollama.
- Score contínuo Gen 3:
  - `src/ptbr_market/gen3_llm.py:280-330` (`extract_score_from_logprobs`)
    aplica `softmax` sobre os logprobs do top-20 do primeiro token,
    restrito aos rótulos da classe (`mercado` vs negativos).
  - Fallback hard label em `gen3_llm.py:425-431` quando o token-alvo
    positivo não aparece no top-20: `y_score ∈ {0,0; 1,0}`.

**Observação — duas distorções que afetam a tabela cross-geração.**

1. **Latência: `test` vs `val`.** As duas partições têm tamanhos próximos
   (16 627 vs 16 701, diferença ≈ 0,4 %) e são adjacentes na linha do
   tempo, então a distribuição de comprimentos de texto é praticamente a
   mesma. A diferença real fica em **dois efeitos secundários** do Gen 3
   medir durante a própria inferência: (a) tempo inclui RTT HTTP local
   + fila do daemon Ollama, ausente em Gen 1/Gen 2; (b) primeira
   requisição paga warm-up de carga de pesos do modelo (~10–30 s diluídos
   em 16 k amostras → +1–2 ms por amostra na média). Magnitudes pequenas
   diante do total (segundos/amostra em Gen 3, ms em Gen 1, ms-100 em
   Gen 2), mas não nulas.
2. **PR-AUC de Gen 3 não é estritamente comparável aos PR-AUCs de Gen 1/
   Gen 2.** O score `y_score` do Gen 3 é um softmax **restrito aos
   tokens-alvo no top-20** — em `bin` (2 alvos) a renormalização é
   inócua; em `mc8` (8 alvos) ela **descarta toda a massa probabilística
   fora dos 8 tokens** e renormaliza, produzindo score sistematicamente
   otimista. Quando o token positivo nem aparece no top-20, o fallback
   bimodal `{0,0; 1,0}` cria platôs na curva PR. **No regime
   desbalanceado de 12,3 % positivos esse otimismo desloca a curva PR
   para cima sem refletir ganho discriminativo real.**

**Por que NÃO mudar agora.**

1. **Mesma filosofia de PA-001/PA-002.** Reescrever o scoring de Gen 3
   (ex.: somar logprob residual via `1 − Σ exp(lp_alvo)` para
   aproximar `P(positivo)` sem renormalização) invalida tudo que for
   rodado depois e cria inconsistência se algum modelo já tiver sido
   executado com a versão atual.
2. **Reinstrumentar latência em Gen 3 sobre `val`** custa uma passada
   extra de inferência por modelo (~horas adicionais por modelo no T4),
   o que num orçamento já apertado de Colab (estimativa 30–55 h totais
   para os 6 modelos × 2 variantes) não se justifica para fechar uma
   diferença de poucos pontos percentuais na latência.
3. **Alternativas em PR-AUC têm trade-offs.** Opção A (somar logprob
   residual): mais defensável estatisticamente mas exige re-rodar todos
   os modelos. Opção B (reportar só F1-min para Gen 3 e omitir PR-AUC):
   honesto mas quebra a simetria de colunas com Gen 1/Gen 2 no paper.
   Manter PR-AUC com ressalva é a opção que preserva a tabela.

**Decisão (2026-04-22, confirmada pelo usuário).**

- **Latência cross-geração**: aceitar a assimetria `test` (Gen 3) vs
  `val` (Gen 1/Gen 2) como diferença incidental. Documentar no paper.
- **PR-AUC de Gen 3**: reportar como está, mas anotar no paper que o
  score vem de softmax restrito ao top-20 logprobs e que em `mc8` é
  otimista por construção. **F1-min é a métrica primária para Gen 3 nas
  comparações pareadas com Gen 1/Gen 2** (McNemar já opera sobre `y_pred`
  binário, então não é afetado pelo viés de PR-AUC).

**Como mencionar no paper.**

- Tabela cross-geração: nota de rodapé única —
  > "Latência reportada em ms / 1 000 artigos sobre o respectivo conjunto
  > de avaliação (val para Gen 1 e Gen 2; test para Gen 3 — diferença
  > incidental, mesma ordem temporal e tamanho equivalente). PR-AUC de
  > Gen 3 é derivado de softmax restrito aos tokens-alvo no top-20
  > logprobs do Ollama; em modo `mc8` esse score é sistematicamente
  > otimista e não deve ser comparado diretamente ao PR-AUC de Gen 1/
  > Gen 2. F1-min e o teste de McNemar pareado (sobre decisões duras)
  > são as métricas primárias para Gen 3."

**Quando revisitar.** Se um revisor pedir explicitamente PR-AUC
comparável cross-geração, considerar a opção A (logprob residual via
`1 − Σ exp(lp_alvo)`) ou abandonar PR-AUC para Gen 3 no rebuttal. Não
revisitar antes da submissão.
