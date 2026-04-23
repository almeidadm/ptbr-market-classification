# Trabalhos Relacionados — Referência Expandida

Este documento expande a seção de trabalhos relacionados do artigo, organizando as referências em oito blocos temáticos que acompanham a estrutura da pipeline (Gen 1 clássica, Gen 2 encoders BERT, Gen 3 LLMs, ensembles Gen 1+2) e a metodologia de avaliação sob desbalanceamento severo (classe positiva "mercado" em ~12,6% dos ~166k artigos do corpus FolhaUOL, com split Out-of-Time 80/10/10 por data de publicação, limiar ajustado em validação e congelado em teste, e reporte conjunto de PR-AUC, F1 da classe minoritária, McNemar pareado e eficiência em ms/1k + VRAM).

Para cada obra citada há (i) uma descrição técnica curta do que ela propõe e (ii) uma declaração explícita de como ela se conecta à nossa pipeline ou protocolo, evitando resumos genéricos. Entradas cujos metadados ainda precisam de confirmação bibliográfica estão marcadas com asterisco.

---

## 1. Classificação sobre FolhaUOL e datasets próximos

Esta seção ancora o trabalho no corpus-alvo (FolhaUOL) e em conjuntos de texto financeiro/jornalístico usados como referência metodológica na literatura de classificação binária e multirrótulo sob desbalanceamento.

### Alcoforado et al., 2022 — ZeroBERTo: Leveraging zero-shot text classification by topic modeling

**O que propõe:** ZeroBERTo é um classificador zero-shot de texto em português que combina uma etapa não-supervisionada de modelagem de tópicos (para gerar rótulos-candidatos e representações compactas) com um classificador NLI pré-treinado que faz o julgamento final por entailment. O método elimina a necessidade de fine-tuning em cada nova taxonomia e foi avaliado, entre outros corpora, sobre o FolhaSP em classificação multirrótulo de seções editoriais, com protocolo de k-fold e F1 ponderado.

**Relevância para este trabalho:** ZeroBERTo é o contraste metodológico mais direto: usa o mesmo corpus FolhaUOL mas sob k-fold (não OOT), F1 ponderado (não F1 da classe minoritária), e resolve zero-shot via NLI — enquanto nossa Gen 3 resolve zero-shot via prompting de LLMs generativos pinados em temperatura 0.0. Citamos ZeroBERTo como ponto de partida e como justificativa para endurecer o protocolo (OOT 80/10/10 por data + PR-AUC + threshold frozen), já que o split aleatório mascara vazamento temporal de vocabulário de evento.

### Malo et al., 2014 — Good debt or bad debt: Detecting semantic orientations in economic texts (Financial PhraseBank)

**O que propõe:** Os autores constroem o Financial PhraseBank, um corpus de sentenças de notícias financeiras anotadas em três polaridades (positivo/neutro/negativo) por múltiplos anotadores com diferentes níveis de concordância, e apresentam uma linha de base léxica especializada em terminologia financeira. O recurso é desde então o benchmark canônico para classificação de sentimento financeiro em inglês.

**Relevância para este trabalho:** Financial PhraseBank não é alvo direto (não avaliamos sobre ele — nosso escopo é FolhaUOL exclusivo), mas é a referência que motiva a existência de FinBERT, FinBERT-PT-BR e toda a família de encoders financeiros que usamos em Gen 2; também ajuda a delimitar que "sentimento financeiro" e "classificação de seção financeira vs. outras" são tarefas parentes mas distintas.

### Maia et al., 2018 — WWW'18 Open Challenge: Financial Opinion Mining and Question Answering (FiQA)

**O que propõe:** FiQA define duas tarefas abertas sobre textos financeiros: classificação de sentimento em nível de aspecto (tweets e headlines) e question answering sobre corpus financeiro. Fornece splits oficiais, anotações de alvo/aspecto e um conjunto de headlines curtas que se tornaram base para avaliação de encoders financeiros.

**Relevância para este trabalho:** FiQA é citado como evidência de que a literatura de PLN financeiro consolidou benchmarks em inglês mas não produziu equivalentes em pt-BR com o mesmo nível de rigor — o que justifica contribuirmos com um protocolo rigoroso (OOT + PR-AUC + McNemar) sobre FolhaUOL como passo nesse sentido, sem pretender substituir FiQA.

### Lewis, 1997; Zhang et al., 2015; Lang, 1995\* — Reuters-21578, AG News e 20 Newsgroups

**O que propõem:** Os três corpora são benchmarks clássicos de classificação de texto: Reuters-21578 (notícias da Reuters anotadas em múltiplas categorias econômicas, com splits ModApte amplamente usados), AG News (notícias em quatro categorias balanceadas, popular como benchmark de CNNs de texto) e 20 Newsgroups (mensagens de fóruns Usenet em 20 tópicos, referência histórica para Naive Bayes e representações BoW). Os três compartilham a característica de ter distribuição de classe aproximadamente balanceada ou moderadamente desbalanceada e avaliação com accuracy/F1 macro.

**Relevância para este trabalho:** Servem apenas como ancoragem na literatura — demarcamos explicitamente que FolhaUOL difere desses benchmarks em três eixos críticos (português, desbalanceamento severo ~12,6% de positivos, e temporalidade que inviabiliza splits aleatórios), e que, por isso, herdar deles métricas como accuracy ou ROC-AUC seria metodologicamente errado em nosso caso.

\*Metadados a verificar: o ano e a forma canônica de citação variam (Reuters-21578 é frequentemente citado via Lewis 1997 ou via ApteMod 1994; AG News como Zhang, Zhao & LeCun 2015; 20 Newsgroups como Lang 1995). Confirmar edição exata antes do envio.

---

## 2. Modelos fundacionais Transformer e NLI

Base teórica sobre a qual toda Gen 2 (encoders BERT) e Gen 3 (LLMs) se apoiam, além dos recursos NLI que sustentam paradigmas zero-shot como o usado por ZeroBERTo.

### Vaswani et al., 2017 — Attention is all you need

**O que propõe:** Introduz a arquitetura Transformer, substituindo recorrência e convolução por mecanismos de auto-atenção multi-cabeça e codificação posicional, com arquitetura encoder-decoder aplicada inicialmente a tradução automática. O trabalho estabelece os blocos (multi-head attention, feed-forward, layer norm, residuais) que viriam a ser a base de BERT, GPT, Llama e toda a geração subsequente.

**Relevância para este trabalho:** É a referência arquitetural única que subjaz tanto os encoders da Gen 2 (BERTimbau, Albertina, XLM-R, DistilBERTimbau, FinBERT-PT-BR, DeB3RTa) quanto os decoders da Gen 3 (Llama 3.1, Qwen 2.5, Gemma 2, Tucano, Bode); toda comparação entre gerações neste artigo é, no fundo, uma comparação entre uso encoder-only vs. decoder-only do mesmo princípio de auto-atenção.

### Devlin et al., 2019 — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**O que propõe:** BERT introduz o paradigma de pré-treino bidirecional mascarado (Masked LM + Next Sentence Prediction) seguido de fine-tuning específico de tarefa, usando um Transformer encoder. O modelo estabeleceu que representações contextuais profundas transferíveis são obtidas eficientemente a partir de grandes corpora não-anotados.

**Relevância para este trabalho:** É o ancestral direto de todos os encoders que fine-tunamos em Gen 2; nosso protocolo de fine-tuning (classificação sequencial com `[CLS]`, LR pequeno, poucas épocas) segue a receita canônica BERT, e nossa comparação Gen 2 vs. Gen 3 mede exatamente o trade-off entre "fine-tune um BERT" e "fazer zero-shot com um LLM" em pt-BR.

### Conneau et al., 2020 — Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)

**O que propõe:** XLM-RoBERTa estende o pré-treino RoBERTa (MLM sem NSP, batches maiores, corpora maiores) para 100 línguas simultaneamente usando CommonCrawl, mostrando que escala de dados multilíngues supera modelos especializados em várias línguas de baixo recurso. Apresenta as variantes base e large.

**Relevância para este trabalho:** XLM-R base e large entram diretamente como backbones Gen 2 na categoria "multilingual giants"; são o baseline que testa se um modelo multilíngue treinado em escala massiva bate encoders dedicados ao português (BERTimbau, Albertina) em tarefa financeira brasileira — questão empírica que nossa tabela resolve.

### Conneau et al., 2018; Williams et al., 2018 — XNLI e MultiNLI

**O que propõem:** MultiNLI (Williams et al., 2018) é um corpus de ~433k pares de sentenças em inglês anotados em entailment/neutral/contradiction, cobrindo múltiplos gêneros textuais, e é o treinamento-padrão para cabeças NLI. XNLI (Conneau et al., 2018) estende o conjunto de desenvolvimento/teste do MultiNLI para 15 línguas via tradução profissional, criando o benchmark canônico para NLI cross-lingual e para zero-shot classification via entailment.

**Relevância para este trabalho:** Não fine-tunamos cabeças NLI em nossa pipeline, mas citamos ambos porque são a base do paradigma zero-shot-via-NLI usado por ZeroBERTo — com o qual nos contrastamos em zero-shot via LLM generativo; mencionar XNLI também explicita por que avaliar zero-shot em pt-BR é possível sem corpus NLI próprio.

### Sanh et al., 2019 — DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter

**O que propõe:** DistilBERT aplica distillation de conhecimento (perda tripla: MLM, alinhamento de embeddings via cosseno, e destilação soft-label da distribuição do mestre) ao BERT, produzindo um modelo com cerca de 40% menos parâmetros e ~60% mais rápido em inferência, retendo a maior parte da qualidade em downstream tasks. Introduz a receita de distilação que foi depois replicada para variantes multilíngues e nacionais.

**Relevância para este trabalho:** Justifica a inclusão de DistilBERTimbau como entrada dedicada na faixa "Efficiency" da Gen 2; o eixo Green-AI do artigo (ms/1k articles e VRAM em cada tabela) depende de ter pelo menos um encoder destilado para mostrar o trade-off capacidade-vs-custo dentro da própria geração BERT.

---

## 3. Modelos pt-BR e pt-PT

Encoders e LLMs focados em português, usados como backbones de Gen 2 (BERTimbau, Albertina, Ulysses, DistilBERTimbau) ou como LLMs de Gen 3 (Tucano, Bode).

### Souza et al., 2020 — BERTimbau: Pretrained BERT Models for Brazilian Portuguese

**O que propõe:** BERTimbau é um BERT pré-treinado do zero sobre o brWaC (corpus web brasileiro), disponibilizado em duas variantes (base e large) seguindo arquitetura e vocabulário WordPiece dedicados a pt-BR. Foi avaliado em NER, STS e RTE brasileiros, estabelecendo baseline forte para tarefas downstream em português brasileiro.

**Relevância para este trabalho:** BERTimbau base e large são os backbones centrais da Gen 2 na categoria "PT-BR Baselines"; o fine-tuning binário sobre FolhaUOL com essas duas variantes é a espinha dorsal da comparação entre encoders pt-BR nativos e alternativas multilíngues/financeiras.

### Rodrigues et al., 2023\* — Advancing Neural Encoding of Portuguese with Transformer Albertina PT-\*

**O que propõe:** Albertina é uma família de encoders pt-PT/pt-BR baseada em DeBERTa-v2, com a variante 900M sendo o modelo pt-BR de maior capacidade aberto no momento de sua publicação, pré-treinado sobre corpora amplos de português. A proposta enfatiza tratamento explícito das duas variantes do idioma e melhoria sobre baselines multilíngues em benchmarks portugueses.

**Relevância para este trabalho:** Albertina 900M entra como backbone Gen 2 na mesma categoria "PT-BR Baselines" de BERTimbau, servindo de teto de capacidade entre os encoders pt-BR; o contraste BERTimbau-large vs. Albertina-900M mede se escala adicional traz ganho real sobre texto jornalístico-financeiro brasileiro ou satura.

\*Metadados a verificar: ano exato (2022 vs. 2023), venue (preprint arXiv ou conferência PROPOR/JURIX) e citação canônica dos autores.

### Ulysses RoBERTa\*

**O que propõe:** Ulysses RoBERTa é um encoder RoBERTa pré-treinado em corpus legislativo-governamental brasileiro (produção do Senado/Câmara), projetado para tarefas de PLN jurídico-legislativo em português brasileiro. Diferencia-se de BERTimbau pelo viés de domínio do corpus de pré-treino, inclinado a textos formais de governo.

**Relevância para este trabalho:** Entra como backbone Gen 2 representando um encoder pt-BR de domínio deslocado do jornalístico-financeiro, permitindo medir o efeito de "domain mismatch": quanto um modelo pré-treinado em corpus legislativo perde (ou ganha) ao ser fine-tunado para classificar notícias de economia na Folha.

\*Metadados a verificar: autores exatos, ano e referência canônica (o modelo circula principalmente como model card no HuggingFace; confirmar existência de paper de acompanhamento).

### Souza et al., 2020 (model card DistilBERTimbau, baseado em Sanh et al., 2019) — DistilBERTimbau

**O que propõe:** DistilBERTimbau é uma versão destilada de BERTimbau produzida pela equipe NeuralMind/Neuralmind-BR, aplicando a receita DistilBERT de Sanh et al. (2019) ao modelo mestre BERTimbau-base, resultando em um encoder pt-BR menor e mais rápido. O modelo preserva o vocabulário e tokenizador do BERTimbau, diferindo apenas em número de camadas e custo de inferência.

**Relevância para este trabalho:** Preenche explicitamente o slot "Efficiency" da Gen 2; junto com DistilBERT genérico, materializa o eixo Green-AI da comparação — cada tabela da Gen 2 reporta ms/1000 artigos e VRAM, e DistilBERTimbau é o ponto mais baixo desse eixo entre encoders pt-BR nativos.

### Garcia et al., 2024\* — Bode: A Portuguese LLM fine-tuned on Alpaca-based instructions

**O que propõe:** Bode é um LLM em português obtido por fine-tuning por instrução (estilo Alpaca) sobre um modelo Llama em versões de 7B e 13B parâmetros, usando traduções pt-BR de instruções e respostas. O objetivo é ter um LLM generativo instruído nativo em português brasileiro.

**Relevância para este trabalho:** Bode é avaliado em Gen 3 em modo zero-shot (mesmo prompt binário, temperatura 0.0, JSON estruturado) via Ollama a partir de um `.GGUF` + Modelfile; serve para testar se um LLM "adaptado a pt-BR por instrução" supera LLMs multilíngues maiores (Llama 3.1 8B, Qwen 2.5 14B) na tarefa mercado-vs-outros.

\*Metadados a verificar: ano exato de publicação (2023 vs. 2024), autores completos e venue/arXiv ID.

### Corrêa et al., 2024\* — Tucano: A Decoder-only Language Model for Brazilian Portuguese

**O que propõe:** Tucano é um LLM decoder-only pré-treinado do zero sobre corpus extenso de português brasileiro, posicionando-se como alternativa nativa aos LLMs multilíngues adaptados por instrução. Enfatiza transparência de dados de pré-treino e disponibilização aberta de checkpoints.

**Relevância para este trabalho:** Entra em Gen 3 (Ollama via `.GGUF` + Modelfile) como contraste direto com Bode: Tucano é pt-BR "nativo desde o pré-treino" vs. Bode é "Llama instruído em pt-BR", e medir ambos sob o mesmo prompt binário e mesmo protocolo OOT isola o efeito de origem do modelo sobre a tarefa.

\*Metadados a verificar: autor líder, ano exato e venue — se preprint arXiv 2024 ou publicação formal.

---

## 4. Modelos de domínio financeiro

Encoders especializados em texto financeiro, incluídos em Gen 2 para testar se especialização de domínio supera especialização de idioma.

### Araci, 2019 — FinBERT: Financial Sentiment Analysis with Pre-trained Language Models

**O que propõe:** FinBERT parte de um BERT base e continua o pré-treino (further pre-training) sobre corpus financeiro em inglês (Reuters TRC2 financial + outros), seguido de fine-tuning em Financial PhraseBank para sentimento de três classes. É uma das primeiras demonstrações de que domain-adaptive pre-training supera BERT genérico em PLN financeiro.

**Relevância para este trabalho:** Entra em Gen 2 (ProsusAI/finbert) como baseline "domain-specialist em inglês"; avalia-se se um encoder financeiro em inglês aplicado a notícias pt-BR via subword/cross-lingual ainda carrega sinal útil — hipótese geralmente negativa, mas necessária de testar para fechar a comparação com FinBERT-PT-BR e DeB3RTa.

### FinBERT-PT-BR\*

**O que propõe:** FinBERT-PT-BR é um encoder BERT adaptado ao domínio financeiro em português brasileiro, tipicamente obtido por further pre-training de BERTimbau sobre notícias financeiras pt-BR e/ou fine-tuning sobre datasets de sentimento financeiro traduzidos. Circula principalmente como model card no HuggingFace, com descrição de corpus e receita de treino.

**Relevância para este trabalho:** É o encoder mais alinhado ao nosso alvo (pt-BR + financeiro) e, portanto, a aposta a priori mais forte da Gen 2; o contraste FinBERT-PT-BR vs. BERTimbau-base sobre FolhaUOL responde se especialização financeira paga o custo de perder cobertura de domínios gerais presentes em notícias "outras". Note que o checkpoint original do FinBERT-PT-BR traz cabeça de 3 classes de sentimento e `problem_type=multi_label_classification` no `config.json`, o que exige `ignore_mismatched_sizes=True` e `problem_type="single_label_classification"` explícitos ao carregar para classificação binária — detalhe já documentado em `src/ptbr_market/gen2_bert.py`.

\*Metadados a verificar: existência de paper formal (não apenas model card), autores, ano e venue. Se for apenas model card, citar explicitamente como tal.

### DeB3RTa\*

**O que propõe:** DeB3RTa é um encoder pt-BR baseado em DeBERTa (provavelmente DeBERTa-v2 ou v3), seguindo a inovação central do DeBERTa original de disentangled attention (posição e conteúdo tratados em matrizes separadas) e enhanced mask decoder. A variante pt-BR busca unir os ganhos arquiteturais DeBERTa com cobertura lexical do português brasileiro.

**Relevância para este trabalho:** Entra em Gen 2 testando se um backbone DeBERTa-like supera RoBERTa/BERT-like (BERTimbau, Ulysses) em pt-BR para a tarefa binária — é o único representante de arquitetura "pós-RoBERTa" entre os encoders pt-BR da nossa comparação.

\*Metadados a verificar: autores, ano, venue canônico e se está posicionado como modelo financeiro ou apenas pt-BR genérico. Confirmar antes de classificá-lo como "domain-specialist".

---

## 5. LLMs generalistas avaliados

Os três LLMs abertos mais robustos rodados via Ollama em Gen 3, mais o bloco teórico que fundamenta o paradigma zero-shot/few-shot com modelos generativos instruídos.

### Dubey et al., 2024 — The Llama 3 Herd of Models

**O que propõe:** Relatório técnico da família Llama 3/3.1 (8B, 70B, 405B parâmetros), detalhando o pré-treino em ~15T tokens multilíngues, o pós-treino com SFT + DPO + rejection sampling, extensão de contexto a 128k, e suporte multilíngue incluindo português. O 8B é posicionado como o modelo "produção-eficiente" da família.

**Relevância para este trabalho:** Llama 3.1 8B é um dos quatro LLMs generalistas da Gen 3, rodado via Ollama nativo com temperatura 0.0 e saída JSON; representa o estado-da-arte open-weights ocidental de menor custo que ainda cabe confortavelmente em uma GPU de pesquisa e serve de âncora de referência para os demais (Qwen, Gemma, Bode, Tucano).

### Qwen Team, 2024 — Qwen2.5 Technical Report

**O que propõe:** Relatório técnico do Qwen 2.5 da Alibaba, família de LLMs decoder-only com variantes de 0.5B a 72B parâmetros, pré-treino em corpus multilíngue amplo com foco em qualidade de dados, e pós-treino em SFT + DPO. Destaca suporte multilíngue forte, contexto longo e variantes especializadas (Coder, Math).

**Relevância para este trabalho:** Avaliamos Qwen 2.5 em duas capacidades (7B e 14B) na Gen 3, também via Ollama; a comparação 7B-vs-14B dentro da mesma família, sob o mesmo prompt binário e protocolo OOT, é o controle que mede o efeito de escala dentro de Gen 3 — dimensão que Gen 1 e Gen 2 não exploram da mesma forma.

### Gemma Team, 2024 — Gemma 2: Improving Open Language Models at a Practical Size

**O que propõe:** Gemma 2 é a segunda geração de LLMs abertos do Google, com variantes de 2B, 9B e 27B parâmetros, pré-treino com distillation (aluno-mestre entre tamanhos) e arquitetura decoder-only com soft-capping de logits e atenção local/global alternada. Foca em alta qualidade a tamanhos práticos de produção.

**Relevância para este trabalho:** Gemma 2 9B é o quarto LLM generalista da Gen 3 rodado via Ollama; ao lado de Llama 3.1 8B e Qwen 2.5 7B, forma a banda "~8-9B parâmetros" que permite comparação justa de custo entre três famílias de origens diferentes (Meta, Alibaba, Google) sobre a mesma tarefa pt-BR.

### Brown et al., 2020; Ouyang et al., 2022; Wei et al., 2022 — Base teórica de zero-shot/few-shot e instruction tuning

**O que propõem:** Brown et al. (2020, "Language Models are Few-Shot Learners") introduz GPT-3 e demonstra que LLMs de grande escala adquirem capacidade de aprender tarefas via prompting sem atualização de parâmetros (in-context learning, few-shot e zero-shot). Ouyang et al. (2022, "Training language models to follow instructions with human feedback", InstructGPT) mostra que RLHF sobre um LLM base produz um modelo que segue instruções de forma mais útil e alinhada. Wei et al. (2022, "Finetuned Language Models Are Zero-Shot Learners", FLAN) estabelece que instruction tuning em múltiplas tarefas melhora generalização zero-shot para tarefas não vistas.

**Relevância para este trabalho:** Os três trabalhos, juntos, justificam teoricamente por que é legítimo avaliar Gen 3 em modo puramente zero-shot (prompt binário + temperatura 0.0 + JSON estruturado, sem fine-tuning) — Brown mostra que a capacidade emerge com escala, Ouyang e Wei mostram que o alinhamento por instrução a torna utilizável sem exemplos. Citamos os três em bloco para fundamentar a decisão de não fazer few-shot nem fine-tuning dos LLMs.

---

## 6. LLMs para finanças e zero-shot financeiro

Evidência de literatura sobre como LLMs se comportam em tarefas financeiras, relevante para contextualizar resultados de Gen 3 mesmo que não avaliemos esses modelos diretamente.

### Wu et al., 2023 — BloombergGPT: A Large Language Model for Finance

**O que propõe:** BloombergGPT é um LLM de 50B parâmetros pré-treinado sobre um corpus misto de ~363B tokens, aproximadamente metade texto financeiro proprietário da Bloomberg e metade texto geral (The Pile, C4, Wikipedia). O modelo é avaliado em benchmarks financeiros (FPB, FiQA, NER financeiro) e gerais, mostrando ganho robusto em tarefas de domínio sem colapsar em tarefas gerais.

**Relevância para este trabalho:** Não avaliamos BloombergGPT (é proprietário, não rodável via Ollama) mas citamos como referência do trade-off entre LLM de domínio treinado do zero vs. LLMs genéricos grandes em zero-shot; é contraste importante para argumentar que nossa Gen 3 é deliberadamente construída só com modelos abertos de ~7-14B acessíveis a laboratórios acadêmicos.

### Yang et al., 2023 — FinGPT: Open-Source Financial Large Language Models

**O que propõe:** FinGPT é um framework aberto de LLMs financeiros baseado em fine-tuning por instrução (LoRA/QLoRA) de modelos base abertos sobre dados financeiros coletados (notícias, relatórios, redes sociais). Foca em democratizar acesso, atualização incremental e transparência, oposto à opção fechada do BloombergGPT.

**Relevância para este trabalho:** Citamos FinGPT como alternativa metodológica considerada e não adotada — deliberadamente não fazemos fine-tuning dos LLMs da Gen 3 em nossa pipeline, porque o escopo do artigo é medir zero-shot de LLMs genéricos contra Gen 1/Gen 2 fine-tunadas, e introduzir fine-tuning financeiro misturaria variáveis.

### Xie et al., 2023 — The Wall Street Neophyte: A Zero-Shot Analysis of ChatGPT Over MultiModal Stock Movement Prediction Challenges

**O que propõe:** Xie et al. avaliam ChatGPT em zero-shot e few-shot em tarefas de previsão de movimento de ações sobre múltiplos corpora financeiros multimodais, medindo se o LLM consegue extrair sinal preditivo de notícias e tweets financeiros sem fine-tuning. O achado central é que zero-shot de LLM generalista performa de forma modesta e frequentemente abaixo de baselines tradicionais em tarefas financeiras reais.

**Relevância para este trabalho:** Ajusta expectativas para nossa Gen 3: é literatura que documenta que LLMs genéricos em zero-shot financeiro tendem a não dominar baselines fine-tunados; se nossos resultados reproduzirem esse padrão sobre FolhaUOL (Gen 2 > Gen 3 em F1-minority mesmo com custo muito menor da Gen 1), Xie et al. é a referência para contextualizar.

### Li et al., 2023\* — Large Language Models in Finance / LLMs for Financial Sentiment

**O que propõe:** Estudo avalia LLMs (família GPT, LLaMA) em tarefas de sentimento financeiro em modo zero-shot e few-shot sobre benchmarks como Financial PhraseBank e FiQA, comparando-os a encoders financeiros fine-tunados como FinBERT. Caracteriza sistematicamente onde LLMs ajudam e onde encoders especializados permanecem superiores.

**Relevância para este trabalho:** Serve de espelho para nossa comparação Gen 2 (FinBERT-PT-BR, DeB3RTa, BERTimbau) vs. Gen 3 (Llama 3.1, Qwen 2.5, Gemma 2) sobre FolhaUOL — se replicarmos o padrão "encoders fine-tunados ainda vencem em F1 mas LLMs vencem em ergonomia de implantação", citamos Li et al. como evidência convergente em inglês.

\*Metadados a verificar: autor exato, título canônico, ano e venue — há múltiplos trabalhos de 2023 com primeiro autor Li sobre LLMs em finanças. Confirmar qual é o referido antes do envio.

---

## 7. Classificadores clássicos e representações (Gen 1)

Fundamentos da Gen 1: representações (TF-IDF, word2vec, fastText), classificadores (LinearSVC, Naive Bayes, gradient boosting, DJINN, TextCNN) e modelagem de tópicos de apoio (BERTopic).

### Salton & Buckley, 1988 — Term-weighting approaches in automatic text retrieval (TF-IDF)

**O que propõe:** Trabalho clássico que formaliza e compara esquemas de ponderação termo-frequência × frequência-inversa-de-documento em recuperação de informação, estabelecendo as variantes canônicas (logarítmica, normalizada, suavizada) de TF e IDF. Define o vocabulário que até hoje é usado em baselines de classificação de texto.

**Relevância para este trabalho:** TF-IDF é a representação principal da Gen 1, combinada com LinearSVC, Logistic Regression, Complement/Multinomial NB e LightGBM/XGBoost; é também o ponto de comparação mais barato em ms/1k artigos contra qualquer encoder Gen 2, central para o eixo Green-AI.

### Mikolov et al., 2013 — Efficient Estimation of Word Representations in Vector Space (word2vec)

**O que propõe:** Introduz as arquiteturas CBOW e Skip-gram que treinam embeddings densos de palavras como tarefa auxiliar de predição contextual, com truques de eficiência (hierarchical softmax, negative sampling) que permitem treinar em bilhões de tokens. Estabelece que embeddings densos capturam regularidades sintáticas e semânticas por operações vetoriais.

**Relevância para este trabalho:** word2vec é uma das três famílias de representações da Gen 1 (ao lado de BoW/TF-IDF e fastText), usada por média de embeddings sobre o documento antes de alimentar LinearSVC/LogReg; é o baseline distribucional "pré-Transformer" que a comparação Gen 1 vs. Gen 2 precisa incluir para ser honesta.

### Bojanowski et al., 2017; Joulin et al., 2017 — Enriching Word Vectors with Subword Information; Bag of Tricks for Efficient Text Classification (fastText)

**O que propõem:** Bojanowski et al. (2017) estende word2vec adicionando representações em nível de n-gramas de caracteres, o que permite obter embeddings robustos para palavras raras e morfologicamente produtivas — especialmente valioso em línguas flexionadas. Joulin et al. (2017) descreve o classificador fastText, que combina embeddings médios e classificação linear treinada end-to-end, competitivo com CNNs a uma fração do custo.

**Relevância para este trabalho:** A combinação justifica duas entradas distintas na Gen 1 — embeddings fastText pré-treinados em pt-BR como input para classificadores lineares e, separadamente, o classificador fastText nativo — ambas relevantes em português, onde morfologia rica faz subword information pagar-se.

### Fan et al., 2008 — LIBLINEAR: A Library for Large Linear Classification

**O que propõe:** LIBLINEAR implementa solvers eficientes (coordinate descent dual e primal, trust region Newton) para SVMs lineares e regressão logística em dados esparsos de alta dimensão, tornando prático treinar esses classificadores sobre matrizes TF-IDF de milhões de documentos e features. É o backend de `sklearn.svm.LinearSVC` e de parte do `LogisticRegression`.

**Relevância para este trabalho:** É a biblioteca por trás de LinearSVC e LogisticRegression na Gen 1; citá-la é obrigatório para reprodutibilidade porque os resultados da Gen 1 sobre matriz TF-IDF (166k × ~50k features) dependem criticamente dos solvers LIBLINEAR que o scikit-learn expõe.

### McCallum & Nigam, 1998 — A Comparison of Event Models for Naive Bayes Text Classification (Multinomial NB)

**O que propõe:** Compara os modelos de evento multi-variate Bernoulli e multinomial para Naive Bayes em texto, concluindo que o multinomial (que modela contagens de termos como eventos independentes de uma multinomial) é consistentemente superior em corpora de tamanho realista. Estabelece o Multinomial NB como baseline canônico de classificação de texto.

**Relevância para este trabalho:** Multinomial NB entra em Gen 1 como baseline clássico; em desbalanceamento severo (12% positivos) é conhecido por degradar especialmente, o que torna interessante a comparação direta com sua variante Complement.

### Rennie et al., 2003 — Tackling the Poor Assumptions of Naive Bayes Text Classifiers (Complement NB)

**O que propõe:** Rennie et al. diagnosticam patologias do Multinomial NB em dados desbalanceados (bias para a classe majoritária) e propõem Complement NB, que estima parâmetros a partir do complemento da classe em vez da própria classe, corrigindo o viés. Adicionam também normalizações (peso-normalização) para estabilizar estimativas.

**Relevância para este trabalho:** Em nosso setting com ~12,6% de positivos, Complement NB é a variante NB que tem chance real de bater Multinomial NB; ambos são reportados em Gen 1 justamente para ilustrar o efeito de desbalanceamento sobre classificadores generativos, eixo central do artigo.

### Ke et al., 2017 — LightGBM: A Highly Efficient Gradient Boosting Decision Tree

**O que propõe:** LightGBM introduz Gradient-based One-Side Sampling (GOSS) e Exclusive Feature Bundling (EFB) para acelerar o treinamento de gradient boosting decision trees, com crescimento leaf-wise (em vez de level-wise) que dá modelos mais profundos em árvores de mesma complexidade. Resulta em treinos muito mais rápidos com qualidade equivalente ou superior a XGBoost em muitos domínios.

**Relevância para este trabalho:** LightGBM é um dos dois gradient boosters da Gen 1 sobre TF-IDF (e, opcionalmente, sobre embeddings médios); é competitivo em eficiência mesmo em features esparsas de alta dimensão e entra na tabela comparativa junto com XGBoost.

### Chen & Guestrin, 2016 — XGBoost: A Scalable Tree Boosting System

**O que propõe:** XGBoost consolida gradient boosting com regularização explícita (L1/L2 em folhas), tratamento nativo de valores faltantes, split finding aproximado baseado em quantis ponderados e sistema de treino paralelizável, estabelecendo-se como baseline forte em competições de ML tabular. Introduz também a formulação de segunda ordem (gradient + hessiana) do objetivo.

**Relevância para este trabalho:** XGBoost é o segundo gradient booster da Gen 1; comparar LightGBM vs. XGBoost sobre a mesma matriz TF-IDF mede se as otimizações de cada biblioteca dão diferença prática em F1-minoritária sobre corpus de ~166k documentos, útil para a discussão de implementação da Gen 1.

### Humbird et al., 2019\* — Deep Jointly-Informed Neural Networks (DJINN)

**O que propõe:** DJINN é um método que constrói redes neurais profundas inicializadas a partir de florestas aleatórias ou ensembles de árvores — cada árvore é mapeada para uma sub-rede cuja topologia reflete splits e profundidade das árvores, e os pesos iniciais derivam das decisões das árvores. O resultado é uma rede com inicialização informada que é depois fine-tunada por backprop.

**Relevância para este trabalho:** DJINN entra na Gen 1 como baseline neural híbrido, deliberadamente não-Transformer, para cobrir o espaço entre gradient boosting (XGBoost/LightGBM) e redes neurais puras (TextCNN) na comparação Gen 1; é também uma escolha metodológica herdada do projeto precursor.

\*Metadados a verificar: ano exato (2018 vs. 2019) e venue canônico — o trabalho tem versões em arXiv e em periódico; confirmar qual citar.

### Kim, 2014 — Convolutional Neural Networks for Sentence Classification (TextCNN)

**O que propõe:** TextCNN aplica CNN 1D sobre embeddings de palavras concatenados em matriz, usando múltiplos tamanhos de filtro (tipicamente 3, 4, 5) seguidos de max-pooling global e classificador denso. Estabelece que uma arquitetura leve e barata de treinar compete com métodos muito mais elaborados em classificação de sentenças/documentos.

**Relevância para este trabalho:** TextCNN é o único representante puramente neural pré-Transformer na Gen 1 (sobre embeddings word2vec ou fastText); é a ponte conceitual entre Gen 1 "rasa" e Gen 2 "profunda", e reportá-lo evita que a comparação Gen 1 vs. Gen 2 vire apenas "linear vs. Transformer".

### Grootendorst, 2022 — BERTopic: Neural topic modeling with a class-based TF-IDF procedure

**O que propõe:** BERTopic é um pipeline de modelagem de tópicos que combina embeddings contextuais (tipicamente Sentence-Transformers), redução de dimensionalidade via UMAP, clustering via HDBSCAN e um procedimento class-based TF-IDF (c-TF-IDF) para extrair termos representativos por cluster-tópico. Oferece tópicos interpretáveis e dinâmicos em cima de embeddings modernos.

**Relevância para este trabalho:** BERTopic não é um classificador da pipeline principal mas pode aparecer como ferramenta de apoio na análise exploratória do FolhaUOL (identificar sub-tópicos dentro de "outros" que confundem os classificadores, por exemplo); citamos por transparência caso seja usado na seção de análise de erros.

---

## 8. Metodologia de avaliação sob desbalanceamento

Referências que fundamentam escolhas metodológicas não-negociáveis do protocolo: PR-AUC em vez de ROC-AUC, e McNemar como teste pareado entre modelos.

### Saito & Rehmsmeier, 2015 — The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets

**O que propõe:** Saito & Rehmsmeier mostram formalmente e empiricamente que, em classificação binária com desbalanceamento severo, a curva ROC e o ROC-AUC são enganosos porque a taxa de falso positivo é dominada pela grande massa de verdadeiros negativos, e pequenas variações absolutas em FP viram variações ínfimas na FPR. A curva precision-recall, ao contrário, responde diretamente ao trade-off entre recall e precisão na classe de interesse, tornando-a estritamente mais informativa.

**Relevância para este trabalho:** É a justificativa canônica de por que reportamos PR-AUC para a classe "mercado" como métrica principal — com ~87,4% de verdadeiros negativos em FolhaUOL, ROC-AUC inflaria todas as comparações e tornaria Gen 1, Gen 2 e Gen 3 visualmente indistinguíveis. Saito & Rehmsmeier é a referência que citamos ao justificar essa escolha no paper.

### Davis & Goadrich, 2006 — The Relationship Between Precision-Recall and ROC Curves

**O que propõe:** Davis & Goadrich estabelecem formalmente a relação entre espaço PR e espaço ROC, mostrando que existe correspondência um-a-um entre pontos (dado o número de positivos e negativos), mas que dominância em ROC não implica dominância em PR e vice-versa. Também caracterizam a interpolação correta em espaço PR (não-linear) e alertam contra interpolações ingênuas.

**Relevância para este trabalho:** Complementa Saito & Rehmsmeier ao fundamentar que PR-AUC não é simplesmente uma "versão alternativa" de ROC-AUC mas uma métrica diferente com propriedades diferentes — isso sustenta que reportar F1 + PR-AUC sem ROC-AUC não é omissão, é escolha metodologicamente defensável em nosso cenário.

### McNemar, 1947; Dietterich, 1998 — Note on the Sampling Error of the Difference Between Correlated Proportions; Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms

**O que propõem:** McNemar (1947) propõe o teste qui-quadrado sobre a tabela 2×2 de discordâncias pareadas entre dois classificadores sobre os mesmos exemplos, avaliando se a diferença entre acertos e erros discordantes é significativa. Dietterich (1998) compara empiricamente cinco testes de significância entre algoritmos de classificação supervisionada e recomenda McNemar como o teste com taxa de erro tipo I controlada quando há um único par train/test — caso em que validação cruzada não é adequada.

**Relevância para este trabalho:** Como nosso protocolo OOT 80/10/10 gera exatamente um par val/test fixo (não cross-validation, porque violaria a temporalidade), McNemar é o teste pareado apropriado para afirmar que um modelo bate outro de forma estatisticamente significativa no test set congelado; Dietterich fornece a justificativa formal dessa escolha, e ambos são citados juntos ao reportar diferenças entre Gen 1, Gen 2, Gen 3 e ensembles.

---

## Observações finais

Entradas marcadas com `*` têm pelo menos um metadado (tipicamente ano, venue ou autoria completa) que precisa de confirmação antes da submissão. São elas: Reuters/AG News/20NG (forma canônica de citação); Albertina 900M (ano e venue); Ulysses RoBERTa (existência de paper de acompanhamento); Bode (ano e arXiv ID); Tucano (autoria e venue); FinBERT-PT-BR (paper formal vs. apenas model card); DeB3RTa (posicionamento financeiro vs. genérico e venue); Li et al., 2023 (identificação canônica entre múltiplos trabalhos homônimos); DJINN (ano exato e venue).

As demais entradas têm referência canônica estabelecida (Vaswani 2017, Devlin 2019, Conneau 2020, Sanh 2019, Souza 2020, Araci 2019, Dubey 2024, Wu 2023, Yang 2023, Xie 2023, Salton & Buckley 1988, Mikolov 2013, Bojanowski 2017, Joulin 2017, Fan 2008, McCallum & Nigam 1998, Rennie 2003, Ke 2017, Chen & Guestrin 2016, Kim 2014, Grootendorst 2022, Saito & Rehmsmeier 2015, Davis & Goadrich 2006, McNemar 1947, Dietterich 1998, Alcoforado 2022, Malo 2014, Maia 2018, Brown 2020, Ouyang 2022, Wei 2022, Qwen Team 2024, Gemma Team 2024).
