# Escopo do Artigo

## 1. O Título e o "Pitch" do Artigo

O seu artigo precisa se vender nos primeiros 30 segundos. O foco não é apenas "classificar economia", mas sim como domar um cenário caótico usando 3 gerações de IA.

- **Sugestão de Título:** _Do Clássico aos LLMs Zero-Shot: Uma Análise Evolutiva e Temporal da Classificação de Textos Financeiros no Cenário Brasileiro Sob Extremo Desbalanceamento._

- **O Argumento Central (Abstract):** Mostrar que classificar notícias financeiras no Brasil é um desafio de "agulha no palheiro" (12% de classe positiva). O artigo propõe não apenas testar a evolução das arquiteturas (SVM → BERT → Llama/Qwen), mas introduz técnicas rigorosas de mitigação de viés temporal e semântico.

## 2. A "Trindade Metodológica" (O seu grande diferencial)

A maioria dos artigos submetidos ao BRACIS peca na metodologia. O seu vai brilhar porque introduzirá três defesas arquiteturais irretocáveis:

1. **Validação Out-of-Time (80/10/10):** Mostrar que você não caiu na armadilha do `train_test_split` aleatório, evitando o _look-ahead bias_ (vazamento de dados do futuro). Isso prova que o modelo aprendeu a semântica da economia, e não apenas decorou eventos isolados.

2. **Paradigma Bimodal de Pré-processamento:** A sacada genial de usar a limpeza agressiva de Garcia (Lematização/SpaCy) para fortalecer a Geração 1, enquanto preserva a integridade sintática (texto bruto) para os mecanismos de atenção do BERT e dos LLMs.

3. **Decomposição da Classe Negativa (Multiclasse para Binário):** O seu insight de ouro. Provar que a melhor forma de isolar a classe "Economia" de uma massa ruidosa é treinando o modelo para reconhecer a topologia de todas as outras classes (esportes, política) antes de binarizar o resultado.

## 3. A Narrativa das Três Gerações (Resultados)

A seção de resultados deve parecer uma "linha do tempo" da IA, culminando no combate final:

- **Geração 1 (O Padrão-Ouro Esparso):** TF-IDF e BoW aliados a LinearSVC, Complement Naive Bayes (o antídoto do desbalanceamento) e LightGBM. Aqui você mostra que matemática pura e matrizes esparsas ainda têm força.

- **Geração 2 (A Revolução do Contexto):** O BERTimbau, mostrando o salto que o ajuste fino (_fine-tuning_) e a compreensão de subpalavras trazem para jargões do mercado.

- **Geração 3 (O Limite do Zero-Shot):** Modelos abertos via Ollama (Llama 3.1 8B, Qwen 2.5 7B e o brasileiro Tucano). O teste de fogo: um modelo generalista "lendo" uma notícia e deduzindo a classe sem nunca ter sido treinado para essa base específica.

- **O Gran Finale (Ensemble Híbrido):** O _Stacking_ unindo Geração 1 e 2. A conclusão provocativa de que um modelo híbrido leve e explicável pode bater de frente com o custo computacional de um LLM bilionário.

## 4. Avaliação Madura e "Green AI"

Revisores destroem artigos que usam apenas a Acurácia em datasets desbalanceados.

- **Métricas Principais:** Focar totalmente em F1-Score (Macro e para a classe minoritária) e PR-AUC.

- **Otimização de Limiar (Threshold):** Mostrar o ganho extra de performance ajustando a probabilidade de corte na validação.

- **Eficiência (Green AI):** Incluir uma discussão sobre latência e custo computacional. Mostrar que não basta ter o melhor F1-Score; se o Llama demora 5 segundos por notícia e o LinearSVC demora 5 milissegundos, a indústria precisa dessa informação.

# Split de Dados

A transição de um corte aleatório clássico para uma **divisão 80/10/10 com ordenação temporal (Out-of-Time Validation)** é, sem exagero, o que separa um experimento acadêmico júnior de uma pesquisa aplicada pronta para a indústria.

Para o domínio de Processamento de Linguagem Natural aplicado a **notícias**, essa abordagem metodológica é o padrão-ouro. Aqui está a discussão detalhada do porquê essa decisão eleva drasticamente o rigor científico do seu trabalho.

### 1. A Falácia da Divisão Aleatória em Notícias

Textos jornalísticos não são eventos independentes; eles são cronológicos. Se você usar um método de amostragem aleatória (como o `train_test_split` tradicional) em notícias, você fatalmente cometerá **vazamento de dados semânticos (Look-ahead Bias)**.

- **O Cenário do Vazamento:** Imagine que o evento "Quebra da Americanas" gerou 50 notícias em janeiro de 2023. Se a divisão for aleatória, 40 dessas notícias vão para o treino e 10 para o teste. O seu modelo vai decorar palavras específicas desse evento (ex: "Rial", "rombo", "PwC") no treino e, quando vir as 10 notícias no teste, acertará facilmente.
    
- **O Falso Positivo:** O seu F1-score ficará altíssimo, mas o modelo não aprendeu a identificar "notícias de mercado" pela sua estrutura gramatical; ele apenas decorou o vocabulário de um evento específico que vazou do passado para o futuro.
    

### 2. A Força da Divisão Temporal (Out-of-Time)

A validação _Out-of-Time_ obriga você a ordenar o dataset por data de publicação e fazer os cortes cronologicamente. Por exemplo:

- **Treino (Passado):** Notícias de 2015 a 2018.
    
- **Validação (Recente):** Notícias de 2019.
    
- **Teste (Futuro):** Notícias de 2020.
    

Isso simula o mundo real de forma perfeita. Quando uma corretora financeira colocar o seu modelo em produção amanhã de manhã, o modelo não saberá quais são as notícias do futuro. Ele terá que deduzir se a notícia sobre uma "nova pandemia" ou "guerra no leste europeu" afeta o mercado usando apenas a bagagem linguística que aprendeu no passado. Se o seu modelo for bem no conjunto de teste temporal, você provou a sua **capacidade de generalização e extração de conceitos**, e não apenas a memorização de eventos.

### 3. A Lógica da Proporção 80/10/10

No seu experimento anterior (64/16/20), reter 20% do dataset para teste custou a você mais de 33 mil amostras. Para um dataset tão rico quanto o da Folha (166 mil artigos), isso é um desperdício de potencial de treino.

- **O Ganho no Treino (80%):** Ao passar o treino de 64% para 80%, você entrega cerca de **133.000 artigos** para os modelos. Redes Neurais e modelos BERT são famintos por dados. Esses 16% adicionais podem conter a variância necessária para que o modelo entenda melhor as fronteiras difusas da classe "outros", melhorando a robustez geral.
    
- **O Tamanho Seguro do Teste (10%):** Mesmo reduzindo o teste para 10%, você ainda terá cerca de **16.600 artigos** para avaliação final. Em termos de estatística (como o Teste de McNemar que você utiliza), 16.600 amostras fornecem um "poder estatístico" colossal, mais do que suficiente para provar com o p-valor (< 0,0001) que o modelo A superou o modelo B.
# Técnicas e Modelos de ML
## ML Classico

### PARTE 1: Métodos de Representação de Texto (Extração de Features)

Antes de passar o dado para o classificador, você testará as três principais escolas de representação de texto pré-Deep Learning:

1. **Bag-of-Words (BoW) com Lematização:**
    
    - **Origem:** Utilizado por Garcia et al.
        
    - **Relevância no seu trabalho:** Serve como a linha de base mais rudimentar. Avalia se apenas a "contagem bruta" de um vocabulário restrito (os termos mais frequentes) é suficiente para separar "mercado" das "outras" categorias, desconsiderando completamente a ordem e a importância global das palavras.
        
2. **TF-IDF (Term Frequency-Inverse Document Frequency):**
    
    - **Origem:** Utilizado no seu trabalho IEEE.
        
    - **Relevância:** A evolução do BoW. Penaliza palavras que aparecem muito em todas as notícias (como "disse", "ano") e dá peso para palavras raras e discriminativas (como "Ibovespa", "Selic"). É o padrão-ouro das matrizes esparsas.
        
3. **Word Embeddings Estáticos (fastText / Word2Vec):**
    
    - **Origem:** Utilizado por Garcia et al.
        
    - **Relevância:** Avalia representações densas. Diferente do TF-IDF, o fastText entende que "dinheiro" e "capital" têm significados próximos. Fazer a média dos vetores fastText da notícia permite testar a semântica sem precisar de um Transformer gigantesco.
        

---

### PARTE 2: Algoritmos de Classificação Clássicos

#### A. Família Linear (Para matrizes BoW e TF-IDF)

São os modelos mais rápidos e robustos para lidar com as milhares de colunas geradas pelo TF-IDF/BoW.

1. **LinearSVC (Support Vector Machine Linear):** O melhor modelo clássico geral para texto. Busca a margem máxima de separação entre as classes.
    
2. **Regressão Logística:** Excelente por fornecer uma calibração de probabilidade (ex: 80% de chance de ser mercado), permitindo ajustar o limiar de corte para combater o desbalanceamento.
    

#### B. Família Bayesiana (Modelos Probabilísticos)

1. **Multinomial Naive Bayes (MNB):** A _baseline_ probabilística tradicional (usada no seu trabalho IEEE).
    
2. **Complement Naive Bayes (CNB):** A adição essencial para o seu novo artigo. Como o seu dataset tem um desbalanceamento severo (87% vs 12%), o CNB corrige o viés do MNB calculando a probabilidade de um texto _não_ pertencer às classes majoritárias.
    

#### C. Família de Árvores e Híbridos (Para relações não-lineares)

1. **LightGBM ou XGBoost:** Substitutos modernos para a Random Forest clássica, extremamente velozes em matrizes esparsas de texto e com capacidade nativa de ponderar a classe minoritária.
    
2. **DJINN (Deep Jointly-Informed Neural Networks):**
    
    - **Origem:** Utilizado por Garcia et al.
        
    - **Relevância:** Um modelo inovador que cria a arquitetura de uma rede neural a partir de uma árvore de decisão pré-treinada. É uma excelente adição se você quiser replicar o método avançado do Garcia na sua tarefa binária.
        

#### D. Redes Neurais Pré-Transformers (Deep Learning Clássico)

1. **TextCNN (Redes Neurais Convolucionais para Texto):**
    
    - **Origem:** Utilizado por Garcia et al. em conjunto com o fastText.
        
    - **Relevância:** Avalia se o uso de filtros convolucionais (que leem o texto em "janelas" de 3, 4 ou 5 palavras por vez) consegue capturar a estrutura do jargão financeiro melhor do que os métodos que ignoram a ordem das palavras (BoW/TF-IDF).

## BERT
### 1. Especialistas no Domínio (O Foco do Problema)

Testam a hipótese: _O pré-treinamento em vocabulário financeiro supera modelos generalistas maiores?_

- **FinBERT-PT-BR** (`lucas-leme/FinBERT-PT-BR`):
    
    - _Justificativa:_ É a _baseline_ natural para finanças no Brasil. Um BERTimbau que continuou seu pré-treinamento em corpora financeiros locais.
        
- **DeB3RTa** (Modelos recentes focados na B3/CVM):
    
    - _Justificativa:_ Representa o estado da arte atual (2025/2026) em _Transformers_ focados no mercado corporativo, contábil e financeiro brasileiro, utilizando a arquitetura DeBERTa (superior ao BERT clássico).
        
- **FinBERT Inglês** (`ProsusAI/finbert`):
    
    - _Justificativa:_ Testa a transferência de aprendizado _cross-lingual_. A pergunta científica é: o conhecimento financeiro estrutural profundo de Wall Street (em inglês) consegue ser transferido para identificar notícias da Faria Lima em português apenas durante a etapa de _fine-tuning_?
        

### 2. Baselines Monolíngues (O Padrão Ouro PT-BR)

Testam a hipótese: _Qual é o limite da compreensão semântica do português brasileiro genérico?_

- **BERTimbau Base** (`neuralmind/bert-base-portuguese-cased` - 110M parâmetros):
    
    - _Justificativa:_ O padrão absoluto da academia brasileira. Obrigatório em qualquer comparação.
        
- **BERTimbau Large** (`neuralmind/bert-large-portuguese-cased` - 330M parâmetros):
    
    - _Justificativa:_ Testa a escalabilidade dentro da mesma arquitetura e corpus. O ganho de F1 justifica triplicar os parâmetros?
        
- **Albertina PT-BR 900M** (`PORTULAN/albertina-900m-portuguese-ptbr-encoder`):
    
    - _Justificativa:_ É um dos maiores _Encoders_ já construídos especificamente para o português. Se ele for batido por um FinBERT de 110M, você prova definitivamente que "domínio > tamanho" para tarefas específicas.
        
- **Ulysses RoBERTa-PT-BR** (`Ulysses-compass/roberta-pt-br`):
    
    - _Justificativa:_ Desenvolvido pelo TC (Tribunal de Contas) da União em parceria com a UnB. Embora seja focado em documentos legais, o vocabulário legal se sobrepõe fortemente ao contábil/financeiro, além de avaliar se a remoção da tarefa NSP (_Next Sentence Prediction_, característica do RoBERTa) beneficia a classificação das notícias da Folha.
        

### 3. Os Gigantes Multilíngues Globais

Testam a hipótese: _Modelos treinados em dezenas de idiomas generalizam melhor a gramática do que os modelos focados apenas em PT-BR?_

- **mBERT** (`bert-base-multilingual-cased`):
    
    - _Justificativa:_ A âncora histórica. É o avô do BERTimbau. Serve para mostrar de onde a área partiu.
        
- **XLM-RoBERTa Base** (`xlm-roberta-base` - 270M parâmetros):
    
    - _Justificativa:_ O padrão ouro da Meta para tarefas multilíngues. Frequentemente supera o BERTimbau em testes de PT-BR devido ao tamanho absurdo do seu corpus de treinamento.
        
- **XLM-RoBERTa Large** (`xlm-roberta-large` - 550M parâmetros):
    
    - _Justificativa:_ Um modelo colossal para os padrões de Encoders. Estabelece o teto absoluto do que um modelo fundacional não-especializado consegue fazer através de _fine-tuning_.
        

### 4. O Foco em Eficiência (A Pegada de Carbono)

Testam a hipótese: _Podemos manter o F1-score acima de 0.80 gastando uma fração da capacidade de processamento?_

- **DistilBERTimbau** (`adalbertojunior/distilbert-portuguese-cased`):
    
    - _Justificativa:_ Retém 97% da compreensão de linguagem do BERTimbau, mas é 60% mais rápido e muito mais leve. Em cenários reais onde um terminal processa milhares de notícias da B3 por minuto, o _trade-off_ de perder 1 ou 2 pontos de F1-score pode ser altamente desejável para empresas. Avaliar este modelo agrega um imenso valor prático e de engenharia de software ao seu artigo.

## LLM

### 1. Modelos Globais Nativos (Instalação Imediata no Ollama)

Estes modelos estão no repositório oficial do Ollama. Eles baixam, quantizam e rodam automaticamente com um único comando de terminal (ex: `ollama run llama3.1`). São os mais indicados para o seu prazo de 1 semana.

- **Llama 3.1 (8B Instruct)**
    
    - **Tag no Ollama:** `llama3.1` (ou `llama3.1:8b`)
        
    - **Por que usar:** É a referência metodológica mundial atual para modelos compactos. Sua capacidade de generalização _zero-shot_ em português é fantástica.
        
    - **Viabilidade:** Roda com máxima fluidez e velocidade na GPU T4 (16GB) do Colab.
        
- **Qwen 2.5 (7B Instruct e 14B Instruct)**
    
    - **Tags no Ollama:** `qwen2.5:7b` e `qwen2.5:14b`
        
    - **Por que usar:** A arquitetura Qwen 2.5 tem demonstrado fluência nativa em português superior à do Llama em muitos _benchmarks_, sendo excelente para interpretação de nuances jornalísticas.
        
    - **Viabilidade:** A versão de 7B voa na T4. A versão de 14B cabe na T4 utilizando a quantização padrão do Ollama (4-bit), mas performa melhor se você conseguir uma instância com GPU L4 (24GB).
        
- **Gemma 2 (9B Instruct)**
    
    - **Tag no Ollama:** `gemma2:9b`
        
    - **Por que usar:** Traz diversidade arquitetural para o seu artigo (um modelo Google frente aos modelos Meta e Alibaba). Possui um desempenho semântico muito forte em extração de informações.
        
    - **Viabilidade:** Excelente encaixe na VRAM da T4.
        

---

### 2. Iniciativas Nacionais (Requerem Importação Customizada)

Estes modelos agregam extremo valor científico para o contexto brasileiro do BRACIS.

**Atenção:** Como eles não estão no repositório padrão do Ollama, você precisará baixar o arquivo `.GGUF` deles diretamente do Hugging Face para o Colab e criar um `Modelfile` simples (um arquivo de texto dizendo ao Ollama como carregar os pesos).

- **Tucano (7B ou 8B)**
    
    - **Por que usar:** É um dos melhores esforços recentes de _fine-tuning_ totalmente focado no português, construído sobre arquiteturas eficientes (como Llama ou Mistral).
        
    - **Objetivo no Artigo:** Responder se a injeção profunda de cultura e gramática brasileira supera o conhecimento genérico de um modelo global do mesmo tamanho.
        
- **Bode (7B ou 13B)**
    
    - **Por que usar:** Criado pela Recogna NLP (Unesp), é um marco da pesquisa acadêmica brasileira em LLMs.
        
    - **Objetivo no Artigo:** Como é baseado na geração anterior (Llama 2), serve para mapear o salto evolutivo das arquiteturas

# Diferenciais para o Futuro

### 1. Métricas Específicas para Desbalanceamento Severo (PR-AUC)

O seu rascunho atual foca fortemente no F1-Score (o que é ótimo), mas revisores de _Machine Learning_ sempre procuram falhas na avaliação de _datasets_ desbalanceados (o seu tem apenas 12,6% na classe alvo).

- **O que adicionar:** A métrica **PR-AUC (Precision-Recall Area Under the Curve)**.
    
- **O porquê:** Enquanto o ROC-AUC pode ser enganoso e parecer altíssimo quando há muitos "Verdadeiros Negativos" (a classe majoritária), a curva PR foca exclusivamente em como o modelo lida com a classe minoritária (mercado). Ter o PR-AUC na sua tabela de resultados demonstra extrema maturidade estatística.
    

### 2. Otimização do Limiar de Decisão (Threshold Tuning)

A maioria dos modelos (como Regressão Logística, LightGBM e BERT) não cospe a classe final de imediato; eles cospem uma probabilidade (ex: "tenho 40% de certeza que é mercado"). O padrão do `scikit-learn` é cortar no limiar de 0,5 (50%).

- **O que adicionar:** Um parágrafo e um gráfico mostrando a busca pelo **limiar ótimo** (usando a curva Precision-Recall) apenas nos dados de Validação.
    
- **O porquê:** Em _datasets_ com 12% de classe positiva, o limiar ideal para maximizar o F1-Score raramente é 0,5. Ele costuma ser menor (ex: 0,3 ou 0,35). Ajustar o _threshold_ na validação e aplicá-lo no teste é uma técnica que dá "pontos de F1 de graça" sem precisar retreinar nada.
    

### 3. Explicabilidade (XAI - Explainable AI) e Transparência

As bancas adoram quando o modelo deixa de ser uma "caixa preta". Como você está usando a Geração 1 (LinearSVC, Regressão Logística), você tem explicabilidade nativa e imediata.

- **O que adicionar:** Uma tabela ou gráfico de barras mostrando as **Top 20 Palavras** com os maiores pesos (coeficientes positivos e negativos) aprendidos pelo seu melhor modelo clássico (LinearSVC).
    
- **O porquê:** Isso prova qualitativamente que o modelo está aprendendo os jargões corretos (ex: "Selic", "Ibovespa", "déficit") e não está decorando ruídos ou viés do _dataset_. Demora 5 minutos para extrair o `.coef_` do modelo e gera uma visualização lindíssima para o artigo.
    

### 4. Análise de Erros Qualitativa (Error Analysis)

Os modelos de _Deep Learning_ e LLMs vão errar, mas o _como_ eles erram é muitas vezes mais interessante do que o acerto.

- **O que adicionar:** Uma seção curta onde você lê manualmente uns 5 ou 10 Falsos Positivos e Falsos Negativos do seu melhor modelo e discute o porquê do erro.
    
- **O porquê:** Com frequência, ao olhar os Falsos Positivos de um LLM no dataset do Kaggle (Folha/UOL), os pesquisadores descobrem que o modelo estava certo e a anotação humana original (_label_) estava errada! Provar que o modelo identificou ruídos no próprio _dataset_ original é um "xeque-mate" científico brilhante.

# Modelos de Ensemble para o futuro

Aqui estão os motivos pelos quais você **deve** manter uma seção de Ensemble, e como adaptá-la para esse novo contexto:

### 1. Você já provou a "Complementaridade de Gerações"

No resumo do seu próprio rascunho do IEEE, você escreveu uma frase de ouro: _"a combinação entre famílias de modelos distintas (TF-IDF + BERT) é mais eficaz que a combinação entre modelos da mesma família"_.

Isso é ciência pura! Você provou que a Geração 1 (que foca na matemática fria das palavras exatas) enxerga coisas que a Geração 2 (que foca no contexto difuso) não enxerga, e vice-versa. O _Stacking_ une essas duas visões. Retirar isso do novo artigo seria jogar fora uma das suas melhores descobertas.

### 2. O Cenário de "Davi contra Golias"

Imagine a seguinte tabela de resultados no final do seu artigo do BRACIS:

- **Melhor Clássico (LinearSVC):** F1 = 0,78
    
- **Melhor BERT (BERTimbau):** F1 = 0,80
    
- **LLM Gigante (Llama 3.1 8B Zero-Shot):** F1 = 0,84
    
- **Stacking Híbrido (LinearSVC + BERTimbau):** F1 = 0,86
    

Se isso acontecer, você tem a conclusão perfeita para um artigo industrial/acadêmico: _"Você não precisa de um LLM caro e pesado de 8 bilhões de parâmetros se você fizer um Stacking inteligente de modelos menores (Geração 1 + 2) que cabem em servidores muito mais baratos."_ O Ensemble serve como o teto de performance da abordagem tradicional.

### 3. Como Enxugar os Ensembles para o Novo Artigo

O seu rascunho original testou 10 estratégias (votações majoritárias, ponderadas, 3 stackings diferentes). Para o BRACIS, considerando o limite de páginas e a introdução dos LLMs, isso poluíria a narrativa.

Você deve reduzir a complexidade e levar apenas os "campeões" para a nova versão:

- **Corte as votações majoritárias puras:** Elas raramente superam um Stacking bem feito.
    
- **Mantenha 1 Votação Ponderada (Soft Voting):** Pegue a média das probabilidades do melhor Clássico e do melhor BERT. É o método híbrido mais rápido e explicável.
    
- **Mantenha 1 Stacking com Meta-Classificador:** Use a Regressão Logística como juiz final aprendendo com os erros do melhor Clássico e do melhor BERT (como você provavelmente já fez na melhor configuração do IEEE).
    

**Atenção:** Não tente colocar os LLMs (Llama/Qwen) dentro do Stacking. Como eles geram texto e não probabilidades bem calibradas (sem engenharia reversa complexa), integrá-los a um ensemble clássico em apenas uma semana é um risco desnecessário. O Ensemble deve ser o ápice da Geração 1 + 2.

# Pontos de Atenção
### 1. Engenharia de Prompts (A "Arquitetura" da Geração 3)

No aprendizado clássico, você otimiza hiperparâmetros (como o _C_ do SVM). Nos LLMs _Zero-Shot_, o seu hiperparâmetro é o **Prompt**. O artigo precisa de uma subseção dedicada a explicar como os LLMs foram instruídos.

- **O que falta incluir:** Você precisa definir o formato do _System Prompt_ (ex: "Você é um classificador financeiro especialista no mercado brasileiro...") e as restrições de saída (ex: "Responda apenas com um JSON estruturado `{"classe": "mercado"}`"). Explicar a técnica de restrição de saída é vital, pois LLMs tendem a ser verbosos ("Claro, aqui está a classificação..."), o que quebra os _scripts_ de avaliação.
    

### 2. O Pré-processamento Específico para Finanças

O dataset do Kaggle (Folha/UOL) é texto jornalístico bruto. O artigo menciona TF-IDF e BoW, mas não detalha a etapa de limpeza.

- **O que falta incluir:** Como você trata **entidades numéricas**? No jargão financeiro, números importam muito (ex: "R$ 10 bilhões", "alta de 5%"). É altamente recomendável (especialmente para a Geração 1 e 2) explicar se você mascarou os números (ex: substituir "5%" por uma tag `[PERCENTUAL]`, ou "R$ 100" por `[VALOR_MONETARIO]`), ou se manteve o texto original. Isso prova um cuidado refinado com o domínio léxico.
    

### 3. Métricas de Eficiência Computacional (Green AI)

O BRACIS e a comunidade acadêmica atual valorizam imensamente discussões sobre pegada de carbono e viabilidade em produção. Comparar o F1-Score do LinearSVC com o do Llama 3.1 8B não é totalmente justo se você não comparar o "preço" dessa predição.

- **O que falta incluir:** Uma coluna na sua tabela de resultados final medindo a **Latência/Incorrência** (ex: tempo médio em milissegundos para classificar 1.000 notícias) e o **Consumo de Memória (VRAM)**. Mostrar que o _Ensemble_ (SVM + BERT) atinge um F1 similar ao do Llama, mas gastando 10x menos energia, é um argumento brilhante.
    

### 4. Hiperparâmetros de Inferência dos LLMs

- **O que falta incluir:** Para garantir a reprodutibilidade exata, o artigo deve citar explicitamente quais foram os parâmetros passados para o Ollama. Por ser uma tarefa de classificação exata, a **Temperatura (Temperature)** deve ser setada para `0.0` (para evitar alucinações e respostas criativas) e o limite de tokens (`max_tokens`) deve ser mínimo.
