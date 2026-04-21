# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository is in a **pre-implementation** stage. At time of writing it contains only `plano_base.md` (the research plan, in Portuguese) and has no commits, no source code, no build system, and no test suite. When code is added, update this file with build/test/run commands.

## Guiding principles for this repo

- **Simplicity and reproducibility over feature-completeness.** Do not introduce abstractions, registries, or scaffolding until they are actually exercised by the experiments in `plano_base.md`. Three straightforward scripts beat a premature framework. A reader should be able to clone the repo and reproduce every table in the paper with minimal ceremony.
- **Data scope is exclusively FolhaSP/UOL** (the Kaggle Folha de São Paulo news corpus). No other datasets, no scraping pipelines, no cross-dataset evaluation, no manifest-based integration. If the work seems to require another source, flag it rather than silently expanding scope.
- **No Claude co-author trailer on commits.** Never append `Co-Authored-By: Claude ...` (or any variant) to commit messages in this repository. Commits must reflect only human authorship.

## Relationship to the predecessor project

There is a prior related project at `/home/diacrono/Documentos/repositorios/economy-classifier/` that tackled a similar classification task with a narrower model set and looser methodology. Treat it as a **conceptual reference only**:

- **Do not port code from it.** Not even modules that look clean (e.g. run-metadata helpers, evaluation utilities, test fixtures, visualization wrappers). If something analogous is needed here, reimplement it from scratch, minimally, for this project's actual requirements.
- **Lessons worth remembering (as ideas, not code):** timestamped run directories with a metadata JSON that includes the git commit and parameters; pytest markers for `slow` / `integration`; stacking with a Logistic Regression meta-classifier trained on validation scores (never on test); and keeping the ensemble surface small (one Soft Voting + one Stacking), not the 10-strategy sweep of the old IEEE draft.
- **Pitfalls of the predecessor to explicitly avoid:** random `train_test_split` on temporal data; reporting only F1/ROC-AUC without PR-AUC; absent or informal threshold tuning that isn't frozen after validation; no latency/VRAM measurement; implicit (undocumented) financial-entity handling rather than a configurable, logged preprocessing step.

## Project purpose

Research pipeline for **binary classification of Brazilian Portuguese news articles** as "market/finance" vs. "other", targeting a BRACIS-style paper. The target dataset is the Folha/UOL news corpus (~166k articles, ~12.6% positive class — severely imbalanced). The plan (`plano_base.md`) compares three generations of NLP approaches on the same task:

- **Gen 1 — Classical ML**: BoW + lemmatization, TF-IDF, fastText/Word2Vec averages; classifiers LinearSVC, Logistic Regression, Multinomial/Complement Naive Bayes, LightGBM/XGBoost, DJINN, TextCNN.
- **Gen 2 — BERT encoders**: domain-specific (FinBERT-PT-BR, DeB3RTa, ProsusAI/finbert), PT-BR baselines (BERTimbau base/large, Albertina 900M, Ulysses RoBERTa), multilingual giants (mBERT, XLM-RoBERTa base/large), and efficient (DistilBERTimbau).
- **Gen 3 — LLMs via Ollama** (zero-shot): Llama 3.1 8B, Qwen 2.5 7B/14B, Gemma 2 9B (native Ollama); Tucano and Bode imported via `.GGUF` + Modelfile. Inference parameters must be pinned (temperature 0.0, minimal `max_tokens`) and outputs constrained to structured JSON.
- **Ensembles** (Gen 1 + Gen 2 only): one Soft Voting and one Stacking with a Logistic Regression meta-classifier. LLMs are explicitly excluded from ensembles in this plan.

## Non-negotiable methodology constraints

These decisions are load-bearing for the paper's scientific validity. Code that violates them is wrong, not just suboptimal.

- **Out-of-Time split, 80/10/10, ordered by publication date.** Never use `train_test_split` with shuffling on this corpus — random splits leak event-specific vocabulary (e.g. "Americanas crash" terms) from past to future and inflate F1. Train = oldest, Val = middle, Test = most recent. All preprocessing that fits on data (TF-IDF vocabulary, fastText averages, threshold tuning) must fit on train (or train+val), never on test.
- **Threshold tuning happens on validation only, then is frozen for test.** Default 0.5 is wrong for a 12% positive class; the F1-optimal threshold is typically 0.3–0.35. Fit threshold on val, apply to test — never refit on test.
- **Evaluation metrics must include PR-AUC alongside F1.** ROC-AUC is misleading on this imbalance ratio because of the large true-negative mass. Report PR-AUC for the positive (market) class. McNemar's test is the planned model-vs-model significance test.
- **Compare models on both F1 and efficiency.** The Green-AI framing is part of the paper's argument, so benchmark tables should carry latency (ms / 1000 articles) and VRAM alongside F1 — otherwise Gen 1 vs. Gen 3 comparisons aren't meaningful.
- **Financial entity handling in preprocessing must be explicit and documented.** Numbers matter in this domain ("R$ 10 bi", "alta de 5%"). Whether they are masked (`[VALOR_MONETARIO]`, `[PERCENTUAL]`) or preserved is a per-generation decision that must be recorded, not a silent default.

## Conventions for this project

- Write user-facing text (reports, notebooks, paper artifacts) in **Portuguese (pt-BR)** to match `plano_base.md`. Code identifiers and comments may be English.
- When adding new models to the comparison, keep the grouping from `plano_base.md` (Domain Specialists / PT-BR Baselines / Multilingual Giants / Efficiency) — the paper's narrative depends on those buckets.
- Keep the ensemble section lean: exactly one Soft Voting and one Stacking, per the plan's explicit instruction to cut the 10-strategy sweep from the prior IEEE draft.

## Reference

`plano_base.md` is the source of truth for scope, model choices, and justifications. Read it before proposing changes to the experimental design.
