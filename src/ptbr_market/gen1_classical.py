"""Classificadores clássicos (Geração 1).

Todos recebem features numéricas (TF-IDF/BoW esparsas ou densas de
fastText) já construídas fora deste módulo. Convenções:

- `SEED=1` (decisão D3) é aplicado onde faz sentido (`random_state`).
- `LinearSVC` é envolvido em `CalibratedClassifierCV` (sigmoid, cv=3)
  para expor `predict_proba` — necessário para `fit_threshold`, que
  opera na grade [0.05, 0.95] de probabilidades.
- `LGBMClassifier` roda em modo determinístico (`deterministic=True`,
  `force_col_wise=True`) para que o `random_state` baste para reprodução.
  É o representante único da família GBDT — XGBoost foi removido da
  matriz por limitações de memória no host de 16 GB durante multiclasse
  (8 classes × histograma por feature estoura 15 GB em representações
  esparsas). `plano_base.md` permite "LightGBM/XGBoost" como opções
  intercambiáveis; ficamos com LightGBM.
- `class_weight` fica em default (None). Balanceamento é escolha do
  experimento, não do builder.
- `LogisticRegression` usa `lbfgs` (suporta binário e multiclasse — o
  pipeline Gen 1 roda ambos). `liblinear` seria mais rápido em binário
  mas não suporta multiclasse nativamente.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

SEED = 1

GEN1_CLASSIFIERS: tuple[str, ...] = (
    "linearsvc",
    "logreg",
    "multinomialnb",
    "complementnb",
    "lightgbm",
)


def build_classifier(name: str) -> ClassifierMixin:
    """Retorna um estimator sklearn com `fit`/`predict_proba` para `name`.

    Nomes aceitos em `GEN1_CLASSIFIERS`.
    """
    if name == "linearsvc":
        base = LinearSVC(random_state=SEED, dual="auto", max_iter=5000)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    if name == "logreg":
        return LogisticRegression(
            random_state=SEED, solver="lbfgs", max_iter=2000
        )
    if name == "multinomialnb":
        return MultinomialNB()
    if name == "complementnb":
        return ComplementNB()
    if name == "lightgbm":
        return LGBMClassifier(
            random_state=SEED,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
        )
    raise ValueError(
        f"Classificador desconhecido: {name!r}. Use um de {GEN1_CLASSIFIERS}."
    )
