import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin


class NaiveBayesPropio(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes para atributos discretos (categóricos), con suavizado
    m-estimate y cálculo de log-probabilidades para estabilidad numérica.

    Parámetros
    ----------
    m : float, default=1.0
        Tamaño equivalente de muestra. m=0 -> sin suavizado (puede dar
        probabilidad 0 si no se vio la combinación). A mayor m, más peso
        recibe el prior p frente a los datos observados.
    """

    def __init__(self, m=1.0):
        self.m = m
        self.clases_ = None
        self.log_prior_ = {}
        self.prob_condicional_ = {}
        self.log_prob_condicional_ = {}
        self.valores_por_atributo_ = {}
        self.atributos_ = None
        self.n_por_clase_ = {}
        self.default_log_prob_ = {}

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        self.atributos_ = list(X.columns)
        self.clases_ = np.unique(y)
        n_total = len(y)

        for c in self.clases_:
            n_c = np.sum(y == c)
            self.n_por_clase_[c] = n_c
            self.log_prior_[c] = np.log(n_c / n_total)

        for atributo in self.atributos_:
            self.valores_por_atributo_[atributo] = X[atributo].unique()

        self.prob_condicional_ = {a: defaultdict(dict) for a in self.atributos_}
        self.log_prob_condicional_ = {a: defaultdict(dict) for a in self.atributos_}
        self.default_log_prob_ = {a: {} for a in self.atributos_}

        for atributo in self.atributos_:
            valores_posibles = self.valores_por_atributo_[atributo]
            k = len(valores_posibles)
            p_uniforme = 1.0 / k if k > 0 else 0.0

            for c in self.clases_:
                mask_c = (y == c)
                n_c = self.n_por_clase_[c]
                columna_c = X.loc[mask_c, atributo]

                for v in valores_posibles:
                    n_vc = np.sum(columna_c == v)
                    prob = (n_vc + self.m * p_uniforme) / (n_c + self.m)
                    prob = max(prob, 1e-12)
                    self.prob_condicional_[atributo][c][v] = prob
                    self.log_prob_condicional_[atributo][c][v] = np.log(prob)

                default_prob = (0 + self.m * p_uniforme) / (n_c + self.m)
                default_prob = max(default_prob, 1e-12)
                self.default_log_prob_[atributo][c] = np.log(default_prob)

        return self

    def _log_prob_clase(self, x_row, c):
        log_prob = self.log_prior_[c]
        for atributo in self.atributos_:
            valor = x_row[atributo]
            probs_atributo = self.prob_condicional_[atributo][c]
            if valor in probs_atributo:
                prob = probs_atributo[valor]
            else:
                k = len(self.valores_por_atributo_[atributo])
                p_uniforme = 1.0 / k if k > 0 else 0.0
                n_c = self.n_por_clase_[c]
                prob = (0 + self.m * p_uniforme) / (n_c + self.m)
            prob = max(prob, 1e-12)
            log_prob += np.log(prob)
        return log_prob

    def predict_log_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.atributos_)

        n_samples = len(X)
        n_classes = len(self.clases_)
        log_probs_matrix = np.zeros((n_samples, n_classes))

        for idx_c, c in enumerate(self.clases_):
            log_prob_c = np.full(n_samples, self.log_prior_[c])
            for atributo in self.atributos_:
                mapping = self.log_prob_condicional_[atributo][c]
                def_val = self.default_log_prob_[atributo][c]
                col_vals = X[atributo].map(mapping).fillna(def_val).to_numpy()
                log_prob_c += col_vals
            log_probs_matrix[:, idx_c] = log_prob_c

        return log_probs_matrix

    def predict(self, X):
        log_probs_matrix = self.predict_log_proba(X)
        best_indices = np.argmax(log_probs_matrix, axis=1)
        return self.clases_[best_indices]

    def predict_proba(self, X):
        log_probs_matrix = self.predict_log_proba(X)
        max_log = np.max(log_probs_matrix, axis=1, keepdims=True)
        probs = np.exp(log_probs_matrix - max_log)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return pd.DataFrame(probs, columns=self.clases_)

