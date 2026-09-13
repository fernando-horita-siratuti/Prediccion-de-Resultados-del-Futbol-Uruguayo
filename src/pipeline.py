import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class DropColumns(BaseEstimator, TransformerMixin):
    """Transformador personalizado para eliminar columnas específicas de un DataFrame."""

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')


class DateTransformer(BaseEstimator, TransformerMixin):
    """Transformador para extraer el año y el mes de una columna de fecha, eliminando la original."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new['year'] = X_new['date'].dt.year
        X_new['month'] = X_new['date'].dt.month
        return X_new.drop(columns=['date'])


class TeamFormTransformer(BaseEstimator, TransformerMixin):
    """
    Calcula, para cada partido, la tasa de victorias reciente (últimos `window_years`
    años) del equipo local y del equipo visitante, usando solo partidos ANTERIORES
    a la fecha del partido actual (sin data leakage).
    """

    def __init__(self, window_years=10):
        self.window_years = window_years

    def fit(self, X, y):
        # Formato "largo": una fila por participación de un equipo en un partido,
        # con 1 si ese equipo ganó ese partido en particular, 0 en caso contrario.
        perspectiva_local = pd.DataFrame({
            'team': X['home'].values,
            'date': X['date'].values,
            'is_win': (np.asarray(y) == 'L').astype(int),
        })
        perspectiva_visitante = pd.DataFrame({
            'team': X['away'].values,
            'date': X['date'].values,
            'is_win': (np.asarray(y) == 'V').astype(int),
        })
        historia = pd.concat([perspectiva_local, perspectiva_visitante], ignore_index=True)

        # Separar el historial por equipo reduce mucho el costo de cada consulta en
        # transform(): en vez de filtrar ~30.000 filas por partido, se filtra solo
        # el historial de ese equipo específico (normalmente unas pocas centenas).
        self.historia_por_equipo_ = {
            team: grupo.sort_values('date').reset_index(drop=True)
            for team, grupo in historia.groupby('team')
        }
        self.ventana_ = pd.Timedelta(days=365 * self.window_years)
        return self

    def _tasa_victoria(self, team, date):
        hist_equipo = self.historia_por_equipo_.get(team)
        if hist_equipo is None:
            return 0.5  # equipo nunca visto en el historial de entrenamiento: valor neutro

        pasado = hist_equipo[(hist_equipo['date'] < date) &
                              (hist_equipo['date'] >= date - self.ventana_)]
        if len(pasado) == 0:
            return 0.5  # sin partidos recientes en la ventana: valor neutro
        return pasado['is_win'].mean()

    def transform(self, X):
        X_new = X.copy()
        """
        Arredondado a 2 decimales: reduce miles de valores únicos posibles a ~100,
        lo que acelera muchísimo la búsqueda de umbral en el árbol propio (que prueba
        cada valor único como candidato de corte). Si aún tarda demasiado, prueben
        con 1 decimal (~10 valores posibles).
        """

        X_new['home_form'] = [round(self._tasa_victoria(t, d), 2) for t, d in zip(X_new['home'], X_new['date'])]
        X_new['away_form'] = [round(self._tasa_victoria(t, d), 2) for t, d in zip(X_new['away'], X_new['date'])]
        return X_new


def get_train_test_data(filepath='../data/raw/futbol_uruguayo.csv'):
    """
    Carga el conjunto de datos, genera la variable objetivo (ganador) y divide
    los datos en conjuntos de entrenamiento (hasta 2023) y prueba (desde 2024).
    """

    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    df['date'] = pd.to_datetime(df['date'])

    def asignar_ganador(row):
        """Asigna 'L' (Local), 'V' (Visitante) o 'E' (Empate) basándose en los goles."""
        if row['gh'] > row['ga']: return 'L'
        elif row['gh'] < row['ga']: return 'V'
        else: return 'E'

    df['ganador'] = df.apply(asignar_ganador, axis=1)
    df = df.sort_values('date').reset_index(drop=True)

    y = df['ganador']
    X = df.drop(columns=['ganador', 'gh', 'ga', 'full_time'])

    mask_train = X['date'].dt.year <= 2023
    mask_test = X['date'].dt.year >= 2024

    return X[mask_train].copy(), y[mask_train].copy(), X[mask_test].copy(), y[mask_test].copy()


columnas_a_eliminar = ['competition', 'level', 'continent', 'home_country',
                       'away_country', 'home_code', 'away_code',
                       'home_continent', 'away_continent', 'home_ident', 'away_ident']

# Pipeline para Random Forest / Naive Bayes: features numéricas (one-hot de equipos
# + tasa de victoria reciente + año/mes).
preprocessing_pipeline = Pipeline([
    ('dropper', DropColumns(columnas_a_eliminar)),
    ('team_form', TeamFormTransformer(window_years=10)),
    ('date_features', DateTransformer()),
    ('encoder', ColumnTransformer([
        ('teams_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['home', 'away'])
    ], remainder='passthrough'))
])

# Pipeline para el árbol propio: mantiene 'home'/'away' como texto (sin one-hot),
# para que el split categórico nativo de DecisionTreeClassifierCustom se use de verdad.
tree_pipeline = Pipeline([
    ('dropper', DropColumns(columnas_a_eliminar)),
    ('team_form', TeamFormTransformer(window_years=10)),
    ('date_features', DateTransformer()),
])