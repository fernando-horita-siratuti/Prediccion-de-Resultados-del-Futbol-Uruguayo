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

# Pipeline principal de preprocesamiento: elimina columnas, procesa fechas y codifica variables

preprocessing_pipeline = Pipeline([
    ('dropper', DropColumns(columnas_a_eliminar)),
    ('date_features', DateTransformer()),
    ('encoder', ColumnTransformer([
        ('teams_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['home', 'away'])
    ], remainder='passthrough'))
])