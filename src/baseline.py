import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ClasificadorBase10Anios(BaseEstimator, ClassifierMixin):
    """Clasificador de línea base que predice el ganador del partido basándose en las estadísticas de los últimos 10 años."""

    def fit(self, X, y):
        """Entrena el clasificador de línea base."""
        
        self.historia_ = X.copy()
        self.historia_['target'] = y.copy()
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predice el ganador del partido basándose en las estadísticas de los últimos 10 años."""
        
        predicciones = []

        for idx, row in X.iterrows():
            fecha_partido = row['date']
            fecha_limite = fecha_partido - pd.DateOffset(years=10)
            eq_local = row['home']
            eq_visit = row['away']

            # Filtro rápido sobre el DataFrame histórico
            mask_tiempo = (self.historia_['date'] >= fecha_limite) & (self.historia_['date'] < fecha_partido)
            historial = self.historia_[mask_tiempo]

            # Estadísticas del equipo local
            partidos_loc = historial[(historial['home'] == eq_local) | (historial['away'] == eq_local)]
            vict_loc = len(partidos_loc[(partidos_loc['home'] == eq_local) & (partidos_loc['target'] == 'L')]) + \
                       len(partidos_loc[(partidos_loc['away'] == eq_local) & (partidos_loc['target'] == 'V')])
            prop_loc = vict_loc / len(partidos_loc) if len(partidos_loc) > 0 else 0

            # Estadísticas del equipo visitante
            partidos_vis = historial[(historial['home'] == eq_visit) | (historial['away'] == eq_visit)]
            vict_vis = len(partidos_vis[(partidos_vis['home'] == eq_visit) & (partidos_vis['target'] == 'L')]) + \
                       len(partidos_vis[(partidos_vis['away'] == eq_visit) & (partidos_vis['target'] == 'V')])
            prop_vis = vict_vis / len(partidos_vis) if len(partidos_vis) > 0 else 0

            if prop_loc > prop_vis:
                predicciones.append('L')
            elif prop_vis > prop_loc:
                predicciones.append('V')
            else:
                predicciones.append('E')

        return np.array(predicciones)