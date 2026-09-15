import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay

CLASES = ['L', 'E', 'V']


def evaluar_modelo(y_true, y_pred, nombre_modelo):
    """
    Imprime precisión, recall y F1-score por clase, así como accuracy y macro-F1.
    Retorna un diccionario con las métricas principales.
    """
    print(f"=== {nombre_modelo} ===")
    reporte = classification_report(y_true, y_pred, labels=CLASES, digits=3)
    print(reporte)

    macro_f1 = f1_score(y_true, y_pred, labels=CLASES, average='macro')
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))

    return {'modelo': nombre_modelo, 'accuracy': accuracy, 'macro_f1': macro_f1}


def graficar_matriz_confusion(y_true, y_pred, nombre_modelo):
    """
    Grafica la matriz de confusión para las clases ['L', 'E', 'V'].
    """
    cm = confusion_matrix(y_true, y_pred, labels=CLASES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASES)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"Matriz de confusión — {nombre_modelo}")
    return fig


def graficar_metrica_vs_hiperparametro(df_resultados, columna_x, columna_y, nombre_modelo):
    """
    Grafica una métrica (ej. Macro-F1) vs un hiperparámetro (ej. m o alpha).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df_resultados[columna_x], df_resultados[columna_y], marker='o')
    ax.set_xlabel(columna_x)
    ax.set_ylabel(columna_y)
    ax.set_title(f"{nombre_modelo}: {columna_y} vs. {columna_x}")
    ax.grid(alpha=0.3)
    return fig
