import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

class Node:
    """Clase que representa un nodo en el árbol de decisión."""
    
    def __init__(self, feature_index=None, feature_name=None, threshold=None, 
                 is_categorical=False, left=None, right=None, 
                 categorical_children=None, value=None):
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.threshold = threshold 
        self.is_categorical = is_categorical 
        
        
        self.left = left
        self.right = right
        
        
        self.categorical_children = categorical_children
        
        
        self.value = value 

class DecisionTreeClassifierCustom(BaseEstimator, ClassifierMixin):
    """Implementación personalizada de un clasificador de árbol de decisión."""
    
    def __init__(self, min_info_gain=0.0):
        
        self.min_info_gain = min_info_gain
        self.tree_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Entrena el árbol de decisión utilizando los datos de entrenamiento (X, y)."""
        
        self.feature_names_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        
        
        if isinstance(X, pd.DataFrame):
            
            self.is_categorical_ = [X.iloc[:, i].dtype.kind in 'O|U' for i in range(X.shape[1])]
            X_arr = X.values
        else:
            X_arr = np.array(X)
            
            self.is_categorical_ = [isinstance(X_arr[0, i], str) for i in range(X_arr.shape[1])]
            
        y_arr = np.array(y)
        self.classes_ = np.unique(y_arr)
        
        
        self.tree_ = self._build_tree(X_arr, y_arr)
        return self

    def _build_tree(self, X, y):
        """Construye el árbol de forma recursiva dividiendo los datos para maximizar la ganancia de información."""
        
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        
        if n_labels <= 1 or n_samples == 0:
            return Node(value=self._most_common_label(y))

        
        best_feature, best_threshold, best_gain = self._best_split(X, y)

        
        if best_feature is None or best_gain <= self.min_info_gain:
            return Node(value=self._most_common_label(y))

        
        is_cat = self.is_categorical_[best_feature]
        
        if is_cat:
            
            categorical_children = {}
            unique_values = np.unique(X[:, best_feature])
            for val in unique_values:
                
                idx = np.where(X[:, best_feature] == val)[0]
                
                categorical_children[val] = self._build_tree(X[idx, :], y[idx])
            return Node(feature_index=best_feature, feature_name=self.feature_names_[best_feature],
                        is_categorical=True, categorical_children=categorical_children, 
                        value=self._most_common_label(y)) 
        else:
            
            left_idx = np.where(X[:, best_feature] <= best_threshold)[0]
            right_idx = np.where(X[:, best_feature] > best_threshold)[0]
            
            left_child = self._build_tree(X[left_idx, :], y[left_idx])
            right_child = self._build_tree(X[right_idx, :], y[right_idx])
            return Node(feature_index=best_feature, feature_name=self.feature_names_[best_feature],
                        threshold=best_threshold, is_categorical=False, 
                        left=left_child, right=right_child)

    def _best_split(self, X, y):
        """Encuentra la mejor característica y el umbral óptimo para dividir los datos en un nodo."""
        
        best_gain = -1
        split_idx, split_threshold = None, None
        current_entropy = self._calculate_entropy(y)

        
        for feature_idx in range(X.shape[1]):
            X_column = X[:, feature_idx]
            is_cat = self.is_categorical_[feature_idx]
            
            if is_cat:
                
                gain = self._information_gain_categorical(X_column, y, current_entropy)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = None
            else:
                
                thresholds = np.unique(X_column)
                
                for i in range(len(thresholds) - 1):
                    thr = (thresholds[i] + thresholds[i+1]) / 2
                    gain = self._information_gain_numerical(X_column, y, thr, current_entropy)
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feature_idx
                        split_threshold = thr

        return split_idx, split_threshold, best_gain

    def _information_gain_numerical(self, X_column, y, threshold, current_entropy):
        """Calcula la ganancia de información para un posible corte en una variable numérica."""
        
        left_idx = np.where(X_column <= threshold)[0]
        right_idx = np.where(X_column > threshold)[0]

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._calculate_entropy(y[left_idx]), self._calculate_entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return current_entropy - child_entropy

    def _information_gain_categorical(self, X_column, y, current_entropy):
        """Calcula la ganancia de información para una variable categórica separando por sus valores únicos."""
        
        unique_values = np.unique(X_column)
        n = len(y)
        child_entropy = 0

        
        for val in unique_values:
            idx = np.where(X_column == val)[0]
            if len(idx) > 0:
                prob = len(idx) / n
                child_entropy += prob * self._calculate_entropy(y[idx])

        return current_entropy - child_entropy

    def _calculate_entropy(self, y):
        """Calcula la entropía de un conjunto de etiquetas, como medida de su nivel de impureza/desorden."""
        
        counts = np.bincount(y) if np.issubdtype(y.dtype, np.integer) else Counter(y).values()
        probabilities = [count / len(y) for count in counts if count > 0]
        return -np.sum([p * np.log2(p) for p in probabilities])

    def _most_common_label(self, y):
        """Devuelve la etiqueta de clase más frecuente, que se utilizará como valor de predicción en los nodos hoja."""
       
        if len(y) == 0:
            return None
        counter = Counter(y)
        
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """Predice las clases de las muestras en X haciendo que cada una recorra el árbol entrenado."""
        
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.array(X)
            
        
        return np.array([self._traverse_tree(x, self.tree_) for x in X_arr])

    def _traverse_tree(self, x, node):
        """Función recursiva para llevar una muestra específica a través del árbol hasta llegar a una hoja."""
        
        if node.value is not None and (not node.is_categorical or node.categorical_children is None):
            return node.value

        feature_val = x[node.feature_index]

        if node.is_categorical:
            
            if feature_val in node.categorical_children:
                return self._traverse_tree(x, node.categorical_children[feature_val])
            else:
                
                return node.value 
        else:
            
            if feature_val <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)