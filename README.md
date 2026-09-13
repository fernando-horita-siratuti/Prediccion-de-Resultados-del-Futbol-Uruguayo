<div align="center">
  <h1>⚽ Uruguayan Football Match Predictor</h1>
</div>

A Machine Learning academic project developed to predict the outcomes of Uruguayan football matches. The system leverages historical data and evaluates multiple classification algorithms, including a custom-built Decision Tree, Scikit-learn's Random Forest, and Naive Bayes. Designed with a strict focus on temporal cross-validation and Macro-F1 optimization.

<div align="center">

![Maintained](https://img.shields.io/badge/Maintained-yes-green.svg)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![UdelaR](https://img.shields.io/badge/UdelaR-Fing-0054A6?style=flat)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

---

## 📑 Table of Contents
- [🎯 Features](#features)
- [📋 Requirements](#requirements)
- [🗄️ Dataset Architecture](#dataset-architecture)
- [🏗️ Project Structure](#project-structure)
- [⚙️ Technical Details](#technical-details)
- [⚠️ Project Limitations](#project-limitations)
- [👥 Team & Roles](#team--roles)
- [📄 License](#license)
- [📬 Contact](#contact)

---

## <a id="features"></a>🎯 Features

- **Custom Decision Tree Implementation**: A from-scratch tree classifier using information gain and entropy calculations to determine the best splits.
- **Ensemble Learning Integration**: Implementation and hyperparameter tuning of Scikit-learn's Random Forest.
- **Probabilistic Modeling**: Naive Bayes implementation with m-estimate smoothing mechanisms.
- **Baseline Heuristic**: A baseline classifier that predicts match winners based on statistics from the last 10 years.
- **Data Leakage Prevention**: Strict chronological train/test splitting and `TimeSeriesSplit` for temporal cross-validation.
- **Automated Hyperparameter Tuning**: Exhaustive searches using `GridSearchCV` optimized specifically for the `f1_macro` metric to handle class imbalances.
- **Overfitting Visual Analysis**: Matplotlib integrations to plot training vs. validation errors across different tree depths and hyperparameters.

## <a id="requirements"></a>📋 Requirements

- **Python 3.x**
- **Libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Environment**: Jupyter Notebook / VS Code with Jupyter extension.

To install the dependencies, run:
`pip install -r requirements.txt`

## <a id="dataset-architecture"></a>🗄️ Dataset Architecture

The project relies on a historical dataset of Uruguayan football matches (`futbol_uruguayo.csv`).

### Data Pipeline (`src/pipeline.py`)
- **Target Variable Creation**: Matches decided by penalties are statistically treated as draws ('E') to prevent high variance from distorting the models' performance in regular time.
- **Team Form Transformer**: A custom transformer that calculates the recent win rate of each team over the last 10 years, exclusively using matches prior to the current fixture.
- **Encoding**: Utilizes `OneHotEncoder` for Scikit-learn models, while keeping categorical string data intact for the custom Decision Tree capable of native categorical splits.

## <a id="project-structure"></a>🏗️ Project Structure

```
proyecto_ml_futbol_uy/
├── data/
│   └── raw/
│       └── futbol_uruguayo.csv      # Historical dataset
├── notebooks/
│   ├── 01_eda_pipeline.ipynb        # Exploratory Data Analysis & Preprocessing
│   ├── 02_sandbox_arbol.ipynb       # Decision Tree & Random Forest experiments
│   └── 03_sandbox_nb.ipynb          # Naive Bayes experiments
├── src/                             # Core Python modules
│   ├── baseline.py                  # Baseline heuristic classifier
│   ├── pipeline.py                  # Custom transformers and data pipelines
│   └── tree_models.py               # From-scratch Decision Tree implementation
├── report/                          # IEEE formatted analysis
├── .gitignore
├── requirements.txt                 # Project dependencies
└── README.md                        # This file
```

## <a id="technical-details"></a>⚙️ Technical Details

### Overfitting & "Wisdom of the Crowds"
The custom Decision Tree experiments revealed structural overfitting when left unpruned (`min_info_gain=0.0`). However, applying the same unconstrained depth (`max_depth=None`) to a Random Forest with 200 estimators demonstrated the power of bagging: diluting individual tree variance to achieve a robust generalization and significantly improving the Macro-F1 score on the test set.

### Optimization Target
Default `accuracy` optimization led to conservative models heavily biased toward the majority class (Home Team Wins). By shifting the `GridSearchCV` scoring target to `f1_macro`, the algorithms were forced to recognize patterns in minority classes (Draws and Away Wins), greatly balancing the Confusion Matrix.

## <a id="project-limitations"></a>⚠️ Project Limitations

- **Domain Stochasticity**: Football results inherently possess a high degree of unpredictability. Factors like red cards, weather, injuries, or referee decisions are not captured in the basic dataset.
- **Feature Limitations**: The current attributes mainly rely on historical team forms. Future improvements should include direct Head-to-Head metrics or relative level differences (`home_form - away_form`) to better capture "balanced" matches and improve Draw (`E`) predictions.

## <a id="team--roles"></a>👥 Team & Roles

This project was developed for the *Aprendizaje Automático* course at Facultad de Ingeniería (Fing), Universidad de la República (UdelaR).

- **Rol 1 (Pipeline & Baseline)**: Anastasia Martucci
- **Rol 2 (Tree Models)**: Fernando Horita Siratuti
- **Rol 3 (Naive Bayes)**: [Nome do Colega]

## <a id="license"></a>📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## <a id="contact"></a>📬 Contact

<div align="center">
  <br><br>
    <i>Fernando Horita Siratuti - Undergraduate - 6th Period, Computer Engineering @ CEFET-MG | Academic Exchange @ Fing - UdelaR</i>
  <br><br>
  
  [![Gmail](https://img.shields.io/badge/-Gmail-c14438?style=flat-square&logo=Gmail&logoColor=white)](mailto:siratutifernando@gmail.com)
  [![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/fernando-siratuti-503ba8301/)
  [![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/fernando-horita-siratuti)
  [![Instagram](https://img.shields.io/badge/-Instagram-E4405F?style=flat-square&logo=instagram&logoColor=white)](https://www.instagram.com/siratuti_/)

</div>
