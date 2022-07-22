# Código de Evaluación
############################################################################

# Import necessary libs
import pandas as pd
import numpy as np
from math import sqrt

import pickle

import warnings
warnings.filterwarnings('ignore')
from scipy import stats

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

## Modelos de regresion
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import statsmodels.api as sm           
import statsmodels.formula.api as smf  

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Métricas de evaluación
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import r2_score

# Separar train and test
from sklearn.model_selection import train_test_split

# Estadística y matemáticas
import scipy.stats as scy
from scipy.stats import kurtosis


# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../data/raw', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['ingres'],axis=1)
    y_test = df[['ingres']]
    y_pred_test=model.predict(X_test)
    


# Validación desde el inicio
def main():
    df = eval_model('model_val.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()