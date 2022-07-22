# Código de Entrenamiento
###########################################################################


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
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw', filename)).set_index('ID')
    X_train = df.drop(['ingres'],axis=1)
    y_train = df[['ingres']]
    print(filename, ' cargado correctamente')
    
    # Entrenamos el modelo con toda la muestra
    lm = LinearRegression()
    y_sqrt = np.sqrt(y_train)
    y_log = np.log(y_train)
    lm.fit(X_train,y_train)

    # Guardamos el modelo entrenado para usarlo en produccion
    filename = '../models/best_model.pkl'
    pickle.dump(lm, open(filename, 'wb'))
    
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('model_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()
