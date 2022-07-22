
# Script de Preparación de Datos
###################################

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

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename)).
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):
    #Aplicar función de logaritmo al Ingreso
    df['ingres'] = np.log(df['ingres'])
    
    print('Transformación de datos completa')
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('model_train.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['edad','AniosDireccion','Gastocoche','Aniosempleo','Aniosresiden','ingres'],'result_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('model_train.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, ['edad','AniosDireccion','Gastocoche','Aniosempleo','Aniosresiden','ingres'],'result_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('model_train.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ['edad','AniosDireccion','Gastocoche','Aniosempleo','Aniosresiden','ingres'],'result_score.csv')
    
if __name__ == "__main__":
    main()
