import scipy.io
import pandas as pd

def get_dataFile():
    data = scipy.io.loadmat('datasets/data.mat')

    #print(data['X'].T)
    
    train_X = data['X'].T
    train_Y = data['y'].T
    val_X = data['Xval'].T
    val_Y = data['yval'].T

    #print('data[\'y\']', data['y'])
    #print('data[\'y\'].T', data['y'].T)

    return train_X, train_Y, val_X, val_Y

def read_csv(path, sep=','):
    return pd.read_csv(path, sep=sep)

def getHiperParametros():        
    hiperparametros = read_csv("Resource/HiperParametros.csv", ';')
    return hiperparametros

def getDataSet():
    dataset = read_csv("Resource/Dataset.csv")
    return dataset

def getMunicipios():
    municipios = read_csv("Resource/Municipios.csv")
    return municipios
