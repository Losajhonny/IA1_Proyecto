from Util.File import get_dataFile
from Util import Plotter
from RedNeuronal.Data import Data
from RedNeuronal.NN_Model import NN_Model
from RedNeuronal.Model import Model
import numpy as np

ONLY_SHOW = False

# Cargando conjunto de datos
train_X, train_Y, val_X, val_Y = get_dataFile()

if ONLY_SHOW:
    Plotter.plot_field_data(train_X, train_Y)
    # Plotter.plot_field_data(val_X, val_Y)
    print("Entradas de entrenamiento:", train_X.shape, sep=' ')
    print("Salidas de entrenamiento:", train_Y.shape, sep=' ')
    print("Entradas de validacion:", val_X.shape, sep=' ')
    print("Salidas de validacion:", val_Y.shape, sep=' ')
    exit()

# Definir los conjuntos de datos
train_set = Data(train_X, train_Y)
val_set = Data(val_X, val_Y)

# Se define las dimensiones de las capas
#capas1 = [Cantidad de variables que tiene el problema, capa 1, capa 2, Capa de salida]
#se tendría una red neuronal de 3 capas, la capa de entrada NO se toma en cuenta
capas1 = [train_set.n, 6, 8, 4, 1]

# Se define el modelo
#0.006735;0;1734;1;
#0.087456;3;10058;0.977470217
nn1 = NN_Model(train_set, capas1, alpha=0.006735, iterations=1734, lambd=0, keep_prob=1)
nn2 = Model(train_set, capas1, alpha=0.006735, iterations=1734, lambd=0, keep_prob=1)

nn3 = NN_Model(train_set, capas1, alpha=0.087456, iterations=10058, lambd=3, keep_prob=0.977470217)
nn4 = Model(train_set, capas1, alpha=0.087456, iterations=10058, lambd=3, keep_prob=0.977470217)

# Se entrena el modelo
nn1.training(False)
nn2.training(False)
nn3.training(False)
nn4.training(False)

# Se analiza el entrenamiento
Plotter.show_Model([nn1, nn2, nn3, nn4])

px = np.array([val_X.T[0]])
py = np.array([val_Y.T[0]])

pset = Data(px.T, py.T)

print('############ n1 ############')
nn1.predict(train_set)
nn1.predict(val_set)
nn1.predict(pset)

print('############ n2 ############')
nn2.predict(train_set)
nn2.predict(val_set)
nn2.predict(pset)

print('########### n3 #############')
nn3.predict(train_set)
nn3.predict(val_set)
nn3.predict(pset)

print('########### n4 #############')
nn4.predict(train_set)
nn4.predict(val_set)
nn4.predict(pset)
