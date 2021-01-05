from Util import File, Plotter
from Main import Main
from RedNeuronal.Data import Data
from RedNeuronal.Model import Model
from RedNeuronal.NN_Model import NN_Model
from RedNeuronal.RModel import RModelo
from Main import Main

import pickle
import numpy as np

m = Main()

# obtener archivos
Hp = File.getHiperParametros()
Ds = File.getDataSet()
Mu = File.getMunicipios()

# construyendo datos
Ds = m.buildDataSet(Ds, Mu)
Ds, vmin, vmax = m.escalamiento("edad", Ds)
Ds, vmin, vmax = m.escalamiento("anio", Ds)
Ds, vmin, vmax = m.escalamiento("distancia", Ds)
nDs = Ds.loc[:, ['genero', 'edad', 'anio', 'distancia', 'estado']]

# aplicando division
res = np.array(nDs)
res = res.T

slice_point = int(res.shape[1] * 0.7)
train, res = m.makeSlicePoint(0.7, res)
valid, test = m.makeSlicePoint(0.5, res)

train_x, train_y = m.makeXY(4, train)
valid_x, valid_y = m.makeXY(4, valid)
test_x, test_y = m.makeXY(4, test)

train_set = Data(train_x, train_y)
valid_set = Data(valid_x, valid_y)
test_set = Data(test_x, test_y)

#capas1 = [train_set.n, 4, 5, 4, 3, 2, 1]
capas1 = [train_set.n, 6, 8, 4, 1]

alpha1 = Hp['alpha'][1]
alpha2 = Hp['alpha'][2]

lambd1 = Hp['lambda'][1]
lambd2 = Hp['lambda'][2]

itera1 = Hp['max_iteration'][1]
itera2 = Hp['max_iteration'][2]

keepp1 = Hp['keep_prob'][1]
keepp2 = Hp['keep_prob'][2]

nn1 = Model(train_set, capas1, alpha=alpha1, iterations=itera1, lambd=lambd1, keep_prob=keepp1)
nn2 = RModelo(train_set, capas1, alpha=alpha1, iterations=itera1, lambd=lambd1, keep_prob=keepp1)

nn3 = Model(train_set, capas1, alpha=alpha2, iterations=itera2, lambd=lambd2, keep_prob=keepp2)
nn4 = RModelo(train_set, capas1, alpha=alpha2, iterations=itera2, lambd=lambd2, keep_prob=keepp2)

# Se entrena el modelo
#nn1.training(False)
#nn2.training(False)
#nn3.training(False)
#nn4.training(False)

# Se analiza el entrenamiento
#Plotter.show_Model([nn1, nn2, nn3, nn4])

px = np.array([valid_x.T[0]])
py = np.array([valid_y.T[0]])

pset = Data(px.T, py.T)
print(px)
print(py)

print('############ n1 ############')
#nn1.predict(train_set)
#nn1.predict(valid_set)
#nn1.predict(test_set)
#nn1.predict(pset)

print('############ n2 ############')
#nn2.predict(train_set)
#nn2.predict(valid_set)
#nn2.predict(test_set)
#nn2.predict(pset)

print('############ n3 ############')
#nn3.predict(train_set)
#nn3.predict(valid_set)
#nn3.predict(test_set)
#nn3.predict(pset)

print('########### n4 #############')
#nn4.predict(train_set)
#nn4.predict(valid_set)
#nn4.predict(test_set)
#nn4.predict(pset)

#with open("modelo.pickle", "wb") as f:
    #pickle.dump(nn4, f)
