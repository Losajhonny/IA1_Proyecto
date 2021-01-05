from Util import File, Plotter
from Main import Main
from RedNeuronal.Data import Data
from RedNeuronal.Model import Model
from AlgoritmoGenetico.Genetico import Genetico
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

capas = [train_set.n, 10, 5, 4, 1]
#capas = [train_set.n, 6, 8, 4, 1]
alpha = Hp['alpha']
lambd = Hp['lambda']
maxx = Hp['max_iteration']
keep = Hp['keep_prob']
g = Genetico(alpha, lambd, maxx, keep, capas, train_set, valid_set, test_set)
mejores = g.ejecutar()

modelos = []
for mejor in mejores:
    modelos.append(mejor.modelo)

Plotter.show_Model(modelos)

with open("modelo.pickle", "wb") as f:
    pickle.dump(mejores[0].modelo, f)

#load = None
#with open("modelo.pickle", "rb") as f:
    #load = pickle.load(f)

#print(load.predict(valid_set))
