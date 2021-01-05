from flask import Flask, request, jsonify
from flask_cors import CORS

from Util import File, Plotter
from Main import Main
from RedNeuronal.Data import Data
from RedNeuronal.Model import Model
from RedNeuronal.NN_Model import NN_Model
from Singleton import Singleton

import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return 'Bienvenido'

@app.route('/deptos', methods=['GET'])
def deptos():
    # recuperar dataset
    Ds = Singleton.getInstance().DataSet
    # eliminar duplicado
    Ds = Ds.drop_duplicates(subset=['cod_depto', 'nombre'])
    # obtener cod y nombre
    nDs = Ds.loc[:, ['cod_depto', 'nombre']]
    nDs = np.array(nDs)
    
    # obtener data
    dataDeptos = []
    for i in range(len(nDs)):
        dataDeptos.append({ 'cod': int(nDs[i][0]), 'name': str(nDs[i][1]) })

    return jsonify({ 'status' : '200', 'data': dataDeptos })

@app.route('/munic', methods=['POST'])
def munic():
    #recuperar json
    data = request.json
    #recuperar depto
    codigo = data['codigo']
    # recuperar dataset
    Ds = Singleton.getInstance().Municipios
    Ds = Ds[Ds.Depto == int(codigo)]
    # eliminar duplicado
    Ds = Ds.drop_duplicates(subset=['Muni', 'Nombre'])
    # obtener cod y nombre
    nDs = Ds.loc[:, ['Muni', 'Nombre']]
    nDs = np.array(nDs)

    # obtener data
    dataMuni = []
    for i in range(len(nDs)):
        dataMuni.append({ 'cod': int(nDs[i][0]), 'name': str(nDs[i][1]) })

    return jsonify({ 'status' : '200', 'data': dataMuni })

@app.route('/consultar', methods=['POST'])
def consultar():
    #recuperar json
    dataJson = request.json
    #recuperar depto
    data = dataJson['data']
    # recuperar dataset
    Ds = Singleton.getInstance().DataSet
    Ds = Ds[(Ds.cod_depto == int(data['depto'])) & (Ds.cod_muni == int(data['municipio']))]
    # eliminar duplicado
    Ds = Ds.drop_duplicates(subset=['cod_depto', 'cod_muni'])

    # datos diagnostico
    genero = [int(data['genero'])]
    edad = [int(data['edad'])]
    anio = [int(data['anio'])]
    estado = [1]
    Ds['genero'] = genero
    Ds['edad'] = edad
    Ds['anio'] = anio
    Ds['estado'] = estado

    # obtener dataset nuevo
    m = Main()
    #print(Ds)
    vminEdad = Singleton.getInstance().vminEdad
    vmaxEdad = Singleton.getInstance().vmaxEdad
    vminAnio = Singleton.getInstance().vminAnio
    vmaxAnio = Singleton.getInstance().vmaxAnio
    print(vminEdad, vmaxEdad, vminAnio, vmaxAnio)
    Ds = m.escalamiento2("edad", Ds, vminEdad, vmaxEdad)
    Ds = m.escalamiento2("anio", Ds, vminAnio, vmaxAnio)
    nDs = Ds.loc[:, ['genero', 'edad', 'anio', 'distancia', 'estado']]
    #print(nDs)
    
    # generar objeto Data
    res = np.array(nDs)
    res = res.T
    data_x, data_y = m.makeXY(4, res)
    data_set = Data(data_x, data_y)
    
    # obtener y leer modelo
    modelo = None
    with open("modelo.pickle", "rb") as f:
        modelo = pickle.load(f)

    #show_Model([modelo])
    print(data_x.T)
    print(data_y.T)
    predict = modelo.predict(data_set)
    #Plotter.show_Model([modelo])
    #print("predict", predict)

    return jsonify({ 'status' : '200', 'data': predict })

def ini():
    print("Obteniendo datos...")
    m = Main()

    # obtener archivos
    Hp = File.getHiperParametros()
    Ds = File.getDataSet()
    Mu = File.getMunicipios()

    # construyendo datos
    Ds = m.buildDataSet(Ds, Mu)
    Ds, vmin1, vmax1 = m.escalamiento("edad", Ds)
    Ds, vmin2, vmax2 = m.escalamiento("anio", Ds)
    Ds, vmin3, vmax3 = m.escalamiento("distancia", Ds)
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

    # guardando datos
    Singleton.getInstance().HiperParametros = Hp
    Singleton.getInstance().DataSet = Ds
    Singleton.getInstance().Municipios = Mu
    Singleton.getInstance().NuevoDataSet = nDs
    Singleton.getInstance().Train = train_set
    Singleton.getInstance().Valid = valid_set
    Singleton.getInstance().Test = test_set
    Singleton.getInstance().vminEdad = vmin1
    Singleton.getInstance().vmaxEdad = vmax1
    Singleton.getInstance().vminAnio = vmin2
    Singleton.getInstance().vmaxAnio = vmax2
    print("Datos guardados!")

if __name__ == "__main__":
    ini()
    app.run(debug=True)
