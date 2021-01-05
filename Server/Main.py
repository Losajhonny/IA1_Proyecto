import numpy as np
import math

class Main:
    def heversine(self, lat1, lon1, lat2, lon2):
        rad = math.pi/180
        dlat = lat2-lat1
        dlon = lon2-lon1
        R = 6372.795477598
        a = (math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
        distancia = 2*R*math.asin(math.sqrt(a))
        return distancia

    def buildDataSet(self, dataset, Mu):
        distancia = []
        estado = []
        genero = []
        for i in range(len(dataset)):
            depto = dataset['cod_depto'][i]
            munic = dataset['cod_muni'][i]
            state = dataset['Estado'][i]
            gener = dataset['Genero'][i]

            nstate = 1 if state == "Traslado" else 0
            ngener = 1 if gener == "MASCULINO" else 0
            estado.append(nstate)
            genero.append(ngener)

            sub = Mu[(Mu.Depto == depto) & (Mu.Muni == munic)]
            dis = self.heversine(sub['Lat'], sub['Lon'], 14.589246, -90.551449)
            distancia.append(dis)

        dataset['distancia'] = distancia
        dataset['estado'] = estado
        dataset['genero'] = genero
        return dataset

    def escalamiento(self, tipo, dataset):
        ds = dataset[tipo]
        vmin = min(ds)
        vmax = max(ds)
        res = (ds - vmin) / (vmax - vmin)
        dataset[tipo] = res
        return dataset, vmin, vmax

    def escalamiento2(self, tipo, dataset, vmin, vmax):
        ds = dataset[tipo]
        res = (ds - vmin) / (vmax - vmin)
        dataset[tipo] = res
        return dataset

    def makeSlicePoint(self, porcent, dataset):
        slice_point = int(dataset.shape[1] * porcent)
        part1 = dataset[:, 0: slice_point]
        part2 = dataset[:, slice_point:]
        return part1, part2
    
    def makeXY(self, noX, dataset):
        x = dataset[0: noX, :]
        y = np.array([dataset[noX, :]])
        return x, y
