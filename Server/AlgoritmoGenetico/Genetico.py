import random
import numpy as np
from RedNeuronal.Model import Model

class Nodo:
    def __init__(self, solucion=[], fitness=0):
        self.solucion = solucion
        self.fitness = fitness
        self.modelo = None

class Genetico:
    def __init__(self, alpha, lambd, maxx, keep, capas, train, valid, test):
        self.maxGeneracion = 10
        self.maxPoblacion = 10
        self.maxIndividuo = 4
        self.maxPadres = 5
        self.alpha = alpha
        self.lambd = lambd
        self.maxx = maxx
        self.keep = keep
        self.train = train
        self.valid = valid
        self.test = test
        self.capas = capas

    def iniciarPoblacion(self):
        poblacion = []
        for i in range(self.maxPoblacion):
            solucion = []
            for j in range(self.maxIndividuo):
                solucion.append(self.getIndice())
            poblacion.append(Nodo(solucion))
        return poblacion

    def getIndice(self):
        return random.randint(0, self.maxPoblacion - 1)

    def evaluarFitness(self, solucion):
        alpha = self.alpha[solucion[0]]
        lambd = self.lambd[solucion[1]]
        itera = self.maxx[solucion[2]]
        keepp = self.keep[solucion[3]]

        m = Model(self.train, self.capas, alpha=alpha, iterations=itera, lambd=lambd, keep_prob=keepp)
        m.training(False)
        prediccion = m.predict(self.valid)
        
        return prediccion, m

    def verificarCriterio(self, poblacion, generacion):
        #actualizar fitness
        for nodo in poblacion:
            prediccion, modelo = self.evaluarFitness(nodo.solucion)
            nodo.fitness = prediccion
            nodo.modelo = modelo
            print("Individuo:", nodo.solucion, "Fitness:", nodo.fitness)
        
        if generacion >= self.maxGeneracion:
            return True
        return None

    def seleccionarPadres(self, poblacion):
        mejores = []

        # ordenar con el mayor fitness (exactitud validacion)
        orden = sorted(poblacion, key=lambda item: item.fitness, reverse=True)

        for i in range(self.maxPadres):
            mejores.append(orden[i])
        return mejores

    def emparejar(self, padres):
        nueva = padres
        noFalta = self.maxPoblacion - self.maxPadres
        mejor1 = sorted(padres, key=lambda item: item.fitness, reverse=True)
        mejor2 = sorted(padres, key=lambda item: item.fitness, reverse=True)
        
        for i in range(noFalta):
            padre1 = mejor1[i]
            padre2 = mejor2[0] if i == (noFalta-1) else mejor2[i + 1]
            hijo = Nodo()
            hijo.solucion = self.cruzar(padre1.solucion, padre2.solucion)
            hijo.solucion = self.mutar(hijo.solucion)
            nueva.append(hijo)
        return nueva

    def cruzar(self, padre1, padre2):
        solucion = []
        for i in range(self.maxIndividuo):
            num = random.random()
            if num <= 0.5:
                solucion.append(padre1[i])
            else:
                solucion.append(padre2[i])
        return solucion

    def mutar(self, solucion):
        pos = random.randint(0, (self.maxIndividuo - 1))
        solucion[pos] = self.getIndice()
        return solucion
    
    def printPoblacion(self, poblacion):
        for i in range(self.maxPoblacion):
            print("~~~~ Poblacion [", i, "] ~~~~")
            print("sol:", poblacion[i].solucion)
            print("fit:", poblacion[i].fitness)
    
    def ejecutar(self):
        generacion = 0
        print("---------- Generacion ----------", generacion, "----------")
        poblacion = self.iniciarPoblacion()
        fin = self.verificarCriterio(poblacion, generacion)

        while fin == None:
            padres = self.seleccionarPadres(poblacion)
            poblacion = self.emparejar(padres)
            generacion += 1
            print("---------- Generacion ----------", generacion, "----------")
            fin = self.verificarCriterio(poblacion, generacion)

        #print("---------- Generacion ----------", generacion, "----------")
        mejor = sorted(poblacion, key=lambda item: item.fitness, reverse=True)
        self.printPoblacion(mejor)

        return mejor
