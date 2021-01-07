import numpy as np
#np.set_printoptions(threshold=100000) #Esto es para que al imprimir un arreglo no me muestre puntos suspensivos

class RModel:

    def __init__(self, train_set, layers, alpha=0.3, iterations=300000, lambd=0, keep_prob=1):
        self.data = train_set
        self.alpha = alpha
        self.max_iteration = iterations
        self.lambd = lambd
        self.kp = keep_prob
        self.layers = layers
        # Se inicializan los pesos
        self.parametros = self.Inicializar(layers)

    def Inicializar(self, layers):
        parametros = {}
        L = len(layers)
        #print('layers:', layers)
        for l in range(1, L):
            #np.random.randn(layers[l], layers[l-1])
            #Crea un arreglo que tiene layers[l] arreglos, donde cada uno de estos arreglos tiene layers[l-1] elementos con valores aleatorios
            #np.sqrt(layers[l-1] se saca la raiz cuadrada positiva de la capa anterior ---> layers[l-1]
            
            #sqr = np.sqrt(layers[l-1])
            #ran = np.random.randn(layers[l], layers[l-1])
            #zer = np.zeros((layers[l], 1))
            #sqr = np.sqrt(layers[l-1])
            parametros['W'+str(l)] = np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1])
            parametros['b'+str(l)] = np.zeros((layers[l], 1))

            #print(layers[l], layers[l-1], np.random.randn(layers[l], layers[l-1]))
            #print(np.sqrt(layers[l-1]))
            #print(np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1]))
        return parametros

    def training(self, show_cost=False):
        self.bitacora = []
        for i in range(0, self.max_iteration):
            y_hat, temp = self.propagacion_adelante(self.data)
            cost = self.cost_function(y_hat)
            gradientes = self.propagacion_atras(temp)
            self.actualizar_parametros(gradientes)
            if i % 50 == 0:
                self.bitacora.append(cost)
                if show_cost:
                    print('Iteracion No.', i, 'Costo:', cost, sep=' ')


    """
    def propagacion_adelante(self, dataSet):
        # Se extraen las entradas
        X = dataSet.x

        Zi = None
        Ai = None
        Di = None

        for i in range(1, len(self.layers) - 1):
            Wi = self.parametros['W' + str(i)]
            bi = self.parametros['b' + str(i)]
            Zi = np.dot(Wi, X) + bi
            X = Ai = self.activation_function('relu', Zi)
            #Se aplica el Dropout Invertido
            D2 = np.random.rand(Ai.shape[0], X.shape[1])
    """

    
    def propagacion_adelante(self, dataSet):
        # Recorrer capas
        L = len(self.layers)
        X = dataSet.x

        Wn, bn = self.dicWB()
        temp = {}

        for i in range(1, L):
            Wi = Wn['W' + str(i)]
            bi = bn['b' + str(i)]
            Ai = 0
            Di = 0
            Zi = 0

            # si el ultimo activar por sigmoide
            if i == (L-1):
                Zi = np.dot(Wi, X) + bi
                Ai = self.activation_function('sigmoide', Zi)
            # sino activar relu y aplicar el dropout invertido
            else:
                Zi = np.dot(Wi, X) + bi
                Ai = self.activation_function('relu', Zi)
                if i == 1:
                    Di = np.random.rand(Ai.shape[0], Ai.shape[1])
                else:
                    Di = np.random.rand(Ai.shape[0], X.shape[1])
                Di = (Di < self.kp).astype(int)
                Ai *= Di
                Ai /= self.kp
            
            X = Ai
            temp['Z' + str(i)] = Zi
            temp['A' + str(i)] = Ai
            temp['D' + str(i)] = Di

        return X, temp
    

    def dicWB(self):
        L = len(self.layers)
        Wn = {}
        bn = {}

        for i in range(1, L):
            Wn['W' + str(i)] = self.parametros['W' + str(i)]
            bn['b' + str(i)] = self.parametros['b' + str(i)]

        return Wn, bn

    def propagacion_atras(self, temp):
        L = i = len(self.layers) - 1

        # Se obtienen los datos
        m = self.data.m
        Y = self.data.y
        X = self.data.x

        Wn, bn = self.dicWB()
        gradientes = {}

        Wiant = 0
        dZiant = 0
        
        while i >= 1:
            Wi = Wn['W' + str(i)]
            bi = bn['b' + str(i)]
            Ai = temp['A' + str(i)]
            Di = temp['D' + str(i)]
            Aij = 0
            dAi = 0
            dZi = 0
            dWi = 0

            if i == L:
                dZi = Ai - Y
            else:
                dAi = np.dot(Wiant.T, dZiant)
                dAi *= Di
                dAi /= self.kp
                dZi= np.multiply(dAi, np.int64(Ai > 0))

            if i == 1:
                dWi = (1 / m) * np.dot(dZi, X.T) + (self.lambd / m) * Wi
            else:
                Aij = temp['A' + str(i - 1)]
                dWi = (1 / m) * np.dot(dZi, Aij.T) + (self.lambd / m) * Wi
            
            dbi = (1 / m) * np.sum(dZi, axis=1, keepdims=True)
            Wiant = Wi
            dZiant = dZi

            if i != L:
                gradientes['dA' + str(i)] = dAi

            gradientes['dZ' + str(i)] = dZi
            gradientes['dW' + str(i)] = dWi
            gradientes['db' + str(i)] = dbi

            i -= 1
            
        return gradientes

    def actualizar_parametros(self, grad):
        # Se obtiene la cantidad de pesos
        L = len(self.parametros) // 2
        for k in range(L):
            self.parametros["W" + str(k + 1)] -= self.alpha * grad["dW" + str(k + 1)]
            self.parametros["b" + str(k + 1)] -= self.alpha * grad["db" + str(k + 1)]

    def cost_function(self, y_hat):
        # Se obtienen los datos
        Y = self.data.y
        m = self.data.m
        # Se hacen los calculos
        temp = np.multiply(-np.log(y_hat), Y) + np.multiply(-np.log(1 - y_hat), 1 - Y)
        result = (1 / m) * np.nansum(temp)
        # Se agrega la regularizacion L2
        if self.lambd > 0:
            L = len(self.parametros) // 2
            suma = 0
            for i in range(L):
                suma += np.sum(np.square(self.parametros["W" + str(i + 1)]))
            result += (self.lambd/(2*m)) * suma
        return result

    def predict(self, dataSet):
        # Se obtienen los datos
        m = dataSet.m
        Y = dataSet.y
        p = np.zeros((1, m), dtype= np.int)
        # Propagacion hacia adelante
        y_hat, temp = self.propagacion_adelante(dataSet)
        # Convertir probabilidad
        for i in range(0, m):
            p[0, i] = 1 if y_hat[0, i] > 0.5 else 0
        exactitud = np.mean((p[0, :] == Y[0, ]))
        #print("Exactitud: " + str(exactitud))
        return exactitud


    def activation_function(self, name, x):
        result = 0
        if name == 'sigmoide':
            result = 1/(1 + np.exp(-x))
        elif name == 'tanh':
            result = np.tanh(x)
        elif name == 'relu':
            result = np.maximum(0, x)
        
        #print('name:', name, 'result:', result)
        return result
