import numpy as np


class RModelo:
    def __init__(self, train_set, capas, alpha=0.3, iterations=300000, lambd=0, keep_prob=1):
        self.data = train_set
        self.alpha = alpha
        self.max_iteration = iterations
        self.lambd = lambd
        self.kp = keep_prob
        self.parametros = self.Inicializar(capas)

    # retorna una lista de listas de parametros
    def Inicializar(self, layers):
        parametros = {}
        L = len(layers)
        print('layers:', layers)
        for l in range(1, L):
            # np.random.randn(layers[l], layers[l-1])
            # Crea un arreglo que tiene layers[l] arreglos, donde cada uno de estos arreglos tiene layers[l-1] elementos con valores aleatorios
            # np.sqrt(layers[l-1] se saca la raiz cuadrada positiva de la capa anterior ---> layers[l-1]

            # Almaceno en diccionario de diccionarios
            parametros['W' + str(l)] = np.random.randn(layers[l], layers[l - 1]) / np.sqrt(layers[l - 1])
            parametros['b' + str(l)] = np.zeros((layers[l], 1))
            # print(layers[l], layers[l-1], np.random.randn(layers[l], layers[l-1]))
            # print(np.sqrt(layers[l-1]))
            # print(np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1]))

        return parametros  # retorna un json con los valores de w y b

    def training(self, show_cost=False):
        self.bitacora = []
        for i in range(0, self.max_iteration):
            y_hat, temp = self.propagacion_adelante(self.data)
            cost = self.cost_function(y_hat)
            gradientes = self.propagacion_atras(temp)  # se le envia un diccionario
            self.actualizar_parametros(gradientes)
            if i % 50 == 0:
                self.bitacora.append(cost)
                if show_cost:
                    print('Iteracion No.', i, 'Costo:', cost, sep=' ')

    def propagacion_adelante(self, dataSet):
        # Se extraen las entradas
        X = dataSet.x
        temp = {}  # contiene todos los : Z, A, D
        No_pesos = len(self.parametros) // 2
        A = None
        Z = None
        D = None
        aux = 0
        for i in range(1, No_pesos):
            aux = i
            Wn = self.parametros['W' + str(i)]
            bn = self.parametros['b' + str(i)]

            if i == 1:  # primera capa
                Z = np.dot(Wn, X) + bn
            else:
                Z = np.dot(Wn, A) + bn

            A = self.activation_function('relu', Z)
            # Se aplica el Dropout Invertido
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < self.kp).astype(int)
            A *= D
            A /= self.kp
            temp['Z' + str(i)] = Z
            temp['A' + str(i)] = A
            temp['D' + str(i)] = D

        # ULTIMA CAPA
        # Aca ya llegue al ultimo peso es por eso que en el for  el limite esta en ( No_pesos - 1)
        Wn = self.parametros['W' + str(aux + 1)]
        bn = self.parametros['b' + str(aux + 1)]
        Z = np.dot(Wn, A) + bn
        A = self.activation_function('sigmoide', Z)
        temp['Z' + str(aux + 1)] = Z
        temp['A' + str(aux + 1)] = A
        return A, temp

    def propagacion_atras(self, temp):
        # Se obtienen los datos
        m = self.data.m
        Y = self.data.y
        X = self.data.x

        No_capas = len(self.parametros) // 2
        dZn = None
        Wn_anterior = None
        gradientes = {}

        while (No_capas != 1):

            A_anterior = temp["A" + str(No_capas - 1)]
            An = temp["A" + str(No_capas)]
            Wn_actual = self.parametros["W" + str(No_capas)]

            if No_capas == (len(self.parametros) // 2):  # es la ultima capa
                dZn = An - Y
                dWn = (1 / m) * np.dot(dZn, A_anterior.T) + (self.lambd / m) * Wn_actual
                dbn = (1 / m) * np.sum(dZn, axis=1, keepdims=True)
                gradientes["dZ" + str(No_capas)] = dZn
                gradientes["dW" + str(No_capas)] = dWn
                gradientes["db" + str(No_capas)] = dbn
            else:
                dAn = np.dot(Wn_anterior.T, dZn)
                dn = temp["D" + str(No_capas)]
                dAn *= dn
                dZn = np.multiply(dAn, np.int64(An > 0))

                dWn = 1. / m * np.dot(dZn, A_anterior.T) + (self.lambd / m) * Wn_actual
                dbn = 1. / m * np.sum(dZn, axis=1, keepdims=True)

                gradientes["dA" + str(No_capas)] = dAn
                gradientes["dZ" + str(No_capas)] = dZn
                gradientes["dW" + str(No_capas)] = dWn
                gradientes["db" + str(No_capas)] = dbn

            Wn_anterior = self.parametros[
                "W" + str(No_capas)]  # de esta manera me aseguro de poder trabajar con un Wn anterior

            No_capas -= 1

        # Cuando llegue aca es porque estoy en la primera capa
        An = temp["A" + str(No_capas)]
        Wn_actual = self.parametros["W" + str(No_capas)]

        dAn = np.dot(Wn_anterior.T, dZn)
        dn = temp["D" + str(No_capas)]
        dAn *= dn
        dAn /= self.kp
        dZn = np.multiply(dAn, np.int64(An > 0))
        dWn = 1. / m * np.dot(dZn, X.T) + (self.lambd / m) * Wn_actual
        db = 1. / m * np.sum(dZn, axis=1, keepdims=True)

        gradientes["dA" + str(No_capas)] = dAn
        gradientes["dZ" + str(No_capas)] = dZn
        gradientes["dW" + str(No_capas)] = dWn
        gradientes["db" + str(No_capas)] = db

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
            result += (self.lambd / (2 * m)) * suma
        return result

    def predict(self, dataSet):
        # Se obtienen los datos
        m = dataSet.m
        Y = dataSet.y
        p = np.zeros((1, m), dtype=np.int)
        # Propagacion hacia adelante
        y_hat, temp = self.propagacion_adelante(dataSet)
        # Convertir probabilidad
        for i in range(0, m):
            p[0, i] = 1 if y_hat[0, i] > 0.5 else 0
        exactitud = np.mean((p[0, :] == Y[0,]))
        print("Exactitud: " + str(exactitud))
        return exactitud

    def activation_function(self, name, x):
        result = 0
        if name == 'sigmoide':
            result = 1 / (1 + np.exp(-x))
        elif name == 'tanh':
            result = np.tanh(x)
        elif name == 'relu':
            result = np.maximum(0, x)

        # print('name:', name, 'result:', result)
        return result