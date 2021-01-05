import math
import matplotlib.pyplot as plt
import numpy as np

class Data:
    def __init__(self, data_set_x, data_set_y, max_value=1):
        self.m = data_set_x.shape[1]
        self.n = data_set_x.shape[0]
        self.x = data_set_x / max_value
        self.y = data_set_y

class NN_Model:

    def __init__(self, train_set, layers, alpha=0.3, iterations=300000, lambd=0, keep_prob=1):
        self.data = train_set
        self.alpha = alpha
        self.max_iteration = iterations
        self.lambd = lambd
        self.kp = keep_prob
        # Se inicializan los pesos
        self.parametros = self.Inicializar(layers)

    def Inicializar(self, layers):
        parametros = {}
        L = len(layers)
        print('layers:', layers)
        for l in range(1, L):
            print(l)
            parametros['W'+str(l)] = np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1])
            parametros['b'+str(l)] = np.zeros((layers[l], 1))

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


    def propagacion_adelante(self, dataSet):
        # Se extraen las entradas
        X = dataSet.x
        
        # Extraemos los pesos
        W1 = self.parametros["W1"]
        b1 = self.parametros["b1"]
        
        W2 = self.parametros["W2"]
        b2 = self.parametros["b2"]

        # ------ Primera capa
        Z1 = np.dot(W1, X) + b1
        A1 = self.activation_function('sigmoide', Z1)
        #Se aplica el Dropout Invertido
        #D1 = np.random.rand(A1.shape[0], A1.shape[1]) #Se generan número aleatorios para cada neurona
        #D1 = (D1 < self.kp).astype(int) #Mientras más alto es kp mayor la probabilidad de que la neurona permanezca
        #A1 *= D1
        #A1 /= self.kp
        
        # ------ Segunda capa
        Z2 = np.dot(W2, A1) + b2
        A2 = self.activation_function('tanh', Z2)

        temp = (Z1, A1, Z2, A2)
        #En A3 va la predicción o el resultado de la red neuronal
        return A2, temp

    def propagacion_atras(self, temp):
        # Se obtienen los datos
        m = self.data.m
        Y = self.data.y
        X = self.data.x
        W1 = self.parametros["W1"]
        W2 = self.parametros["W2"]
        (Z1, A1, Z2, A2) = temp

        # Derivadas parciales de la segunda capa
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T) + (self.lambd / m) * W2
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Derivadas parciales de la primera capa
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1./m * np.dot(dZ1, X.T) + (self.lambd / m) * W1
        db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

        #Se guardan todas la derivadas parciales
        gradientes = {"dZ2": dZ2, "dW2": dW2, "db2": db2,
                     "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

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
        #temp = np.multiply(-np.log(y_hat), Y) + np.multiply(-np.log(1 - y_hat), 1 - Y)
        #result = (1 / m) * np.nansum(temp)
        temp = (1 / 2) * np.multiply(y_hat - Y, y_hat - Y)
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
        print("Exactitud: " + str(exactitud))
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

def show_Model(models):
    for model in models:
        plt.plot(model.bitacora, label=str(model.alpha))
    plt.ylabel('Costo')
    plt.xlabel('Iteraciones')
    legend = plt.legend(loc='upper center', shadow=True)
    plt.show()

def funcion(x):
    return ((x ** 2) / 100) - math.sin((x)) + 1

def dataSet():
    ds = []
    for i in range(101):
        x = i - 50
        y = funcion(x)
        ds.append([x, y])

    print(ds)
    result = np.array(ds)
    np.random.shuffle(result)
    result = result.T

    slice_point = int(result.shape[1] * 0.7)
    train_set = result[:, 0: slice_point]
    test_set = result[:, slice_point:]

    #separar entradas y salidas
    train_x = train_set[0: 1, :]
    train_y = np.array([train_set[1, :]])

    test_x = test_set[0: 1, :]
    test_y = np.array([test_set[1, :]])

    return train_x, train_y, test_x, test_y

# Datos
train_x, train_y, test_x, test_y = dataSet()
train_set = Data(train_x, train_y)
val_set = Data(test_x, test_y)

# capas
capas = [train_set.n, 2, 1]

# definir modelo
m = NN_Model(train_set, capas, alpha=0.00001, iterations=10000, lambd=0.01, keep_prob=1)
m.training(False)

print('Entrenamiento')
m.predict(train_set)
print('Validacion')
m.predict(val_set)

show_Model([m])
