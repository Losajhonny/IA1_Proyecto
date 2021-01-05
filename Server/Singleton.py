import csv

class Singleton:
    __instance = None

    HiperParametros = None
    DataSet = None
    Municipios = None
    NuevoDataSet = None

    Train = None
    Valid = None
    Test = None

    vminEdad = 0
    vmaxEdad = 0
    vminAnio = 0
    vmaxAnio = 0

    @staticmethod 
    def getInstance():
        if Singleton.__instance == None:
            Singleton()
        return Singleton.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Singleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Singleton.__instance = self
