import pandas as pd
import random
import numpy as np

# Aquí cargo el dataset
dataset = pd.read_excel('2024.05.22 dataset 8A.xlsx')

# Aquí almaceno las variables de entrada para cada fila
x1 = dataset['x1'].tolist() # Convierto la columna x1 en una lista 31 elementos
x2 = dataset['x2'].tolist() # Convierto la columna x2 en una lista 31 elementos
x3 = dataset['x3'].tolist() # Convierto la columna x3 en una lista 31 elementos
x4 = dataset['x4'].tolist() # Convierto la columna x4 en una lista 31 elementos

# Aquí almaceno la "y" deseada para cada fila
yd = dataset['y'].tolist()

def generar_constantes(min_rango=0.0, max_rango=1.0):
    return [round(random.uniform(min_rango, max_rango), 2) for i in range(5)]
    
def calcular_y_deseada(x1, x2, x3, x4, constantes):
    a, b, c, d, e = constantes
    return [round(a + b*x1[i] + c*x2[i] + d*x3[i] + e*x4[i], 2) for i in range(len(x1))]

def calcular_error(y_deseada, y_calculada):
    return [abs(y_deseada[i] - y_calculada[i]) for i in range(len(y_deseada))]

def calcular_norma_error(error):
     return round(np.linalg.norm(error), 2)
 
