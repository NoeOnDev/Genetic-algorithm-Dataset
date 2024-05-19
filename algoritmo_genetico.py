import math
import os
import random
from sympy import symbols, lambdify
import matplotlib.pyplot as plt   
import cv2
import numpy as np


class DNA:
    delta = 0
    numeroBits = 1
    rango = 0
    numeroRango = 0
    resolucion = 0
    limiteInferior = 0
    limiteSuperior = 0
    poblacionInicial = 0
    poblacionMaxima = 0
    tipoProblema = ""
    poblacionGeneral = []
    probMutacionInd = 0
    probMutacionGen = 0
    num_generaciones = 0 

formula = "((x**3*(log(0.1 + abs(x**2))+3*cos(x)))/x**2+1)" 
 
class Individuo:
    identificador = 0
    def __init__(self, binario, i, x, y):
        Individuo.identificador += 1
        self.id = Individuo.identificador
        self.binario = binario
        self.i = i
        self.x = round(x, 4)
        self.y = round(y, 4)
    def __str__(self):
        return f"id: {self.id}, i: {self.i}, num.binario: {self.binario}, posición en X: {self.x}, posición en Y: {self.y}"

class Estadisticas:
    promedio = []
    peor_individuo = []
    mejor_individuo = []

    @classmethod
    def agregar_promedio(cls, generacion, promedio):
        cls.promedio.append((generacion, promedio))

    @classmethod
    def agregar_mejor_individuo(cls, generacion, mejor_individuo):
        cls.mejor_individuo.append((generacion, mejor_individuo))

    @classmethod
    def agregar_peor_individuo(cls, generacion, peor_individuo):
        cls.peor_individuo.append((generacion, peor_individuo))

def ordenar_por_generacion(filename):
    return int(filename.split('_')[-1].split('.')[0])

def crear_video():
    folder_path = 'generation_plots'
    output_folder = 'video_output'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_array = []
    for filename in sorted(os.listdir(folder_path), key=ordenar_por_generacion):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path)
            img_array.append(img)

    height, width, layers = img_array[0].shape
    video_path = os.path.join(output_folder, 'generation_video.avi')
    
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 2, (width, height))

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

    print(f"Video creado en: {video_path}")

def plot_generation(generation):
    plt.clf()
    plt.xlim(DNA.limiteInferior, DNA.limiteSuperior) 
    plt.title(f'Generacion {generation}')
    plt.xlabel('X')
    plt.ylabel('Y')
 
    x_values = [individuo.x for individuo in DNA.poblacionGeneral]
    y_values = [individuo.y for individuo in DNA.poblacionGeneral]

    plt.scatter(x_values, y_values, label="individuos", s=90, c="#45aaf2", alpha=0.4)
    
    if DNA.tipoProblema == "Maximizacion":
        mejorIndividuoY = max(DNA.poblacionGeneral, key=lambda individuo:individuo.y)
        mejorIndividuoX = mejorIndividuoY.x
        peorIndividuoY = min (DNA.poblacionGeneral, key=lambda individuo:individuo.y)
        peorIndividuoX = peorIndividuoY.x
        
        x_func =np.linspace(DNA.limiteInferior, DNA.limiteSuperior, 200)
        x = symbols ('x')
        expresion = lambdify(x, formula, 'numpy')
        y_func = expresion(x_func)
        plt.plot(x_func, y_func)
        
        plt.scatter(mejorIndividuoX, mejorIndividuoY.y, c='green', label='Mejor Individuo', s=90)
        plt.scatter(peorIndividuoX, peorIndividuoY.y, c='red', label='Peor Individuo', s=90)
    else:
        mejorIndividuoY = min(DNA.poblacionGeneral, key=lambda individuo:individuo.y)
        mejorIndividuoX = mejorIndividuoY.x
        peorIndividuoY = max (DNA.poblacionGeneral, key=lambda individuo:individuo.y)
        peorIndividuoX = peorIndividuoY.x
        
        x_func =np.linspace(DNA.limiteInferior, DNA.limiteSuperior, 200)
        x = symbols ('x')
        expresion = lambdify(x, formula, 'numpy')
        y_func = expresion(x_func)
        plt.plot(x_func, y_func)
        
        plt.scatter(mejorIndividuoX, mejorIndividuoY.y, c='green', label='Mejor Individuo', s=90)
        plt.scatter(peorIndividuoX, peorIndividuoY.y,  c='red', label='Peor Individuo', s=90)

    plt.legend()

    folder_path = 'generation_plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, f'generation_{generation}.png'))
    plt.close()
    
def calcular_x(num_generado):
    num_generado = max(min(num_generado, DNA.numeroRango), 0)
    valor_x = DNA.limiteInferior + num_generado * DNA.delta
    valor_x = max(min(valor_x, DNA.limiteSuperior), DNA.limiteInferior)
    return valor_x

def primerPoblacion():
        for i in range(DNA.poblacionInicial):
            num_generado = (random.randint(1, DNA.numeroRango))
            num_generado_binario = (bin(num_generado)[2:]).zfill(DNA.numeroBits)
            valor_x = calcular_x(num_generado)
            valor_y = calcular_funcion(formula, valor_x)
            individuo = Individuo(i=num_generado, binario=num_generado_binario, x=valor_x, y= valor_y)
            DNA.poblacionGeneral.append(individuo)

def calcular_funcion(funcion, valor_x):
    x = symbols('x')
    expresion = lambdify(x, funcion, 'numpy')
    resultado = expresion(valor_x)
    return resultado

def calculoDatos():
    DNA.rango = DNA.limiteSuperior - DNA.limiteInferior
    saltos = DNA.rango/DNA.resolucion
    puntos = saltos + 1
    numeroBits = math.log2(puntos)
    numeroBits = math.ceil(numeroBits)
    DNA.delta = DNA.rango/((2**numeroBits)-1)
    
    DNA.numeroRango= 2**numeroBits  
    DNA.numeroBits = numeroBits
    
def algoritmo_genetico(data):
    DNA.poblacionInicial = int(data.pob_inicial)
    DNA.poblacionMaxima = int(data.pob_max)
    DNA.resolucion = float(data.resolucion)
    DNA.limiteInferior = float(data.lim_inf)
    DNA.limiteSuperior = float(data.lim_sup)
    DNA.probMutacionInd = float(data.mut_ind)
    DNA.probMutacionGen = float(data.mut_gen)
    DNA.tipoProblema = data.problema
    DNA.num_generaciones = int(data.num_generaciones)
    
    calculoDatos()
    primerPoblacion()
    
    for generacion in range(1, DNA.num_generaciones + 1):
        print(f"\ngeneracion {generacion}:")
        inicializar(generacion)
        plot_generation(generacion)
        podar()

    for individuo in DNA.poblacionGeneral:
        print(f"id: {individuo.id}, i: {individuo.i}, num.binario: {individuo.binario}, el punto en X: {individuo.x}, el punto en Y: {individuo.y}")
    plot_stats()
    
    crear_video() 

def inicializar(generacion):
    mejor_ind_act, peor_ind_act = optimizar()

    Estadisticas.agregar_mejor_individuo(generacion, mejor_ind_act)
    Estadisticas.agregar_peor_individuo(generacion, peor_ind_act)

    suma_y = sum(individuo.y for individuo in DNA.poblacionGeneral)
    promedio = suma_y / len(DNA.poblacionGeneral)
    Estadisticas.agregar_promedio(generacion, promedio)

def optimizar():
    bandera = True
    if DNA.tipoProblema == "Minimizacion":
        bandera = False
    ordenIndividuos = sorted(DNA.poblacionGeneral, key=lambda x: x.y, reverse=bandera)
    
    mitad = int(len(ordenIndividuos) / 2)
    mejor_aptitud = ordenIndividuos[:mitad] 
    menor_aptitud = ordenIndividuos[mitad:]
    
    resto_poblacion = []
    for individuo in menor_aptitud:
        resto_poblacion.append(individuo)
        
    emparejar(resto_poblacion, mejor_aptitud)
    
    return mejor_aptitud[0], resto_poblacion[-1]

def plot_stats():
    generaciones = [generacion for generacion, _ in Estadisticas.mejor_individuo]
    mejores_y = [mejor_individuo.y for _, mejor_individuo in Estadisticas.mejor_individuo]
    peores_y = [peor_individuo.y for _, peor_individuo in Estadisticas.peor_individuo]
    promedio_y = [promedio for _, promedio in Estadisticas.promedio]

    plt.plot(generaciones, mejores_y, label='Mejor Individuo')
    plt.plot(generaciones, peores_y, label='Peor Individuo')
    plt.plot(generaciones, promedio_y, label='Promedio')

    plt.title('Evolución del fitness')
    plt.xlabel('Generación')
    plt.ylabel('Valor de la Función Objetivo')
    plt.legend()

    folder_path = 'stats_plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, 'population_stats.png'))
    plt.close()
    
def emparejar(resto_poblacion, mejor_aptitud):
    new_poblation = []
    for individuo in resto_poblacion:
        new_poblation.append(individuo)
    
    for individuo in mejor_aptitud:
        new_poblation.append(individuo)

    for mejor_individuo in mejor_aptitud:
        for individuo in resto_poblacion:
            
            new_individuo1, new_individuo2 = cruzar(mejor_individuo, individuo)
            new_poblation.append(new_individuo1)
            new_poblation.append(new_individuo2)

def cruzar(mejor_individuo, individuo): 
    puntoDeCruza = int(DNA.numeroBits / 2)
    
    p1 = mejor_individuo.binario[:puntoDeCruza]
    p2 = mejor_individuo.binario[puntoDeCruza:]
    p3 = individuo.binario[:puntoDeCruza]
    p4 = individuo.binario[puntoDeCruza:]
    
    new_individuo_1 = p1 + p4
    new_individuo_2 = p3 + p2
    
    if(random.randint(1,100))/100 <= DNA.probMutacionInd:
        new_individuo_1 = mutar(new_individuo_1)
        
    if(random.randint(1,100))/100 <= DNA.probMutacionInd:
        new_individuo_2 = mutar(new_individuo_2)
    
    nuevosIndividuos(new_individuo_1, new_individuo_2)
    
    return new_individuo_1, new_individuo_2

def mutar(individuo):
    binarioSeparado = list(individuo)
    
    for i in range(len(binarioSeparado)):
        if (random.randint(1,100))/100 <= DNA.probMutacionGen:
            binarioSeparado[i] = '1' if binarioSeparado[i] == '0' else '0'
    new_binario = ''.join(binarioSeparado)
    
    return new_binario

def nuevosIndividuos(individuo1, individuo2):
    numero_decimal1 = int(individuo1, 2)
    numero_decimal2 = int(individuo2, 2)
    x1 = DNA.limiteInferior + numero_decimal1*DNA.delta
    x2 = DNA.limiteInferior + numero_decimal2*DNA.delta
    y1 = calcular_funcion(formula, x1)
    y2 = calcular_funcion(formula, x2)
    
    individuo1 = Individuo(i=numero_decimal1, binario=individuo1, x=x1, y= y1)
    individuo2 = Individuo(i=numero_decimal2, binario=individuo2, x=x2, y= y2)
    
    DNA.poblacionGeneral.append(individuo1)
    DNA.poblacionGeneral.append(individuo2)
    
def podar():
    poblacionUnica = []
    iConjunta = set()

    for individuo in DNA.poblacionGeneral:
        if individuo.i not in iConjunta:
            iConjunta.add(individuo.i)
            poblacionUnica.append(individuo)

    for individuo in poblacionUnica:
        print(individuo)
        
    DNA.poblacionGeneral = poblacionUnica

    bandera = True
    if DNA.tipoProblema == "Minimizacion":
        bandera = False
    individuos_ordenados = sorted(DNA.poblacionGeneral, key=lambda x: x.y, reverse=bandera)
    
    nuevaPoblacion = [individuos_ordenados[0]] 
    
    if len(individuos_ordenados) > 1:
        nuevaPoblacion.extend(random.sample(individuos_ordenados[1:], min(len(individuos_ordenados)-1, DNA.poblacionMaxima -1)))

    DNA.poblacionGeneral = nuevaPoblacion
    
    print("Población después de la poda:")
    for individuo in DNA.poblacionGeneral:
        print(individuo) 