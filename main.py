# main.py
import pandas as pd
import random
import numpy as np
from graphics import crear_grafica, crear_grafica_error, crear_graficas_constante, crear_video

dataset = pd.read_excel('Dataset01.xlsx')

x1 = dataset['x1'].tolist()
x2 = dataset['x2'].tolist()
x3 = dataset['x3'].tolist()
x4 = dataset['x4'].tolist()
x5 = dataset['x5'].tolist()
yd = dataset['y'].tolist()

def generar_constantes(min_rango=-4, max_rango=8.5):
    return [random.uniform(min_rango, max_rango) for _ in range(6)]

def calcular_y_deseada(x1, x2, x3, x4, x5, constantes):
    a, b, c, d, e, f = constantes
    return [a + b*x1[i] + c*x2[i] + d*x3[i] + e*x4[i] + f*x5[i] for i in range(len(x1))]

def calcular_error(y_deseada, y_calculada):
    return [abs(y_deseada[i] - y_calculada[i]) for i in range(len(y_deseada))]

def calcular_norma_error(error):
    return np.linalg.norm(error)

# Selecciono un punto de cruzamiento y combino partes de dos individuos para crear dos nuevos individuos (hijos).
def cruza(pareja1, pareja2):
    posicion = random.randint(1, len(pareja1) - 1)
    hijo1 = pareja1[:posicion] + pareja2[posicion:]
    hijo2 = pareja2[:posicion] + pareja1[posicion:]
    return hijo1, hijo2

# Aplico mutacion a un individuo con una cierta probabilidad, alterando ligeramente una o mas de las constantes.
def mutacion(individuo, probabilidad_mutacion_gen):
    nuevo = individuo.copy()
    muto = False
    while not muto:
        for i, constante in enumerate(nuevo):
            if random.random() < probabilidad_mutacion_gen:
                nuevo[i] = constante * (1.0 + random.uniform(-100, 100) / 1000)
                muto = True
    return nuevo

def definir_mutacion(hijo, probabilidad_mutacion_individual, probabilidad_mutacion_gen):
    if random.random() < probabilidad_mutacion_individual:
        hijo = mutacion(hijo, probabilidad_mutacion_gen)
    return hijo

def podar(poblacion, poblacion_maxima):
    return poblacion[:poblacion_maxima]

# Aquí genero pares de individuos para el cruzamiento, seleccionando aleatoriamente parejas de la población.
def generar_parejas(poblacion):
    parejas_cruce = []
    for i in range(len(poblacion)):
        cantidad_parejas = random.randint(1, len(poblacion) - 1)
        parejas = random.sample(poblacion[:i] + poblacion[i + 1:], cantidad_parejas)
        parejas_cruce.append((poblacion[i], parejas))
    return parejas_cruce

def algoritmo_genetico(p_mutacion, p_mutaciong, n_generaciones, poblacion_max, poblacion_min, actualizar_tabla):
    probabilidad_mutacion_individual = float(p_mutacion)
    probabilidad_mutacion_gen = float(p_mutaciong)
    cantidad_generaciones = int(n_generaciones)
    poblacion_maxima = int(poblacion_max)
    poblacion_minima = int(poblacion_min)
    individuos_iniciales = random.randint(poblacion_minima, poblacion_maxima)
    
    poblacion = [generar_constantes() for _ in range(individuos_iniciales)]
    generaciones = []
    mejores = []
    errores_menores = []
    promedio_errores = []
    peores = []

    for gen in range(cantidad_generaciones):
        ysc = [calcular_y_deseada(x1, x2, x3, x4, x5, individuo) for individuo in poblacion]
        fitnes = [calcular_norma_error(calcular_error(yd, yc)) for yc in ysc]
        
        # Guardar la información del mejor individuo de cada generación
        mejor_fitnes = min(fitnes)
        mejor_individuo = poblacion[fitnes.index(mejor_fitnes)]
        errores_menores.append(mejor_fitnes)
        mejores.append({'fitness': mejor_fitnes, 'error': errores_menores[-1], 'constantes': mejor_individuo, 'Generacion': gen + 1})

        crear_grafica(yd, calcular_y_deseada(x1, x2, x3, x4, x5, mejor_individuo), gen + 1)
        cruces = generar_parejas(poblacion)
        nueva_poblacion = [mejor_individuo]
        
        # Realizar cruzamientos y posibles mutaciones para crear nueva población
        for pareja1, parejas in cruces:
            for pareja2 in parejas:
                hijo1, hijo2 = cruza(pareja1, pareja2)
                nueva_poblacion.append(definir_mutacion(hijo1, probabilidad_mutacion_individual, probabilidad_mutacion_gen))
                if len(nueva_poblacion) < poblacion_maxima:
                    nueva_poblacion.append(definir_mutacion(hijo2, probabilidad_mutacion_individual, probabilidad_mutacion_gen))

        promedio_errores.append(round(sum(fitnes) / len(fitnes), 2))
        peores.append(max(fitnes))
        
        # Actualizar la población seleccionando los mejores individuos
        poblacion = podar(nueva_poblacion, poblacion_maxima)
        generaciones.append(poblacion)

    # Crear gráficos y video final
    actualizar_tabla([mejores[-1]])
    crear_grafica_error(errores_menores, promedio_errores, peores)
    a, b, c, d, e, f = zip(*[mejor['constantes'] for mejor in mejores])
    crear_graficas_constante(a, b, c, d, e, f)
    crear_video()

