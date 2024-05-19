# Descripción: Este archivo contiene las funciones necesarias para graficar los resultados de las generaciones

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify

from algoritmo_genetico import DNA

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
        peorIndividuoY = min(DNA.poblacionGeneral, key=lambda individuo:individuo.y)
        peorIndividuoX = peorIndividuoY.x
        
        x_func = np.linspace(DNA.limiteInferior, DNA.limiteSuperior, 200)
        x = symbols('x')
        expresion = lambdify(x, DNA.formula, 'numpy')
        y_func = expresion(x_func)
        plt.plot(x_func, y_func)
        
        plt.scatter(mejorIndividuoX, mejorIndividuoY.y, c='green', label='Mejor Individuo', s=90)
        plt.scatter(peorIndividuoX, peorIndividuoY.y, c='red', label='Peor Individuo', s=90)
    else:
        mejorIndividuoY = min(DNA.poblacionGeneral, key=lambda individuo:individuo.y)
        mejorIndividuoX = mejorIndividuoY.x
        peorIndividuoY = max(DNA.poblacionGeneral, key=lambda individuo:individuo.y)
        peorIndividuoX = peorIndividuoY.x
        
        x_func = np.linspace(DNA.limiteInferior, DNA.limiteSuperior, 200)
        x = symbols('x')
        expresion = lambdify(x, DNA.formula, 'numpy')
        y_func = expresion(x_func)
        plt.plot(x_func, y_func)
        
        plt.scatter(mejorIndividuoX, mejorIndividuoY.y, c='green', label='Mejor Individuo', s=90)
        plt.scatter(peorIndividuoX, peorIndividuoY.y, c='red', label='Peor Individuo', s=90)

    plt.legend()

    folder_path = 'generation_plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, f'generation_{generation}.png'))
    plt.close()

def plot_stats(estadisticas):
    generaciones = [generacion for generacion, _ in estadisticas.mejor_individuo]
    mejores_y = [mejor_individuo.y for _, mejor_individuo in estadisticas.mejor_individuo]
    peores_y = [peor_individuo.y for _, peor_individuo in estadisticas.peor_individuo]
    promedio_y = [promedio for _, promedio in estadisticas.promedio]

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
