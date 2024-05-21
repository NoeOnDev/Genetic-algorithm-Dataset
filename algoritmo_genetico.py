import os
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import six
import tkinter as tk
from tkinter import messagebox

def binario_a_decimal(cadena_binaria, x_min, x_max):
    valor_entero = int(cadena_binaria, 2)
    return x_min + (x_max - x_min) * valor_entero / (2**len(cadena_binaria) - 1)

def decimal_a_binario(valor, x_min, x_max, num_bits=16):
    valor_entero = int((valor - x_min) / (x_max - x_min) * (2**num_bits - 1))
    return format(valor_entero, f'0{num_bits}b')

def inicializar_poblacion(tam_poblacion, x_min, x_max, num_bits=16):
    poblacion = []
    for _ in range(tam_poblacion):
        valor = random.uniform(x_min, x_max)
        cadena_binaria = decimal_a_binario(valor, x_min, x_max, num_bits)
        poblacion.append(cadena_binaria)
    return poblacion

def funcion_aptitud(cadena_binaria, x_min, x_max, minimizar=False):
    x = binario_a_decimal(cadena_binaria, x_min, x_max)
    aptitud = x * np.cos(x)
    return -aptitud if minimizar else aptitud

def formar_parejas(poblacion):
    parejas = []
    tam_poblacion = len(poblacion)
    for i in range(tam_poblacion):
        m = random.randint(0, tam_poblacion - 1)
        companeros = random.sample(range(tam_poblacion), m)
        if i in companeros:
            companeros.remove(i)
        parejas.append((i, companeros))
    return parejas

def cruza(padre1, padre2, num_bits=16):
    num_puntos = random.randint(1, num_bits - 1)
    puntos_cruza = sorted(random.sample(range(1, num_bits), num_puntos))
    hijo1, hijo2 = list(padre1), list(padre2)
    for i in range(0, len(puntos_cruza), 2):
        if i + 1 < len(puntos_cruza):
            hijo1[puntos_cruza[i]:puntos_cruza[i+1]] = padre2[puntos_cruza[i]:puntos_cruza[i+1]]
            hijo2[puntos_cruza[i]:puntos_cruza[i+1]] = padre1[puntos_cruza[i]:puntos_cruza[i+1]]
    return ''.join(hijo1), ''.join(hijo2)

def crear_descendencia(poblacion, parejas, num_bits=16):
    descendencia = []
    for i, companeros in parejas:
        padre1 = poblacion[i]
        for companero in companeros:
            padre2 = poblacion[companero]
            hijo1, hijo2 = cruza(padre1, padre2, num_bits)
            descendencia.append(hijo1)
            descendencia.append(hijo2)
    return descendencia

def mutar(individuo, tasa_mutacion_individuo, tasa_mutacion_gen, num_bits=16):
    if random.random() < tasa_mutacion_individuo:
        individuo = list(individuo)
        for i in range(num_bits):
            if random.random() < tasa_mutacion_gen:
                j = random.randint(0, num_bits - 1)
                individuo[i], individuo[j] = individuo[j], individuo[i]
        individuo = ''.join(individuo)
    return individuo

def aplicar_mutaciones(descendencia, tasa_mutacion_individuo, tasa_mutacion_gen, num_bits=16):
    return [mutar(individuo, tasa_mutacion_individuo, tasa_mutacion_gen, num_bits) for individuo in descendencia]

def podar_poblacion(poblacion, aptitud, tam_max, mejor_individuo):
    poblacion_unica, indices_unicos = np.unique(poblacion, return_index=True)
    aptitud_unica = [aptitud[i] for i in indices_unicos]
    
    mejor_indice = np.argmax(aptitud_unica)
    mejor_individuo = poblacion_unica[mejor_indice]
    
    indices = list(range(len(poblacion_unica)))
    indices.remove(mejor_indice)
    random.shuffle(indices)
    indices = indices[:tam_max-1]
    
    nueva_poblacion = [mejor_individuo] + [poblacion_unica[i] for i in indices]
    nueva_aptitud = [aptitud_unica[mejor_indice]] + [aptitud_unica[i] for i in indices]
    
    return nueva_poblacion, nueva_aptitud

def crear_directorio(directorio):
    if not os.path.exists(directorio):
        os.makedirs(directorio)

def guardar_grafica(generacion, poblacion, aptitudes, x_min, x_max, directorio, minimizar):
    xs = [binario_a_decimal(ind, x_min, x_max) for ind in poblacion]
    mejores = np.argmin(aptitudes) if minimizar else np.argmax(aptitudes)
    peores = np.argmax(aptitudes) if minimizar else np.argmin(aptitudes)
    
    plt.figure()
    plt.scatter(xs, aptitudes, color='lightblue', label='Individuos')
    plt.scatter([xs[mejores]], [aptitudes[mejores]], color='green', label='Mejor Individuo')
    plt.scatter([xs[peores]], [aptitudes[peores]], color='red', label='Peor Individuo')
    plt.xlabel('x')
    plt.ylabel('Aptitud')
    plt.title(f'Generación {generacion + 1}')
    plt.legend()
    plt.grid(True)
    
    filename = os.path.join(directorio, f'grafica_generacion_{generacion + 1}.png')
    plt.savefig(filename)
    plt.close()

def crear_video(directorio, output_file, fps=4):
    images = [img for img in os.listdir(directorio) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    frame = cv2.imread(os.path.join(directorio, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(directorio, image)))

    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic Algorithm for maximizing and minimizing 2D functions.')
    parser.add_argument('--tam_poblacion', type=int, default=10, help='Initial population size')
    parser.add_argument('--tam_max_poblacion', type=int, default=50, help='Maximum population size')
    parser.add_argument('--x_min', type=float, default=-10, help='Lower limit of x')
    parser.add_argument('--x_max', type=float, default=40, help='Upper limit of x')
    parser.add_argument('--tasa_mutacion_individuo', type=float, default=0.7, help='Individual mutation probability')
    parser.add_argument('--tasa_mutacion_gen', type=float, default=0.6, help='Gene mutation probability')
    parser.add_argument('--generaciones', type=int, default=10, help='Number of generations')
    parser.add_argument('--minimizar', action='store_false', help='Minimize the function instead of maximizing it')
    parser.add_argument('--directorio_graficas', type=str, default='graficas', help='Directory to save the graphs')
    parser.add_argument('--directorio_evolucion', type=str, default='evolucion', help='Directory to save the evolution graph')
    parser.add_argument('--output_video', type=str, default='evolucion.mp4', help='Output video file name')
    parser.add_argument('--output_csv', type=str, default='mejores_individuos.csv', help='Output CSV file name')
    
    args = parser.parse_args()

    crear_directorio(args.directorio_graficas)
    crear_directorio(args.directorio_evolucion)

    # Inicialización de la población
    poblacion = inicializar_poblacion(args.tam_poblacion, args.x_min, args.x_max)
    aptitud = [funcion_aptitud(ind, args.x_min, args.x_max, args.minimizar) for ind in poblacion]

    mejor_aptitud_hist = []
    peor_aptitud_hist = []
    promedio_aptitud_hist = []
    mejores_individuos = []

    for generacion in range(args.generaciones):
        parejas = formar_parejas(poblacion)
        descendencia = crear_descendencia(poblacion, parejas)
        descendencia_mutada = aplicar_mutaciones(descendencia, args.tasa_mutacion_individuo, args.tasa_mutacion_gen)
        aptitud_descendencia = [funcion_aptitud(ind, args.x_min, args.x_max, args.minimizar) for ind in descendencia_mutada]

        poblacion_combinada = poblacion + descendencia_mutada
        aptitud_combinada = aptitud + aptitud_descendencia

        poblacion_unica, indices_unicos = np.unique(poblacion_combinada, return_index=True)
        aptitud_unica = [aptitud_combinada[i] for i in indices_unicos]

        mejor_indice = np.argmin(aptitud_unica) if args.minimizar else np.argmax(aptitud_unica)
        mejor_individuo = poblacion_unica[mejor_indice]

        # Guardo la estadisticas antes de podar
        mejor_aptitud_hist.append(min(aptitud_unica) if args.minimizar else max(aptitud_unica))
        peor_aptitud_hist.append(max(aptitud_unica) if args.minimizar else min(aptitud_unica))
        promedio_aptitud_hist.append(np.mean(aptitud_unica) if args.minimizar else np.mean(aptitud_unica))

        # Almaceno al mejor individuo de la generación
        mejor_x = binario_a_decimal(mejor_individuo, args.x_min, args.x_max)
        mejores_individuos.append((generacion + 1, mejor_individuo, mejor_indice, mejor_x, aptitud_unica[mejor_indice]))

        guardar_grafica(generacion, poblacion_unica, aptitud_unica, args.x_min, args.x_max, args.directorio_graficas, args.minimizar)

        poblacion, aptitud = podar_poblacion(poblacion_unica, aptitud_unica, args.tam_poblacion, mejor_individuo)

        print(f"Generación {generacion + 1}: Mejor individuo = {mejor_individuo}, Índice = {mejor_indice}, x = {mejor_x}, Aptitud = {aptitud_unica[mejor_indice]}")

    print("Optimización completada.")

    generaciones = list(range(1, args.generaciones + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(generaciones, mejor_aptitud_hist, label='Mejor Aptitud')
    plt.plot(generaciones, peor_aptitud_hist, label='Peor Aptitud')
    plt.plot(generaciones, promedio_aptitud_hist, label='Aptitud Promedio')
    plt.xlabel('Generaciones')
    plt.ylabel('Aptitud')
    plt.title('Evolución de la Aptitud de la Población')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.directorio_evolucion, 'evolucion_aptitud.png'))
    plt.show()
    
    crear_video(args.directorio_graficas, args.output_video)

    def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=7,
                        header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                        bbox=[0, 0, 1, 1], header_columns=0,
                        ax=None, **kwargs):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors)])
        return ax

    df = pd.DataFrame(mejores_individuos, columns=['Generación', 'Mejor Individuo (Binario)', 'Índice', 'x', 'Aptitud'])
    render_mpl_table(df, header_columns=0, col_width=2.0)
    plt.savefig('tabla.png')
