import os
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import cv2

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

def funcion_aptitud(cadena_binaria, x_min, x_max, maximizar=True):
    x = binario_a_decimal(cadena_binaria, x_min, x_max)
    aptitud = x * np.cos(x)
    return aptitud if maximizar else -aptitud

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

def guardar_grafica(generacion, poblacion, aptitudes, x_min, x_max, directorio):
    xs = [binario_a_decimal(ind, x_min, x_max) for ind in poblacion]
    mejores = np.argmax(aptitudes)
    peores = np.argmin(aptitudes)
    
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
    parser = argparse.ArgumentParser(description='Algoritmo Genético para maximizar y minimizar funciones 2D.')
    parser.add_argument('--tam_poblacion', type=int, default=3, help='Tamaño de la población inicial')
    parser.add_argument('--tam_max_poblacion', type=int, default=50, help='Tamaño máximo de la población')
    parser.add_argument('--x_min', type=float, default=-10, help='Límite inferior de x')
    parser.add_argument('--x_max', type=float, default=40, help='Límite superior de x')
    parser.add_argument('--tasa_mutacion_individuo', type=float, default=0.7, help='Probabilidad de mutación del individuo')
    parser.add_argument('--tasa_mutacion_gen', type=float, default=0.6, help='Probabilidad de mutación del gen')
    parser.add_argument('--generaciones', type=int, default=100, help='Número de generaciones')
    parser.add_argument('--maximizar', action='store_true', help='Maximizar la función en lugar de minimizarla')
    parser.add_argument('--directorio_graficas', type=str, default='graficas', help='Directorio para guardar las gráficas')
    parser.add_argument('--directorio_evolucion', type=str, default='evolucion', help='Directorio para guardar la gráfica de evolución')
    parser.add_argument('--output_video', type=str, default='evolucion.mp4', help='Nombre del archivo de salida del video')
    
    args = parser.parse_args()

    crear_directorio(args.directorio_graficas)
    crear_directorio(args.directorio_evolucion)

    poblacion = inicializar_poblacion(args.tam_poblacion, args.x_min, args.x_max)
    aptitud = [funcion_aptitud(ind, args.x_min, args.x_max, args.maximizar) for ind in poblacion]

    mejor_aptitud_hist = []
    peor_aptitud_hist = []
    promedio_aptitud_hist = []

    for generacion in range(args.generaciones):
        parejas = formar_parejas(poblacion)
        descendencia = crear_descendencia(poblacion, parejas)
        descendencia_mutada = aplicar_mutaciones(descendencia, args.tasa_mutacion_individuo, args.tasa_mutacion_gen)
        aptitud_descendencia = [funcion_aptitud(ind, args.x_min, args.x_max, args.maximizar) for ind in descendencia_mutada]

        poblacion_combinada = poblacion + descendencia_mutada
        aptitud_combinada = aptitud + aptitud_descendencia

        poblacion_unica, indices_unicos = np.unique(poblacion_combinada, return_index=True)
        aptitud_unica = [aptitud_combinada[i] for i in indices_unicos]

        mejor_indice = np.argmax(aptitud_unica) if args.maximizar else np.argmin(aptitud_unica)
        mejor_individuo = poblacion_unica[mejor_indice]

        mejor_aptitud_hist.append(max(aptitud_unica))
        peor_aptitud_hist.append(min(aptitud_unica))
        promedio_aptitud_hist.append(np.mean(aptitud_unica))

        guardar_grafica(generacion, poblacion_unica, aptitud_unica, args.x_min, args.x_max, args.directorio_graficas)

        poblacion, aptitud = podar_poblacion(poblacion_unica, aptitud_unica, args.tam_poblacion, mejor_individuo)

        mejor_x = binario_a_decimal(mejor_individuo, args.x_min, args.x_max)
        print(f"Generación {generacion + 1}: Mejor individuo = {mejor_individuo}, Índice = {mejor_indice}, x = {mejor_x}, Aptitud = {aptitud_unica[mejor_indice]}")

    print("Optimización finalizada.")

    generaciones = list(range(1, args.generaciones + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(generaciones, mejor_aptitud_hist, label='Mejor Aptitud')
    plt.plot(generaciones, peor_aptitud_hist, label='Peor Aptitud')
    plt.plot(generaciones, promedio_aptitud_hist, label='Promedio Aptitud')
    plt.xlabel('Generaciones')
    plt.ylabel('Aptitud')
    plt.title('Evolución de la Aptitud de la Población')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.directorio_evolucion, 'evolucion_aptitud.png'))
    plt.show()
    
    crear_video(args.directorio_graficas, args.output_video)
