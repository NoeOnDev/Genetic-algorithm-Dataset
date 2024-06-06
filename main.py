import pandas as pd
import random
import numpy as np
import os
import re
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, ttk

# Aquí cargo el dataset
dataset = pd.read_excel('2024.05.22 dataset 8A.xlsx')

# Aquí almaceno las variables de entrada para cada fila
x1 = dataset['x1'].tolist() # Convierto la columna x1 en una lista (31 elementos)
x2 = dataset['x2'].tolist() # Convierto la columna x2 en una lista (31 elementos)
x3 = dataset['x3'].tolist() # Convierto la columna x3 en una lista (31 elementos)
x4 = dataset['x4'].tolist() # Convierto la columna x4 en una lista (31 elementos)

# Aquí almaceno la "y" deseada para cada fila
yd = dataset['y'].tolist() # Convierto la columna y en una lista (31 elementos)

def generar_constantes(min_rango=0.0, max_rango=1.0):
    return [round(random.uniform(min_rango, max_rango), 2) for i in range(5)]
    
def calcular_y_deseada(x1, x2, x3, x4, constantes):
    a, b, c, d, e = constantes
    return [round(a + b*x1[i] + c*x2[i] + d*x3[i] + e*x4[i], 2) for i in range(len(x1))]

def calcular_error(y_deseada, y_calculada):
    return [abs(y_deseada[i] - y_calculada[i]) for i in range(len(y_deseada))]

def calcular_norma_error(error):
    return round(np.linalg.norm(error), 2)

def algoritmo_genetico():
    probabilidad_mutacion_individual = float(p_mutacion.get())
    probabilidad_mutacion_gen = float(p_mutaciong.get())
    tgeneraciones = int(n_generaciones.get())
    max_poblacion = int(poblacion_maxima.get())
    min_poblacion = int(poblacion_minima.get())
    individuos_iniciales = random.randint(min_poblacion, max_poblacion)
    poblacion = []
    generaciones = []
    mejores = []
    errores_menores = []
    promedio_errores = []
    peores = []
    
    for _ in range(individuos_iniciales):
        poblacion.append(generar_constantes())

    for gen in range(tgeneraciones):
        ysc = [calcular_y_deseada(x1, x2, x3, x4, individuo) for individuo in poblacion]
        fitnes = [calcular_norma_error(calcular_error(yd, yc)) for yc in ysc]
        mejor_fitnes = min(fitnes)
        mejor_individuo = poblacion[fitnes.index(mejor_fitnes)]
        errores_menores.append(mejor_fitnes)
        mejores.append({'fitness': mejor_fitnes,
                        'error': errores_menores[-1],
                        'constantes': mejor_individuo, 'Generacion': gen + 1})

        crear_grafica(yd, calcular_y_deseada(x1, x2, x3, x4, mejor_individuo), gen + 1)
        cruces = generar_parejas(poblacion)
        nueva_poblacion = [mejor_individuo]
        
        for pareja1, parejas in cruces:
            for pareja2 in parejas:
                hijo1, hijo2 = cruza(pareja1, pareja2)
                nueva_poblacion.append(definir_mutacion(hijo1, probabilidad_mutacion_individual, probabilidad_mutacion_gen))
                if len(nueva_poblacion) < max_poblacion:
                    nueva_poblacion.append(definir_mutacion(hijo2, probabilidad_mutacion_individual, probabilidad_mutacion_gen))
        
        promedio_errores.append(round(sum(fitnes) / len(fitnes), 2))
        peores.append(max(fitnes))
        poblacion = podar(nueva_poblacion, max_poblacion)
        generaciones.append(poblacion)

    mostrar_tabla(mejores)
    crear_grafica_error(errores_menores, promedio_errores, peores)
    a, b, c, d, e = zip(*[mejor['constantes'] for mejor in mejores])
    crear_graficas_constante(a, b, c, d, e)
    crear_video()

def crear_video():
    img_dir = "imagenes_graficas_generadas"
    video_dir = "video_generado"
    os.makedirs(video_dir, exist_ok=True)
    video_filename = os.path.join(video_dir, 'generations_video.mp4')

    images = [img for img in os.listdir(img_dir) if img.endswith(".png")]
    images.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    frame = cv2.imread(os.path.join(img_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(img_dir, image)))

    cv2.destroyAllWindows()
    video.release()


def crear_grafica_error(norm_errores, promedio_errores, peores):
    plt.figure(figsize=(10, 10))
    plt.plot(norm_errores, color='blue', label='Generacion norma de error')
    plt.plot(promedio_errores, color='black', label='Promedio de |error|')
    plt.plot(peores, color='red', label='Peores de cada generacion')
    plt.title('Norma de errores')
    plt.xlabel('Generaciones')
    plt.ylabel('Norma de error')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    plt.tight_layout()
    plt.show()

def crear_graficas_constante(a, b, c, d, e):
    img_dir = "imagen_constantes"
    os.makedirs(img_dir, exist_ok=True)

    def save_plots(a, b, c, d, e):
        x = range(len(a))
        plt.figure(figsize=(10, 10))
        plt.plot(x, a, color='blue', label='A')
        plt.plot(x, b, color='green', label='B')
        plt.plot(x, c, color='red', label='C')
        plt.plot(x, d, color='gray', label='D')
        plt.plot(x, e, color='black', label='E')
        plt.title('Constantes')
        plt.xlabel('Generación')
        plt.ylabel('Valor constante')
        plt.grid(True)
        plt.legend()
        filename = f"{img_dir}/constantes.png"
        plt.savefig(filename)
        plt.show()
    
    save_plots(a, b, c, d, e)
    
def generar_nombre_archivo_generacion(num_generacion):
    return f"generation_{num_generacion:03d}.png"

def crear_grafica(yd, fx, i):
    img_dir = "imagenes_graficas_generadas"
    os.makedirs(img_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 10))
    plt.plot(fx, color='green', label='Resultado obtenido')
    plt.plot(yd, color='black', label='Resultado deseado')
    plt.scatter(range(len(fx)), fx, color='green', s=100, label='Resultados obtenidos')
    plt.scatter(range(len(yd)), yd, color='black', s=20, label='Resultados deseados')
    plt.title(f'Generación {i}')
    plt.xlabel('Cantidad de generaciones')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    plt.tight_layout()
    filename = os.path.join(img_dir, generar_nombre_archivo_generacion(i))
    plt.savefig(filename)
    plt.close()

def mostrar_tabla(mejores):
    for item in treeview.get_children():
        treeview.delete(item)
    for mejor in mejores:
        treeview.insert("", "end", values=(mejor['fitness'], mejor['Generacion'], mejor['error'], 
                                           ':'.join(map(str, mejor['constantes']))))

def cruza(pareja1, pareja2):
    posicion = random.randint(1, len(pareja1) - 1)
    hijo1 = pareja1[:posicion] + pareja2[posicion:]
    hijo2 = pareja2[:posicion] + pareja1[posicion:]
    return hijo1, hijo2

def mutacion(individuo, pmutacion):
    nuevo = individuo.copy()
    muto = False
    while not muto:
        for i, constante in enumerate(nuevo):
            if random.random() < pmutacion:
                nuevo[i] = round(constante * (1 + np.random.normal(0, 0.4)), 2)
                muto = True
    return nuevo

def definir_mutacion(hijo, probabilidad_mutacion_individual, probabilidad_mutacion_gen):
    if random.random() < probabilidad_mutacion_individual:
        hijo = mutacion(hijo, probabilidad_mutacion_gen)
    return hijo

def podar(poblacion, max_individuos):
    return poblacion[:max_individuos]

def generar_parejas(poblacion):
    parejas_cruce = []
    for i in range(len(poblacion)):
        cantidad_parejas = random.randint(1, len(poblacion) - 1)
        parejas = random.sample(poblacion[:i] + poblacion[i + 1:], cantidad_parejas)
        parejas_cruce.append((poblacion[i], parejas))
    return parejas_cruce

def mostrar_ventana():
    global ventana, p_mutacion, p_mutaciong, n_generaciones, poblacion_maxima, poblacion_minima, treeview
    ventana = Tk()
    ventana.title("Ingrese valores")
    
    Label(ventana, text="Población mínima:").grid(row=1, column=0)
    Label(ventana, text="Población máxima:").grid(row=2, column=0)
    Label(ventana, text="Valor de probabilidad de mutación del individuo:").grid(row=3, column=0)
    Label(ventana, text="Valor de probabilidad de mutación del gen:").grid(row=4, column=0)
    Label(ventana, text="Generaciones:").grid(row=5, column=0)

    poblacion_minima = Entry(ventana)
    poblacion_maxima = Entry(ventana)
    p_mutacion = Entry(ventana)
    p_mutaciong = Entry(ventana)
    n_generaciones = Entry(ventana)
    
    poblacion_minima.grid(row=1, column=1)
    poblacion_maxima.grid(row=2, column=1)
    p_mutacion.grid(row=3, column=1)
    p_mutaciong.grid(row=4, column=1)
    n_generaciones.grid(row=5, column=1)

    Button(ventana, text="Aceptar", command=algoritmo_genetico).grid(row=6, column=0, columnspan=3)
    
    treeview = ttk.Treeview(ventana, columns=("Fitness", "Generación", "Error", "Constantes"), show="headings")
    treeview.heading("Fitness", text="Fitness")
    treeview.heading("Generación", text='Generación')
    treeview.heading("Error", text="Error")
    treeview.heading("Constantes", text="Constantes")
    treeview.grid(row=7, column=0, columnspan=3)

    ventana.mainloop()

mostrar_ventana()
