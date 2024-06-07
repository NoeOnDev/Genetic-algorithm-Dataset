import pandas as pd
import random
import numpy as np
import os
import re
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, ttk

dataset = pd.read_excel('2024.05.22 dataset 8A.xlsx')

x1 = dataset['x1'].tolist()
x2 = dataset['x2'].tolist()
x3 = dataset['x3'].tolist()
x4 = dataset['x4'].tolist()
yd = dataset['y'].tolist()

def generar_constantes(min_rango=-5, max_rango=5):
    return [random.uniform(min_rango, max_rango) for _ in range(5)]

def calcular_y_deseada(x1, x2, x3, x4, constantes):
    a, b, c, d, e = constantes
    return [a + b*x1[i] + c*x2[i] + d*x3[i] + e*x4[i] for i in range(len(x1))]

def calcular_error(y_deseada, y_calculada):
    return [abs(y_deseada[i] - y_calculada[i]) for i in range(len(y_deseada))]

def calcular_norma_error(error):
    return np.linalg.norm(error)

# Algoritmo Genético
def algoritmo_genetico():
    probabilidad_mutacion_individual = float(p_mutacion.get())
    probabilidad_mutacion_gen = float(p_mutaciong.get())
    cantidad_generaciones = int(n_generaciones.get())
    poblacion_maxima = int(poblacion_max.get())
    poblacion_minima = int(poblacion_min.get())
    individuos_iniciales = random.randint(poblacion_minima, poblacion_maxima)
    
    poblacion = [generar_constantes() for _ in range(individuos_iniciales)]
    generaciones = []
    mejores = []
    errores_menores = []
    promedio_errores = []
    peores = []

    for gen in range(cantidad_generaciones):
        ysc = [calcular_y_deseada(x1, x2, x3, x4, individuo) for individuo in poblacion]
        fitnes = [calcular_norma_error(calcular_error(yd, yc)) for yc in ysc]
        mejor_fitnes = min(fitnes)
        mejor_individuo = poblacion[fitnes.index(mejor_fitnes)]
        errores_menores.append(mejor_fitnes)
        mejores.append({'fitness': mejor_fitnes, 'error': errores_menores[-1], 'constantes': mejor_individuo, 'Generacion': gen + 1})

        crear_grafica(yd, calcular_y_deseada(x1, x2, x3, x4, mejor_individuo), gen + 1)
        cruces = generar_parejas(poblacion)
        nueva_poblacion = [mejor_individuo]

        for pareja1, parejas in cruces:
            for pareja2 in parejas:
                hijo1, hijo2 = cruza(pareja1, pareja2)
                nueva_poblacion.append(definir_mutacion(hijo1, probabilidad_mutacion_individual, probabilidad_mutacion_gen))
                if len(nueva_poblacion) < poblacion_maxima:
                    nueva_poblacion.append(definir_mutacion(hijo2, probabilidad_mutacion_individual, probabilidad_mutacion_gen))

        promedio_errores.append(round(sum(fitnes) / len(fitnes), 2))
        peores.append(max(fitnes))
        poblacion = podar(nueva_poblacion, poblacion_maxima)
        generaciones.append(poblacion)

    mostrar_tabla(mejores)
    crear_grafica_error(errores_menores, promedio_errores, peores)
    a, b, c, d, e = zip(*[mejor['constantes'] for mejor in mejores])
    crear_graficas_constante(a, b, c, d, e)
    crear_video()

# Funciones genéticas
def cruza(pareja1, pareja2):
    posicion = random.randint(1, len(pareja1) - 1)
    hijo1 = pareja1[:posicion] + pareja2[posicion:]
    hijo2 = pareja2[:posicion] + pareja1[posicion:]
    return hijo1, hijo2

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

def generar_parejas(poblacion):
    parejas_cruce = []
    for i in range(len(poblacion)):
        cantidad_parejas = random.randint(1, len(poblacion) - 1)
        parejas = random.sample(poblacion[:i] + poblacion[i + 1:], cantidad_parejas)
        parejas_cruce.append((poblacion[i], parejas))
    return parejas_cruce

def mostrar_tabla(mejores):
    for item in treeview.get_children():
        treeview.delete(item)
    for mejor in mejores:
        treeview.insert("", "end", values=(mejor['fitness'], mejor['Generacion'], mejor['error'], ':'.join(map(str, mejor['constantes']))))

# Funciones para gráficos y video
def crear_video():
    img_dir = "imagenes_graficas_generadas"
    video_dir = "video_generado"
    os.makedirs(video_dir, exist_ok=True)
    video_filename = os.path.join(video_dir, 'generations_video.mp4')

    images = [img for img in os.listdir(img_dir) if img.endswith(".png")]
    images.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    frame = cv2.imread(os.path.join(img_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 3, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(img_dir, image)))

    cv2.destroyAllWindows()
    video.release()

def crear_grafica_error(norm_errores, promedio_errores, peores):
    plt.figure(figsize=(12, 8))
    plt.plot(norm_errores, color='blue', label='Mejores de cada generacion')
    plt.plot(promedio_errores, color='black', label='Promedio de cada generacion')
    plt.plot(peores, color='red', label='Peores de cada generacion')
    plt.title('Evolución de las aptitudes de la población')
    plt.xlabel('Generación')
    plt.ylabel('Aptitud de la población')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    plt.tight_layout()
    plt.show()

def crear_graficas_constante(a, b, c, d, e):
    img_dir = "imagen_constantes"
    os.makedirs(img_dir, exist_ok=True)

    def save_plots(a, b, c, d, e):
        x = range(len(a))
        plt.figure(figsize=(12, 8))
        plt.scatter(x, a, color='blue', label='')
        plt.plot(x, a, color='blue', label='A')
        plt.scatter(x, b, color='green', label='')
        plt.plot(x, b, color='green', label='B')
        plt.scatter(x, c, color='red', label='')
        plt.plot(x, c, color='red', label='C')
        plt.scatter(x, d, color='gray', label='')
        plt.plot(x, d, color='gray', label='D')
        plt.scatter(x, e, color='black', label='')
        plt.plot(x, e, color='black', label='E')
        plt.title('Evolución de los parámetros')
        plt.xlabel('Generación')
        plt.ylabel('Parámetros del mejor individuo')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"{img_dir}/constantes.png"
        plt.savefig(filename)
        plt.show()

    save_plots(a, b, c, d, e)

def generar_nombre_archivo_generacion(num_generacion):
    return f"generation_{num_generacion:03d}.png"

def crear_grafica(yd, fx, i):
    img_dir = "imagenes_graficas_generadas"
    os.makedirs(img_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
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

# Interfaz gráfica
def mostrar_ventana():
    global ventana, p_mutacion, p_mutaciong, n_generaciones, poblacion_max, poblacion_min, treeview
    ventana = Tk()
    ventana.title("Ingrese valores")

    estilo = ttk.Style()
    estilo.configure("TLabel", font=("Arial", 14))
    estilo.configure("TButton", font=("Arial", 14))
    estilo.configure("TEntry", font=("Arial", 14))
    estilo.configure("Treeview.Heading", font=("Arial", 14, "bold"))
    estilo.configure("Treeview", font=("Arial", 12))

    Label(ventana, text="Población mínima:", font=("Arial", 14)).grid(row=1, column=0, padx=10, pady=10)
    Label(ventana, text="Población máxima:", font=("Arial", 14)).grid(row=2, column=0, padx=10, pady=10)
    Label(ventana, text="Probabilidad de mutación del individuo:", font=("Arial", 14)).grid(row=3, column=0, padx=10, pady=10)
    Label(ventana, text="Probabilidad de mutación del gen:", font=("Arial", 14)).grid(row=4, column=0, padx=10, pady=10)
    Label(ventana, text="Generaciones:", font=("Arial", 14)).grid(row=5, column=0, padx=10, pady=10)

    poblacion_min = Entry(ventana, font=("Arial", 14))
    poblacion_max = Entry(ventana, font=("Arial", 14))
    p_mutacion = Entry(ventana, font=("Arial", 14))
    p_mutaciong = Entry(ventana, font=("Arial", 14))
    n_generaciones = Entry(ventana, font=("Arial", 14))

    poblacion_min.grid(row=1, column=1, padx=10, pady=10)
    poblacion_max.grid(row=2, column=1, padx=10, pady=10)
    p_mutacion.grid(row=3, column=1, padx=10, pady=10)
    p_mutaciong.grid(row=4, column=1, padx=10, pady=10)
    n_generaciones.grid(row=5, column=1, padx=10, pady=10)

    Button(ventana, text="Iniciar", command=algoritmo_genetico, font=("Arial", 14)).grid(row=6, column=0, columnspan=3, pady=20)

    treeview = ttk.Treeview(ventana, columns=("Fitness", "Generación", "Error", "Constantes"), show="headings", height=10, selectmode="browse")
    treeview.heading("Fitness", text="Fitness")
    treeview.heading("Generación", text='Generación')
    treeview.heading("Error", text="Error")
    treeview.heading("Constantes", text="Constantes")
    treeview.grid(row=7, column=0, columnspan=3, padx=0)

    ventana.geometry("801x600")
    ventana.mainloop()

# Iniciar la interfaz gráfica
mostrar_ventana()
