
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from tkinter import Tk, Label, Entry, Button, ttk

def prinsipal ():
    pmutacion = float(p_mutacion.get())
    pmutaciong = float(p_mutaciong.get())
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
    X, Y = obtener_variables()
    x1, x2, x3, x4 = X
    for _ in range(individuos_iniciales):
        poblacion.append(crear_individuo())
    
    while tgeneraciones>len(generaciones):
        ysc = [definir_y(individuo, x1, x2, x3, x4) for individuo in poblacion]
        fitnes = [fitness(yc, Y) for yc in ysc]
        mejor_fitnes = min(fitnes)
        mejor_individuo = poblacion[fitnes.index(mejor_fitnes)]
        errores_menores.append(fitnes[fitnes.index(mejor_fitnes)])
        mejores.append({'fitness': fitnes[fitnes.index(mejor_fitnes)],
                        'error': errores_menores[len(errores_menores)-1],
                        'constantes': mejor_individuo, 'Generacion': len(generaciones)+1})

        crear_grafica(Y, definir_y(mejor_individuo, x1, x2, x3, x4), len(generaciones)+1)
        cruces = generar_parejas(poblacion)
        poblacion=[]
        poblacion.append(mejor_individuo)
        for pareja1, parejas in cruces:
            for pareja2 in parejas:
                hijo1, hijo2 = cruza(pareja1, pareja2)
                poblacion.append(definir_mutacion(hijo1, pmutacion, pmutaciong))
                poblacion.append(definir_mutacion(hijo2, pmutacion, pmutaciong))
        promedio_errores.append(round(sum(fitnes) / len(fitnes), 2))
        peores.append(max(fitnes))
        poblacion = podar(poblacion, max_poblacion)
        generaciones.append(poblacion)

    mostrar_tabla(mejores)
    crear_grafica_error(errores_menores, promedio_errores, peores)
    a =[]
    b = []
    c = []
    d = []
    e =[]
    for mejor in mejores:
        a.append(mejor['constantes'][0])
        b.append(mejor['constantes'][1])
        c.append(mejor['constantes'][2])
        d.append(mejor['constantes'][3])
        e.append(mejor['constantes'][4])
    crear_graficas_constante(a, b, c, d, e)

def crear_individuo():
    constantes = [round(float(random.random()), 2),
                  round(float(random.random()), 2),
                  round(float(random.random()), 2),
                  round(float(random.random()), 2),
                  round(float(random.random()), 2)]
    return constantes

def crear_video():
    clip = ImageSequenceClip('gen_images_resultado', fps=1)
    video_filename = 'generations_video.mp4'
    clip.write_videofile(video_filename, codec='libx264')

def obtener_variables():
    file_path = '2024.05.22 dataset 8A.xlsx'
    df = pd.read_excel(file_path)

    # Obtener las columnas específicas
    columnas_interes = ['x1', 'x2', 'x3', 'x4']
    datos = df[columnas_interes]
    resultados = df['y']
    xs = []
    for dato in datos:
        xs.append(datos[dato])
    return xs, resultados

def crear_grafica_error(norm_errores, promedio_errores, peores):
    plt.figure(figsize=(10, 10))
    plt.plot(norm_errores, color='blue', label='Generacion norma de error')
    plt.plot(promedio_errores, color = 'black', label = 'Promedio de |error|')
    plt.plot(peores, color ='red', label = 'Peores de cada generacion')
    plt.title('Norma de errores')
    plt.xlabel('Generaciones')
    plt.ylabel('Norma de error')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    plt.tight_layout()
    plt.show()

def crear_graficas_constante(a, b, c, d, e):
    img_dir = "gen_images_resultado"
    os.makedirs(img_dir, exist_ok=True)

    def save_plots(a, b, c, d, e):
        x_a = np.linspace(0, len(a)-1, len(a))
        x_b = np.linspace(0, len(b)-1, len(b))
        x_c = np.linspace(0, len(c)-1, len(c))
        x_d = np.linspace(0, len(d)-1, len(d))
        x_e = np.linspace(0, len(e)-1, len(e))
        plt.figure(figsize=(8, 6))
        plt.scatter(x_a, a, color='blue', label='A')
        plt.plot(a, color='blue', label='A')
        plt.scatter(x_b, b, color='red', label='B')
        plt.plot(b, color='red', label='A')
        plt.scatter(x_c, c, color='green', label='C')
        plt.plot(c, color='green', label='A')
        plt.scatter(x_d, d, color='black', label='D')
        plt.plot(d, color='black', label='A')
        plt.scatter(x_e, e, color='skyblue', label='E')
        plt.plot(e, color='skyblue', label='A')
        plt.title('Constantes')
        plt.xlabel('Valor constante')
        plt.ylabel('Generación')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"{img_dir}/constantes.png"
        plt.savefig(filename)
        plt.show()
    
    save_plots(a, b, c, d, e)

def crear_grafica(yd, fx, i):
    img_dir = "gen_images_resultado"
    os.makedirs(img_dir, exist_ok=True)
    
    x_yd = np.linspace(0, len(yd)-1, len(yd))
    x_fx = np.linspace(0, len(fx)-1, len(fx))

    plt.plot(fx, color='red', label='Resultado obtenido')
    plt.plot(yd, label='Resultado esperado')
    plt.scatter(x_fx, fx, color='red', s= 100, label='Resultado obtenidos')
    plt.scatter(x_yd, yd, color='blue', s= 20, label='Resultado deseados')
    plt.title(f'Generación {i}')
    plt.xlabel('Numero de generación')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    plt.tight_layout()
    filename = f"{img_dir}/generation{i}.png"
    plt.savefig(filename)
    plt.close()

def mostrar_tabla(mejores):
    for item in treeview.get_children():
        treeview.delete(item)
    for mejor in mejores:
        treeview.insert("", "end", values=(mejor['fitness'], mejor['Generacion'], mejor['error'], 
                                                                    (mejor['constantes'][0],":" , 
                                                                    mejor['constantes'][1],":" , 
                                                                    mejor['constantes'][2],":" , 
                                                                    mejor['constantes'][3],":" , 
                                                                    mejor['constantes'][4])))
        
def definir_y(individuo, x1, x2, x3, x4):
    a, b, c, d, e = individuo
    return round((a+b*x1+c*x2+d*x3+e*x4), 2)

def cruza(pareja1, pareja2):
    posicion = random.randint(1, len(pareja1) - 1)
    hijo1 = pareja1[:posicion] + pareja2[posicion:]
    hijo2 = pareja2[:posicion] + pareja1[posicion:]
    return hijo1, hijo2

def fitness(y, resultado):
    return round(np.linalg.norm(abs(resultado - y)), 2)

def mutacion(individuo, pmutacion):
    nuevo = individuo
    muto = False
    while not muto:
        for i, constante in enumerate(nuevo):
            if random.randint(1, 99) / 100 < pmutacion:
                nuevo[i] = round(constante * (1 + (np.random.normal(0, 0.4))), 2)
                muto = True
    return nuevo

def definir_mutacion(hijo, probabilidad_mutacioni, probabilidad_mutaciong):
    if random.randint(1, 99) / 100 < probabilidad_mutacioni:
        hijo = mutacion(hijo, probabilidad_mutaciong)
    return hijo

def podar(poblacion, max_individuos):
    nueva_poblacion = []
    i = 0
    while len(nueva_poblacion) < max_individuos and i < len(poblacion):
        nueva_poblacion.append(poblacion[i])
        i += 1
    return nueva_poblacion

def generar_parejas(poblacion):
    parejas_cruce = []
    poblacion_indices = list(range(len(poblacion)))
    
    for i in range(len(poblacion)):
        cantidad_parejas = random.randint(0, len(poblacion)-1)
        parejas = random.sample(poblacion_indices[:i] + poblacion_indices[i + 1:], cantidad_parejas)
        parejas_cruce.append((poblacion[i], [poblacion[j] for j in parejas]))
    
    return parejas_cruce

def mostrar_ventana ():
    global ventana
    ventana = Tk()
    ventana.title("Ingrese valores")
    
    Label(ventana, text="Valor de probabilidad de mutación del individuo:").grid(row=1, column=0)
    Label(ventana, text="Valor de probabilidad de mutación del gen:").grid(row=2, column=0)
    Label(ventana, text="Generaciones:").grid(row=3, column=0)
    Label(ventana, text="Población máxima:").grid(row=4, column=0)
    Label(ventana, text="Población mínima:").grid(row=5, column=0)

    global p_mutacion, n_generaciones, poblacion_maxima, poblacion_minima, p_mutaciong, treeview
    p_mutacion = Entry(ventana)
    p_mutaciong = Entry(ventana)
    n_generaciones = Entry(ventana)
    poblacion_maxima = Entry(ventana)
    poblacion_minima = Entry(ventana)

    p_mutacion.grid(row=1, column=1)
    p_mutaciong.grid(row=2, column=1)
    n_generaciones.grid(row=3, column=1)
    poblacion_maxima.grid(row=4, column=1)
    poblacion_minima.grid(row=5, column=1)

    Button(ventana, text="Aceptar", command=prinsipal).grid(row=6, column=0, columnspan=3)

    # Crear un Treeview para la tabla
    treeview = ttk.Treeview(ventana, columns=("Fitness", "Generación", "Error", "constantes"), show="headings")
    treeview.heading("Fitness", text="Fitness")
    treeview.heading("Generación", text='Generación')
    treeview.heading("Error", text="Error")
    treeview.heading("constantes", text="Constantes")
    treeview.grid(row=7, column=0, columnspan=3)

    ventana.mainloop()

mostrar_ventana()
