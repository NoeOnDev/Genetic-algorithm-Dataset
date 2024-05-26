import numpy as np
import random
import tkinter as tk
from tkinter import ttk

class AlgoritmoGeneticoDTO:
    def __init__(self, tam_poblacion, max_poblacion, prob_mutacion_individuo, prob_mutacion_gen, x_min, x_max, num_generaciones, modo):
        self.tam_poblacion = tam_poblacion
        self.max_poblacion = max_poblacion
        self.prob_mutacion_individuo = prob_mutacion_individuo
        self.prob_mutacion_gen = prob_mutacion_gen
        self.x_min = x_min
        self.x_max = x_max
        self.num_generaciones = num_generaciones
        self.modo = modo

def funcion_fitness(x):
    return x * np.cos(x)

def inicializar_poblacion(tam_poblacion, x_min, x_max):
    return np.random.uniform(x_min, x_max, tam_poblacion)

def evaluar_poblacion(poblacion, funcion_fitness):
    return np.array([funcion_fitness(ind) for ind in poblacion])

def seleccionar_parejas(poblacion, n):
    num_individuos = len(poblacion)
    padres = []
    for i in range(num_individuos):
        m = random.randint(0, min(n, num_individuos - 1))
        indices_seleccionados = random.sample(range(num_individuos), m)
        if i in indices_seleccionados:
            indices_seleccionados.remove(i)
        padres.append((i, indices_seleccionados))
    return padres

def cruce(padres, poblacion):
    descendientes = []
    for padre, parejas in padres:
        for pareja in parejas:
            if pareja != padre:
                padre1 = poblacion[padre]
                padre2 = poblacion[pareja]
                num_puntos = random.randint(1, 15)
                puntos = sorted(random.sample(range(1, 16), num_puntos))
                hijo = cruce_bits(padre1, padre2, puntos)
                descendientes.append(hijo)
    return np.array(descendientes)

def cruce_bits(padre1, padre2, puntos):
    bin_padre1 = list(bin(int(padre1) & 0xFFFF)[2:].zfill(16))
    bin_padre2 = list(bin(int(padre2) & 0xFFFF)[2:].zfill(16))
    hijo = bin_padre1.copy()
    for i, punto in enumerate(puntos):
        if i % 2 == 0:
            hijo[punto:] = bin_padre2[punto:]
        else:
            hijo[punto:] = bin_padre1[punto:]
    return int(''.join(hijo), 2)

def mutar(descendientes, prob_mutacion_individuo, prob_mutacion_gen):
    for idx in range(len(descendientes)):
        if np.random.rand() < prob_mutacion_individuo:
            bin_descendiente = list(bin(int(descendientes[idx]) & 0xFFFF)[2:].zfill(16))
            for gen in range(len(bin_descendiente)):
                if np.random.rand() < prob_mutacion_gen:
                    swap_idx = random.randint(0, len(bin_descendiente) - 1)
                    bin_descendiente[gen], bin_descendiente[swap_idx] = bin_descendiente[swap_idx], bin_descendiente[gen]
            descendientes[idx] = int(''.join(bin_descendiente), 2)
    return descendientes

def podar_poblacion(poblacion, descendientes, funcion_fitness):
    nueva_poblacion = np.concatenate((poblacion, descendientes))
    fitness_nueva_poblacion = evaluar_poblacion(nueva_poblacion, funcion_fitness)
    mejor_individuo = nueva_poblacion[np.argmax(fitness_nueva_poblacion)]
    indices = np.random.choice(range(len(nueva_poblacion)), size=len(poblacion) - 1, replace=False)
    nueva_poblacion = nueva_poblacion[indices]
    nueva_poblacion = np.append(nueva_poblacion, mejor_individuo)
    return nueva_poblacion

def algoritmo_genetico(dto):
    poblacion = inicializar_poblacion(dto.tam_poblacion, dto.x_min, dto.x_max)
    for generacion in range(dto.num_generaciones):
        fitness = evaluar_poblacion(poblacion, funcion_fitness)
        
        mejor_individuo_idx = np.argmax(fitness) if dto.modo == 'maximizar' else np.argmin(fitness)
        mejor_individuo = poblacion[mejor_individuo_idx]
        mejor_individuo_bin = bin(int(mejor_individuo) & 0xFFFF)[2:].zfill(16)
        mejor_fitness = fitness[mejor_individuo_idx]
        
        print(f"Generación {generacion}: Mejor individuo: {mejor_individuo_bin} (índice: {mejor_individuo_idx}, x: {mejor_individuo:.2f}, aptitud: {mejor_fitness:.2f})")
        
        padres = seleccionar_parejas(poblacion, dto.max_poblacion)
        descendientes_cruza = cruce(padres, poblacion)
        descendientes_mutados = mutar(descendientes_cruza, dto.prob_mutacion_individuo, dto.prob_mutacion_gen)
        poblacion = podar_poblacion(poblacion, descendientes_mutados, funcion_fitness)
    
    fitness_final = evaluar_poblacion(poblacion, funcion_fitness)
    mejor_solucion_idx = np.argmax(fitness_final) if dto.modo == 'maximizar' else np.argmin(fitness_final)
    return poblacion[mejor_solucion_idx]

def ejecutar_algoritmo():
    tam_poblacion = int(entry_tam_poblacion.get())
    max_poblacion = int(entry_max_poblacion.get())
    prob_mutacion_individuo = float(entry_prob_mutacion_individuo.get())
    prob_mutacion_gen = float(entry_prob_mutacion_gen.get())
    x_min = float(entry_x_min.get())
    x_max = float(entry_x_max.get())
    num_generaciones = int(entry_num_generaciones.get())
    modo = modo_var.get()

    dto = AlgoritmoGeneticoDTO(tam_poblacion, max_poblacion, prob_mutacion_individuo, prob_mutacion_gen, x_min, x_max, num_generaciones, modo)
    mejor_solucion = algoritmo_genetico(dto)
    resultado.set(f"La mejor solución es: {mejor_solucion:.2f} con valor: {funcion_fitness(mejor_solucion):.2f}")

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Algoritmo Genético")

ttk.Label(root, text="Tamaño de Población:").grid(row=0, column=0)
entry_tam_poblacion = ttk.Entry(root)
entry_tam_poblacion.grid(row=0, column=1)

ttk.Label(root, text="Máximo de Población:").grid(row=1, column=0)
entry_max_poblacion = ttk.Entry(root)
entry_max_poblacion.grid(row=1, column=1)

ttk.Label(root, text="Probabilidad de Mutación (Individuo):").grid(row=2, column=0)
entry_prob_mutacion_individuo = ttk.Entry(root)
entry_prob_mutacion_individuo.grid(row=2, column=1)

ttk.Label(root, text="Probabilidad de Mutación (Gen):").grid(row=3, column=0)
entry_prob_mutacion_gen = ttk.Entry(root)
entry_prob_mutacion_gen.grid(row=3, column=1)

ttk.Label(root, text="Valor mínimo de x:").grid(row=4, column=0)
entry_x_min = ttk.Entry(root)
entry_x_min.grid(row=4, column=1)

ttk.Label(root, text="Valor máximo de x:").grid(row=5, column=0)
entry_x_max = ttk.Entry(root)
entry_x_max.grid(row=5, column=1)

ttk.Label(root, text="Número de Generaciones:").grid(row=6, column=0)
entry_num_generaciones = ttk.Entry(root)
entry_num_generaciones.grid(row=6, column=1)

modo_var = tk.StringVar()
modo_var.set('maximizar')
ttk.Label(root, text="Modo:").grid(row=7, column=0)
ttk.Radiobutton(root, text='Maximizar', variable=modo_var, value='maximizar').grid(row=7, column=1)
ttk.Radiobutton(root, text='Minimizar', variable=modo_var, value='minimizar').grid(row=7, column=2)

ttk.Button(root, text="Ejecutar Algoritmo", command=ejecutar_algoritmo).grid(row=8, column=0, columnspan=3)

resultado = tk.StringVar()
ttk.Label(root, textvariable=resultado).grid(row=9, column=0, columnspan=3)

root.mainloop()
