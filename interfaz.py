# Description: Este archivo contiene la interfaz gráfica del algoritmo genético.

from tkinter import *
from tkinter import ttk
from algoritmo_genetico import algoritmo_genetico

root = Tk()
root.title("ALGORITMO GENÉTICO")

class DataInfo:
    def __init__(self, pob_inicial, pob_max, resolucion, lim_inf, lim_sup, mut_ind, mut_gen,num_generaciones, problema):
        self.pob_inicial = pob_inicial
        self.pob_max = pob_max
        self.resolucion = resolucion
        self.lim_inf = lim_inf
        self.lim_sup = lim_sup
        self.mut_ind = mut_ind
        self.mut_gen = mut_gen
        self.num_generaciones = num_generaciones
        self.problema = problema

mainframe = ttk.Frame(root, padding="20 20 50 50")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
for i in range(15):
    root.rowconfigure(i, weight=1)
    for j in range(3):
        mainframe.columnconfigure(j, weight=1) 

def save_data():
    p_inicial_value = pob_inicial.get()
    p_max_value = pob_max.get()
    resolucion_value = resolucion.get()
    lim_inf_value = lim_inf.get()
    lim_sup_value = lim_sup.get()
    mut_ind_value = mut_ind.get()
    mut_gen_value = mut_gen.get()
    num_generaciones_value = num_generaciones.get()
    problema_value = combobox_var.get()
    data = DataInfo(p_inicial_value, p_max_value, resolucion_value, lim_inf_value, lim_sup_value, mut_ind_value, mut_gen_value, num_generaciones_value, problema_value)
    algoritmo_genetico(data)        

pob_inicial = StringVar()
ttk.Label(mainframe, text="Poblacion inicial:").grid(column=1, row=1)
ttk.Spinbox(mainframe, textvariable=pob_inicial).grid(column=3, row=1) 

pob_max = StringVar()
ttk.Label(mainframe, text="Poblacion maxima:").grid(column=1, row=2)
ttk.Spinbox(mainframe, textvariable=pob_max).grid(column=3, row=2)

lim_inf = StringVar()
ttk.Label(mainframe, text="Limite inferior:").grid(column=1, row=3)
ttk.Spinbox(mainframe, textvariable=lim_inf).grid(column=3, row=3)

lim_sup = StringVar()
ttk.Label(mainframe, text="Limite superior:").grid(column=1, row=4)
ttk.Spinbox(mainframe, textvariable=lim_sup).grid(column=3, row=4)

resolucion = StringVar()
ttk.Label(mainframe, text="Resolucion:").grid(column=1, row=5)
ttk.Spinbox(mainframe, textvariable=resolucion).grid(column=3, row=5)

num_generaciones = StringVar()
ttk.Label(mainframe, text="Generaciones:").grid(column=1, row=6)
ttk.Spinbox(mainframe, textvariable=num_generaciones).grid(column=3, row=6)

mut_gen = StringVar()
ttk.Label(mainframe, text="P. mutación del gen:").grid(column=1, row=7)
ttk.Spinbox(mainframe, textvariable=mut_gen).grid(column=3, row=7)

mut_ind = StringVar()
ttk.Label(mainframe, text="P. mutación del individuo:").grid(column=1, row=8)
ttk.Spinbox(mainframe, textvariable=mut_ind).grid(column=3, row=8)

ttk.Label(mainframe, text="selecciones uno:").grid(column=1, row=10, sticky=W)
combobox_var = StringVar(value="Minimizacion")
combobox=ttk.Combobox(mainframe, values=["Maximizacion","Minimizacion"],textvariable=combobox_var, state='readonly')
combobox.grid(column=3, row=10, sticky=W)

ttk.Button(mainframe, text="Iniciar", command=save_data).grid(column=2, row=12, sticky=W)

for child in mainframe.winfo_children():
    child.grid_configure(pady=5)
    if isinstance(child, ttk.Label):
        child.grid_configure(padx=10)

root.update()

window_width = root.winfo_reqwidth() + 50
window_height = root.winfo_reqheight() + 50
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_coordinate = int((screen_width - window_width) / 2)
y_coordinate = int((screen_height - window_height) / 2)

root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

root.mainloop()