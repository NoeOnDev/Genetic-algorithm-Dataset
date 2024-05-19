# Description: Este archivo contiene la interfaz gráfica del algoritmo genético.

from tkinter import *
from tkinter import ttk
from algoritmo_genetico import algoritmo_genetico

class GeneticAlgorithmUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Algoritmo Genético")
        self.create_widgets()
        self.configure_grid()
        self.center_window()

    def create_widgets(self):
        self.mainframe = ttk.Frame(self.root, padding="20 20 50 50")
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        self.fields = [
            ("Poblacion inicial:", StringVar()),
            ("Poblacion maxima:", StringVar()),
            ("Limite inferior:", StringVar()),
            ("Limite superior:", StringVar()),
            ("Resolucion:", StringVar()),
            ("Generaciones:", StringVar()),
            ("P. mutación del gen:", StringVar()),
            ("P. mutación del individuo:", StringVar()),
        ]

        for i, (label, var) in enumerate(self.fields, start=1):
            ttk.Label(self.mainframe, text=label).grid(column=1, row=i, sticky=W, padx=10)
            ttk.Spinbox(self.mainframe, textvariable=var).grid(column=3, row=i, sticky=(W, E))

        ttk.Label(self.mainframe, text="Seleccione uno:").grid(column=1, row=len(self.fields)+1, sticky=W, padx=10)
        self.combobox_var = StringVar(value="Minimización")
        self.combobox = ttk.Combobox(self.mainframe, values=["Maximización", "Minimización"], textvariable=self.combobox_var, state='readonly')
        self.combobox.grid(column=3, row=len(self.fields)+1, sticky=(W, E))

        ttk.Button(self.mainframe, text="Iniciar", command=self.save_data).grid(column=2, row=len(self.fields)+2, sticky=W, pady=10)

    def configure_grid(self):
        for i in range(len(self.fields) + 3):
            self.root.rowconfigure(i, weight=1)
            self.mainframe.columnconfigure(i, weight=1)
        
        for child in self.mainframe.winfo_children():
            child.grid_configure(pady=5)
            if isinstance(child, ttk.Label):
                child.grid_configure(padx=10)

    def save_data(self):
        data_values = [var.get() for _, var in self.fields]
        problema_value = self.combobox_var.get()
        
        try:
            data = DataInfo(*data_values, problema_value)
            algoritmo_genetico(data)
        except ValueError as e:
            print(f"Error: {e}")

    def center_window(self):
        self.root.update()
        window_width = self.root.winfo_reqwidth() + 50
        window_height = self.root.winfo_reqheight() + 50
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_coordinate = int((screen_width - window_width) / 2)
        y_coordinate = int((screen_height - window_height) / 2)
        self.root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

class DataInfo:
    def __init__(self, pob_inicial, pob_max, lim_inf, lim_sup, resolucion, num_generaciones, mut_gen, mut_ind, problema):
        self.pob_inicial = int(pob_inicial)
        self.pob_max = int(pob_max)
        self.resolucion = float(resolucion)
        self.lim_inf = float(lim_inf)
        self.lim_sup = float(lim_sup)
        self.mut_ind = float(mut_ind)
        self.mut_gen = float(mut_gen)
        self.num_generaciones = int(num_generaciones)
        self.problema = problema.lower()

        if self.lim_inf >= self.lim_sup:
            raise ValueError("El límite inferior debe ser menor que el límite superior")

if __name__ == "__main__":
    root = Tk()
    app = GeneticAlgorithmUI(root)
    root.mainloop()
