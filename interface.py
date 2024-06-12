from tkinter import Tk, Label, Entry, Button, ttk
from main import algoritmo_genetico

def mostrar_tabla(mejores):
    for item in treeview.get_children():
        treeview.delete(item)
    for mejor in mejores:
        treeview.insert("", "end", values=(mejor['Generacion'], mejor['error'], ':'.join(map(str, mejor['constantes']))))

def iniciar_algoritmo():
    algoritmo_genetico(p_mutacion.get(), p_mutaciong.get(), n_generaciones.get(), poblacion_max.get(), poblacion_min.get(), mostrar_tabla)

def mostrar_ventana():
    global ventana, p_mutacion, p_mutaciong, n_generaciones, poblacion_max, poblacion_min, treeview
    ventana = Tk()
    ventana.title("Algoritmo Genético individual 8A")

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

    Button(ventana, text="Iniciar", command=iniciar_algoritmo, font=("Arial", 14)).grid(row=6, column=0, columnspan=3, pady=20)

    treeview = ttk.Treeview(ventana, columns=("Generación", "Error", "Constantes"), show="headings", height=20, selectmode="browse")
    treeview.heading("Generación", text='Generación')
    treeview.heading("Error", text="Error")
    treeview.heading("Constantes", text="Constantes")
    treeview.column("Generación", width=150)
    treeview.column("Error", width=200)
    treeview.column("Constantes", width=830)
    treeview.grid(row=7, column=0, columnspan=3, padx=0, pady=20)

    ventana.geometry("1182x765")
    ventana.mainloop()

if __name__ == "__main__":
    mostrar_ventana()
