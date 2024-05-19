class DNA:
    poblacionInicial = 0
    poblacionMaxima = 0
    limiteInferior = 0.0
    limiteSuperior = 0.0
    resolucion = 0.0
    delta = 0.0
    numeroBits = 0
    numeroRango = 0
    probMutacionInd = 0.0
    probMutacionGen = 0.0
    tipoProblema = ''
    num_generaciones = 0
    poblacionGeneral = []
    formula = "((x**3*(log(0.1 + abs(x**2))+3*cos(x)))/x**2+1)" 
