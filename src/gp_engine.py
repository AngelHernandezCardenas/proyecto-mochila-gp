import operator
import random

import multiprocessing
import numpy as np
import pandas as pd
from deap import base, creator, tools, gp, algorithms
from knapsack import Item, KnapsackState, KnapsackInstance

# Identificador de la iteracion final (Se actualiza en el bucle principal)
FASE_ACTUAL = 1

# Definicion de funciones matematicas seguras para el arbol
def div_segura(izq, der):
    if der == 0:
        return 1.0 # Previene la division por cero asignando un valor neutral
    return izq / der

# Configuracion de la Programacion Genetica (GP)
pset = gp.PrimitiveSet("MAIN", 3) # Define tres variables de entrada para la formula generada
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div_segura, 2)
pset.renameArguments(ARG0='P', ARG1='W', ARG2='PW')

# Creacion de las estructuras para maximizar la aptitud (Fitness)
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Indica que el objetivo es Maximizar el valor
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax) # Crea el molde del arbol genético

# Registro de herramientas fundamentales de DEAP
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Funcion de evaluacion de la hiper-heuristica (Simulacion de empaquetado en una instancia)
def evaluar_hiper_heuristica(individuo, instancia): 
    #Evalúa una regla matemática generada por GP contra una instancia específica.
    #Retorna el Fitness ajustado (Ganancia - Penalización).
    
    rutina_puntuacion = toolbox.compile(expr=individuo) # Compila el árbol en una funcion ejecutable
    mochila = KnapsackState(capacity=instancia.capacity)
    items_puntuados = []

    for item in instancia.items:
        # La IA califica el objeto usando su formula generada (P, W, PW)
        puntuacion = rutina_puntuacion(item.profit, item.weight, item.ratio)
        items_puntuados.append((puntuacion, item))

    items_puntuados.sort(key=lambda x: x[0], reverse=True) # Ordena por puntuación descendente

    # Empaquetado fisico respetando la capacidad de la mochila
    for puntuacion, item in items_puntuados:
        mochila.pack(item)

    # Penalizacion por longitud del árbol (Bloat penalty): Controla el consumo de memoria y complejidad
    penalizacion_longitud = len(individuo) * 0.01

    return mochila.current_profit - penalizacion_longitud, # Retorna la ganancia ajustada


# Operadores Geneticos y control de crecimiento
toolbox.register("select", tools.selTournament, tournsize=3) # Selecciona padres mediante torneo
toolbox.register("mate", gp.cxOnePoint) # Realiza el cruce
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset) # Introduce variabilidad genética

# Limita la altura máxima del árbol en cruce y mutación para controlar el Bloat estructural
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8)) 
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

# Generador Estocastico de bases de datos (Simula el entorno cambiante)
def generar_base_datos_aleatoria(num_instancias=10, num_objetos=50):
    #Genera un lote masivo y aleatorio de problemas de mochila para la prueba.
    instancias = []
    for i in range(num_instancias):
        capacidad = random.uniform(50.0, 150.0) 
        objetos = []
        for j in range(num_objetos):
            peso = random.uniform(1.0, 20.0)
            ganancia = random.uniform(10.0, 100.0)
            objetos.append(Item(j, peso, ganancia))
        instancias.append(KnapsackInstance(f"Inst_{i}", capacidad, objetos))
    return instancias

# Función principal del motor evolutivo (Sin KMeans; enfocado en la robustez de las múltiples instancias)
def clasificar_y_evolucionar(lista_instancias, generaciones=20):
    #Ejecuta el proceso evolutivo sobre un conjunto completo de instancias para obtener un fitness promedio robusto."
    
    print("Iniciando evolucion con enfoque multi-instancia sin diagnostico previo.")

    # Configuracion de las estadisticas: Captura Promedio, Max_Ganancia y Desviacion Estándar (Smoothed Analysis)
    estadisticas = tools.Statistics(lambda ind: ind.fitness.values[0]) # Mide el rendimiento promedio de la poblacion
    estadisticas.register("Promedio", np.mean) 
    estadisticas.register("Max_Ganancia", np.max) # Rastrea la mejor regla encontrada hasta la fecha
    estadisticas.register("Desviacion", np.std) # Mide la estabilidad y homogeneizacion de las reglas en la poblacion

    # Herramienta para guardar al mejor individuo historico (Elitismo)
    salon_fama = tools.HallOfFame(1) 

    # PROCESO DE VALIDACIÓN ROBUSTA POR INSTANCIA
    print("Evaluando el rendimiento de cada individuo contra todas las instancias.")
    resultados_por_individuo = {} # Diccionario para guardar los resultados de fitness en todas las instancias.

    for instancia in lista_instancias:
        # Registra la funcion de evaluacion por esta instancia específica
        toolbox.register("evaluate", evaluar_hiper_heuristica, instancia=instancia) 
        
        # Evalua cada individuo contra esta instancia para recolectar resultados individuales
        results = []
        for ind in toolbox.population(n=50): # Evaluamos toda la poblacion contra esta única instancia.
            fitness_tuple, = evaluar_hiper_heuristica(ind, instancia) # Calculo manual y directo de fitness.
            results.append(fitness_tuple) 

        # Nota: Si se desea registrar el rendimiento en este punto, se haría aquí.
    
    print("Evaluacion masiva de instancias completada.")

    # Ejecucion del ciclo generacional completo usando eaSimple para la optimizacion final.
    poblacion_final, bitacora = algorithms.eaSimple(
        toolbox.population(n=50), toolbox, cxpb=0.7, mutpb=0.2, ngen=generaciones, 
        stats=estadisticas, halloffame=salon_fama, verbose=False # Ejecucion del ciclo de generaciones sin logs en consola
    )

    # Análisis Post-Ejecución y Logging de Resultados
    print("\n Resultados guardados del analisis")
    mejor_individuo = salon_fama[0] # El individuo ganador es la mejor regla encontrada.
    mejores_reglas = {} 
    # Se guarda la regla ganadora generalizada de la fase actual.
    mejores_reglas[FASE_ACTUAL] = str(mejor_individuo) 

    # Automatizacion de exportacion a CSV usando Pandas para la trazabilidad
    df_log = pd.DataFrame(bitacora)
    df_log.to_csv(f"Fase{FASE_ACTUAL}_bitacora_global.csv", index=False)
    print(f"Log de la convergencia guardado en: Fase{FASE_ACTUAL}_bitacora_global.csv")

    # Guardado automatico de las formulas ganadoras en archivo de texto plano
    with open(f"Fase{FASE_ACTUAL}_mejores_reglas.txt", "w") as archivo_texto:
        archivo_texto.write(f"REPORTE DE FORMULAS EVOLUTIVAS - FASE {FASE_ACTUAL}\n")
        for cluster_id, regla in mejores_reglas.items():
            archivo_texto.write(f"Mejor regla encontrada:\n{regla}\n\n")

    return mejores_reglas # Retorna las reglas ganadoras de la fase


# Bloque de ejecucion principal automatizado
if __name__ == "__main__":
    for nueva_fase in range(1, 11): # Bucle configurado para iniciar en Fase 1 y correr hasta 10.
        print(f" INICIANDO SIMULACIÓN DE FASE {nueva_fase}")
        FASE_ACTUAL = nueva_fase # Actualiza la variable global de seguimiento

        # Generacion de datos estocásticos masivos para el experimento (10 instancias x 50 objetos)
        base_de_datos = generar_base_datos_aleatoria(num_instancias=10, num_objetos=50)

        # Ejecucion del motor: Ahora usa la funcionalidad de múltiples instancias por fase.
        clasificar_y_evolucionar(base_de_datos, generaciones=20)

    print("Todas las iteraciones simuladas han finalizado exitosamente.")

import numpy as np
import pandas as pd
from deap import base, creator, tools, gp, algorithms
# DEAP: base/creator = estructuras, tools = operadores,

# Parámetros principales de la simulación
NUM_FASES       = 10   # entornos distintos a recorrer
GEN_POR_FASE    = 5    # generaciones por fase
TAM_POBLACION   = 50   # individuos por generación
TAM_ELITE       = 10   # semilla transferida entre fases
MAX_TREE_HEIGHT = 8    # límite de altura, controla bloat
NUM_INSTANCIAS  = 10   # instancias knapsack por fase
NUM_OBJETOS     = 50   # objetos por instancia

FASE_ACTUAL = 1

# Clases del problema knapsack
class Item:
    # Objeto candidato: peso, ganancia y ratio P/W (argumento PW del árbol)
    def __init__(self, id_item, weight, profit):
        self.id     = id_item
        self.weight = weight
        self.profit = profit
        self.ratio  = profit / weight

class KnapsackState:
    # Estado de la mochila: acumula peso y ganancia al empacar
    def __init__(self, capacity):
        self.capacity       = capacity
        self.current_weight = 0.0
        self.current_profit = 0.0

    def pack(self, item):
        if self.current_weight + item.weight <= self.capacity:
            self.current_weight += item.weight
            self.current_profit += item.profit

class KnapsackInstance:
    # Problema completo: una mochila con su lista de objetos
    def __init__(self, id_instancia, capacity, items):
        self.id       = id_instancia
        self.capacity = capacity
        self.items    = items

# Configuración del árbol GP: 4 operadores binarios + 3 argumentos (P, W, PW)
def div_segura(izq, der):
    return izq / der if abs(der) > 1e-6 else 1.0

pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div_segura,   2)
pset.renameArguments(ARG0='P', ARG1='W', ARG2='PW')  # P=profit, W=weight, PW=ratio

# Estructuras DEAP: fitness de maximización e individuo árbol GP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Toolbox: generación, compilación y operadores evolutivos
toolbox = base.Toolbox()
toolbox.register("expr",       gp.genHalfAndHalf,  pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate,   creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat,    list, toolbox.individual)
toolbox.register("compile",    gp.compile,          pset=pset)

# Selección torneo, cruce de subárboles, mutación uniforme
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate",   gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# Decoradores: descartan árboles que excedan MAX_TREE_HEIGHT
toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))

def evaluar_robusto(individuo, lista_instancias):
    # Fitness = ganancia promedio sobre todas las instancias, penalizado por tamaño del árbol
    try:
        rutina_puntuacion = toolbox.compile(expr=individuo)
    except Exception:
        return (-np.inf,)

    ganancias = []
    for instancia in lista_instancias:
        mochila = KnapsackState(capacity=instancia.capacity)
        items_puntuados = []
        for item in instancia.items:
            try:
                score = rutina_puntuacion(item.profit, item.weight, item.ratio)
                items_puntuados.append((score, item))
            except Exception:
                continue
        items_puntuados.sort(key=lambda x: x[0], reverse=True)
        for _, item in items_puntuados:
            mochila.pack(item)
        ganancias.append(mochila.current_profit)

    if not ganancias:
        return (-np.inf,)

    penalizacion = len(individuo) * 0.01
    return (np.mean(ganancias) - penalizacion,)

def registrar_evaluador(lista_instancias):
    # Vincula evaluador a las instancias de la fase actual
    toolbox.register("evaluate", evaluar_robusto, lista_instancias=lista_instancias)

def generar_base_datos_aleatoria(num_instancias=NUM_INSTANCIAS, num_objetos=NUM_OBJETOS):
    # Genera instancias aleatorias para simular entorno cambiante
    instancias = []
    for i in range(num_instancias):
        capacidad = random.uniform(50.0, 150.0)
        objetos   = [Item(j, random.uniform(1.0, 20.0), random.uniform(10.0, 100.0))
                     for j in range(num_objetos)]
        instancias.append(KnapsackInstance(f"Inst_{i}", capacidad, objetos))
    return instancias

def clasificar_y_evolucionar(lista_instancias, generaciones=GEN_POR_FASE, elite_anterior=None):
    global FASE_ACTUAL

    # Seeding: elite heredada + nuevos individuos para diversidad
    if elite_anterior:
        poblacion = [toolbox.clone(ind) for ind in elite_anterior]
        while len(poblacion) < TAM_POBLACION:
            poblacion.append(toolbox.individual())
    else:
        poblacion = toolbox.population(n=TAM_POBLACION)

    registrar_evaluador(lista_instancias)

    # Invalidar fitness heredado: nuevo entorno, nueva evaluación
    for ind in poblacion:
        del ind.fitness.values

    estadisticas = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else -np.inf)
    estadisticas.register("Promedio",     np.mean)
    estadisticas.register("Max_Ganancia", np.max)
    estadisticas.register("Desviacion",   np.std)
    salon_fama = tools.HallOfFame(TAM_ELITE)

    # Iteraciones generacionales: evaluar → seleccionar → cruzar → mutar
    poblacion_final, bitacora = algorithms.eaSimple(
        poblacion, toolbox,
        cxpb=0.7, mutpb=0.2,
        ngen=generaciones,
        stats=estadisticas,
        halloffame=salon_fama,
        verbose=True
    )

    # Exportar bitácora y mejor regla de la fase
    df_log = pd.DataFrame(bitacora)
    df_log.insert(0, "fase", FASE_ACTUAL)
    df_log.to_csv(f"Fase{FASE_ACTUAL:02d}_bitacora.csv", index=False)

    mejor = salon_fama[0]
    with open(f"Fase{FASE_ACTUAL:02d}_mejor_regla.txt", "w") as f:
        f.write(f"Fase {FASE_ACTUAL} - Mejor hiper-heuristica\n")
        f.write(str(mejor) + "\n\n")
        f.write(f"Fitness: {mejor.fitness.values[0]:.4f}\n")
        f.write(f"Nodos: {len(mejor)}  |  Altura: {mejor.height}\n")

    return list(salon_fama)

if __name__ == "__main__":
    import glob, os

    # Eliminar archivos de ejecuciones anteriores
    patrones = ["Fase*_bitacora.csv", "Fase*_mejor_regla.txt", "resumen_completo.csv"]
    for patron in patrones:
        for archivo in glob.glob(patron):
            os.remove(archivo)

    pool_elite = None

    # Bucle principal: una fase = un entorno nuevo + ciclo evolutivo
    for nueva_fase in range(1, NUM_FASES + 1):
        FASE_ACTUAL   = nueva_fase
        base_de_datos = generar_base_datos_aleatoria()
        pool_elite    = clasificar_y_evolucionar(
            base_de_datos,
            generaciones=GEN_POR_FASE,
            elite_anterior=pool_elite
        )

    # Consolidar bitácoras de todas las fases
    frames = []
    for f in range(1, NUM_FASES + 1):
        try:
            frames.append(pd.read_csv(f"Fase{f:02d}_bitacora.csv"))
        except FileNotFoundError:
            pass

    if frames:
        resumen = pd.concat(frames, ignore_index=True)
        resumen.to_csv("resumen_completo.csv", index=False)
        print(resumen.groupby("fase")["Max_Ganancia"].max().to_string())

    print("\nMejor individuo final:")
    print(pool_elite[0])

