import operator
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, gp, algorithms

# ─── Parámetros principales de la simulación ────────────────────────────────
NUM_FASES       = 10   # Entornos distintos a recorrer
GEN_POR_FASE    = 5    # Generaciones por fase
TAM_POBLACION   = 50   # Individuos por generación
TAM_ELITE       = 10   # Semilla transferida entre fases
MAX_TREE_HEIGHT = 8    # Límite de altura del árbol, controla bloat
NUM_INSTANCIAS  = 10   # Instancias knapsack por fase
NUM_OBJETOS     = 50   # Objetos por instancia

# ─── Clases del problema Knapsack ───────────────────────────────────────────
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

# ─── Árbol GP: operadores y argumentos ──────────────────────────────────────
def div_segura(izq, der):
    # División protegida contra cero
    return izq / der if abs(der) > 1e-6 else 1.0

pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div_segura,   2)
pset.renameArguments(ARG0='P', ARG1='W', ARG2='PW')  # P=profit, W=weight, PW=ratio

# ─── Estructuras DEAP ───────────────────────────────────────────────────────
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr",       gp.genHalfAndHalf,  pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate,   creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat,    list, toolbox.individual)
toolbox.register("compile",    gp.compile,          pset=pset)

# Selección torneo, cruce de subárboles, mutación uniforme
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate",   gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# Decoradores: descartan árboles que excedan MAX_TREE_HEIGHT (control de bloat)
toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))

# ─── Funciones del motor evolutivo ──────────────────────────────────────────
def evaluar_robusto(individuo, lista_instancias):
    # Fitness = ganancia promedio sobre todas las instancias, penalizado por tamaño del árbol
    try:
        rutina_puntuacion = toolbox.compile(expr=individuo)
    except Exception:
        return (-float('inf'),)

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
        return (-float('inf'),)

    penalizacion = len(individuo) * 0.01
    return (float(np.mean(ganancias)) - penalizacion,)

def registrar_evaluador(lista_instancias):
    # Vincula el evaluador a las instancias de la fase actual
    toolbox.register("evaluate", evaluar_robusto, lista_instancias=lista_instancias)

def generar_base_datos_aleatoria(num_instancias=NUM_INSTANCIAS, num_objetos=NUM_OBJETOS):
    # Genera instancias aleatorias para simular un entorno cambiante
    instancias = []
    for i in range(num_instancias):
        capacidad = random.uniform(50.0, 150.0)
        objetos   = [Item(j, random.uniform(1.0, 20.0), random.uniform(10.0, 100.0))
                     for j in range(num_objetos)]
        instancias.append(KnapsackInstance(f"Inst_{i}", capacidad, objetos))
    return instancias

def clasificar_y_evolucionar(lista_instancias, generaciones=GEN_POR_FASE, elite_anterior=None):
    # Seeding: reutiliza la élite de la fase anterior y completa con nuevos individuos
    if elite_anterior:
        poblacion = [toolbox.clone(ind) for ind in elite_anterior]
        while len(poblacion) < TAM_POBLACION:
            poblacion.append(toolbox.individual())
    else:
        poblacion = toolbox.population(n=TAM_POBLACION)

    registrar_evaluador(lista_instancias)

    # Invalidar fitness heredado: nuevo entorno requiere nueva evaluación
    for ind in poblacion:
        del ind.fitness.values

    estadisticas = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else -float('inf'))
    estadisticas.register("Promedio",     np.mean)
    estadisticas.register("Max_Ganancia", np.max)
    estadisticas.register("Desviacion",   np.std)
    salon_fama = tools.HallOfFame(TAM_ELITE)

    # Ciclo generacional: evaluar → seleccionar → cruzar → mutar
    algorithms.eaSimple(
        poblacion, toolbox,
        cxpb=0.7, mutpb=0.2,
        ngen=generaciones,
        stats=estadisticas,
        halloffame=salon_fama,
        verbose=True
    )

    return list(salon_fama)

# ─── Bloque principal ────────────────────────────────────────────────────────
if __name__ == "__main__":
    pool_elite = None

    for fase in range(1, NUM_FASES + 1):
        print(f"\n── FASE {fase}/{NUM_FASES} ──────────────────────────────────────")
        base_de_datos = generar_base_datos_aleatoria()
        pool_elite    = clasificar_y_evolucionar(
            base_de_datos,
            generaciones=GEN_POR_FASE,
            elite_anterior=pool_elite
        )

    print("\n Todas las fases completadas.")
    print(f" Mejor individuo final:\n{pool_elite[0]}")
    print(f" Fitness: {pool_elite[0].fitness.values[0]:.4f}")
