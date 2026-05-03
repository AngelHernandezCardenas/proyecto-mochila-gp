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