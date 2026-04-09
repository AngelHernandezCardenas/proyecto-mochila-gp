import operator                                  # Operadores matematicos basicos
import sys                                       # Para manejo de rutas
import os                                        # Para obtener rutas del sistema
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) # Agregar ruta actual al path
import numpy as np                               # Manejo de arreglos y matematicas

try:
    from deap import base, creator, tools, gp        # Libreria principal de evolucion (GP)
except ImportError:
    print("Advertencia: DEAP no está instalado. Instale con: pip install deap")
    
try:
    from sklearn.cluster import KMeans               # Libreria para el amontonamiento inteligente
except ImportError:
    print("Advertencia: scikit-learn no está instalado. Instale con: pip install scikit-learn")

# 1. Definicion de funciones matematicas seguras
def div_segura(izq, der):                        # Evita divisiones por cero en los arboles
    if der == 0:                                 # Si el denominador (peso) es cero
        return 1.0                               # Retorna un valor neutral para no quebrar el programa
    return izq / der                             # Retorna la division normal

# 2. Configuracion de la Programacion Genetica (GP)
pset = gp.PrimitiveSet("MAIN", 3)                # Define 3 variables de entrada para la IA
pset.addPrimitive(operator.add, 2)               # Agrega operacion de suma al arbol
pset.addPrimitive(operator.sub, 2)               # Agrega operacion de resta al arbol
pset.addPrimitive(operator.mul, 2)               # Agrega operacion de multiplicacion al arbol
pset.addPrimitive(div_segura, 2)                 # Agrega operacion de division segura al arbol
pset.renameArguments(ARG0='P', ARG1='W', ARG2='PW') # Renombra variables a Profit, Weight y Ratio

creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Le indica al sistema que busque maximizar
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax) # Crea la estructura del arbol

toolbox = base.Toolbox()                         # Inicia la caja de herramientas de DEAP
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3) # Generador de arboles aleatorios
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) # Molde del individuo
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Molde de la poblacion
toolbox.register("compile", gp.compile, pset=pset) # Traductor de arbol a codigo Python ejecutable

# 3. Funcion de Evaluacion de la Heuristica (Simulacion de la Mochila)
def evaluar_heuristica(individuo, elementos, capacidad): # Funcion que pone a prueba la regla generada
    rutina = toolbox.compile(expr=individuo)     # Convierte la regla de la IA en una funcion real
    elementos_puntuados = []                     # Lista para guardar las calificaciones de los objetos
    
    for p, w in elementos:                       # Recorre cada elemento disponible en la instancia
        ratio = p / w if w > 0 else 0            # Calcula la proporcion valor/peso
        puntuacion = rutina(p, w, ratio)         # La IA califica el elemento usando su propia formula
        elementos_puntuados.append((puntuacion, p, w)) # Guarda la calificacion, la ganancia y el peso
        
    elementos_puntuados.sort(reverse=True, key=lambda x: x[0]) # Ordena los elementos del mejor al peor
    
    ganancia_total = 0                           # Inicia la ganancia de la mochila en cero
    peso_actual = 0                              # Inicia el peso de la mochila en cero
    
    for _, p, w in elementos_puntuados:       # Intenta empaquetar en el orden que dicto la IA
        if peso_actual + w <= capacidad:         # Si el objeto aun cabe en la mochila
            ganancia_total += p                  # Suma la ganancia
            peso_actual += w                     # Suma el peso
            
    return ganancia_total,                       # Retorna la ganancia final (en formato tupla para DEAP)

# 4. Operadores Evolutivos
toolbox.register("select", tools.selTournament, tournsize=3) # Metodo de seleccion de padres (Torneo)
toolbox.register("mate", gp.cxOnePoint)          # Metodo de cruce (Intercambio de ramas del arbol)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset) # Metodo de mutacion aleatoria

# 5. Modulo K-Means y Bucle Principal
def clasificar_y_evolucionar(instancias):        # Funcion principal que une Clustering y GP
    caracteristicas = []                         # Lista para extraer rasgos matematicos de los problemas
    for elementos, _cap in instancias:            # Recorre las bases de datos de prueba
        promedio_p = np.mean([e[0] for e in elementos]) # Extrae la ganancia promedio de la instancia
        promedio_w = np.mean([e[1] for e in elementos]) # Extrae el peso promedio de la instancia
        caracteristicas.append([promedio_p, promedio_w]) # Guarda los rasgos en un vector
        
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) # Configura el agrupamiento en 2 clusters
    etiquetas = kmeans.fit_predict(caracteristicas) # Ejecuta K-Means y etiqueta cada problema
    
    print("Agrupamiento K-Means completado. Etiquetas:", etiquetas) # Muestra los grupos creados
    
    # Prueba de evaluacion para un grupo especifico
    poblacion = toolbox.population(n=20)         # Crea una poblacion inicial de 20 reglas aleatorias
    problema_prueba = instancias[0]              # Toma el primer problema matematico para probar
    elementos = problema_prueba[0]               # Extrae los objetos
    capacidad = problema_prueba[1]               # Extrae el limite de peso
    
    for ind in poblacion:                        # Ciclo para medir el rendimiento de cada regla
        toolbox.register("evaluate", evaluar_heuristica, elementos=elementos, capacidad=capacidad)
        ind.fitness.values = toolbox.evaluate(ind) # Asigna la calificacion basada en la ganancia
        
    mejores = tools.selBest(poblacion, k=1)      # Selecciona al mejor individuo de la generacion
    mejor_regla = mejores[0]                     # Extrae la formula matematica ganadora
    
    print("La mejor hiper-heuristica generada es:", mejor_regla) # Imprime la regla en consola
    return mejor_regla                           # Finaliza y retorna la regla

# 6. Datos de Prueba (Instancias Simples de la Mochila)
if __name__ == "__main__":      # Bloque de ejecucion principal
    # Formato: [ [ (Ganancia1, Peso1), (Ganancia2, Peso2), (GananciaX, PesoX),... ], Capacidad ]
    instancia_1 = [[(10, 5), (40, 4), (30, 6), (50, 3)], 10] # Instancia tipo 1
    instancia_2 = [[(15, 2), (20, 5), (25, 8), (10, 1)], 12] # Instancia tipo 2
    
    base_de_datos = [instancia_1, instancia_2]   # Agrupa las instancias
    clasificar_y_evolucionar(base_de_datos)      # Inicia el algoritmo completo