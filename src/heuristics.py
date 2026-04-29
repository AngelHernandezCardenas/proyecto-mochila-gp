import pandas as pd
from knapsack import Item, KnapsackState, KnapsackInstance

# 1. Definición de la heurística clásica (Greedy)
def heuristica_max_pw(mochila, objetos):
    # Ordena los objetos por su ratio (ganancia/peso) de mayor a menor
    mejores = sorted(objetos, key=lambda x: x.ratio, reverse=True)
    for obj in mejores:
        if mochila.can_pack(obj):
            return obj
    return None 

# Mapa de heurísticas estáticas disponibles
HEURISTIC_MAP = {
    "MaxPW": heuristica_max_pw
}

# 2. Extractor masivo de línea base (Benchmark)
def evaluar_instancias_baseline(lista_instancias):
    resultados = []
    heuristica_activa = HEURISTIC_MAP["MaxPW"]
    
    print("INICIANDO EXTRACCIÓN DE LÍNEA BASE (HEURÍSTICA VORAZ)")
    
    for instancia in lista_instancias:
        mochila = KnapsackState(capacity=instancia.capacity)
        # Creamos una copia de la lista de objetos para ir eliminándolos al empacar
        objetos_disponibles = list(instancia.items) 
        
        # Bucle de empaquetado para la instancia actual
        while True:
            mejor_objeto = heuristica_activa(mochila, objetos_disponibles)
            if mejor_objeto is None:
                break # Rompe el ciclo si ya no cabe nada
            
            mochila.pack(mejor_objeto)
            objetos_disponibles.remove(mejor_objeto)
            
        # Guardado de métricas por instancia
        resultados.append({
            "Instancia": instancia.instance_id,
            "Capacidad_Max": instancia.capacity,
            "Peso_Ocupado": mochila.current_weight,
            "Ganancia_Total": mochila.current_profit
        })
        print(f"Instancia procesada: {instancia.instance_id} | Ganancia lograda: {mochila.current_profit}")

    # 3. Automatización de reporte CSV mediante Pandas
    df_resultados = pd.DataFrame(resultados)
    nombre_archivo = "baseline_heuristics.csv"
    df_resultados.to_csv(nombre_archivo, index=False)
    
    print(f"\nReporte estadístico guardado exitosamente en: {nombre_archivo}")

# Bloque de ejecución principal
from gp_engine import generar_base_datos_aleatoria # Asegurar que esta función está disponible

if __name__ == "__main__":
    # Generar el set de datos estocástico masivo para una comparación justa
    num_pruebas = 10 # Coincidir con la cantidad usada en gp_engine por fase
    objetos_por_instancia = 50
    base_de_datos = generar_base_datos_aleatoria(num_instancias=num_pruebas, num_objetos=objetos_por_instancia)
    
    # Ejecutamos el extractor masivo
    evaluar_instancias_baseline(base_de_datos)