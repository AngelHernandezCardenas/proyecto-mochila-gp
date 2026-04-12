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
if __name__ == "__main__":
    # Utilizamos exactamente las mismas instancias de prueba que en gp_engine.py
    items_instancia_1 = [Item(1, 5, 10), Item(2, 4, 40), Item(3, 6, 30), Item(4, 3, 50), Item(5, 2, 15)]
    instancia_1 = KnapsackInstance("Instancia_1", 12, items_instancia_1)
    
    items_instancia_2 = [Item(1, 2, 15), Item(2, 5, 20), Item(3, 8, 25), Item(4, 1, 10), Item(5, 4, 30)]
    instancia_2 = KnapsackInstance("Instancia_2", 10, items_instancia_2)
    
    base_de_datos = [instancia_1, instancia_2]
    
    # Ejecutamos el extractor masivo
    evaluar_instancias_baseline(base_de_datos)