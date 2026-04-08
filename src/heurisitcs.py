from knapsack import Item, KnapsackState

# Definimos la heurística para que la prueba pueda funcionar
def heuristica_max_pw(mochila, objetos):
    # Ordena los objetos por su ratio (ganancia/peso) de mayor a menor
    mejores = sorted(objetos, key=lambda x: x.ratio, reverse=True)
    for obj in mejores:
        if mochila.can_pack(obj): # Si el mejor objeto cabe, lo recomienda
            return obj
    return None # Si ya no cabe nada, devuelve None

# Mapa de heurísticas disponibles
HEURISTIC_MAP = {
    "MaxPW": heuristica_max_pw
}

def test_manual_knapsack():
    print("=== INICIANDO PRUEBA DE MOCHILA ===")
    
    # 1. Creamos una mochila con capacidad de 15 kg
    mochila = KnapsackState(capacity=15.0)
    
    # 2. Creamos nuestro "universo" de objetos (ID, Peso, Ganancia)
    objetos_disponibles = [
        Item(1, 12.0, 4.0),   # Objeto pesado, poco valor
        Item(2, 2.0, 2.0),    # Objeto ligero, valor medio
        Item(3, 1.0, 1.0),    # Objeto muy ligero, poco valor
        Item(4, 4.0, 10.0),   # Objeto medio, MUY valioso (El mejor)
        Item(5, 1.0, 2.0)     # Objeto ligero, buen valor
    ]
    
    print(f"Estado inicial: {mochila}")
    
    # 3. Simulamos que la IA eligió la heurística "MaxPW" (Max Profit/Weight)
    heuristica_activa = HEURISTIC_MAP["MaxPW"]
    print("\n--- Seleccionando objetos con la heurística MaxPW ---")
    
    paso = 1
    while True:
        # La heurística analiza el estado y los objetos, y nos recomienda el mejor
        mejor_objeto = heuristica_activa(mochila, objetos_disponibles)
        
        # Si nos devuelve None, significa que ya no cabe nada
        if mejor_objeto is None:
            print("Ya no caben más objetos o se acabaron.")
            break
            
        # Empacamos el objeto
        mochila.pack(mejor_objeto)
        # Lo quitamos de la mesa para no volver a agarrarlo
        objetos_disponibles.remove(mejor_objeto)
        
        print(f"Paso {paso}: Se empacó el {mejor_objeto}")
        print(f"  -> {mochila}")
        paso += 1

    print("\n=== RESULTADO FINAL ===")
    print(f"Ganancia total: ${mochila.current_profit}")
    print(f"Peso final: {mochila.current_weight} / {mochila.capacity} kg")

if __name__ == "__main__":
    test_manual_knapsack()