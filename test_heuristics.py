import sys
import os

# --- INYECCIÓN DE RUTA (Soluciona problemas de Thonny) ---
# Esto le dice a Python: "La carpeta donde está este archivo es la raíz del proyecto"
ruta_raiz = os.path.abspath(os.path.dirname(__file__))
if ruta_raiz not in sys.path:
    sys.path.insert(0, ruta_raiz)
# ---------------------------------------------------------

from src.knapsack import Item, KnapsackState
from src.heuristics import HEURISTIC_MAP

def test_manual_knapsack():
    print("=== INICIANDO PRUEBA DE MOCHILA ===")
    
    mochila = KnapsackState(capacity=15.0)
    
    objetos_disponibles = [
        Item(1, 12.0, 4.0),
        Item(2, 2.0, 2.0),
        Item(3, 1.0, 1.0),
        Item(4, 4.0, 10.0),
        Item(5, 1.0, 2.0)
    ]
    
    print(f"Estado inicial: {mochila}")
    
    heuristica_activa = HEURISTIC_MAP["MaxPW"]
    print("\n--- Seleccionando objetos con la heurística MaxPW ---")
    
    paso = 1
    while True:
        mejor_objeto = heuristica_activa(mochila, objetos_disponibles)
        
        if mejor_objeto is None:
            print("Ya no caben más objetos o se acabaron.")
            break
            
        mochila.pack(mejor_objeto)
        objetos_disponibles.remove(mejor_objeto)
        
        print(f"Paso {paso}: Se empacó el {mejor_objeto}")
        print(f"  -> {mochila}")
        paso += 1

    print("\n=== RESULTADO FINAL ===")
    print(f"Ganancia total: ${mochila.current_profit}")
    print(f"Peso final: {mochila.current_weight} / {mochila.capacity} kg")

if __name__ == "__main__":
    test_manual_knapsack()