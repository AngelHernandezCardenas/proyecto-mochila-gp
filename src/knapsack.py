from typing import List

class Item:
    #Representa un objeto individual que puede ser empacado.
    def __init__(self, item_id: int, weight: float, profit: float):
        self.item_id = item_id
        self.weight = weight
        self.profit = profit
        self.ratio = profit / weight if weight > 0 else 0.0

    def __repr__(self):
        return f"Item(id={self.item_id}, p={self.profit}, w={self.weight})"


class KnapsackState:
    #Representa el estado actual de la mochila durante el proceso de llenado."
    def __init__(self, capacity: float):
        self.capacity = capacity
        self.current_weight = 0.0
        self.current_profit = 0.0
        # CORRECCIÓN: Le decimos a Python que esta lista solo aceptará 'Items'
        self.packed_items: List[Item] = []

    @property
    def remaining_capacity(self) -> float:
        return self.capacity - self.current_weight

    def can_pack(self, item: Item) -> bool:
        return self.remaining_capacity >= item.weight

    def pack(self, item: Item) -> bool:
        if self.can_pack(item):
            self.packed_items.append(item)
            self.current_weight += item.weight
            self.current_profit += item.profit
            return True
        return False

    def __repr__(self):
        return f"Mochila(Ganancia={self.current_profit}, Peso={self.current_weight}/{self.capacity}, Ítems={len(self.packed_items)})"

class KnapsackInstance:#Representa una instancia completa del problema.
    # CORRECCIÓN: Tipamos la lista y nos aseguramos de asignar todas las variables
    def __init__(self, instance_id: str, capacity: float, items: List[Item]):
        self.instance_id = instance_id
        self.capacity = capacity
        self.items = items