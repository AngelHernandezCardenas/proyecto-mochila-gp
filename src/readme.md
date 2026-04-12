# Framework de Programación Genética y K-Means para el Problema de la Mochila 0/1

Este repositorio contiene el código fuente y la arquitectura algorítmica de mi proyecto de investigación. El objetivo principal es resolver el Problema de la Mochila 0/1 (Knapsack Problem) en entornos de alta complejidad matemática, sustituyendo los algoritmos tradicionales por una **Hiper-heurística Evolutiva Híbrida**.

#Descripción del Proyecto

A diferencia de los enfoques voraces (Greedy) que se estancan en óptimos locales cuando no existe correlación entre el peso y la ganancia de los objetos, este sistema utiliza Inteligencia Artificial para evolucionar sus propias reglas matemáticas. 

El entorno somete a la IA a bases de datos estocásticas (aleatorias) de 50 objetos, obligándola a encontrar soluciones creativas y eficientes mediante selección natural computacional.

#Características Principales
* **Clasificación Inteligente (K-Means)**: El sistema no ataca los problemas a ciegas. Primero agrupa matemáticamente las mochilas en clústeres según sus vectores de peso y ganancia promedio.
* **Evolución Dirigida (Programación Genética)**: Utiliza árboles de sintaxis para crear fórmulas matemáticas (heurísticas) que evalúan qué objetos empacar.
* **Procesamiento Paralelo**: Integración de la librería `multiprocessing` para evaluar múltiples poblaciones genéticas simultáneamente, optimizando el uso del CPU.
* **Penalización Estructural (Efecto Bloat)**: Algoritmo de control de memoria que castiga el tamaño del código, obligando a la IA a diseñar reglas matemáticas cortas y elegantes.

# Entorno de Desarrollo y Tecnologías

Todo el desarrollo, depuración y ejecución de métricas de este proyecto fue realizado de manera local utilizando exclusivamente las siguientes herramientas:
* **Entornos de Desarrollo (IDEs)**: Visual Studio Code y Thonny.
* **Lenguaje Base**: Python 3.10+
* **Librerías Principales**:
  * `deap` (Motor de algoritmos evolutivos)
  * `scikit-learn` (Modelo K-Means para Machine Learning)
  * `pandas` & `numpy` (Manejo de datos masivos y matemáticas estructuradas)
  * `matplotlib` (Renderizado de visualizaciones científicas)

#Estructura del Repositorio

* `knapsack.py`: Modelo de dominio físico. Contiene las leyes inquebrantables del simulador (Objetos, Mochila e Instancias).
* `gp_engine.py`: El núcleo del proyecto. Contiene la configuración de DEAP, el ciclo de multiprocesamiento y la extracción de datos a CSV.
* `heuristics.py`: Algoritmos comparativos de línea base (Greedy) controlados por humanos.
* `analisis_correlacional.py`: Analizador estadístico que demuestra el coeficiente de Pearson nulo del espacio de búsqueda.
* `graficas_avanzadas.py`: Herramienta de renderizado para exportar las curvas de convergencia poblacional y los mapas de agrupamiento vectorial.

##Instalación y Ejecución

Para replicar este entorno de investigación en tu máquina local:

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/AngelHernandezCardenas/proyecto-mochila-gp.git](https://github.com/AngelHernandezCardenas/proyecto-mochila-gp.git)
2. Funciona principalmente en el software Thonny donde fue pensado para funcionar correctamente https://www.bing.com/ck/a?!&&p=f7350b3ceaa286fbcba11d69c0891db987408227166e6f802f90cc08b3684ca6JmltdHM9MTc3NTg2NTYwMA&ptn=3&ver=2&hsh=4&fclid=22358fa5-d0ad-6759-3a94-999fd1046625&psq=Thonny+&u=a1aHR0cHM6Ly90aG9ubnkub3JnLw
3. Si se desea descargar las librerías en Thonny que fueron utilizadas en el proyecto, enteonces hacer lo siguiente:
- Ir a "Herramientas/Tools"
- Dar clic en 'Gestionar paquetes'.
- Buscar 'matplotlib', 'deap' y 'pandas'. Buscar en PyPi y descargar las librerías.