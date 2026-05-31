# Optimización Heurística y Bio-inspirada: De Funciones Continuas al Problema del Viajero (TSP)

Este repositorio contiene la implementación modular, visualización dinámica y análisis comparativo para el proyecto de optimización de la asignatura de **Redes Neuronales y Algoritmos Bio-inspirados**. El trabajo abarca desde la exploración de superficies continuas no convexas complejas hasta la resolución de problemas combinatorios NP-difíciles utilizando heurísticas y metaheurísticas de última generación.

---

##  Contenido del Proyecto

El proyecto se divide en dos grandes ejes temáticos y metodológicos:

### 1. Optimización de Funciones Continuas (Parte 1)
Estudio detallado de la convergencia y eficiencia computacional en paisajes de búsqueda complejos con dimensiones 2D y 3D, evaluando:
* **Función de Rosenbrock:** Caracterizada por su valle estrecho, alargado y curvado. Aunque parece sencilla, el gradiente es extremadamente pequeño cerca del mínimo global, lo que ralentiza la convergencia de la mayoría de algoritmos.
* **Función de Schwefel:** Un paisaje altamente no lineal y multimodal con numerosos mínimos locales y colinas oscilantes que ponen a prueba el equilibrio entre exploración y explotación de los optimizadores.

#### Algoritmos Continuos Implementados:
* **Descenso de Gradiente Cuasi-Newton (BFGS):** Enfoque clásico basado en derivadas matemáticas locales para explotación de precisión.
* **Evolución Diferencial (DE):** Algoritmo evolutivo basado en vectores de diferencia mutada para búsquedas globales robustas.
* **Particle Swarm Optimization (PSO):** Heurística bio-inspirada basada en la inteligencia colectiva y dinámica social de enjambres.
* **Algoritmo Evolutivo / Genético (EA):** Metaheurística que simula la selección natural de poblaciones a través de generaciones de individuos (cruzamiento, mutación y selección elitista con PyGAD).

### 2. Problema del Agente Viajero (TSP) y Extracción de Datos (Parte 2)
Solución heurística y bio-inspirada aplicada al problema clásico de optimización combinatoria (TSP) para encontrar la ruta más óptima que visite un conjunto de coordenadas geográficas y retorne al origen, utilizando:
* **Algoritmo de Colonia de Hormigas (Ant Colony Optimization - ACO):** Simulación del rastro de feromonas depositado por hormigas artificiales para descubrir caminos óptimos a través de grafos complejos.
* **Algoritmo Genético (GA):** Evolución cromosómica de rutas codificadas para seleccionar cruces con distancias mínimas acumuladas.
* **Scraper de Datos Reales de Carretera (CAPUFE):** Extracción automatizada y estructurada mediante Selenium y BeautifulSoup de datos reales de casetas, combustible, costos de peaje y tiempos de traslado interurbanos, modelados en matrices reales en la subcarpeta `extracción_datos`.

---

##  Estructura del Repositorio (Windows Tree)

A continuación se detalla la estructura física del proyecto generada a través del comando `tree /F` de Windows (excluyendo entornos virtuales y metadatos de Git):

```text
C:.
│   AntColonyOptimizer.py               # Motor del optimizador de Colonia de Hormigas para el TSP
│   antcolony_new.py                    # Implementación alternativa optimizada de ACO
│   clases.py                           # Lógica principal y clases modulares (SGD, PSO, DE, EA)
│   GeneticAlgorithmOptimizer.py        # Motor del optimizador Genético para el TSP
│   probar_clases.py                    # Script de pruebas globales para validación rápida
│   README.md                           # Documentación principal del repositorio
│   report.log                          # Registros y logs de ejecución de algoritmos
│   requirement.txt                     # Archivo de dependencias de Python
│   resultados_optimizacion_final.csv   # Base de datos con los resultados y métricas recolectadas
│   RunGeneticAlgorithmOptimizer.py     # Orquestador ejecutable del Algoritmo Genético para el TSP
│   ruta_aco_optima.gif                 # Animación del progreso de rutas encontradas por ACO
│   ruta_final.txt                      # Reporte plano de la ruta geográfica definitiva
│   ruta_ga_animada.gif                 # Animación del progreso de rutas encontradas por el GA
│   ruta_ga_estatica.png                # Gráfico estático de convergencia final del GA
│
├───extracción_datos                    # Módulo extractor de información geográfica y vial real
│       matriz_casetas_mxn.csv          # Datos de tarifas de casetas e infraestructura vial
│       matriz_combustible_mxn.csv      # Datos de costos de combustible calculados por tramo
│       matriz_costo_total_mxn.csv      # Matriz combinada de costo total de viaje
│       matriz_tiempo_horas.csv         # Matriz de tiempos reales de conducción interurbana
│       progreso.csv                    # Registro temporal del progreso del scraper
│       scraper_capufe.py               # Script automatizado de extracción de datos de CAPUFE
│
├───Graficas_punto_1                    # Registro visual y resultados gráficos de la Parte 1
│   ├───Algoritmo Evolutivo             # Resultados y animaciones del Algoritmo Genético (PyGAD)
│   │   ├───Rosenbrock
│   │   │       curvas_nivel.png
│   │   │       Evolucion de optimizacion.png
│   │   │       grafica_3d.png
│   │   │       rosenbrock_ea_2d.gif
│   │   │       rosenbrock_ea_3d.gif
│   │   │
│   │   └───Schwefel
│   │           curvas_nivel.png
│   │           Evolucion de optimizacion.png
│   │           grafica_3d.png
│   │           schwefel_ea_2d.gif
│   │           schwefel_ea_3d.gif
│   │
│   ├───Descenso por gradiente          # Resultados del método de descenso clásico (BFGS)
│   │   ├───Rosenbrock
│   │   │       Cruvas_nivel.png
│   │   │       Evolucion de optimizacion.png
│   │   │       Rosenbrock_3d.png
│   │   │       rosenbrock_descenso_2d.gif
│   │   │       rosenbrock_descenso_3d.gif
│   │   │
│   │   └───Schwefel
│   │           animacion_3d_desde_arriba.png
│   │           curvas_nivel.png
│   │           Evolucion de optimizacion.png
│   │           grafica_3d.png
│   │           schwefel_descenso_2d_sgd.gif
│   │           schwefel_descenso_3d_sgd.gif
│   │
│   ├───Enjambre de particulas          # Resultados y simulaciones dinámicas del PSO
│   │   ├───Rosenbrock
│   │   │       animacion_2d.png
│   │   │       curvas_nivel.png
│   │   │       Evolucion de optimizacion.png
│   │   │       grafica_3d.png
│   │   │       rosenbrock_pso_2d.gif
│   │   │       rosenbrock_pso_3d.gif
│   │   │
│   │   └───Schwefel
│   │           animacion_2d.png
│   │           curvas_nivel.png
│   │           Evolucion de optimizacion.png
│   │           grafica_3d.png
│   │           schwefel_pso_2d.gif
│   │           schwefel_pso_3d.gif
│   │
│   └───Evolucion diferencial          # Resultados y simulaciones de Convergencia por DE
│       ├───Rosenbrock
│       │       curvas_nivel.png
│       │       Evolucion de optimizacion.png
│       │       rosenbrock_3d.png
│       │       rosenbrock_evolucion_difrencial_2d.gif
│       │       rosenbrock_evolucion_difrencial_3d.gif
│       │
│       └───Schwefel
│               curvas_nivel.png
│               Evolucion de optimizacion.png
│               grafica_3d.png
│               schwefel_evolucion_diferencial_2d.gif
│               schwefel_evolucion_diferencial_3d.gif
│
├───Optimizacion_funciones
│       init                            # Inicializador de la biblioteca interna
│
└───reporte_html                        # Reportes enriquecidos para la web
        blog_optimizacion.html          # Artículo/Blog completo documentando la investigación
        ruta_aco_optima.gif             # Animación en formato GIF de la convergencia de la ruta ACO
        ruta_ga_animada.gif             # Animación del progreso de convergencia del GA
        ruta_ga_estatica.png            # Visualización estática del mapa vial óptimo
```

---

##  Visualizaciones y Resultados

Uno de los pilares del proyecto es la representación visual enriquecida y la interpretación intuitiva de la optimización:
* **Gráficas de Superficie 3D y Contornos 2D:** Visualizan las complejas geografías matemáticas de prueba.
* **Animaciones de Convergencia en Tiempo Real:** Archivos GIFs generados dinámicamente mediante `FuncAnimation` que rastrean el desplazamiento exacto de las partículas del enjambre (PSO), la reducción evolutiva de la población cromosómica (EA y DE) y las líneas de descenso directo basadas en gradiente (BFGS).
* **Análisis de Eficiencia:** Comparativa rigurosa entre **Iteraciones Lógicas** (generaciones/pasos de actualización) y **Evaluaciones Reales de la Función (`nfev`)**, revelando el verdadero costo computacional y los recursos lógicos empleados por cada algoritmo.

---

##  Hallazgos Clave

* **La Trampa de la Planitud:** En los algoritmos evolutivos y poblacionales, la función de Rosenbrock llega a requerir hasta **4 veces más iteraciones/evaluaciones** que Schwefel. Esto se debe a la extrema planitud de su valle central; a pesar de tener un diseño topográfico intuitivo sin mínimos locales alternativos, la ausencia de gradientes fuertes hace que los optimizadores converjan lentamente y de forma oscilante.
* **Exploración vs. Explotación (PSO y DE al Rescate):** El Descenso de Gradiente (BFGS) falla drásticamente al quedar atrapado en los profundos mínimos locales oscilantes de la función de Schwefel a menos que se inicialice en un punto extremadamente cercano al global. En contraste, las heurísticas poblacionales basadas en el comportamiento social (`PSO`) y evolutivo diferencial (`DE`) superaron las trampas locales gracias a sus altas capacidades de exploración global combinadas con dinámicas estocásticas.
* **ACO vs. Algoritmos Genéticos en el TSP:** El algoritmo de colonia de hormigas (`ACO`) demostró un alto nivel de eficiencia al consolidar caminos estables gracias a las trazas de feromonas acumuladas en aristas críticas, mientras que los algoritmos genéticos (`GA`) mostraron una excelente capacidad de paralelismo para explorar variadas alternativas topológicas en fases tempranas.

---

##  Tecnologías Utilizadas

* **Python 3.x**
* **NumPy:** Soporte para computación de arreglos matriciales multidimensionales rápidos y operaciones vectoriales de aptitud.
* **Matplotlib:** Generador premium de gráficos estadísticos interactivos, mapas 3D y animaciones fluidas (`FuncAnimation`).
* **SciPy:** Implementación integrada de optimizadores industriales (BFGS en `scipy.optimize.minimize` y `differential_evolution`).
* **PySwarms:** Framework dedicado a la orquestación del enjambre de partículas (PSO).
* **PyGAD:** Framework especializado en la optimización de algoritmos genéticos y evolutivos poblacionales.

---

##  Instalación y Uso

### 1. Clonar el Repositorio:
```bash
git clone https://github.com/JhanuarC/RNA-2026-1.git
cd RNA-2026-1
```

### 2. Instalar Dependencias del Sistema:
Puedes utilizar el archivo `requirement.txt` para instalar de forma directa todas las dependencias necesarias:
```bash
pip install -r requirement.txt
```
O de forma manual:
```bash
pip install numpy matplotlib scipy pyswarms pygad selenium beautifulsoup4
```

### 3. Ejecutar Pruebas Continuas (Parte 1):
Ejecuta la suite modular de descenso clásico, enjambres de partículas, algoritmos diferenciales e híbridos evolutivos:
```bash
python clases.py
```

### 4. Ejecutar Solución Combinatoria y TSP (Parte 2):
Para correr la optimización del problema del agente viajero (TSP) mediante algoritmo genético:
```bash
python RunGeneticAlgorithmOptimizer.py
```
O corre el script modular de Colonia de Hormigas:
```bash
python antcolony_new.py
```

---

##  Autores y Metadata

* **Autores:** 
  * Daniel Felipe Garzón Acosta
  * Jhanuar Castro Lopez
  * Juan Felipe Moreno Ruiz
* **Materia:** Redes Neuronales y Algoritmos Bio-inspirados
* **Fecha:** Abril 2026
