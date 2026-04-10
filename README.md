# RNA-2026-1

# Borrador del readme hecho con IA


Optimización Heurística: De Funciones Matemáticas al Problema del Viajero (TSP)
Este repositorio contiene la implementación y el análisis del Punto 1 del proyecto de optimización, enfocado en comparar métodos de descenso por gradiente y algoritmos heurísticos aplicados a funciones de prueba estándar y problemas de optimización combinatoria.

🚀 Contenido del Proyecto
El proyecto se divide en dos grandes ejes temáticos:

1. Optimización de Funciones Continuas
Estudio de la convergencia y eficiencia en paisajes de búsqueda complejos utilizando las funciones de:

Rosenbrock (The Banana Function): Caracterizada por su valle estrecho y curvado donde el gradiente es casi nulo cerca del mínimo global.

Schwefel: Una función altamente multimodal con numerosos mínimos locales que ponen a prueba la capacidad de exploración de los algoritmos.

Algoritmos Implementados:

Descenso de Gradiente (BFGS): Enfoque basado en derivadas para explotación local.

Evolución Diferencial (DE): Algoritmo evolutivo para optimización global.

Particle Swarm Optimization (PSO): Heurística basada en el comportamiento social de enjambres.

2. Problema del Agente Viajero (TSP)
Aplicación de métodos heurísticos para resolver el problema combinatorio de encontrar la ruta más corta que visite un conjunto de ciudades y regrese al punto de origen.

📊 Visualizaciones y Resultados
Uno de los pilares de este proyecto es la interpretación visual de la optimización:

Gráficas 3D y Contornos: Representación de la topografía de las funciones.

Animaciones de Convergencia: GIFs generados con Matplotlib que muestran el rastro de los algoritmos (trayectorias de BFGS y movimiento del enjambre en PSO).

Análisis de Desempeño: Comparativa entre Iteraciones (pasos lógicos) vs. Evaluaciones Reales (nfev), destacando el costo computacional real de cada método.

🛠️ Tecnologías Utilizadas
Python 3.x

NumPy: Gestión de vectores y operaciones matriciales.

Matplotlib: Generación de gráficos estáticos y animaciones (FuncAnimation).

SciPy: Motores de optimización (BFGS, Differential Evolution).

PySwarms: Framework para la implementación de PSO.

📁 Estructura del Código
clases.py: Contiene la lógica modular del proyecto.

Rosenbrock_sgd / Schwefel_sgd: Implementaciones basadas en gradiente.

Rosenbrock_pso: Implementación de enjambre de partículas.

animar_descenso(): Métodos personalizados para la generación de visualizaciones dinámicas.

💡 Hallazgos Clave
La Trampa de la Planitud: Se identificó que la función de Rosenbrock requiere hasta 4 veces más iteraciones que Schwefel en métodos evolutivos debido a la pérdida de gradiente en su valle central, a pesar de tener una estructura aparentemente "más simple".

Exploración vs. Explotación: Las heurísticas (PSO/DE) demostraron ser superiores en la función de Schwefel para evitar quedar atrapados en los profundos mínimos locales que engañan al descenso de gradiente.

🔧 Instalación y Uso
Clonar el repositorio:

Bash
git clone https://github.com/tu-usuario/proyecto-optimizacion.git
Instalar dependencias:

Bash
pip install numpy matplotlib scipy pyswarms
Ejecutar las pruebas:

Bash
python clases.py
Autor: [Tu Nombre]

Materia: Optimización

Fecha: Abril 2026
