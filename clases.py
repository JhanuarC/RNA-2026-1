import numpy as np
import pygad
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen
from scipy.optimize import differential_evolution
from matplotlib import cm
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher, Designer, Animator
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors



class Rosenbrock_sgd:#Clase para la función de Rosenbrock, usando metodos descenso por gradiente
    def __init__(self, a=1.0, b=100.0):
        self.a = a
        self.b = b
        self.historia_2d = []
        self.historia_3d = []
        self.trayectoria_2d = []
        self.trayectoria_3d = []

    def evaluate(self, x):#Función de Rosenbrock, se puede usar tanto para 2D como para 3D dependiendo del tamaño de x
            return np.sum(self.b*(x[1:] - x[:-1]**2.0)**2.0 + (self.a - x[:-1])**2.0)

    def callback_2d(self, xk):
        self.historia_2d.append(self.evaluate(xk))
        self.trayectoria_2d.append(np.copy(xk)) # GUARDAMOS UNA COPIA DEL PUNTO [X, Y] EN CADA ITERACIÓN PARA LA ANIMACIÓN
    
    def callback_3d(self, xk):
        self.historia_3d.append(self.evaluate(xk))
        self.trayectoria_3d.append(np.copy(xk)) # GUARDAMOS UNA COPIA DEL PUNTO [X, Y, Z] EN CADA ITERACIÓN PARA LA ANIMACIÓN

    # Método para optimizar en 2D
    def optimizar_2d(self):
        res_2d = minimize(self.evaluate, x0=np.random.uniform(-2, 2, 2), method='BFGS', callback=self.callback_2d)
        return res_2d
    # Método para optimizar en 3D
    def optimizar_3d(self):
        res_3d = minimize(self.evaluate, x0=np.random.uniform(-2, 2, 3), method='BFGS', callback=self.callback_3d)
        return res_3d
    
    # Comparación de resultados
    def resultados(self, res_2d, res_3d):
        print(f"--- RESULTADOS 2D ---")
        print(f"X óptimo: {res_2d.x}")
        print(f"Evaluaciones reales de la función: {res_2d.nfev}") 

        print(f"\n--- RESULTADOS 3D ---")
        print(f"X óptimo: {res_3d.x}")
        print(f"Evaluaciones reales de la función: {res_3d.nfev}")
    # Graficar la evolución
    def graficar_evo(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.historia_2d, label='Rosenbrock 2D')
        plt.plot(self.historia_3d, label='Rosenbrock 3D')
        plt.yscale('log') # Escala logarítmica porque Rosenbrock baja de valores muy altos a casi 0
        plt.xlabel('Iteraciones')
        plt.ylabel('Valor de la función (log)')
        plt.title('Evolución de la Optimización (BFGS)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def grafica_3d(self,res_3d):#Grafica 3d de la función de Rosenbrock
            x = np.linspace(-2, 2, 250)
            y = np.linspace(-1, 3, 250)
            X, Y = np.meshgrid(x, y)
            Z = (self.a - X)**2 + self.b * (Y - X**2)**2#Evaluamos la función en cada punto de la malla
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            #cmap es el mapa de colores, antialiased para suavizar la superficie, alpha para transparencia
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, antialiased=False, alpha=0.8)

            fig.colorbar(surf, shrink=0.5, aspect=5) #Barra de colores para entender los valores de Z


            ax.set_title('Función de Rosenbrock en 3D')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Rosenbrock(X, Y)')
            plt.show()

            #Comparar punto óptimo con la gráfica

            try:#Grafica el punto óptimo encontrado en 3D, si es que se encuentra dentro del rango de la gráfica
                ax.scatter(res_3d.x[0], res_3d.x[1], self.evaluate(res_3d.x), color='red', s=100, label='Óptimo encontrado', marker='*')
                ax.legend()
            
            except Exception as e:
                print(f"No se pudo graficar el punto óptimo: {e}")
    
    def grafica_2d(self,res_2d):
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = (self.a - X)**2 + self.b * (Y - X**2)**2#Evaluamos la función en cada punto de la malla

        plt.figure(figsize=(8, 6))
        
        cp = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='magma')
        plt.colorbar(cp)
        plt.title('Curvas de Nivel - Rosenbrock')
        
        plt.plot(1, 1, 'go', markersize=15, label='Mínimo Global (1,1)',zorder =4)#Optimo global conocido de la función de Rosenbrock
    

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Graficar el punto óptimo encontrado
        try:
            plt.scatter(res_2d.x[0], res_2d.x[1], color='red', s=100, label='Óptimo encontrado', marker='*', zorder =5)
            plt.legend()
        except Exception as e:
            print(f"No se pudo graficar el punto óptimo: {e}")

        plt.show()

  
    def animar_descenso_2d(self):
        # Usamos la nueva lista con las coordenadas [X, Y]
        puntos = np.array(self.trayectoria_2d) 
        if len(puntos) == 0:
            print("No hay trayectoria para animar.")
            return

        # Dibujar el bendito fondo
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = (self.a - X)**2 + self.b * (Y - X**2)**2 # Rosenbrock
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='magma', alpha=0.5)       

        # Lo que se va a animar
        linea, = ax.plot([], [], 'r--', alpha=0.6) # Línea del rastro
        punto, = ax.plot([], [], 'ro', markersize=8) # Punto actual
        
        # Marcar el óptimo global para referencia
        ax.plot(1, 1, 'g*', markersize=15, label='Óptimo Global')

        ax.set_title("Animación del Descenso de Gradiente (BFGS)")
        ax.legend()

        def init():
            linea.set_data([], [])
            punto.set_data([], [])
            return linea, punto
        
        def update(i):
            # Actualizamos la línea con todos los puntos hasta i (inclusive)
            linea.set_data(puntos[:i+1, 0], puntos[:i+1, 1])
            # El punto actual recibe una lista con la coordenada X y otra con la Y
            punto.set_data([puntos[i, 0]], [puntos[i, 1]])
            return linea, punto

        # Crear la animación
        anim = FuncAnimation(fig, update, frames=len(puntos), init_func=init, blit=True, interval=100, repeat=False)

        return anim
    def animar_descenso_3d(self):
        
        puntos = np.array(self.trayectoria_3d)# Usamos la trayectoria 3D generada por la optimización 3D
        if len(puntos) == 0:
            print("No hay trayectoria para animar.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Dibujar la superficie de fondo (Rosenbrock)
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z_surf = (self.a - X)**2 + self.b * (Y - X**2)**2
        
        # Usamos alpha=0.6 para que la superficie sea semitransparente y se vea el punto
        ax.plot_surface(X, Y, Z_surf, cmap='viridis', alpha=0.6, edgecolor='none')

        # 2. Elementos que se van a animar
        # Nota: Inicializamos con listas vacías en 3D
        linea, = ax.plot([], [], [], 'r-', linewidth=2, label='Trayectoria') 
        punto, = ax.plot([], [], [], 'go', markersize=10, label='Posición Actual')
        
        # Marcar el óptimo global
        ax.plot([1], [1], [0], 'b*', markersize=15, label='Óptimo Global')

        ax.set_title("Animación 3D del Descenso (BFGS)")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Costo)')
        ax.legend()

        def init():
            linea.set_data([], [])
            linea.set_3d_properties([])
            punto.set_data([], [])
            punto.set_3d_properties([])
            return linea, punto
        
        def update(i):
            # Extraemos las coordenadas X e Y hasta el paso i
            x_data = puntos[:i+1, 0]
            y_data = puntos[:i+1, 1]
            
            # Calculamos la altura Z para cada punto de la trayectoria
            z_data = [self.evaluate(p) for p in puntos[:i+1]]

            # Actualizamos X e Y
            linea.set_data(x_data, y_data)
            # Actualizamos Z (exclusivo de mplot3d)
            linea.set_3d_properties(z_data)

            # Actualizamos el punto actual (el último de la lista actual)
            punto.set_data([puntos[i, 0]], [puntos[i, 1]])
            punto.set_3d_properties([z_data[-1]])
            
            return linea, punto

        # Crear la animación
        anim_3d = FuncAnimation(fig, update, frames=len(puntos), init_func=init, blit=False, interval=100, repeat=False)

        return anim_3d

    def ejecutar(self):
        res_2d = self.optimizar_2d()
        res_3d = self.optimizar_3d()
        self.resultados(res_2d, res_3d)
        self.graficar_evo()
        self.grafica_3d(res_3d)
        self.grafica_2d(res_2d)
        self.animacion = self.animar_descenso_2d() 
        """print("Guardando animación 2D como GIF... (esto puede tardar unos segundos)")
        # fps=10 controla la velocidad, writer='pillow' es el motor que crea el gif
        self.animacion.save('rosenbrock_descenso_2d.gif', writer='pillow', fps=12)
        print("¡Animación guardada con éxito!")"""
        plt.show()
        """#Descomenten las lineas de arriba si quieren guardar la animación como un archivo GIF, 
        pero tengan en cuenta que puede tardar un poco dependiendo de la cantidad de iteraciones y la velocidad de su computadora."""
        self.animacion_3d = self.animar_descenso_3d()
        # Guardar en GIF
        """print("Guardando animación 3D... (toma un poco más de tiempo)")
        self.animacion_3d.save('rosenbrock_descenso_3d.gif', writer='pillow', fps=10)
        print("¡GIF 3D guardado!")"""
        plt.show()
        #Lo mismo de antes, descomentar si quieren guardar la animación 3D.
    
class Schwefel_sgd:#Clase para la función de Schwefel, usando metodos descenso por gradiente
    def __init__(self):
        pass
        self.historia_2d = []
        self.historia_3d = []
        self.trayectoria_2d = []
        self.trayectoria_3d = []

    def evaluate(self, x):
        # Si x es un vector (optimización), axis=1 (o sumatoria total) suma sus elementos.
        # Si x es una matriz (pyswarms), axis=1 suma las dimensiones por cada partícula.
            return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))),axis=0)
    
    def callback_2d(self,xk):
        self.historia_2d.append(self.evaluate(xk))
        self.trayectoria_2d.append(np.copy(xk)) # GUARDAMOS UNA COPIA DEL PUNTO [X, Y] EN CADA ITERACIÓN PARA LA ANIMACIÓN
        
    def callback_3d(self,xk):
        self.historia_3d.append(self.evaluate(xk))
        self.trayectoria_3d.append(np.copy(xk)) # GUARDAMOS UNA COPIA DEL PUNTO [X, Y, Z] EN CADA ITERACIÓN PARA LA ANIMACIÓN
    
    def optimizar_2d(self):
        res_2d = minimize(self.evaluate, x0=np.random.uniform(-20, 20, 2), method='BFGS', callback=self.callback_2d)
        return res_2d

    def optimizar_3d(self):
        res_3d = minimize(self.evaluate, x0=np.random.uniform(-20, 20, 3), method='BFGS', callback=self.callback_3d)
        return res_3d
    
        # Comparación de resultados

    def resultados(self, res_2d, res_3d):
        print(f"--- RESULTADOS 2D ---")
        print(f"X óptimo: {res_2d.x}")
        print(f"Evaluaciones reales de la función: {res_2d.nfev}") 

        print(f"\n--- RESULTADOS 3D ---")
        print(f"X óptimo: {res_3d.x}")
        print(f"Evaluaciones reales de la función: {res_3d.nfev}")

    # Graficar la evolución
    def graficar_evo(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.historia_2d, label='schwefel 2D')
        plt.plot(self.historia_3d, label='schwefel 3D')
        plt.yscale('log') #Igualmente escala logaritmica
        plt.xlabel('Iteraciones')
        plt.ylabel('Valor de la función (log)')
        plt.title('Evolución de la Optimización (BFGS)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def grafica_3d(self,res_3d):
        x = np.linspace(-500, 500, 250)
        y = np.linspace(-500, 500, 250)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate(np.array([X, Y]))

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, antialiased=False, alpha=0.8)

        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_title('Función de Schwefel en 3D')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Schwefel(X, Y)')
        plt.show()

        try:
            ax.scatter(res_3d.x[0], res_3d.x[1], self.evaluate(res_3d.x), color='red', s=100, label='Óptimo encontrado', marker='*')
            ax.legend()
        
        except Exception as e:
            print(f"No se pudo graficar el punto óptimo: {e}")

    def grafica_2d(self,res_2d):
        x = np.linspace(-500, 500, 250)
        y = np.linspace(-500, 500, 250)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate(np.array([X, Y]))

        plt.figure(figsize=(8, 6))
        
        cp = plt.contour(X, Y, Z, levels=np.logspace(1, 5, 20), cmap='magma')
        plt.colorbar(cp)
        plt.title('Curvas de Nivel - Schwefel')
        
        plt.plot(420.9687, 420.9687, 'go', markersize=15, label='Mínimo Global (420.9687,420.9687)',zorder =4)#Optimo global conocido de la función de Schwefel
    

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # Graficar el punto óptimo encontrado
        try:
            plt.scatter(res_2d.x[0], res_2d.x[1], color='red', s=100, label='Óptimo encontrado', marker='*', zorder =5)
            plt.legend()
        except Exception as e:
            print(f"No se pudo graficar el punto óptimo: {e}")

        plt.show()

    def animar_descenso_2d(self):
        # Usamos la nueva lista con las coordenadas [X, Y]
        puntos = np.array(self.trayectoria_2d) 
        if len(puntos) == 0:
            print("No hay trayectoria para animar.")
            return

        # Dibujar el bendito fondo
        fig, ax = plt.subplots(figsize=(8, 6))
        x = x = np.linspace(-500, 500, 250)
        y = x = np.linspace(-500, 500, 250)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate(np.array([X, Y])) # Schwefel
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='magma', alpha=0.5)       

        # Lo que se va a animar
        linea, = ax.plot([], [], 'r--', alpha=0.6) # Línea del rastro
        punto, = ax.plot([], [], 'ro', markersize=8) # Punto actual
            
        # Marcar el óptimo global para referencia
        ax.plot(420.9687, 420.9687, 'g*', markersize=15, label='Óptimo Global')

        ax.set_title("Animación del Descenso de Gradiente (BFGS)")
        ax.legend()

        def init():
            linea.set_data([], [])
            punto.set_data([], [])
            return linea, punto
        
        def update(i):
            # Actualizamos la línea con todos los puntos hasta i (inclusive)
            linea.set_data(puntos[:i+1, 0], puntos[:i+1, 1])
            # El punto actual recibe una lista con la coordenada X y otra con la Y
            punto.set_data([puntos[i, 0]], [puntos[i, 1]])
            return linea, punto

        # Crear la animación
        anim = FuncAnimation(fig, update, frames=len(puntos), init_func=init, blit=True, interval=100, repeat=False)

        return anim

    def animar_descenso_3d(self):
        puntos = np.array(self.trayectoria_3d)# Usamos la trayectoria 3D generada por la optimización 3D
        if len(puntos) == 0:
            print("No hay trayectoria para animar.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Dibujar la superficie de fondo (Rosenbrock)
        x = np.linspace(-500, 500, 250)
        y = np.linspace(-500, 500, 250)
        X, Y = np.meshgrid(x, y)
        Z_surf = self.evaluate(np.array([X, Y]))
        
        # Usamos alpha=0.6 para que la superficie sea semitransparente y se vea el punto
        ax.plot_surface(X, Y, Z_surf, cmap='viridis', alpha=0.6, edgecolor='none')

        # 2. Elementos que se van a animar
        # Nota: Inicializamos con listas vacías en 3D
        linea, = ax.plot([], [], [], 'r-', linewidth=2, label='Trayectoria') 
        punto, = ax.plot([], [], [], 'ro', markersize=15, label='Posición Actual')
        
        # Marcar el óptimo global
        ax.plot([420.9687], [420.9687], [420.9687], 'b*', markersize=15, label='Óptimo Global')

        ax.set_title("Animación 3D del Descenso (BFGS)")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Costo)')
        ax.legend()

        def init():
            linea.set_data([], [])
            linea.set_3d_properties([])
            punto.set_data([], [])
            punto.set_3d_properties([])
            return linea, punto
        
        def update(i):
            # Extraemos las coordenadas X e Y hasta el paso i
            x_data = puntos[:i+1, 0]
            y_data = puntos[:i+1, 1]
            
            # Calculamos la altura Z para cada punto de la trayectoria
            z_data = [self.evaluate(p) for p in puntos[:i+1]]

            # Actualizamos X e Y
            linea.set_data(x_data, y_data)
            # Actualizamos Z (exclusivo de mplot3d)
            linea.set_3d_properties(z_data)

            # Actualizamos el punto actual (el último de la lista actual)
            punto.set_data([puntos[i, 0]], [puntos[i, 1]])
            punto.set_3d_properties([z_data[-1]])
            
            return linea, punto

        # Crear la animación
        anim_3d = FuncAnimation(fig, update, frames=len(puntos), init_func=init, blit=False, interval=100, repeat=False)

        return anim_3d
     
    def ejecutar(self):
        res_2d = self.optimizar_2d()
        res_3d = self.optimizar_3d()
        self.resultados(res_2d, res_3d)
        self.graficar_evo()
        self.grafica_3d(res_3d)
        self.grafica_2d(res_2d)
        self.animacion = self.animar_descenso_2d()
        print("Guardando animación 2D como GIF... (esto puede tardar unos segundos)")
        """# fps=10 controla la velocidad, writer='pillow' es el motor que crea el gif
        self.animacion.save('schwefel_descenso_2d_sgd.gif', writer='pillow', fps=12)
        print("¡Animación guardada con éxito!")"""
        plt.show()
        self.animacion_3d = self.animar_descenso_3d()
        """# Guardar en GIF
        print("Guardando animación 3D... (toma un poco más de tiempo)")
        self.animacion_3d.save('schwefel_descenso_3d_sgd.gif', writer='pillow', fps=12)
        print("¡GIF 3D guardado!")"""
        plt.show()
        

"""
A partir de aqui se usaran metodos de optimización más avanzados, como algoritmos evolutivos
o PSO, para intentar encontrar el mínimo global de estas funciones, 
ya que el método BFGS es un método de optimización local y puede quedarse atrapado en mínimos locales, 
especialmente en funciones tan complejas como Schwefel.
"""

#Algoritmo para evolución diferencial

class Rosenbrock_de(Rosenbrock_sgd):#Clase para la función de Rosenbrock, usando algoritmo de evolución diferencial
    def __init__(self,bounds=None):
        super().__init__()
        self.bounds = bounds
        self.historia_2d = []
        self.historia_3d = []
        self.trayectoria_2d = []
        self.trayectoria_3d = []


    def monitor_progreso_2d(self,xk,convergencia):#Función de callback para evolución diferencial, se llama en cada iteración con el punto actual y la convergencia
        self.historia_2d.append(self.evaluate(xk))
        self.trayectoria_2d.append(np.copy(xk))

    def monitor_progreso_3d(self,xk,convergencia):
        self.historia_3d.append(self.evaluate(xk))
        self.trayectoria_3d.append(np.copy(xk)) # GUARDAMOS UNA COPIA DEL PUNTO [X, Y, Z] EN CADA ITERACIÓN PARA LA ANIMACIÓN

    def optimizar_2d(self):
        self.bounds = [(-2, 2)]*2 #Rangos para la optimización en 2D
        res_2d = differential_evolution(self.evaluate, bounds=self.bounds,callback=self.monitor_progreso_2d)#Optimimzacion

        return res_2d
    
    def optimizar_3d(self):
        self.bounds = [(-2, 2)] * 3
        res_3d = differential_evolution(self.evaluate, bounds=self.bounds,callback=self.monitor_progreso_3d)
        return res_3d
    
    def resultados(self, res_2d, res_3d):
        return super().resultados(res_2d, res_3d)
    
    def graficar_evo(self): 
        plt.figure(figsize=(10, 5))
        plt.plot(self.historia_2d, label='Rosenbrock 2D')
        plt.plot(self.historia_3d, label='Rosenbrock 3D')
        plt.yscale('log') # Escala logarítmica porque Rosenbrock baja de valores muy altos a casi 0
        plt.xlabel('Iteraciones')
        plt.ylabel('Valor de la función (log)')
        plt.title('Evolución de la Optimización (Evolución Diferencial)')
        plt.legend()
        plt.grid(True)
        plt.show()
    def grafica_3d(self,res_3d):    
        return super().grafica_3d(res_3d)
    def grafica_2d(self,res_2d):
        return super().grafica_2d(res_2d)

    def animar_descenso_2d(self):
        # Usamos la nueva lista con las coordenadas [X, Y]
        puntos = np.array(self.trayectoria_2d) 
        if len(puntos) == 0:
            print("No hay trayectoria para animar.")
            return

        # Dibujar el bendito fondo
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = (self.a - X)**2 + self.b * (Y - X**2)**2 # Rosenbrock
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='magma', alpha=0.5)       

        # Lo que se va a animar
        linea, = ax.plot([], [], 'r--', alpha=0.6) # Línea del rastro
        punto, = ax.plot([], [], 'ro', markersize=8) # Punto actual
        
        # Marcar el óptimo global para referencia
        ax.plot(1, 1, 'g*', markersize=15, label='Óptimo Global')

        ax.set_title("Animación de la evolución diferencial")
        ax.legend()

        def init():
            linea.set_data([], [])
            punto.set_data([], [])
            return linea, punto
        
        def update(i):
            # Actualizamos la línea con todos los puntos hasta i (inclusive)
            linea.set_data(puntos[:i+1, 0], puntos[:i+1, 1])
            # El punto actual recibe una lista con la coordenada X y otra con la Y
            punto.set_data([puntos[i, 0]], [puntos[i, 1]])
            return linea, punto

        # Crear la animación
        anim = FuncAnimation(fig, update, frames=len(puntos), init_func=init, blit=True, interval=100, repeat=False)

        return anim
        
        
    def animar_descenso_3d(self):
        puntos = np.array(self.trayectoria_3d)# Usamos la trayectoria 3D generada por la optimización 3D
        if len(puntos) == 0:
            print("No hay trayectoria para animar.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Dibujar la superficie de fondo (Rosenbrock)
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z_surf = (self.a - X)**2 + self.b * (Y - X**2)**2
        
        # Usamos alpha=0.6 para que la superficie sea semitransparente y se vea el punto
        ax.plot_surface(X, Y, Z_surf, cmap='viridis', alpha=0.6, edgecolor='none')

        # 2. Elementos que se van a animar
        # Nota: Inicializamos con listas vacías en 3D
        linea, = ax.plot([], [], [], 'r-', linewidth=2, label='Trayectoria') 
        punto, = ax.plot([], [], [], 'go', markersize=10, label='Posición Actual')
        
        # Marcar el óptimo global
        ax.plot([1], [1], [0], 'b*', markersize=15, label='Óptimo Global')

        ax.set_title("Animación 3D de la evolución diferencial)")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Costo)')
        ax.legend()

        def init():
            linea.set_data([], [])
            linea.set_3d_properties([])
            punto.set_data([], [])
            punto.set_3d_properties([])
            return linea, punto
        
        def update(i):
            # Extraemos las coordenadas X e Y hasta el paso i
            x_data = puntos[:i+1, 0]
            y_data = puntos[:i+1, 1]
            
            # Calculamos la altura Z para cada punto de la trayectoria
            z_data = [self.evaluate(p) for p in puntos[:i+1]]

            # Actualizamos X e Y
            linea.set_data(x_data, y_data)
            # Actualizamos Z (exclusivo de mplot3d)
            linea.set_3d_properties(z_data)

            # Actualizamos el punto actual (el último de la lista actual)
            punto.set_data([puntos[i, 0]], [puntos[i, 1]])
            punto.set_3d_properties([z_data[-1]])
            
            return linea, punto
        
        # Crear la animación
        anim_3d = FuncAnimation(fig, update, frames=len(puntos), init_func=init, blit=False, interval=100, repeat=False)

        return anim_3d
    
    def ejecutar(self):
        res_2d = self.optimizar_2d()
        res_3d = self.optimizar_3d()
        self.resultados(res_2d, res_3d)
        self.graficar_evo()
        self.grafica_3d(res_3d)
        self.grafica_2d(res_2d)
        self.animacion = self.animar_descenso_2d()
       
        """print("Guardando animación 2D como GIF... (esto puede tardar unos segundos)")
        # fps=10 controla la velocidad, writer='pillow' es el motor que crea el gif
        self.animacion.save('rosenbrock_evolucion_difrencial_2d.gif', writer='pillow', fps=12)
        print("¡Animación guardada con éxito!")"""
        plt.show()

        self.animacion_3d = self.animar_descenso_3d()
        # Guardar en GIF
        """print("Guardando animación 3D... (toma un poco más de tiempo)")
        self.animacion_3d.save('rosenbrock_evolucion_difrencial_3d.gif', writer='pillow', fps=12)
        print("¡GIF 3D guardado!")"""
        plt.show()

class Schwefel_de(Schwefel_sgd):#Clase para la función de Schwefel, usando algoritmo de evolución diferencial
    def __init__(self,bounds=None):
        super().__init__()
        self.bounds = bounds
        self.historia_2d = []
        self.historia_3d = []
        self.trayectoria_2d = []
        self.trayectoria_3d = []
        

        

    def monitor_progreso_2d(self,xk,convergencia):
        self.historia_2d.append(self.evaluate(xk))
        self.trayectoria_2d.append(np.copy(xk))
        

    def monitor_progreso_3d(self,xk,convergencia):
        self.historia_3d.append(self.evaluate(xk))
        self.trayectoria_3d.append(np.copy(xk))

    def optimizar_2d(self):
        self.bounds = [(-500, 500)]*2 #Rangos para la optimización en 2D
        res_2d = differential_evolution(self.evaluate, bounds=self.bounds,callback=self.monitor_progreso_2d)
        return res_2d
    
    def optimizar_3d(self):
        self.bounds = [(-500, 500)] * 3
        res_3d = differential_evolution(self.evaluate, bounds=self.bounds,callback=self.monitor_progreso_3d)
        return res_3d
    
    def resultados(self, res_2d, res_3d):
        return super().resultados(res_2d, res_3d)
    
    def graficar_evo(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.historia_2d, label='schwefel 2D')
        plt.plot(self.historia_3d, label='schwefel 3D')
        plt.yscale('log') #Igualmente escala logaritmica
        plt.xlabel('Iteraciones')
        plt.ylabel('Valor de la función (log)')
        plt.title('Evolución de la Optimización (Evolución Diferencial)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def grafica_3d(self,res_3d):    
        return super().grafica_3d(res_3d)
    
    def grafica_2d(self,res_2d):
        return super().grafica_2d(res_2d)
    
    def animar_descenso_2d(self):
        # Usamos la nueva lista con las coordenadas [X, Y]
        puntos = np.array(self.trayectoria_2d) 
        if len(puntos) == 0:
            print("No hay trayectoria para animar.")
            return

        # Dibujar el bendito fondo
        fig, ax = plt.subplots(figsize=(8, 6))
        x = x = np.linspace(-500, 500, 250)
        y = x = np.linspace(-500, 500, 250)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate(np.array([X, Y])) # Schwefel
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='magma', alpha=0.5)       

        # Lo que se va a animar
        linea, = ax.plot([], [], 'r--', alpha=0.6) # Línea del rastro
        punto, = ax.plot([], [], 'ro', markersize=8) # Punto actual
            
        # Marcar el óptimo global para referencia
        ax.plot(420.9687, 420.9687, 'g*', markersize=15, label='Óptimo Global')

        ax.set_title("Animación de la evolución diferencial")
        ax.legend()

        def init():
            linea.set_data([], [])
            punto.set_data([], [])
            return linea, punto
        
        def update(i):
            # Actualizamos la línea con todos los puntos hasta i (inclusive)
            linea.set_data(puntos[:i+1, 0], puntos[:i+1, 1])
            # El punto actual recibe una lista con la coordenada X y otra con la Y
            punto.set_data([puntos[i, 0]], [puntos[i, 1]])
            return linea, punto

        # Crear la animación
        anim = FuncAnimation(fig, update, frames=len(puntos), init_func=init, blit=True, interval=100, repeat=False)

        return anim

    def animar_descenso_3d(self):
        puntos = np.array(self.trayectoria_3d)#Usamos la misma trayectoria 2D pero la graficamos en 3D, con la altura Z dada por el valor de la función en cada punto (X, Y)
        if len(puntos) == 0:
            print("No hay trayectoria para animar.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Dibujar la superficie de fondo (Rosenbrock)
        x = np.linspace(-500, 500, 250)
        y = np.linspace(-500, 500, 250)
        X, Y = np.meshgrid(x, y)
        Z_surf = self.evaluate(np.array([X, Y]))
        
        # Usamos alpha=0.6 para que la superficie sea semitransparente y se vea el punto
        ax.plot_surface(X, Y, Z_surf, cmap='viridis', alpha=0.6, edgecolor='none')

        # 2. Elementos que se van a animar
        # Nota: Inicializamos con listas vacías en 3D
        linea, = ax.plot([], [], [], 'r-', linewidth=2, label='Trayectoria') 
        punto, = ax.plot([], [], [], 'ro', markersize=15, label='Posición Actual')
        
        # Marcar el óptimo global
        ax.plot([420.9687], [420.9687], [420.9687], 'b*', markersize=15, label='Óptimo Global')

        ax.set_title("Animación 3D dela evolución diferencial")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Costo)')
        ax.legend()

        def init():
            linea.set_data([], [])
            linea.set_3d_properties([])
            punto.set_data([], [])
            punto.set_3d_properties([])
            return linea, punto
        
        def update(i):
            # Extraemos las coordenadas X e Y hasta el paso i
            x_data = puntos[:i+1, 0]
            y_data = puntos[:i+1, 1]
            
            # Calculamos la altura Z para cada punto de la trayectoria
            z_data = [self.evaluate(p) for p in puntos[:i+1]]

            # Actualizamos X e Y
            linea.set_data(x_data, y_data)
            # Actualizamos Z (exclusivo de mplot3d)
            linea.set_3d_properties(z_data)

            # Actualizamos el punto actual (el último de la lista actual)
            punto.set_data([puntos[i, 0]], [puntos[i, 1]])
            punto.set_3d_properties([z_data[-1]])
            
            return linea, punto

        # Crear la animación
        anim_3d = FuncAnimation(fig, update, frames=len(puntos), init_func=init, blit=False, interval=100, repeat=False)

        return anim_3d

        
    def ejecutar(self):
        res_2d = self.optimizar_2d()
        res_3d = self.optimizar_3d()
        self.resultados(res_2d, res_3d)
        self.graficar_evo()
        self.grafica_3d(res_3d)
        self.grafica_2d(res_2d)
        
        self.animacion = self.animar_descenso_2d()
        """print("Guardando animación 2D como GIF... (esto puede tardar unos segundos)")
        # fps=10 controla la velocidad, writer='pillow' es el motor que crea el gif
        self.animacion.save('schwefel_evolucion_diferencial_2d.gif', writer='pillow', fps=12)
        print("¡Animación guardada con éxito!")"""
        plt.show()
        self.animacion_3d = self.animar_descenso_3d()
        # Guardar en GIF
        """print("Guardando animación 3D... (toma un poco más de tiempo)")
        self.animacion_3d.save('schwefel_evolucion_diferencial_3d.gif', writer='pillow', fps=12)
        print("¡GIF 3D guardado!")"""
        plt.show()

#Optimizacion por enjambre de partículas (PSO)
#No voy a hacer eso a mano hermano, mejor uso la librería pyswarms que ya tiene implementado el algoritmo de PSO y es fácil de usar.

class Rosenbrock_pso(Rosenbrock_sgd):
    def __init__(self,bounds=None):
        super().__init__()
        self.bounds = bounds

    def evaluate(self, x):
        if x.ndim == 1:
            return np.sum(self.b*(x[1:] - x[:-1]**2.0)**2.0 + (self.a - x[:-1])**2.0)
        else:
            return np.sum(self.b*(x[:, 1:] - x[:, :-1]**2.0)**2.0 + (self.a - x[:, :-1])**2.0, axis=1)

    def optimizar_2d(self):
        self.bounds = (np.array([-2, -2]), np.array([2, 2]))
        options = {'c1': 0.5, 'c2': 0.9, 'w': 0.6} #Parámetros de PSO: c1 y c2 son los coeficientes de aprendizaje, w es el factor de inercia
        optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options, bounds=self.bounds)
        best_cost, best_pos = optimizer.optimize(self.evaluate, iters=1000, verbose=False)
        return best_pos, best_cost ,optimizer.cost_history, optimizer.pos_history
    
    def optimizar_3d(self):
        self.bounds = (np.array([-2, -2, -2]), np.array([2, 2, 2]))
        options = {'c1': 0.5, 'c2': 0.9, 'w': 0.7}
        optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=3, options=options, bounds=self.bounds)
        best_cost, best_pos = optimizer.optimize(self.evaluate, iters=1500, verbose=False)
        return best_pos, best_cost , optimizer.cost_history, optimizer.pos_history
    
    def resultados(self, res_2d, res_3d):
        print(f"--- RESULTADOS 2D ---")
        print(f"X óptimo: {res_2d[0]}")
        print(f"Valor de la función en el óptimo: {res_2d[1]}") 

        print(f"\n--- RESULTADOS 3D ---")
        print(f"X óptimo: {res_3d[0]}")
        print(f"Valor de la función en el óptimo: {res_3d[1]}")

    def graficar_evo(self,res_2d, res_3d):
        plt.figure(figsize=(10, 5))
        plt.plot(res_2d[2], label='Rosenbrock 2D PSO')
        plt.plot(res_3d[2], label='Rosenbrock 3D PSO')
        plt.yscale('log')
        plt.xlabel('Iteraciones')
        plt.ylabel('Costo (Valor de la función)')
        plt.title('Evolución de la Optimización (PSO)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def grafica_3d(self,res_3d):   
            x = np.linspace(-2, 2, 250)
            y = np.linspace(-1, 3, 250)
            X, Y = np.meshgrid(x, y)
            Z = (self.a - X)**2 + self.b * (Y - X**2)**2#Evaluamos la función en cada punto de la malla
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            #cmap es el mapa de colores, antialiased para suavizar la superficie, alpha para transparencia
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, antialiased=False, alpha=0.8)


            pos_final = np.array(res_3d[3][-1]) # Convertimos a array de numpy
            z_particulas = (self.a - pos_final[:,0])**2 + self.b * (pos_final[:,1] - pos_final[:,0]**2)**2
            
            ax.scatter(pos_final[:,0], pos_final[:,1], z_particulas, color='green', s=20, label='Enjambre Final')
            ax.scatter(1, 1, 1, color='red', s=100, label='Mínimo Global', marker='*')

            fig.colorbar(surf, shrink=0.5, aspect=5) #Barra de colores para entender los valores de Z


            ax.set_title('Convergencia Final del Enjambre (PSO)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Rosenbrock(X, Y)')
            
            plt.show()

            #Comparar punto óptimo con la gráfica
            try:#Grafica el punto óptimo encontrado en 3D, si es que se encuentra dentro del rango de la gráfica
                ax.scatter(res_3d[0][0], res_3d[0][1], self.evaluate(res_3d[0]), color='red', s=100, label='Óptimo encontrado', marker='*')
                ax.legend()
            
            except Exception as e:
                print(f"No se pudo graficar el punto óptimo: {e}")
        
    def grafica_2d(self,res_2d):
        
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = (self.a - X)**2 + self.b * (Y - X**2)**2#Evaluamos la función en cada punto de la malla

        plt.figure(figsize=(8, 6))
        
        cp = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='magma')
        plt.colorbar(cp)
        plt.title('Curvas de Nivel - Rosenbrock')
        
        plt.plot(1, 1, 'go', markersize=15, label='Mínimo Global (1,1)',zorder =4)#Optimo global conocido de la función de Rosenbrock
        plt.plot(res_2d[0][0], res_2d[0][1], 'r*', markersize=15, label='Óptimo encontrado PSO', zorder=5)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def animacion_2d(self,res_2d):#Funciopn para animar la evolución del PSO en 2D, usando la librería pyswarms y matplotlib
        m = Mesher(func=self.evaluate,
           limits=[(-2,2), (-2,2)]) #Crea una malla para graficar la función de Rosenbrock
        historia = res_2d[3] #La historia de posiciones de las partículas durante la optimización, se obtiene del resultado de la optimización PSO
        historia_reducida = historia[::10] #Reducimos la cantidad de puntos para la animación
        
        d = Designer(limits=[(-2,2), (-2,2), (-0.1,1)],
                    label=['x-axis', 'y-axis', 'z-axis'],
                    )
         
        faster_anim = Animator(interval=80,repeat = False) #Intervalo entre frames en milisegundos, ajusta según la velocidad deseada
        
        animation = plot_contour(pos_history=historia, mesher=m, designer=d, mark=(0,0),animator=faster_anim)
        return animation
            

    def animacion_3d(self, res_3d):
        # res_3d[3] es la historia de posiciones (optimizer.pos_history)
        historia = res_3d[3]
        # Reducimos la historia para que la animación sea fluida y no tarde demasiado
        historia_reducida = historia[::5] # Cada 5 iteraciones
        if len(historia_reducida) == 0:
            historia_reducida = historia

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Límites para Rosenbrock (-2 a 2 en x, y, z)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])

        ax.set_xlabel('X (x0)')
        ax.set_ylabel('Y (x1)')
        ax.set_zlabel('Z (x2)')
        ax.set_title('Animación 3D del Enjambre de Partículas (PSO - Rosenbrock)')

        # Marcar el óptimo global (1, 1, 1) en Rosenbrock 3D
        ax.scatter([1], [1], [1], color='red', marker='*', s=200, label='Mínimo Global (1,1,1)', zorder=10)
        
        # Inicializar el scatter de las partículas
        scatter = ax.scatter([], [], [], c=[], cmap='viridis', s=30, alpha=0.8, edgecolors='k')

        # Configurar la barra de colores para representar el costo de las partículas
        todos_los_costos = []
        for pos in historia_reducida:
            todos_los_costos.extend(self.evaluate(pos))
        min_cost = min(todos_los_costos)
        max_cost = max(todos_los_costos)
        
        # Usamos escala logarítmica para el costo (Rosenbrock baja de miles a casi 0)
        norm = mcolors.LogNorm(vmin=max(1e-5, min_cost), vmax=max_cost)
        mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        fig.colorbar(mappable, ax=ax, label='Costo de las partículas (Escala Log)')

        ax.legend()

        def init():
            scatter._offsets3d = ([], [], [])
            return scatter,

        def update(frame):
            posiciones = np.array(historia_reducida[frame]) # (n_particles, 3)
            x_data = posiciones[:, 0]
            y_data = posiciones[:, 1]
            z_data = posiciones[:, 2]

            # Calculamos costo para colorear las partículas
            costos = self.evaluate(posiciones)
            
            # Actualizar coordenadas y colores
            scatter._offsets3d = (x_data, y_data, z_data)
            scatter.set_array(costos)
            
            ax.set_title(f'Animación 3D del Enjambre (PSO) - Iteración {frame * 5}')
            return scatter,

        # Crear la animación
        anim = FuncAnimation(fig, update, frames=len(historia_reducida), init_func=init, blit=False, interval=80, repeat=False)
        return anim

    def ejecutar(self):
        res_2d = self.optimizar_2d()
        res_3d = self.optimizar_3d()
        self.resultados(res_2d, res_3d)
        self.graficar_evo(res_2d, res_3d)
        self.grafica_3d(res_3d)
        self.grafica_2d(res_2d)
        self.anim_2d = self.animacion_2d(res_2d)
        self.anim_3d = self.animacion_3d(res_3d)
        plt.show()

        #Guardar animación 2D
        """print("Guardando animación 2D... (toma un poco más de tiempo)")
        self.anim_2d.save('rosenbrock_pso_2d.gif', writer='pillow', fps=12)
        print("¡GIF 2D guardado!")"""

        #Guardar animación 3D
        """print("Guardando animación 3D... (toma un poco más de tiempo)")
        self.anim_3d.save('rosenbrock_pso_3d.gif', writer='pillow', fps=12)
        print("¡GIF 3D guardado!")"""

class Schwefel_pso(Schwefel_sgd):
    def __init__(self,bounds = None):
        super().__init__()
        self.bounds = bounds    

    def evaluate(self, x):#redefinimos la función de evaluación para que funcione con PSO, ya que PSO puede pasarle una matriz de partículas, entonces hay que evaluar cada partícula por separado
        """
        Función de evaluación robusta para Schwefel.
        Maneja:
        - (D,)      : Una sola partícula (vector)
        - (N, D)    : Enjambre de N partículas (PSO y Mesher)
        - (2, M, N) : Rejilla de meshgrid (grafica_2d/3d)
        """
        if x.ndim == 1:
                # Caso (D,)
            return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=0)
            
        if x.ndim == 3 and x.shape[0] == 2:
                # Caso (2, M, N) - Meshgrid manual
            return 418.9829 * 2 - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=0)
                
            # Caso (N, D) - Estándar para PSO y Mesher
        return 418.9829 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
    
    def optimizar_2d(self):
        self.bounds = (np.array([-500, -500]), np.array([500, 500]))
        options = {'c1': 0.5, 'c2': 0.93, 'w': 0.56}
        optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options, bounds=self.bounds)
        best_cost, best_pos = optimizer.optimize(self.evaluate, iters=1000, verbose=False)
        return best_pos, best_cost ,optimizer.cost_history, optimizer.pos_history

    def optimizar_3d(self):
        self.bounds = (np.array([-500, -500,-500]), np.array([500, 500, 500]))
        options = {'c1': 0.5, 'c2': 0.9, 'w': 0.7}
        optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=3, options=options, bounds=self.bounds)
        best_cost, best_pos = optimizer.optimize(self.evaluate, iters=1500, verbose=False)
        return best_pos, best_cost , optimizer.cost_history, optimizer.pos_history
    
    def resultados(self, res_2d, res_3d):
        print(f"--- RESULTADOS 2D ---")
        print(f"X óptimo: {res_2d[0]}")
        print(f"Valor de la función en el óptimo: {res_2d[1]}") 

        print(f"\n--- RESULTADOS 3D ---")
        print(f"X óptimo: {res_3d[0]}")
        print(f"Valor de la función en el óptimo: {res_3d[1]}")
    
    def graficar_evo(self,res_2d, res_3d):
        plt.figure(figsize=(10, 5))
        plt.plot(res_2d[2], label='Schwefel 2D PSO')
        plt.plot(res_3d[2], label='Schwefel 3D PSO')
        plt.yscale('log')
        plt.xlabel('Iteraciones')
        plt.ylabel('Costo (Valor de la función)')
        plt.title('Evolución de la Optimización (PSO)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def grafica_2d(self,res_2d):
        
        x = np.linspace(-500, 500, 250)
        y = np.linspace(-500, 500, 250)
        X, Y = np.meshgrid(x, y)

        f = lambda x: 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))),axis=0)#Se maneja como una función lambda para evaluar la 
        #función de Schwefel en cada punto de la malla, ya que Z = f(X,Y) no funciona directamente porque f espera un vector y no una matriz, 
        # entonces se le pasa np.array([X,Y]) para que lo evalúe correctamente. Esto es un poco hack pero funciona.
        Z = f(np.array([X, Y])) #Evaluamos la función en cada punto de la malla

        plt.figure(figsize=(8, 6))
        
        cp = plt.contour(X, Y, Z, levels=np.logspace(1, 5, 20), cmap='magma')
        plt.colorbar(cp)
        plt.title('Curvas de Nivel - Schwefel')
        
        plt.plot(420.9687, 420.9687, 'go', markersize=15, label='Mínimo Global (420.9687,420.9687)',zorder =4)#Optimo global conocido de la función de Schwefel
        plt.plot(res_2d[0][0], res_2d[0][1], 'r*', markersize=15, label='Óptimo encontrado PSO', zorder=5)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


    def grafica_3d(self, res_3d):
        x = np.linspace(-500,500, 250)
        y = np.linspace(-500,500, 250)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate(np.array([X, Y]))#Evaluamos la función en cada punto de la malla
            
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        #cmap es el mapa de colores, antialiased para suavizar la superficie, alpha para transparencia
        surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, antialiased=False, alpha=0.8)


        pos_final = np.array(res_3d[3][-1]) # Convertimos a array de numpy
        z_particulas = self.evaluate(pos_final)
            
        ax.scatter(pos_final[:,0], pos_final[:,1], z_particulas, color='red', s=70, label='Enjambre Final')
        ax.scatter(420.9687, 420.9687, 420.9687, color='red', s=200, label='Mínimo Global', marker='*')

        fig.colorbar(surf, shrink=0.5, aspect=5) #Barra de colores para entender los valores de Z


        ax.set_title('Convergencia Final del Enjambre (PSO)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Rosenbrock(X, Y)')
            
        plt.show()

        #Comparar punto óptimo con la gráfica
        try:#Grafica el punto óptimo encontrado en 3D, si es que se encuentra dentro del rango de la gráfica
            ax.scatter(res_3d[0][0], res_3d[0][1], self.evaluate(res_3d[0]), color='red', s=100, label='Óptimo encontrado', marker='*')
            ax.legend()
            
        except Exception as e:
            print(f"No se pudo graficar el punto óptimo: {e}")

    def animacion_2d(self,res_2d):
        m = Mesher(func=self.evaluate,
           limits=[(-500,500), (-500,500)],delta=2.0) #Crea una malla para graficar la función de Schwefel
        
        #La historia de posiciones de las partículas durante la optimización, se obtiene del resultado de la optimización PSO
        historia_reducida =res_2d[3][::10] #Reducimos la cantidad de puntos para la animación
        
        d = Designer(limits=[(-500,500), (-500,500), (0, 1700)],
                    label=['x-axis', 'y-axis', 'z-axis'],
                    )
         
        faster_anim = Animator(interval=80,repeat = False) #Intervalo entre frames en milisegundos, ajusta según la velocidad deseada
        
        animation = plot_contour(pos_history=historia_reducida, 
                                designer=d, mesher=m, 
                                mark=(420.9687,420.9687),animator=faster_anim,
                               )
        return animation

    def animacion_3d(self, res_3d):
        # res_3d[3] es la historia de posiciones (optimizer.pos_history)
        historia = res_3d[3]
        # Reducimos la historia para que la animación sea fluida y no tarde demasiado
        historia_reducida = historia[::5] # Cada 5 iteraciones
        if len(historia_reducida) == 0:
            historia_reducida = historia

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Límites para Schwefel (-500 a 500 en x, y, z)
        ax.set_xlim([-500, 500])
        ax.set_ylim([-500, 500])
        ax.set_zlim([-500, 500])

        ax.set_xlabel('X (x0)')
        ax.set_ylabel('Y (x1)')
        ax.set_zlabel('Z (x2)')
        ax.set_title('Animación 3D del Enjambre de Partículas (PSO - Schwefel)')

        # Marcar el óptimo global (420.9687, 420.9687, 420.9687) en Schwefel 3D
        ax.scatter([420.9687], [420.9687], [420.9687], color='red', marker='*', s=200, label='Mínimo Global', zorder=10)
        
        # Inicializar el scatter de las partículas
        scatter = ax.scatter([], [], [], c=[], cmap='plasma', s=30, alpha=0.8, edgecolors='k')

        # Configurar la barra de colores para representar el costo de las partículas
        import matplotlib.colors as mcolors
        todos_los_costos = []
        for pos in historia_reducida:
            todos_los_costos.extend(self.evaluate(pos))
        min_cost = min(todos_los_costos)
        max_cost = max(todos_los_costos)
        
        # Usamos escala lineal o logarítmica
        norm = mcolors.Normalize(vmin=min_cost, vmax=max_cost)
        mappable = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        fig.colorbar(mappable, ax=ax, label='Costo de las partículas')

        ax.legend()

        def init():
            scatter._offsets3d = ([], [], [])
            return scatter,

        def update(frame):
            posiciones = np.array(historia_reducida[frame]) # (n_particles, 3)
            x_data = posiciones[:, 0]
            y_data = posiciones[:, 1]
            z_data = posiciones[:, 2]

            # Calculamos costo para colorear las partículas
            costos = self.evaluate(posiciones)
            
            # Actualizar coordenadas y colores
            scatter._offsets3d = (x_data, y_data, z_data)
            scatter.set_array(costos)
            
            ax.set_title(f'Animación 3D del Enjambre (PSO) - Iteración {frame * 5}')
            return scatter,

        # Crear la animación
        anim = FuncAnimation(fig, update, frames=len(historia_reducida), init_func=init, blit=False, interval=80, repeat=False)
        return anim

    def ejecutar(self):
        res_2d = self.optimizar_2d()
        res_3d = self.optimizar_3d()
        self.resultados(res_2d, res_3d)
        self.graficar_evo(res_2d, res_3d)
        self.grafica_3d(res_3d)
        self.grafica_2d(res_2d)
        self.anim_2d = self.animacion_2d(res_2d)
        self.anim_3d = self.animacion_3d(res_3d)
        plt.show()

        # Guardar animación 2D
        print("Guardando animación 2D... (toma un poco más de tiempo)")
        self.anim_2d.save('schwefel_pso_2d.gif', writer='pillow', fps=12)
        print("¡GIF 2D guardado!")

        # Guardar animación 3D
        print("Guardando animación 3D... (toma un poco más de tiempo)")
        self.anim_3d.save('schwefel_pso_3d.gif', writer='pillow', fps=12)
        print("¡GIF 3D guardado!")

#Optimizacion por Algoritmos Evolutivos
class Rosenbrock_ea(Rosenbrock_sgd):
    def __init__(self):
        super().__init__()
        self.historia_poblacion_2d = []
        self.historia_poblacion_3d = []

    def evaluate(self, x):
        if x.ndim == 1:
            return np.sum(self.b*(x[1:] - x[:-1]**2.0)**2.0 + (self.a - x[:-1])**2.0)
        else:
            return np.sum(self.b*(x[:, 1:] - x[:, :-1]**2.0)**2.0 + (self.a - x[:, :-1])**2.0, axis=1)
    
    def optimizar_2d(self):
        self.historia_poblacion_2d = []
        
        def fitness_func(ga_instance,solution,solution_idx):
            func = self.evaluate(solution)#Solution es el array que genera PyGAD
            
            #Debido a que Pygad maximiza y nosotros queremos mnimizar, tenemos que usar el inverso de la función como fitness
            fitness = 1.0 / (func + 1e-6) #Agregamos un pequeño valor para evitar división por cero
            return fitness
        
        def callback_generacion(ga_instance):
            self.historia_poblacion_2d.append(np.copy(ga_instance.population))
        
        #Configuracion de dimensionalidad
        num_dimensiones = 2
        
        ga_instance = pygad.GA(
        num_generations=1000,
        num_parents_mating=30,
        fitness_func=fitness_func,
        sol_per_pop=100,
        num_genes=num_dimensiones, # Aquí defines si es 2, 3 o N dimensiones
        init_range_low=-2.0,
        init_range_high=2.0,
        mutation_percent_genes=20,
        mutation_num_genes=1,
        on_generation=callback_generacion
                                )
        ga_instance.run()
        solution, solution_fitness, _ = ga_instance.best_solution()
        #Solution es el array con la mejor solución encontrada
        #Solution_fitness es el valor de fitness de esa solución, pero como fitness es el inverso de la función, para obtener el valor real de la función en esa solución, tenemos que hacer 1/fitness
        return solution, self.evaluate(solution) , ga_instance.best_solutions_fitness
    
    def optimizar_3d(self):
        self.historia_poblacion_3d = []
        
        def fitness_func(ga_instance,solution,solution_idx):
            func = self.evaluate(solution)
            fitness = 1.0 / (func + 1e-6)
            return fitness
        
        def callback_generacion(ga_instance):
            self.historia_poblacion_3d.append(np.copy(ga_instance.population))
        
        num_dimensiones = 3
        
        ga_instance = pygad.GA(
        num_generations=2500,
        num_parents_mating=50,
        fitness_func=fitness_func,
        sol_per_pop=250,
        num_genes=num_dimensiones,
        init_range_low=-2.0,
        init_range_high=2.0,
        mutation_percent_genes=20,
        mutation_num_genes=2,
        on_generation=callback_generacion
                            )
        ga_instance.run()
        solution, solution_fitness, _ = ga_instance.best_solution()
        return solution, self.evaluate(solution) , ga_instance.best_solutions_fitness

    def resultados(self, res_2d, res_3d):
        print(f"--- RESULTADOS 2D ---")
        print(f"X óptimo: {res_2d[0]}")
        print(f"Valor de la función en el óptimo: {res_2d[1]}")#En Rosenbrock, esto debe ser un valor cercano a 0, ya que el mínimo global es 0 en (1,1) para 2D y (1,1,1) para 3D

        print(f"\n--- RESULTADOS 3D ---")
        print(f"X óptimo: {res_3d[0]}")
        print(f"Valor de la función en el óptimo: {res_3d[1]}")
        
        """#En este caso no es evaluaciones reales, sino el valor de la función en el óptimo encontrado,
        Porque Pygad no nos da el número de evaluaciones, 
        pero si nos da el valor de la función en la mejor solución encontrada, que es lo que realmente nos interesa."""

    def graficar_evo(self,res_2d,res_3d):
        #Pygad tiene su propia función para graficar la evolución del fitness, pero vamos a hacer una gráfica personalizada para comparar 2D y 3D
        plt.figure(figsize=(10, 5))
        plt.plot(res_2d[2], label='Rosenbrock 2D EA')
        plt.plot(res_3d[2], label='Rosenbrock 3D EA')
        plt.yscale('log')
        plt.xlabel('')
        plt.ylabel('Fitness (1/Valor de la función)')
        plt.title('Evolución de la Optimización (Algoritmo Evolutivo)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def grafica_2d(self,res_2d):
        x = np.linspace(-2, 2, 250)
        y = np.linspace(-1, 3, 250)
        X, Y = np.meshgrid(x, y)
        Z = (self.a - X)**2 + self.b * (Y - X**2)**2#Evaluamos la función en cada punto de la malla

        plt.figure(figsize=(8, 6))
        
        cp = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='magma')
        plt.colorbar(cp)
        plt.title('Curvas de Nivel - Rosenbrock')
        
        plt.plot(1, 1, 'go', markersize=15, label='Mínimo Global (1,1)',zorder =4)#Optimo global conocido de la función de Rosenbrock
        plt.plot(res_2d[0][0], res_2d[0][1], 'r*', markersize=15, label='Óptimo encontrado EA', zorder=5)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def grafica_3d(self,res_3d):
            x = np.linspace(-2, 2, 250)
            y = np.linspace(-1, 3, 250)
            X, Y = np.meshgrid(x, y)
            Z = (self.a - X)**2 + self.b * (Y - X**2)**2#Evaluamos la función en cada punto de la malla
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            #cmap es el mapa de colores, antialiased para suavizar la superficie, alpha para transparencia
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, antialiased=False, alpha=0.8)

            fig.colorbar(surf, shrink=0.5, aspect=5) #Barra de colores para entender los valores de Z


            ax.set_title('Función de Rosenbrock en 3D')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Rosenbrock(X, Y)')
            plt.show()

            #Comparar punto óptimo con la gráfica

            try:#Grafica el punto óptimo encontrado en 3D, si es que se encuentra dentro del rango de la gráfica
                ax.scatter(res_3d[0][0], res_3d[0][1], self.evaluate(res_3d[0]), color='red', s=100, label='Óptimo encontrado', marker='*')
                ax.legend()
            
            except Exception as e:
                print(f"No se pudo graficar el punto óptimo: {e}")

    def animacion_2d(self, res_2d):
        historia = self.historia_poblacion_2d
        historia_reducida = historia[::10]
        if len(historia_reducida) == 0:
            historia_reducida = historia

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Grid para contorno de Rosenbrock
        x = np.linspace(-2, 2, 250)
        y = np.linspace(-1, 3, 250)
        X, Y = np.meshgrid(x, y)
        Z = (self.a - X)**2 + self.b * (Y - X**2)**2
        
        cp = ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='magma')
        fig.colorbar(cp, ax=ax, label='Valor de la función (Rosenbrock)')
        
        # Óptimo global
        ax.plot(1, 1, 'go', markersize=12, label='Mínimo Global (1,1)', zorder=4)
        
        # Inicializar scatter
        scatter = ax.scatter([], [], c=[], cmap='viridis', s=25, alpha=0.7, edgecolors='none', zorder=5)
        
        # Calcular costos para normalizar colores
        todos_los_costos = []
        for pop in historia_reducida:
            for ind in pop:
                todos_los_costos.append(self.evaluate(ind))
        min_cost = min(todos_los_costos)
        max_cost = max(todos_los_costos)
        
        norm = mcolors.LogNorm(vmin=max(1e-5, min_cost), vmax=max_cost)
        mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        fig.colorbar(mappable, ax=ax, label='Costo de los individuos (Escala Log)')
        
        ax.set_xlim([-2, 2])
        ax.set_ylim([-1, 3])
        ax.set_xlabel('X (x0)')
        ax.set_ylabel('Y (x1)')
        ax.set_title('Animación 2D de la Población (EA - Rosenbrock)')
        ax.legend()
        
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([]))
            return scatter,
            
        def update(frame):
            pop = historia_reducida[frame]
            scatter.set_offsets(pop)
            costos = np.array([self.evaluate(ind) for ind in pop])
            scatter.set_array(costos)
            ax.set_title(f'Animación 2D de la Población (EA) - Generación {frame * 10}')
            return scatter,
            
        anim = FuncAnimation(fig, update, frames=len(historia_reducida), init_func=init, blit=False, interval=80, repeat=False)
        return anim

    def animacion_3d(self, res_3d):
        historia = self.historia_poblacion_3d
        historia_reducida = historia[::25]
        if len(historia_reducida) == 0:
            historia_reducida = historia

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        
        ax.set_xlabel('X (x0)')
        ax.set_ylabel('Y (x1)')
        ax.set_zlabel('Z (x2)')
        ax.set_title('Animación 3D de la Población (EA - Rosenbrock)')
        
        # Mínimo global (1, 1, 1)
        ax.scatter([1], [1], [1], color='red', marker='*', s=200, label='Mínimo Global (1,1,1)', zorder=10)
        
        # Scatter de la población
        scatter = ax.scatter([], [], [], c=[], cmap='viridis', s=25, alpha=0.7, edgecolors='none')
        
        # Normalizar colores
        todos_los_costos = []
        for pop in historia_reducida:
            for ind in pop:
                todos_los_costos.append(self.evaluate(ind))
        min_cost = min(todos_los_costos)
        max_cost = max(todos_los_costos)
        
        norm = mcolors.LogNorm(vmin=max(1e-5, min_cost), vmax=max_cost)
        mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        fig.colorbar(mappable, ax=ax, label='Costo de los individuos (Escala Log)')
        
        ax.legend()
        
        def init():
            scatter._offsets3d = ([], [], [])
            return scatter,
            
        def update(frame):
            pop = historia_reducida[frame]
            x_data = pop[:, 0]
            y_data = pop[:, 1]
            z_data = pop[:, 2]
            costos = np.array([self.evaluate(ind) for ind in pop])
            scatter._offsets3d = (x_data, y_data, z_data)
            scatter.set_array(costos)
            ax.set_title(f'Animación 3D de la Población (EA) - Generación {frame * 25}')
            return scatter,
            
        anim = FuncAnimation(fig, update, frames=len(historia_reducida), init_func=init, blit=False, interval=80, repeat=False)
        return anim
    
    def ejecutar(self):
        res_2d = self.optimizar_2d()
        res_3d = self.optimizar_3d()
        self.resultados(res_2d, res_3d)
        self.graficar_evo(res_2d, res_3d)
        self.grafica_3d(res_3d)
        self.grafica_2d(res_2d)
        self.anim_2d = self.animacion_2d(res_2d)
        self.anim_3d = self.animacion_3d(res_3d)
        plt.show()

        # Guardar animación 2D
        """print("Guardando animación 2D... (toma un poco más de tiempo)")
        self.anim_2d.save('rosenbrock_ea_2d.gif', writer='pillow', fps=12)
        print("¡GIF 2D guardado!")"""

        # Guardar animación 3D
        """print("Guardando animación 3D... (toma un poco más de tiempo)")
        self.anim_3d.save('rosenbrock_ea_3d.gif', writer='pillow', fps=12)
        print("¡GIF 3D guardado!")"""
   
class Schwefel_ea(Schwefel_sgd):
    def __init__(self):
        super().__init__()
        self.historia_poblacion_2d = []
        self.historia_poblacion_3d = []

    def evaluate(self, x):
        if x.ndim == 1:
            return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=0)
        if x.ndim == 3 and x.shape[0] == 2:
            return 418.9829 * 2 - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=0)
        return 418.9829 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
    
    def optimizar_2d(self):
        self.historia_poblacion_2d = []
        
        #Función de fitness para Schwefel, similar a la de Rosenbrock pero adaptada a la función de Schwefel
        def fitness_func(ga_instance,solution,solution_idx):
            func = self.evaluate(solution)
            fitness = 1.0 / (func + 1e-6)
            return fitness
        
        def callback_generacion(ga_instance):
            self.historia_poblacion_2d.append(np.copy(ga_instance.population))
        
        num_dimensiones = 2
        
        ga_instance = pygad.GA(
        num_generations=1000,
        num_parents_mating=30,
        fitness_func=fitness_func,
        sol_per_pop=100,
        num_genes=num_dimensiones,
        init_range_low=-500.0,
        init_range_high=500.0,
        mutation_percent_genes=20,
        mutation_num_genes=1,
        on_generation=callback_generacion
                                )
        ga_instance.run()
        solution, solution_fitness, _ = ga_instance.best_solution()
        return solution, self.evaluate(solution) , ga_instance.best_solutions_fitness
    
    def optimizar_3d(self):
        self.historia_poblacion_3d = []
        
        def fitness_func(ga_instance,solution,solution_idx):
            func = self.evaluate(solution)
            fitness = 1.0 / (func + 1e-6)
            return fitness
        
        def callback_generacion(ga_instance):
            self.historia_poblacion_3d.append(np.copy(ga_instance.population))
        
        num_dimensiones = 3
        
        ga_instance = pygad.GA(
        num_generations=2500,
        num_parents_mating=50,
        fitness_func=fitness_func,
        sol_per_pop=250,
        num_genes=num_dimensiones,
        init_range_low=-500.0,
        init_range_high=500.0,
        mutation_percent_genes=15,
        mutation_num_genes=1,
        on_generation=callback_generacion
                            )
        ga_instance.run()
        solution, solution_fitness, _ = ga_instance.best_solution()
        return solution, self.evaluate(solution) , ga_instance.best_solutions_fitness

    def resultados(self, res_2d, res_3d):
        print(f"--- RESULTADOS 2D ---")
        print(f"X óptimo: {res_2d[0]}")
        print(f"Valor de la función en el óptimo: {res_2d[1]}") 

        print(f"\n--- RESULTADOS 3D ---")
        print(f"X óptimo: {res_3d[0]}")
        print(f"Valor de la función en el óptimo: {res_3d[1]}")

    def graficar_evo(self,res_2d, res_3d):
        plt.figure(figsize=(10, 5))
        plt.plot(res_2d[2], label='Schwefel 2D EA')
        plt.plot(res_3d[2], label='Schwefel 3D EA')
        plt.yscale('log')
        plt.xlabel('')
        plt.ylabel('Fitness (1/Valor de la función)')
        plt.title('Evolución de la Optimización (Algoritmo Evolutivo)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def grafica_2d(self,res_2d):
        x = np.linspace(-500, 500, 250)
        y = np.linspace(-500, 500, 250)
        X, Y = np.meshgrid(x, y)

        f = lambda x: 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))),axis=0)
        Z = f(np.array([X, Y]))

        plt.figure(figsize=(8, 6))
        
        cp = plt.contour(X, Y, Z, levels=np.logspace(1, 5, 20), cmap='magma')
        plt.colorbar(cp)
        plt.title('Curvas de Nivel - Schwefel')
        
        plt.plot(420.9687, 420.9687, 'go', markersize=15, label='Mínimo Global (420.9687,420.9687)',zorder =4)#Optimo global conocido de la función de Schwefel
        plt.plot(res_2d[0][0], res_2d[0][1], 'r*', markersize=15, label='Óptimo encontrado EA', zorder=5)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def grafica_3d(self, res_3d):
        x = np.linspace(-500,500, 250)
        y = np.linspace(-500,500, 250)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate(np.array([X, Y]))#Evaluamos la función en cada punto de la malla
            
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        #cmap es el mapa de colores, antialiased para suavizar la superficie, alpha para transparencia
        surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, antialiased=False, alpha=0.8)

        fig.colorbar(surf, shrink=0.5, aspect=5) #Barra de colores para entender los valores de Z


        ax.set_title('Función de Schwefel en 3D')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Schwefel(X, Y)')
        plt.show()

        #Comparar punto óptimo con la gráfica
        try:#Grafica el punto óptimo encontrado en 3D, si es que se encuentra dentro del rango de la gráfica
            ax.scatter(res_3d[0][0], res_3d[0][1], self.evaluate(res_3d[0]), color='red', s=100, label='Óptimo encontrado', marker='*')
            ax.legend()
            
        except Exception as e:
            print(f"No se pudo graficar el punto óptimo: {e}")

    def animacion_2d(self, res_2d):
        historia = self.historia_poblacion_2d
        historia_reducida = historia[::10]
        if len(historia_reducida) == 0:
            historia_reducida = historia

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Grid para contorno de Schwefel
        x = np.linspace(-500, 500, 250)
        y = np.linspace(-500, 500, 250)
        X, Y = np.meshgrid(x, y)
        f = lambda x: 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=0)
        Z = f(np.array([X, Y]))
        
        cp = ax.contour(X, Y, Z, levels=np.logspace(1, 5, 20), cmap='magma')
        fig.colorbar(cp, ax=ax, label='Valor de la función (Schwefel)')
        
        # Óptimo global (420.9687, 420.9687)
        ax.plot(420.9687, 420.9687, 'go', markersize=12, label='Mínimo Global (420.9687,420.9687)', zorder=4)
        
        # Inicializar scatter
        scatter = ax.scatter([], [], c=[], cmap='plasma', s=25, alpha=0.7, edgecolors='none', zorder=5)
        
        # Calcular costos para normalizar colores (escala lineal para Schwefel)
        todos_los_costos = []
        for pop in historia_reducida:
            for ind in pop:
                todos_los_costos.append(self.evaluate(ind))
        min_cost = min(todos_los_costos)
        max_cost = max(todos_los_costos)
        
        norm = mcolors.Normalize(vmin=min_cost, vmax=max_cost)
        mappable = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        fig.colorbar(mappable, ax=ax, label='Costo de los individuos')
        
        ax.set_xlim([-500, 500])
        ax.set_ylim([-500, 500])
        ax.set_xlabel('X (x0)')
        ax.set_ylabel('Y (x1)')
        ax.set_title('Animación 2D de la Población (EA - Schwefel)')
        ax.legend()
        
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([]))
            return scatter,
            
        def update(frame):
            pop = historia_reducida[frame]
            scatter.set_offsets(pop)
            costos = np.array([self.evaluate(ind) for ind in pop])
            scatter.set_array(costos)
            ax.set_title(f'Animación 2D de la Población (EA) - Generación {frame * 10}')
            return scatter,
            
        anim = FuncAnimation(fig, update, frames=len(historia_reducida), init_func=init, blit=False, interval=80, repeat=False)
        return anim

    def animacion_3d(self, res_3d):
        historia = self.historia_poblacion_3d
        historia_reducida = historia[::25]
        if len(historia_reducida) == 0:
            historia_reducida = historia

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlim([-500, 500])
        ax.set_ylim([-500, 500])
        ax.set_zlim([-500, 500])
        
        ax.set_xlabel('X (x0)')
        ax.set_ylabel('Y (x1)')
        ax.set_zlabel('Z (x2)')
        ax.set_title('Animación 3D de la Población (EA - Schwefel)')
        
        # Mínimo global (420.9687, 420.9687, 420.9687)
        ax.scatter([420.9687], [420.9687], [420.9687], color='red', marker='*', s=200, label='Mínimo Global', zorder=10)
        
        # Scatter de la población
        scatter = ax.scatter([], [], [], c=[], cmap='plasma', s=25, alpha=0.7, edgecolors='none')
        
        # Normalizar colores (escala lineal para Schwefel)
        todos_los_costos = []
        for pop in historia_reducida:
            for ind in pop:
                todos_los_costos.append(self.evaluate(ind))
        min_cost = min(todos_los_costos)
        max_cost = max(todos_los_costos)
        
        norm = mcolors.Normalize(vmin=min_cost, vmax=max_cost)
        mappable = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        fig.colorbar(mappable, ax=ax, label='Costo de los individuos')
        
        ax.legend()
        
        def init():
            scatter._offsets3d = ([], [], [])
            return scatter,
            
        def update(frame):
            pop = historia_reducida[frame]
            x_data = pop[:, 0]
            y_data = pop[:, 1]
            z_data = pop[:, 2]
            costos = np.array([self.evaluate(ind) for ind in pop])
            scatter._offsets3d = (x_data, y_data, z_data)
            scatter.set_array(costos)
            ax.set_title(f'Animación 3D de la Población (EA) - Generación {frame * 25}')
            return scatter,
            
        anim = FuncAnimation(fig, update, frames=len(historia_reducida), init_func=init, blit=False, interval=80, repeat=False)
        return anim
    
    def ejecutar(self):
        res_2d = self.optimizar_2d()
        res_3d = self.optimizar_3d()
        self.resultados(res_2d, res_3d)
        self.graficar_evo(res_2d, res_3d)
        self.grafica_3d(res_3d)
        self.grafica_2d(res_2d)
        self.anim_2d = self.animacion_2d(res_2d)
        self.anim_3d = self.animacion_3d(res_3d)
        plt.show()

        # Guardar animación 2D
        """print("Guardando animación 2D... (toma un poco más de tiempo)")
        self.anim_2d.save('schwefel_ea_2d.gif', writer='pillow', fps=12)
        print("¡GIF 2D guardado!")"""

        # Guardar animación 3D
        """print("Guardando animación 3D... (toma un poco más de tiempo)")
        self.anim_3d.save('schwefel_ea_3d.gif', writer='pillow', fps=12)
        print("¡GIF 3D guardado!")"""



