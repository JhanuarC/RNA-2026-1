import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import cm

class G_rosenbrock:#Clase para la función de Rosenbrock, usando metodos descenso por gradiente
    def __init__(self, a=1.0, b=100.0):
        self.a = a
        self.b = b
        self.historia_2d = []
        self.historia_3d = []

    def evaluate(self, x):#Función de Rosenbrock, se puede usar tanto para 2D como para 3D dependiendo del tamaño de x
        return np.sum(self.b*(x[1:] - x[:-1]**2.0)**2.0 + (self.a - x[:-1])**2.0)
    
    def callback_2d(self, xk):
        self.historia_2d.append(self.evaluate(xk))
    
    def callback_3d(self, xk):
        self.historia_3d.append(self.evaluate(xk))

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

    def ejecutar(self):
        res_2d = self.optimizar_2d()
        res_3d = self.optimizar_3d()
        self.resultados(res_2d, res_3d)
        self.graficar_evo()
        self.grafica_3d(res_3d)
        self.grafica_2d(res_2d)

class G_schwefel:#Clase para la función de Schwefel, usando metodos descenso por gradiente
    def __init__(self):
        pass
        self.historia_2d = []
        self.historia_3d = []
    
    def evaluate(self,x):   
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    def callback_2d(self,xk):
        self.historia_2d.append(self.evaluate(xk))
        
    def callback_3d(self,xk):
        self.historia_3d.append(self.evaluate(xk))
    
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
    def graficar(self):
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

    def ejecutar(self):
        res_2d = self.optimizar_2d()
        res_3d = self.optimizar_3d()
        self.resultados(res_2d, res_3d)
        self.graficar()



rosen = G_rosenbrock()
rosen.ejecutar()