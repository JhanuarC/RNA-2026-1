#Codigo simple para probar las clases creadas en el archivo clases.py
import pandas as pd
import numpy as np
from clases import (
    Rosenbrock_sgd, Schwefel_sgd,
    Rosenbrock_de, Schwefel_de,
    Rosenbrock_pso, Schwefel_pso,
    Rosenbrock_ea, Schwefel_ea
)

def probar_optimizacion():
    print("=== INICIANDO BATERÍA DE PRUEBAS DE OPTIMIZACIÓN ===\n")

    # --- 1. DESCENSO POR GRADIENTE (BFGS) ---
    print("\n>>> [1/4] Probando Métodos de Gradiente...")
    
    print("\n>>> Probando Rosenbrock con BFGS...")
    r_sgd = Rosenbrock_sgd() 
    r_sgd.ejecutar() 
    print("\n>>> Probando Schwefel con BFGS...")
    s_sgd = Schwefel_sgd() 
    s_sgd.ejecutar() 

    # --- 2. EVOLUCIÓN DIFERENCIAL ---
    print("\n>>> [2/4] Probando Evolución Diferencial...")
    
    print("\n>>> Probando Rosenbrock con DE...")
    r_de = Rosenbrock_de()
    r_de.ejecutar()
    
    print(">>> Probando Schwefel con DE...")
    s_de = Schwefel_de()
    s_de.ejecutar() # Probamos esta porque es la que corrige el error del gradiente

    # --- 3. ENJAMBRE DE PARTÍCULAS (PSO) ---
    print("\n>>> [3/4] Probando Particle Swarm Optimization...")
  
    print("\n>>> Probando Rosenbrock con PSO...")
    r_pso = Rosenbrock_pso()
    r_pso.ejecutar() # Esta generará la animación 2D que pide la tarea
   
    print("\n>>> Probando Schwefel con PSO...")
    s_pso = Schwefel_pso()
    s_pso.ejecutar() 

    # --- 4. ALGORITMOS EVOLUTIVOS (PyGAD) ---
    print("\n>>> [4/4] Probando Algoritmos Genéticos...")
    
    print("\n>>> Probando Rosenbrock con EA...")
    r_ea = Rosenbrock_ea()
    r_ea.ejecutar()
    
    print("\n>>> Probando Schwefel con EA...")
    s_ea = Schwefel_ea()
    s_ea.ejecutar()

    print("\n=== PRUEBAS FINALIZADAS ===")

def formatear_x(x_array):
    # Toma el arreglo de coordenadas y lo formatea como un string corto
    # Ejemplo: [420.9687, 420.9687] -> "[420.97, 420.97]"
    return "[" + ", ".join([f"{val:.2f}" for val in x_array]) + "]"

def recolectar_tabla_completa():
    print("\n>>> Generando Tabla Comparativa Extendida (Esto tomará unos minutos)...")
    data = []

    # --- VALORES TEÓRICOS ---
    val_teorico = 0.0
    
    # Coordenadas teóricas Rosenbrock
    xr_teorico_2d = "[1.00, 1.00]"
    xr_teorico_3d = "[1.00, 1.00, 1.00]"
    
    # Coordenadas teóricas Schwefel
    xs_teorico_2d = "[420.97, 420.97]"
    xs_teorico_3d = "[420.97, 420.97, 420.97]"

    # --- 1. ROSENBROCK ---
    print("Evaluando Rosenbrock...")
    r_sgd = Rosenbrock_sgd()
    res_r2 = r_sgd.optimizar_2d()
    res_r3 = r_sgd.optimizar_3d()
    data.append(["Rosenbrock", "2D", "SGD (BFGS)", xr_teorico_2d, formatear_x(res_r2.x), val_teorico, res_r2.fun, res_r2.nfev])
    data.append(["Rosenbrock", "3D", "SGD (BFGS)", xr_teorico_3d, formatear_x(res_r3.x), val_teorico, res_r3.fun, res_r3.nfev])

    r_de = Rosenbrock_de()
    res_rde2 = r_de.optimizar_2d()
    res_rde3 = r_de.optimizar_3d()
    data.append(["Rosenbrock", "2D", "Evolución Dif.", xr_teorico_2d, formatear_x(res_rde2.x), val_teorico, res_rde2.fun, res_rde2.nfev])
    data.append(["Rosenbrock", "3D", "Evolución Dif.", xr_teorico_3d, formatear_x(res_rde3.x), val_teorico, res_rde3.fun, res_rde3.nfev])

    r_pso = Rosenbrock_pso()
    res_rpso2 = r_pso.optimizar_2d()
    res_rpso3 = r_pso.optimizar_3d()
    # En la clase PSO, el índice 0 tiene la mejor posición (x) y el índice 1 el costo (función)
    data.append(["Rosenbrock", "2D", "PSO", xr_teorico_2d, formatear_x(res_rpso2[0]), val_teorico, res_rpso2[1], 50000])
    data.append(["Rosenbrock", "3D", "PSO", xr_teorico_3d, formatear_x(res_rpso3[0]), val_teorico, res_rpso3[1], 75000])

    r_ea = Rosenbrock_ea()
    res_rea2 = r_ea.optimizar_2d()
    res_rea3 = r_ea.optimizar_3d()
    # En la clase EA, el índice 0 tiene la solución (x) y el índice 1 el valor de la función
    data.append(["Rosenbrock", "2D", "Alg. Genético", xr_teorico_2d, formatear_x(res_rea2[0]), val_teorico, res_rea2[1], 100000])
    data.append(["Rosenbrock", "3D", "Alg. Genético", xr_teorico_3d, formatear_x(res_rea3[0]), val_teorico, res_rea3[1], 625000])

    # --- 2. SCHWEFEL ---
    print("Evaluando Schwefel...")
    s_sgd = Schwefel_sgd()
    res_s2 = s_sgd.optimizar_2d()
    res_s3 = s_sgd.optimizar_3d()
    data.append(["Schwefel", "2D", "SGD (BFGS)", xs_teorico_2d, formatear_x(res_s2.x), val_teorico, res_s2.fun, res_s2.nfev])
    data.append(["Schwefel", "3D", "SGD (BFGS)", xs_teorico_3d, formatear_x(res_s3.x), val_teorico, res_s3.fun, res_s3.nfev])

    s_de = Schwefel_de()
    res_sde2 = s_de.optimizar_2d()
    res_sde3 = s_de.optimizar_3d()
    data.append(["Schwefel", "2D", "Evolución Dif.", xs_teorico_2d, formatear_x(res_sde2.x), val_teorico, res_sde2.fun, res_sde2.nfev])
    data.append(["Schwefel", "3D", "Evolución Dif.", xs_teorico_3d, formatear_x(res_sde3.x), val_teorico, res_sde3.fun, res_sde3.nfev])

    s_pso = Schwefel_pso()
    res_spso2 = s_pso.optimizar_2d()
    res_spso3 = s_pso.optimizar_3d()
    data.append(["Schwefel", "2D", "PSO", xs_teorico_2d, formatear_x(res_spso2[0]), val_teorico, res_spso2[1], 50000])
    data.append(["Schwefel", "3D", "PSO", xs_teorico_3d, formatear_x(res_spso3[0]), val_teorico, res_spso3[1], 75000])

    s_ea = Schwefel_ea()
    res_sea2 = s_ea.optimizar_2d()
    res_sea3 = s_ea.optimizar_3d()
    data.append(["Schwefel", "2D", "Alg. Genético", xs_teorico_2d, formatear_x(res_sea2[0]), val_teorico, res_sea2[1], 100000])
    data.append(["Schwefel", "3D", "Alg. Genético", xs_teorico_3d, formatear_x(res_sea3[0]), val_teorico, res_sea3[1], 625000])

    # --- FORMATO DEL DATAFRAME ---
    columnas = ["Función", "Dim", "Método", "X Teórico", "X Alcanzado", "f(x) Teórico", "f(x) Alcanzado", "Evaluaciones"]
    df = pd.DataFrame(data, columns=columnas)
    
    # Formatear la vista para la consola
    df["f(x) Alcanzado"] = df["f(x) Alcanzado"].map("{:.4e}".format)
    df["f(x) Teórico"] = df["f(x) Teórico"].map("{:.1f}".format)
    
    print("\n" + "="*115)
    print("TABLA FINAL PARA EL REPORTE TÉCNICO (Parte 1)")
    print("="*115)
    print(df.to_string(index=False))
    print("="*115)
    
    # Exportar a CSV
    df.to_csv("resultados_optimizacion_final.csv", index=False)
    return df
if __name__ == "__main__":
    recolectar_tabla_completa()
    #probar_optimizacion() # Descomenta esta línea si quieres ejecutar la batería de pruebas completa (sin la tabla)












