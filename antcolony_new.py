import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.animation as animation
import matplotlib.patches as mpatches

coordenadas = np.array([
    [-102.2967,  21.8805],  # Aguascalientes
    [-115.4526,  32.6245],  # Mexicali
    [-110.3098,  24.1426],  # La Paz
    [ -90.5353,  19.8454],  # Campeche
    [ -93.1130,  16.7521],  # Tuxtla Gutiérrez
    [-106.0691,  28.6329],  # Chihuahua
    [ -99.1332,  19.4326],  # Ciudad de México
    [-100.9737,  25.4232],  # Saltillo
    [-103.7241,  19.2452],  # Colima
    [-104.6532,  24.0277],  # Victoria de Durango
    [-101.2574,  21.0190],  # Guanajuato
    [ -99.5009,  17.5506],  # Chilpancingo
    [ -98.7591,  20.1011],  # Pachuca
    [-103.3496,  20.6597],  # Guadalajara
    [ -99.6557,  19.2826],  # Toluca
    [-101.1844,  19.7060],  # Morelia
    [ -99.2216,  18.9242],  # Cuernavaca
    [-104.8948,  21.5001],  # Tepic
    [-100.3161,  25.6866],  # Monterrey
    [ -96.7266,  17.0732],  # Oaxaca
    [ -98.2063,  19.0414],  # Puebla
    [-100.3899,  20.5888],  # Querétaro
    [ -88.2963,  18.5001],  # Chetumal
    [-100.9855,  22.1565],  # San Luis Potosí
    [-107.3940,  24.7994],  # Culiacán
    [-110.9559,  29.0729],  # Hermosillo
    [ -92.9376,  17.9892],  # Villahermosa
    [ -99.1409,  23.7369],  # Ciudad Victoria
    [ -98.2370,  19.3139],  # Tlaxcala
    [ -96.9270,  19.5438],  # Xalapa
    [ -89.6230,  20.9674],  # Mérida
    [-102.5832,  22.7709],  # Zacatecas
])

repo_path = os.path.abspath("Ant-Colony-Optimization")
if repo_path not in sys.path:
    sys.path.append(repo_path)

tiempo_horas    = pd.read_csv(r"extracción_datos\matriz_tiempo_horas.csv",    index_col=0)
combustible_mxn = pd.read_csv(r"extracción_datos\matriz_combustible_mxn.csv", index_col=0)
casetas_mxn     = pd.read_csv(r"extracción_datos\matriz_casetas_mxn.csv",     index_col=0)

costo_hora_vendedor = 187.5
df_costo_total = (tiempo_horas * costo_hora_vendedor) + combustible_mxn + casetas_mxn

plt.scatter(coordenadas[:,0],coordenadas[:,1])
plt.title("Ciudades a visitar")
plt.grid()
plt.show()

from AntColonyOptimizer import AntColonyOptimizer

matrix = df_costo_total.to_numpy().copy()
np.fill_diagonal(matrix, np.inf)

optimizer = AntColonyOptimizer(
    ants=200,
    evaporation_rate=0.3,
    intensification=3,
    alpha=1,
    beta=3,
    beta_evaporation_rate=0,
    choose_best=0.2
)

best = optimizer.fit(matrix, 500)
optimizer.plot()

ciudades = df_costo_total.index.tolist()
ruta_nombres = [ciudades[i] for i in optimizer.best_path]
print("Mejor ruta encontrada:")
print(" → ".join(ruta_nombres))
print(f"Costo total: ${best:,.2f} MXN")

# Recorrido resultante con la mejor solución:
plt.scatter(coordenadas[:,0],coordenadas[:,1])
plt.plot(coordenadas[optimizer.best_path,0],coordenadas[optimizer.best_path,1],'-r')
plt.grid()
plt.title("Ciudades a visitar")
plt.show()

print(optimizer.best_path)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

#GENERACIÓN DEL GIF ANIMADO 

'''ciudades = [
    "Aguascalientes", "Mexicali", "La Paz", "Campeche", "Tuxtla Gutiérrez",
    "Chihuahua", "Ciudad de México", "Saltillo", "Colima", "Durango",
    "Guanajuato", "Chilpancingo", "Pachuca", "Guadalajara", "Toluca",
    "Morelia", "Cuernavaca", "Tepic", "Monterrey", "Oaxaca",
    "Puebla", "Querétaro", "Chetumal", "San Luis Potosí", "Culiacán",
    "Hermosillo", "Villahermosa", "Cd. Victoria", "Tlaxcala", "Xalapa",
    "Mérida", "Zacatecas"
]

# ── Usar la mejor ruta del optimizador ────────────────────────────────────────
# optimizer.best_path es la lista de índices en orden óptimo
ruta = list(optimizer.best_path)

# ── Configuración de la figura ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

def init():
    ax.clear()
    ax.set_facecolor("#f0f4f8")
    ax.set_xlim(-118, -86)
    ax.set_ylim(14, 35)
    ax.set_title("Ruta óptima — ACO", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True, alpha=0.3)

    # Graficar todas las ciudades de fondo
    ax.scatter(coordenadas[:, 0], coordenadas[:, 1],
               color="steelblue", s=60, zorder=5)

    # Nombres de todas las ciudades
    for i, nombre in enumerate(ciudades):
        ax.annotate(nombre, (coordenadas[i, 0], coordenadas[i, 1]),
                    textcoords="offset points", xytext=(4, 4), fontsize=6.5,
                    color="#333333")

def animate(frame):
    init()

    # Cuántos segmentos dibujar en este frame
    segmentos = frame + 1

    # Dibujar segmentos ya recorridos en rojo
    for k in range(min(segmentos, len(ruta) - 1)):
        ciudad_a = ruta[k]
        ciudad_b = ruta[k + 1]
        ax.plot(
            [coordenadas[ciudad_a, 0], coordenadas[ciudad_b, 0]],
            [coordenadas[ciudad_a, 1], coordenadas[ciudad_b, 1]],
            '-r', linewidth=1.5, alpha=0.8, zorder=4
        )

    # Ciudad actual (frente de la hormiga)
    ciudad_actual = ruta[min(frame, len(ruta) - 1)]
    ax.scatter(coordenadas[ciudad_actual, 0], coordenadas[ciudad_actual, 1],
               color="red", s=120, zorder=6)

    # Último segmento: cerrar el ciclo de vuelta al inicio
    if frame >= len(ruta) - 1:
        ax.plot(
            [coordenadas[ruta[-1], 0], coordenadas[ruta[0], 0]],
            [coordenadas[ruta[-1], 1], coordenadas[ruta[0], 1]],
            '-r', linewidth=1.5, alpha=0.8, zorder=4
        )
        ax.set_title(f"Ruta óptima — ACO  |  Costo: ${best:,.0f} MXN",
                     fontsize=14, fontweight="bold")

    # Contador de progreso
    ax.text(0.02, 0.97,
            f"Visitadas: {min(frame+1, len(ruta))}/{len(ruta)} ciudades",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", color="#333333")

# ── Generar animación ──────────────────────────────────────────────────────────
# total de frames = número de ciudades + 5 frames finales para pausar al terminar
frames_totales = len(ruta) + 5

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=frames_totales,
    init_func=init,
    interval=300,   # milisegundos entre frames (300 = ~3 ciudades/segundo)
    repeat=True
)

# ── Guardar como GIF ───────────────────────────────────────────────────────────
ani.save("ruta_optima.gif", writer="pillow", fps=3, dpi=120)
print("GIF guardado como ruta_optima.gif")

# ── También mostrar en pantalla ────────────────────────────────────────────────
plt.show()'''
# GENERACIÓN DEL GIF ANIMADO
fig, ax = plt.subplots(figsize=(12, 8))
ruta = list(optimizer.best_path)

def animate(frame):
    ax.clear()
    ax.set_facecolor("#f0f4f8")
    ax.set_xlim(-118, -86); ax.set_ylim(14, 35)
    ax.grid(True, alpha=0.3)
    ax.scatter(coordenadas[:, 0], coordenadas[:, 1], color="steelblue", s=60, zorder=5)

    for i, nombre in enumerate(ciudades):
        ax.annotate(nombre, (coordenadas[i, 0], coordenadas[i, 1]),
                    textcoords="offset points", xytext=(4, 4), fontsize=6.5)

    segs = min(frame + 1, len(ruta) - 1)
    for k in range(segs):
        a, b = ruta[k], ruta[k + 1]
        ax.plot([coordenadas[a, 0], coordenadas[b, 0]],
                [coordenadas[a, 1], coordenadas[b, 1]], '-r', lw=1.5, alpha=0.8)

    curr = ruta[min(frame, len(ruta) - 1)]
    ax.scatter(coordenadas[curr, 0], coordenadas[curr, 1], color="red", s=120, zorder=6)

    if frame >= len(ruta) - 1:
        a, b = ruta[-1], ruta[0]
        ax.plot([coordenadas[a, 0], coordenadas[b, 0]],
                [coordenadas[a, 1], coordenadas[b, 1]], '-r', lw=1.5, alpha=0.8)
        ax.set_title(f"Ruta óptima — ACO  |  Costo: ${best:,.0f} MXN", fontsize=14, fontweight="bold")
    else:
        ax.set_title("Ruta óptima — ACO", fontsize=14, fontweight="bold")

    ax.text(0.02, 0.97, f"Visitadas: {min(frame+1, len(ruta))}/{len(ruta)}",
            transform=ax.transAxes, fontsize=9, va="top")

ani = animation.FuncAnimation(fig, animate, frames=len(ruta) + 5, interval=300, repeat=True)
ani.save("ruta_optima.gif", writer="pillow", fps=3, dpi=120)
print("GIF guardado como ruta_optima.gif")
plt.show()