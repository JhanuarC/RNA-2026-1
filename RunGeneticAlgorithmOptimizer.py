"""
main_ga.py
----------
Parte 2 – Algoritmos Genéticos (TSP, capitales de México)
Usa exactamente la misma estructura de datos que main_aco.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys, os

# ── Coordenadas de las 32 capitales ──────────────────────────────────────────
coordenadas = np.array([
    [-102.2967,  21.8805],  # 0  Aguascalientes
    [-115.4526,  32.6245],  # 1  Mexicali
    [-110.3098,  24.1426],  # 2  La Paz
    [ -90.5353,  19.8454],  # 3  Campeche
    [ -93.1130,  16.7521],  # 4  Tuxtla Gutiérrez
    [-106.0691,  28.6329],  # 5  Chihuahua
    [ -99.1332,  19.4326],  # 6  Ciudad de México
    [-100.9737,  25.4232],  # 7  Saltillo
    [-103.7241,  19.2452],  # 8  Colima
    [-104.6532,  24.0277],  # 9  Victoria de Durango
    [-101.2574,  21.0190],  # 10 Guanajuato
    [ -99.5009,  17.5506],  # 11 Chilpancingo
    [ -98.7591,  20.1011],  # 12 Pachuca
    [-103.3496,  20.6597],  # 13 Guadalajara
    [ -99.6557,  19.2826],  # 14 Toluca
    [-101.1844,  19.7060],  # 15 Morelia
    [ -99.2216,  18.9242],  # 16 Cuernavaca
    [-104.8948,  21.5001],  # 17 Tepic
    [-100.3161,  25.6866],  # 18 Monterrey
    [ -96.7266,  17.0732],  # 19 Oaxaca
    [ -98.2063,  19.0414],  # 20 Puebla
    [-100.3899,  20.5888],  # 21 Querétaro
    [ -88.2963,  18.5001],  # 22 Chetumal
    [-100.9855,  22.1565],  # 23 San Luis Potosí
    [-107.3940,  24.7994],  # 24 Culiacán
    [-110.9559,  29.0729],  # 25 Hermosillo
    [ -92.9376,  17.9892],  # 26 Villahermosa
    [ -99.1409,  23.7369],  # 27 Ciudad Victoria
    [ -98.2370,  19.3139],  # 28 Tlaxcala
    [ -96.9270,  19.5438],  # 29 Xalapa
    [ -89.6230,  20.9674],  # 30 Mérida
    [-102.5832,  22.7709],  # 31 Zacatecas
])

ciudades = [
    "Aguascalientes", "Mexicali", "La Paz", "Campeche", "Tuxtla Gutiérrez",
    "Chihuahua", "Ciudad de México", "Saltillo", "Colima", "Durango",
    "Guanajuato", "Chilpancingo", "Pachuca", "Guadalajara", "Toluca",
    "Morelia", "Cuernavaca", "Tepic", "Monterrey", "Oaxaca",
    "Puebla", "Querétaro", "Chetumal", "San Luis Potosí", "Culiacán",
    "Hermosillo", "Villahermosa", "Cd. Victoria", "Tlaxcala", "Xalapa",
    "Mérida", "Zacatecas"
]

# ── Cargar matrices de costo ──────────────────────────────────────────────────
tiempo_horas    = pd.read_csv(r"extracción_datos\matriz_tiempo_horas.csv",    index_col=0)
combustible_mxn = pd.read_csv(r"extracción_datos\matriz_combustible_mxn.csv", index_col=0)
casetas_mxn     = pd.read_csv(r"extracción_datos\matriz_casetas_mxn.csv",     index_col=0)

costo_hora_vendedor = 187.5          # MXN / hora (parámetro del equipo)
df_costo_total = (tiempo_horas * costo_hora_vendedor) + combustible_mxn + casetas_mxn

# ── Preparar matriz NumPy ─────────────────────────────────────────────────────
matrix = df_costo_total.to_numpy().copy()
np.fill_diagonal(matrix, np.inf)     # sin auto-ciclos

# ── Importar el optimizador ───────────────────────────────────────────────────
from GeneticAlgorithmOptimizer import GeneticAlgorithmOptimizer

# ── Crear y ajustar el GA ─────────────────────────────────────────────────────
optimizer = GeneticAlgorithmOptimizer(
    population_size  = 300,   # más individuos → mejor exploración
    elite_size       = 30,    # ~10 % de élite
    mutation_rate    = 0.05,  # 2 % de probabilidad de mutación
    tournament_size  = 5,
    crossover_method = 'order',     # OX crossover
    mutation_method  = 'inversion'  # inversión de subsecuencia
)

best = optimizer.fit(matrix, iterations=1000, mode='min',
                     early_stopping_count=80, verbose=True)

optimizer.plot()

# ── Imprimir mejor ruta ───────────────────────────────────────────────────────
ruta_nombres = [ciudades[i] for i in optimizer.best_path]
print("\nMejor ruta encontrada (GA):")
print(" → ".join(ruta_nombres))
print(f"  + regreso a {ruta_nombres[0]}")
print(f"\nCosto total: ${best:,.2f} MXN")
print(f"Índices   : {optimizer.best_path}")

# ── Visualización estática ────────────────────────────────────────────────────
fig_static, ax_s = plt.subplots(figsize=(12, 8))
ax_s.set_facecolor("#f0f4f8")
ax_s.scatter(coordenadas[:, 0], coordenadas[:, 1], color="steelblue", s=70, zorder=5)

ruta_ciclica = optimizer.best_path + [optimizer.best_path[0]]
ax_s.plot(coordenadas[ruta_ciclica, 0], coordenadas[ruta_ciclica, 1],
          '-r', linewidth=1.5, alpha=0.8, zorder=4)

for i, nombre in enumerate(ciudades):
    ax_s.annotate(nombre, (coordenadas[i, 0], coordenadas[i, 1]),
                  textcoords="offset points", xytext=(4, 4), fontsize=7)

ax_s.set_xlim(-118, -86); ax_s.set_ylim(14, 35)
ax_s.set_xlabel("Longitud"); ax_s.set_ylabel("Latitud")
ax_s.set_title(f"Mejor ruta GA  |  Costo: ${best:,.0f} MXN", fontsize=13, fontweight='bold')
ax_s.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ruta_ga_estatica.png", dpi=150)
plt.show()
print("Imagen estática guardada como ruta_ga_estatica.png")

# ── GIF animado ───────────────────────────────────────────────────────────────
ruta        = optimizer.best_path          # lista de índices
ruta_cicl   = ruta + [ruta[0]]             # cerrar el ciclo para el frame final

fig_gif, ax_g = plt.subplots(figsize=(12, 8))

def animate(frame):
    ax_g.clear()
    ax_g.set_facecolor("#f0f4f8")
    ax_g.set_xlim(-118, -86); ax_g.set_ylim(14, 35)
    ax_g.grid(True, alpha=0.3)

    # Todas las ciudades de fondo
    ax_g.scatter(coordenadas[:, 0], coordenadas[:, 1],
                 color="steelblue", s=60, zorder=5)
    for i, nombre in enumerate(ciudades):
        ax_g.annotate(nombre, (coordenadas[i, 0], coordenadas[i, 1]),
                      textcoords="offset points", xytext=(4, 4), fontsize=6.5)

    # Segmentos ya recorridos
    segs = min(frame + 1, len(ruta))
    for k in range(segs):
        a = ruta_cicl[k]
        b = ruta_cicl[k + 1]
        ax_g.plot([coordenadas[a, 0], coordenadas[b, 0]],
                  [coordenadas[a, 1], coordenadas[b, 1]],
                  '-r', lw=1.8, alpha=0.85, zorder=4)

    # Ciudad actual (punto rojo grande)
    curr = ruta_cicl[min(frame, len(ruta_cicl) - 1)]
    ax_g.scatter(coordenadas[curr, 0], coordenadas[curr, 1],
                 color="red", s=130, zorder=6)

    # Título
    if frame >= len(ruta):
        ax_g.set_title(f"Ruta óptima — GA  |  Costo: ${best:,.0f} MXN",
                       fontsize=13, fontweight='bold')
    else:
        ax_g.set_title("Ruta óptima — Algoritmo Genético", fontsize=13, fontweight='bold')

    ax_g.text(0.02, 0.97,
              f"Visitadas: {min(frame + 1, len(ruta))}/{len(ruta)} ciudades",
              transform=ax_g.transAxes, fontsize=9, va='top', color='#333')

frames_totales = len(ruta_cicl) + 5   # + 5 frames de pausa al final

ani = animation.FuncAnimation(
    fig_gif, animate,
    frames=frames_totales,
    interval=320,     # ms entre frames  (~3 fps)
    repeat=True
)

ani.save("ruta_ga_animada.gif", writer="pillow", fps=3, dpi=120)
print("GIF guardado como ruta_ga_animada.gif")
plt.show()
