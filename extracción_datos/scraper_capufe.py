"""
scraper_capufe.py  v9
======================
Nombres de ciudad verificados directamente desde el endpoint
admin-ajax.php del sitio. Son los nombres exactos que aparecen
en el autocomplete.

INSTALACIÓN:
  pip install playwright pandas
  playwright install chromium

USO:
  python scraper_capufe.py
"""

import asyncio
import csv
import re
import time
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# CONFIGURACIÓN

CONSUMO_KML         = 18.8
TIPO_COMBUSTIBLE    = "Gasolina magna"
TIPO_VEHICULO       = "Auto"
VALOR_HORA_VENDEDOR = 187.5

HEADLESS          = False
TIMEOUT_RESULTADO = 30
PAUSA             = 1.0

OUT_DIR          = Path(".")
FILE_TIEMPO      = OUT_DIR / "matriz_tiempo_horas.csv"
FILE_CASETAS     = OUT_DIR / "matriz_casetas_mxn.csv"
FILE_COMBUSTIBLE = OUT_DIR / "matriz_combustible_mxn.csv"
FILE_COSTO_TOTAL = OUT_DIR / "matriz_costo_total_mxn.csv"
FILE_PROGRESO    = OUT_DIR / "progreso.csv"

# 32 CAPITALES — verificadas contra el endpoint real
# "ciudad"  = nombre en la matriz final
# "buscar"  = texto a escribir en el campo
# "elegir"  = texto exacto que aparece en el dropdown para hacer click

CAPITALES = [
    {"ciudad": "Aguascalientes",     "buscar": "Aguascalientes",  "elegir": "Aguascalientes"},
    {"ciudad": "Mexicali",           "buscar": "Mexicali",        "elegir": "Mexicali"},
    {"ciudad": "La Paz",             "buscar": "La Paz",          "elegir": "La Paz"},
    {"ciudad": "Campeche",           "buscar": "Campeche",        "elegir": "San Francisco de Campeche"},
    {"ciudad": "Tuxtla Gutiérrez",   "buscar": "Tuxtla",          "elegir": "Tuxtla Gutiérrez"},
    {"ciudad": "Chihuahua",          "buscar": "Chihuahua",       "elegir": "Chihuahua"},
    {"ciudad": "Ciudad de México",   "buscar": "Ciudad de Mexico","elegir": "Ciudad de México"},
    {"ciudad": "Saltillo",           "buscar": "Saltillo",        "elegir": "Saltillo"},
    {"ciudad": "Colima",             "buscar": "Colima",          "elegir": "Colima"},
    {"ciudad": "Victoria de Durango","buscar": "Durango",         "elegir": "Victoria de Durango"},
    {"ciudad": "Guanajuato",         "buscar": "Guanajuato",      "elegir": "Guanajuato"},
    {"ciudad": "Chilpancingo",       "buscar": "Chilpancingo",    "elegir": "Chilpancingo de los Bravo"},
    {"ciudad": "Pachuca",            "buscar": "Pachuca",         "elegir": "Pachuca de Soto"},
    {"ciudad": "Guadalajara",        "buscar": "Guadalajara",     "elegir": "Guadalajara"},
    {"ciudad": "Toluca",             "buscar": "Toluca",          "elegir": "Toluca de Lerdo"},
    {"ciudad": "Morelia",            "buscar": "Morelia",         "elegir": "Morelia"},
    {"ciudad": "Cuernavaca",         "buscar": "Cuernavaca",      "elegir": "Cuernavaca"},
    {"ciudad": "Tepic",              "buscar": "Tepic",           "elegir": "Tepic"},
    {"ciudad": "Monterrey",          "buscar": "Monterrey",       "elegir": "Monterrey"},
    {"ciudad": "Oaxaca",             "buscar": "Oaxaca",          "elegir": "Oaxaca de Juárez"},
    {"ciudad": "Puebla",             "buscar": "Puebla",          "elegir": "Heroica Puebla de Zaragoza"},
    {"ciudad": "Querétaro",          "buscar": "Queretaro",       "elegir": "Santiago de Querétaro"},
    {"ciudad": "Chetumal",           "buscar": "Chetumal",        "elegir": "Chetumal"},
    {"ciudad": "San Luis Potosí",    "buscar": "San Luis Potosi", "elegir": "San Luis Potosí"},
    {"ciudad": "Culiacán",           "buscar": "Culiacan",        "elegir": "Culiacán Rosales"},
    {"ciudad": "Hermosillo",         "buscar": "Hermosillo",      "elegir": "Hermosillo"},
    {"ciudad": "Villahermosa",       "buscar": "Villahermosa",    "elegir": "Villahermosa"},
    {"ciudad": "Ciudad Victoria",    "buscar": "Ciudad Victoria", "elegir": "Ciudad Victoria"},
    {"ciudad": "Tlaxcala", "buscar": "Tlaxcala de Xicoht", "elegir": "Tlaxcala de Xicohténcatl"},
    {"ciudad": "Xalapa",             "buscar": "Xalapa",          "elegir": "Xalapa-Enríquez"},
    {"ciudad": "Mérida",             "buscar": "Merida",          "elegir": "Mérida"},
    {"ciudad": "Zacatecas",          "buscar": "Zacatecas",       "elegir": "Zacatecas"},
]

# ── MODO PRUEBA: descomenta para probar solo 3 ciudades ──
'''CAPITALES = [
     {"ciudad": "Guadalajara",      "buscar": "Guadalajara",      "elegir": "Guadalajara"},
     {"ciudad": "Ciudad de México", "buscar": "Ciudad de Mexico", "elegir": "Ciudad de México"},
     {"ciudad": "Monterrey",        "buscar": "Monterrey",        "elegir": "Monterrey"},
     {"ciudad": "Tlaxcala", "buscar": "Tlaxcala de Xicoht", "elegir": "Tlaxcala de Xicohténcatl"}
 ]'''

N       = len(CAPITALES)
NOMBRES = [c["ciudad"] for c in CAPITALES]

# UTILIDADES

def tiempo_a_horas(texto: str) -> float:
    horas = minutos = 0
    m = re.search(r"(\d+)\s*hora", texto)
    if m:
        horas = int(m.group(1))
    m = re.search(r"(\d+)\s*minuto", texto)
    if m:
        minutos = int(m.group(1))
    return round(horas + minutos / 60, 4)

def extraer_numero(texto: str) -> float:
    texto = texto.replace(",", "")
    m = re.search(r"[\d]+(?:\.\d+)?", texto)
    return float(m.group()) if m else 0.0

def cargar_progreso() -> dict:
    done = {}
    if FILE_PROGRESO.exists():
        with open(FILE_PROGRESO, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["origen"], row["destino"])
                done[key] = {
                    "tiempo_horas":    float(row["tiempo_horas"]),
                    "casetas_mxn":     float(row["casetas_mxn"]),
                    "combustible_mxn": float(row["combustible_mxn"]),
                }
    return done

def guardar_fila(origen, destino, tiempo, casetas, combustible):
    existe = FILE_PROGRESO.exists()
    with open(FILE_PROGRESO, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not existe:
            w.writerow(["origen", "destino",
                        "tiempo_horas", "casetas_mxn", "combustible_mxn"])
        w.writerow([origen, destino, tiempo, casetas, combustible])

# BLOQUEAR RECURSOS PESADOS

BLOQUEAR = re.compile(
    r"(googlesyndication|doubleclick|googletagmanager|google-analytics"
    r"|adsbygoogle|pagead|fonts\.googleapis|fonts\.gstatic|gravatar\.com"
    r"|tile\.openstreetmap|maps\.googleapis)",
    re.IGNORECASE
)

async def bloquear_recursos(context):
    async def interceptar(route):
        url  = route.request.url
        tipo = route.request.resource_type
        if tipo in ("image", "font", "media"):
            await route.abort()
            return
        if BLOQUEAR.search(url):
            await route.abort()
            return
        await route.continue_()
    await context.route("**/*", interceptar)

# CERRAR POPUPS

async def cerrar_popups(page):
    selectores = [
        "button:has-text('Aceptar')",
        "button:has-text('Accept')",
        "button:has-text('Cerrar')",
        "button:has-text('Close')",
        "button:has-text('OK')",
        "[class*='close']:visible",
        "[id*='cookie'] button:visible",
        "[class*='cookie'] button:visible",
        "[class*='modal'] button:visible",
    ]
    for sel in selectores:
        try:
            el = page.locator(sel).first
            if await el.is_visible():
                await el.click()
                await page.wait_for_timeout(400)
        except Exception:
            pass

# LLENAR CIUDAD

async def llenar_ciudad(page, campo_id: str, buscar: str, elegir: str) -> bool:
    # Cerrar cualquier dropdown previo
    await page.evaluate("document.body.click()")
    await page.wait_for_timeout(300)

    # Limpiar y enfocar
    await page.evaluate(f"""
        const el = document.getElementById('{campo_id}');
        el.value = '';
        el.focus();
    """)
    await page.wait_for_timeout(300)

    # Escribir letra por letra — jQuery UI necesita esto
    await page.type(f"#{campo_id}", buscar, delay=120)
    await page.wait_for_timeout(1500)

    try:
        # Esperar dropdown visible
        await page.wait_for_selector(
            "ul.ui-autocomplete:visible li.ui-menu-item",
            state="visible", timeout=7000
        )
        await page.wait_for_timeout(300)

        # Buscar coincidencia exacta con "elegir"
        items = await page.locator("ul.ui-autocomplete:visible li.ui-menu-item").all()
        elegir_lower = elegir.lower()
        clicked = False

        for item in items:
            texto_item = (await item.inner_text()).strip().lower()
            if elegir_lower in texto_item:
                await item.click()
                clicked = True
                break

        if not clicked:
            # Tomar la primera si no hay coincidencia exacta
            await page.locator("ul.ui-autocomplete:visible li.ui-menu-item").first.click()
            clicked = True

        await page.wait_for_timeout(500)
        return clicked

    except PWTimeout:
        print(f"    ✗ Sin dropdown para '{buscar}'")
        return False

# MARCAR CHECKBOX VIA JS

async def marcar_checkbox(page, checkbox_id: str):
    await page.evaluate(f"""
        const cb = document.getElementById('{checkbox_id}');
        if (!cb.checked) {{
            cb.checked = true;
            cb.dispatchEvent(new Event('change', {{bubbles: true}}));
            cb.dispatchEvent(new Event('click',  {{bubbles: true}}));
        }}
    """)
    await page.wait_for_timeout(300)

# CONSULTAR RUTA

async def consultar_ruta(page, origen: dict, destino: dict) -> dict | None:
    URL = "https://tarifascapufe.com.mx/traza-tu-ruta/"
    await page.goto(URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(1500)

    await cerrar_popups(page)

    ok_origen = await llenar_ciudad(
        page, "ciudad-origen", origen["buscar"], origen["elegir"]
    )
    ok_destino = await llenar_ciudad(
        page, "ciudad-destino", destino["buscar"], destino["elegir"]
    )

    if not ok_origen or not ok_destino:
        return None

    await page.select_option("#tipo-vehiculo", label=TIPO_VEHICULO)
    await marcar_checkbox(page, "incluir-combustible")
    await page.wait_for_timeout(300)
    await page.select_option("#tipo-combustible", label=TIPO_COMBUSTIBLE)
    await marcar_checkbox(page, "conozco-gasto-combustible")
    await page.evaluate(
        f"document.getElementById('gasto-combustible').value = '{CONSUMO_KML}'"
    )

    await page.evaluate("document.getElementById('miBoton').scrollIntoView()")
    await page.wait_for_timeout(400)
    await page.evaluate("document.getElementById('miBoton').click()")

    try:
        await page.wait_for_function(
            """() => document.body.innerText.includes('Costo peajes')
                   || document.body.innerText.includes('distancia de')""",
            timeout=TIMEOUT_RESULTADO * 1000
        )
    except PWTimeout:
        return None

    await page.wait_for_timeout(600)
    texto = await page.inner_text("body")

    tiempo = 0.0
    m = re.search(r"trayecto es de\s*([^\n.]+)", texto, re.IGNORECASE)
    if m:
        tiempo = tiempo_a_horas(m.group(1))

    casetas = 0.0
    m = re.search(r"Costo peajes?[:\s]+([\d,. ]+)\s*MXN", texto, re.IGNORECASE)
    if m:
        casetas = extraer_numero(m.group(1))

    combustible = 0.0
    m = re.search(r"Costo combustible[:\s]+([\d,. ]+)\s*MXN", texto, re.IGNORECASE)
    if m:
        combustible = extraer_numero(m.group(1))

    return {
        "tiempo_horas":    tiempo,
        "casetas_mxn":     casetas,
        "combustible_mxn": combustible,
    }

# MAIN

async def main():
    progreso = cargar_progreso()
    total    = N * (N - 1)
    hechos   = len(progreso)

    print(f"\n{'='*58}")
    print(f"  Vehículo   : {TIPO_VEHICULO} · {CONSUMO_KML} km/l · {TIPO_COMBUSTIBLE}")
    print(f"  Valor hora : ${VALOR_HORA_VENDEDOR} MXN/h")
    print(f"  Modo       : {'sin ventana' if HEADLESS else 'con ventana'}")
    print(f"  Pares      : {total} total | {hechos} hechos | {total-hechos} pendientes")
    print(f"{'='*58}\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=HEADLESS,
            slow_mo=0,
            args=["--window-size=1280,900", "--no-sandbox"]
        )
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        await bloquear_recursos(context)
        page    = await context.new_page()

        errores  = []
        t_inicio = time.time()

        for i, origen in enumerate(CAPITALES):
            for j, destino in enumerate(CAPITALES):
                if i == j:
                    continue

                key = (origen["ciudad"], destino["ciudad"])
                if key in progreso:
                    continue

                pendientes = total - hechos
                if hechos > 0:
                    seg_por_par = (time.time() - t_inicio) / hechos
                    eta = round(pendientes * seg_por_par / 60)
                    eta_str = f"  ETA ~{eta} min"
                else:
                    eta_str = ""

                print(f"[{hechos+1}/{total}] {origen['ciudad']} → {destino['ciudad']}{eta_str}")

                datos = None
                for intento in range(3):
                    try:
                        datos = await consultar_ruta(page, origen, destino)
                        if datos is not None:
                            break
                        print(f"    ⚠ Timeout intento {intento+1}/3")
                    except Exception as e:
                        print(f"    ✗ Error intento {intento+1}/3: {e}")

                if datos is None:
                    print(f"    ✗ Falló tras 3 intentos — guardando como 0")
                    datos = {"tiempo_horas": 0, "casetas_mxn": 0, "combustible_mxn": 0}
                    errores.append(key)

                progreso[key] = datos
                guardar_fila(
                    origen["ciudad"], destino["ciudad"],
                    datos["tiempo_horas"],
                    datos["casetas_mxn"],
                    datos["combustible_mxn"]
                )
                hechos += 1

                costo = (datos["casetas_mxn"] + datos["combustible_mxn"]
                         + datos["tiempo_horas"] * VALOR_HORA_VENDEDOR)
                print(f"    ✓ {datos['tiempo_horas']}h | "
                      f"casetas ${datos['casetas_mxn']} | "
                      f"combustible ${datos['combustible_mxn']} | "
                      f"total ${costo:.2f} MXN")

                await asyncio.sleep(PAUSA)

        await browser.close()

    t_total = round((time.time() - t_inicio) / 60, 1)
    print(f"\nTiempo total: {t_total} minutos")

    print("Construyendo matrices 32×32...")
    mat_time = pd.DataFrame(0.0, index=NOMBRES, columns=NOMBRES)
    mat_cas  = pd.DataFrame(0.0, index=NOMBRES, columns=NOMBRES)
    mat_comb = pd.DataFrame(0.0, index=NOMBRES, columns=NOMBRES)

    for (orig, dest), datos in progreso.items():
        if orig in NOMBRES and dest in NOMBRES:
            mat_time.loc[orig, dest] = datos["tiempo_horas"]
            mat_cas.loc[orig, dest]  = datos["casetas_mxn"]
            mat_comb.loc[orig, dest] = datos["combustible_mxn"]

    mat_total = mat_cas + mat_comb + mat_time * VALOR_HORA_VENDEDOR

    mat_time.to_csv(FILE_TIEMPO)
    mat_cas.to_csv(FILE_CASETAS)
    mat_comb.to_csv(FILE_COMBUSTIBLE)
    mat_total.to_csv(FILE_COSTO_TOTAL)

    print(f"\n✅ 3 tablas base:")
    print(f"   {FILE_TIEMPO}")
    print(f"   {FILE_CASETAS}")
    print(f"   {FILE_COMBUSTIBLE}")
    print(f"\n✅ Tabla de costo total (valor_hora=${VALOR_HORA_VENDEDOR} MXN/h):")
    print(f"   {FILE_COSTO_TOTAL}")

    if errores:
        print(f"\n⚠ {len(errores)} pares fallidos — vuelve a correr para reintentar:")
        for e in errores:
            print(f"   {e[0]} → {e[1]}")

if __name__ == "__main__":
    asyncio.run(main())