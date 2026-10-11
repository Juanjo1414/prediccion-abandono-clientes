"""
App Gradio: riesgo de abandono de clientes (proyecto de Fundamentos de IA, Universidad EIA).

Carga el modelo que dejó el notebook 04 (models/modelo_final.joblib y models/metadata_modelo.json) y deja:
  1. Evaluar un cliente: llenar sus datos a mano (o cargar un cliente real de la muestra de prueba) y ver su
     probabilidad calibrada de abandono, su nivel de riesgo y qué factores empujan ese riesgo.
  2. Priorizar contactos: subir un CSV con varios clientes y obtener la lista ordenada de mayor a menor riesgo.
  3. Sobre el modelo: qué modelo es, con qué umbral y qué tan bien funcionó en prueba.

Cómo correrla:
  - Local:  uv run python app.py   (se abre en http://127.0.0.1:7860)
  - Colab:  poner EN_COLAB = True aquí abajo y ejecutar  %run app.py  desde la carpeta del proyecto en Drive.

Es un proyecto académico. Sirve para priorizar a quién contactar; no estima si una oferta evitará el abandono
ni promete ahorros.
"""
import os
import json
import math
import tempfile
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Si corremos en Google Colab ponemos True; en el computador (VS Code), False
EN_COLAB = False

if EN_COLAB:
    # En Colab los archivos viven en Drive, así que primero lo montamos
    from google.colab import drive
    drive.mount("/content/drive")
    RUTA_BASE = "/content/drive/MyDrive/churn/"
else:
    # Así la app funciona sin importar desde qué carpeta se lance
    RUTA_BASE = os.path.dirname(os.path.abspath(__file__)) + "/"

# Todas las rutas de la app salen de aquí, para no tener rutas regadas por el código
RUTA_DATOS = RUTA_BASE + "data/"
RUTA_MODELOS = RUTA_BASE + "models/"
RUTA_RESULTADOS = RUTA_BASE + "resultados/"

# Campos del formulario, en el orden en que se muestran. La edad no se pide: en estos datos es una función
# exacta del grupo de edad y se calcula sola.
NOMBRES_CAMPOS = ["segundos_uso", "frecuencia_uso", "frecuencia_sms", "numeros_distintos", "fallas_llamadas",
                  "antiguedad", "nivel_cargo", "plan_tarifario", "quejas", "grupo_edad"]

ETIQUETAS = {
    "segundos_uso": "Segundos de uso",
    "frecuencia_uso": "Número de llamadas",
    "frecuencia_sms": "Número de mensajes",
    "numeros_distintos": "Números distintos llamados",
    "fallas_llamadas": "Llamadas fallidas",
    "antiguedad": "Antigüedad (meses)",
    "nivel_cargo": "Nivel de cargo",
    "plan_tarifario": "Plan tarifario",
    "quejas": "Queja presentada",
    "grupo_edad": "Grupo de edad",
    "edad": "Edad",
}

DESCRIPCIONES = {
    "segundos_uso": "Segundos de llamadas en los primeros nueve meses",
    "frecuencia_uso": "Cantidad de llamadas realizadas",
    "frecuencia_sms": "Cantidad de mensajes de texto enviados",
    "numeros_distintos": "Cantidad de números distintos a los que llamó",
    "fallas_llamadas": "Llamadas que fallaron",
    "antiguedad": "Meses desde que se suscribió",
    "nivel_cargo": "Nivel de cargo de 0 a 9 (no son unidades monetarias)",
    "plan_tarifario": "Tipo de plan contratado",
    "quejas": "Si el cliente presentó alguna queja (no cuenta cuántas)",
    "grupo_edad": "Grupo de edad del cliente, de 1 a 5",
}

# El archivo original de UCI trae los nombres en inglés (con espacios dobles); aceptamos ambos en el CSV
TRADUCCION = {
    "Call  Failure": "fallas_llamadas", "Complains": "quejas", "Subscription  Length": "antiguedad",
    "Charge  Amount": "nivel_cargo", "Seconds of Use": "segundos_uso", "Frequency of use": "frecuencia_uso",
    "Frequency of SMS": "frecuencia_sms", "Distinct Called Numbers": "numeros_distintos",
    "Age Group": "grupo_edad", "Tariff Plan": "plan_tarifario", "Age": "edad",
}

OPCIONES_QUEJA = {"No": 0, "Sí": 1}
OPCIONES_PLAN = {"Pago por uso": 1, "Contractual": 2}

# Por debajo de este riesgo el cliente es de riesgo bajo. Es una convención de la app (no del modelo) para
# separar a los clientes sin alerta en dos grupos; el umbral de alerta sí es el del modelo.
RIESGO_BAJO_HASTA = 0.10

COLOR_BAJO = "#0F8B7A"
COLOR_MEDIO = "#D97706"
COLOR_ALTO = "#DC2626"

CABECERA_HTML = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
                 '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                 '<link href="https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&'
                 'family=Fira+Sans+Condensed:wght@500;600;700&family=Fira+Code:wght@500;600&display=swap" rel="stylesheet">')

ESTILOS_CSS = """
:root {
  --tinta: #0B1B33; --azul: #1E40AF; --azul2: #3B82F6; --ambar: #F59E0B; --papel: #F4F7FB; --borde: #DCE4EF;
  --texto: #0F172A; --suave: #475569; --bajo: #0F8B7A; --medio: #D97706; --alto: #DC2626;
  --titulo: 'Fira Sans Condensed', 'Arial Narrow', 'Roboto Condensed', sans-serif;
  --numero: 'Fira Code', Consolas, monospace;
}
body, .gradio-container, .dark .gradio-container { background: var(--papel) !important; }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; font-family: 'Fira Sans', 'Segoe UI', Arial, sans-serif !important; }
.gradio-container, .gradio-container .prose, .gradio-container label span { color: var(--texto); }
footer { display: none !important; }

/* Encabezado: barras de señal que suben una sola vez al abrir la app */
.appbar { display: flex; gap: 28px; align-items: center; justify-content: space-between; flex-wrap: wrap;
  background: var(--tinta); color: #E8EEF8; border-radius: 16px; padding: 26px 34px; margin-bottom: 14px; }
.appbar .marca { display: flex; gap: 22px; align-items: center; }
.appbar .senal { width: 62px; height: 52px; flex: none; }
.appbar .senal rect { fill: var(--ambar); transform-origin: bottom; transform-box: fill-box; animation: subir .7s cubic-bezier(.2,.8,.2,1) backwards; }
.appbar .senal rect:nth-child(1) { animation-delay: .05s; } .appbar .senal rect:nth-child(2) { animation-delay: .15s; }
.appbar .senal rect:nth-child(3) { animation-delay: .25s; } .appbar .senal rect:nth-child(4) { animation-delay: .35s; }
.appbar .senal rect:nth-child(5) { animation-delay: .45s; }
@keyframes subir { from { transform: scaleY(0); } }
.appbar h1 { font-family: var(--titulo); font-weight: 700; font-size: 38px; line-height: 1; margin: 0 0 6px; color: #fff; }
.appbar p { margin: 0; color: #B5C4DD; font-size: 16px; max-width: 58ch; line-height: 1.4; }
.appbar .datos { display: flex; gap: 10px; flex-wrap: wrap; }
.appbar .dato { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); border-radius: 10px; padding: 8px 14px; min-width: 112px; }
.appbar .dato b { display: block; font-family: var(--numero); font-size: 22px; font-weight: 600; color: #fff; }
.appbar .dato span { font-size: 13px; color: #B5C4DD; }

/* Pestañas */
button[role="tab"] { font-family: var(--titulo) !important; font-size: 19px !important; font-weight: 600 !important; color: var(--suave) !important; cursor: pointer; }
button[role="tab"][aria-selected="true"] { color: var(--tinta) !important; border-bottom: 3px solid var(--ambar) !important; }

/* Tarjetas */
.tarjeta { background: #fff !important; border: 1px solid var(--borde) !important; border-radius: 14px !important; padding: 20px 22px !important; }
.tarjeta .form, .tarjeta .row { border: none !important; box-shadow: none !important; background: transparent !important; }
.titulo-tarjeta { font-family: var(--titulo); font-size: 24px; font-weight: 600; line-height: 1.1; margin: 0 0 2px; color: var(--tinta); }
.sub-tarjeta { color: var(--suave); font-size: 15px; margin: 0 0 8px; line-height: 1.4; }
.grupo-form { font-family: var(--titulo); font-size: 17px; font-weight: 600; color: var(--azul); margin: 10px 0 0; padding-bottom: 4px; border-bottom: 1px solid var(--borde); }
.tarjeta label span, .tarjeta .block-info { font-size: 14px; }
.tarjeta input[type="number"] { font-family: var(--numero); font-size: 17px !important; }

/* Botones */
.gradio-container button.primary { background: var(--ambar) !important; border: 2px solid var(--ambar) !important; color: var(--tinta) !important; font-weight: 600 !important; font-size: 17px !important; cursor: pointer; transition: background .2s; }
.gradio-container button.primary:hover { background: #FBBF24 !important; border-color: #FBBF24 !important; }
.gradio-container button.secondary { background: #fff !important; border: 2px solid var(--azul) !important; color: var(--azul) !important; font-weight: 600 !important; font-size: 15px !important; cursor: pointer; transition: background .2s, color .2s; }
.gradio-container button.secondary:hover { background: var(--azul) !important; color: #fff !important; }
.gradio-container button:focus-visible { outline: 3px solid var(--azul2) !important; outline-offset: 2px; }

/* Resultado */
.resultado { text-align: center; padding: 4px 4px 0; }
.nivel { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 6px 16px; font-weight: 600; font-size: 16px; margin-bottom: 6px; color: #fff; }
.nivel.bajo { background: var(--bajo); } .nivel.medio { background: var(--medio); } .nivel.alto { background: var(--alto); }
.nivel svg { width: 18px; height: 18px; }
.resultado .anillo { width: 100%; max-width: 250px; height: auto; display: block; margin: 4px auto 0; }
.resultado .anillo .progreso { animation: llenar .9s cubic-bezier(.2,.8,.2,1); }
@keyframes llenar { from { stroke-dasharray: 0 600; } }
.resultado .accion { color: var(--texto); font-size: 17px; font-weight: 500; margin: 4px auto 0; max-width: 36ch; line-height: 1.35; }
.resultado .fichas { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin: 12px 0 6px; }
.resultado .ficha { background: var(--papel); border-radius: 999px; padding: 4px 12px; font-size: 13px; color: var(--suave); }
.factores { text-align: left; margin-top: 12px; border-top: 1px solid var(--borde); padding-top: 12px; }
.factores h4 { font-family: var(--titulo); font-size: 17px; font-weight: 600; margin: 0 0 8px; color: var(--tinta); }
.factor { display: grid; grid-template-columns: 1fr 120px 52px; gap: 8px; align-items: center; font-size: 14px; margin: 6px 0; }
.factor .barra { height: 10px; background: var(--papel); border-radius: 5px; position: relative; overflow: hidden; }
.factor .barra i { position: absolute; top: 0; bottom: 0; border-radius: 5px; }
.factor .barra i.sube { left: 50%; background: var(--alto); } .factor .barra i.baja { right: 50%; background: var(--bajo); }
.factor .valor { font-family: var(--numero); font-size: 13px; text-align: right; }
.nota-factores { color: var(--suave); font-size: 12.5px; margin-top: 6px; line-height: 1.35; }
.vacio { text-align: center; color: var(--suave); padding: 40px 16px 30px; }
.vacio .titulo-tarjeta { color: var(--suave); margin-bottom: 8px; }
.vacio .punto { width: 84px; height: 84px; border-radius: 50%; border: 3px dashed var(--borde); margin: 0 auto 14px; }
.real { margin-top: 10px; padding: 10px 14px; border-radius: 10px; font-size: 15px; text-align: left; }
.real.si { background: #FDE8E8; border-left: 4px solid var(--alto); } .real.no { background: #E1F4F0; border-left: 4px solid var(--bajo); }
.aviso { background: #FEF3C7; border-left: 4px solid var(--ambar); border-radius: 0 10px 10px 0; padding: 8px 12px; font-size: 13.5px; text-align: left; margin-top: 10px; }

/* Resumen de la lista y del modelo */
.resumen-lista { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 10px 0; }
.kpi { background: var(--papel); border-radius: 12px; padding: 12px 14px; }
.kpi .nombre { color: var(--suave); font-size: 13px; }
.kpi .valor { font-family: var(--numero); font-size: 26px; font-weight: 600; color: var(--tinta); line-height: 1.15; }
.kpi .detalle { color: var(--suave); font-size: 12.5px; }
.fichas-modelo { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 16px; }
.fichas-modelo span { background: var(--tinta); color: #E8EEF8; border-radius: 8px; padding: 5px 12px; font-size: 14px; }
.nota-modelo { border-left: 4px solid var(--ambar); background: #FEF9E7; border-radius: 0 10px 10px 0; padding: 12px 16px; margin-top: 14px; font-size: 15px; line-height: 1.45; }
.gradio-container table { font-variant-numeric: tabular-nums; }
.pie { text-align: center; color: var(--suave); font-size: 13px; padding: 14px 0 4px; }

@media (max-width: 760px) { .appbar { padding: 20px; } .appbar h1 { font-size: 30px; } .factor { grid-template-columns: 1fr 80px 48px; } }
@media (prefers-reduced-motion: reduce) { .appbar .senal rect, .resultado .anillo .progreso { animation: none; } }
"""

ICONO_BAJO = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" '
              'stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>')
ICONO_ALERTA = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" '
                'stroke-linejoin="round" aria-hidden="true"><path d="M12 9v4M12 17h.01"/>'
                '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>')


def cargar_artefactos():
    """
    Lee lo que dejaron los notebooks: el modelo final (pipeline, calibrador, umbral y columnas), la metadata,
    los datos de entrenamiento (para los valores típicos y los rangos) y los clientes reales de prueba que
    sirven de ejemplo. Devuelve las cuatro cosas, en ese orden.
    """
    artefacto = joblib.load(RUTA_MODELOS + "modelo_final.joblib")
    with open(RUTA_MODELOS + "metadata_modelo.json", encoding="utf-8") as archivo:
        metadata = json.load(archivo)
    train = pd.read_csv(RUTA_DATOS + "train.csv")
    clientes_ejemplo = pd.read_csv(RUTA_DATOS + "test.csv")
    return artefacto, metadata, train, clientes_ejemplo


# Se cargan una sola vez, al abrir la app. Los botones de Gradio no pueden recibir parámetros extra, así que
# las funciones de abajo leen estas variables globales (solo las leen, nunca las modifican)
ARTEFACTO, METADATA, TRAIN, EJEMPLOS = cargar_artefactos()
COLUMNAS = ARTEFACTO["columnas_entrada"]
UMBRAL = ARTEFACTO["umbral"]


def obtener_puntaje(pipeline, X):
    """
    Devuelve un puntaje continuo para cada fila de X: la función de decisión si el modelo la tiene y, si no,
    el logit de la probabilidad que predice. Es lo que recibe el calibrador (copia del notebook 04).
    """
    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(X)).ravel()
    probabilidades = np.clip(pipeline.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(probabilidades / (1 - probabilidades))


def predecir_probabilidad(X):
    """
    Recibe un DataFrame con las columnas de entrada del modelo y devuelve la probabilidad calibrada de abandono
    de cada fila: primero el puntaje del pipeline y luego el calibrador, igual que en el notebook 04.
    """
    puntajes = obtener_puntaje(ARTEFACTO["pipeline"], X[COLUMNAS])
    return ARTEFACTO["calibrador"].predict_proba(puntajes.reshape(-1, 1))[:, 1]


def calcular_referencias(train):
    """
    Calcula, con los datos de entrenamiento, el valor típico (mediana) y el rango de cada entrada, y la edad que
    corresponde a cada grupo de edad. Devuelve tres diccionarios: medianas, rangos y edad por grupo.
    """
    medianas = {}
    rangos = {}
    for columna in COLUMNAS:
        medianas[columna] = float(train[columna].median())
        rangos[columna] = (float(train[columna].min()), float(train[columna].max()))
    edad_por_grupo = {}
    for grupo, edad in train.groupby("grupo_edad")["edad"].first().items():
        edad_por_grupo[int(grupo)] = int(edad)
    return medianas, rangos, edad_por_grupo


MEDIANAS, RANGOS, EDAD_POR_GRUPO = calcular_referencias(TRAIN)

# Probabilidad de cada cliente de prueba, calculada una vez; sirve para elegir clientes de ejemplo según su riesgo
PROBABILIDADES_EJEMPLOS = predecir_probabilidad(EJEMPLOS)


def formato_numero(valor, decimales):
    """Devuelve un número como texto con coma decimal (6,6) y no con punto, como escribimos en todo el proyecto."""
    return ("{:." + str(decimales) + "f}").format(valor).replace(".", ",")


def nivel_de_riesgo(probabilidad):
    """
    Clasifica una probabilidad en bajo, medio o alto. Alto es desde el umbral del modelo (la alerta); medio, entre
    RIESGO_BAJO_HASTA y el umbral; bajo, por debajo. Devuelve el nombre del nivel, su color y la acción sugerida.
    """
    if probabilidad >= UMBRAL:
        return "alto", COLOR_ALTO, "Incluir en la lista de contacto prioritario de retención."
    if probabilidad >= RIESGO_BAJO_HASTA:
        return "medio", COLOR_MEDIO, "Mantener en seguimiento; contactar si hay capacidad de sobra."
    return "bajo", COLOR_BAJO, "Sin acción especial de retención por ahora."


def anillo_svg(probabilidad, color):
    """
    Dibuja el anillo del riesgo como SVG: una pista gris, un arco de progreso del color del nivel y el porcentaje
    en el centro. Lo hacemos a mano para poder animarlo con CSS sin depender de ninguna librería gráfica.
    """
    radio = 78
    circunferencia = 2 * math.pi * radio
    lleno = circunferencia * min(max(probabilidad, 0.0), 1.0)
    return ('<svg class="anillo" viewBox="0 0 200 200" role="img" aria-label="Probabilidad de abandono: {texto}%">'
            '<circle cx="100" cy="100" r="{r}" fill="none" stroke="#E6ECF5" stroke-width="18"/>'
            '<circle class="progreso" cx="100" cy="100" r="{r}" fill="none" stroke="{color}" stroke-width="18" '
            'stroke-linecap="round" stroke-dasharray="{lleno:.1f} {total:.1f}" transform="rotate(-90 100 100)"/>'
            '<text x="100" y="106" text-anchor="middle" font-size="44" font-weight="700" fill="#0B1B33" '
            'style="font-family:var(--titulo)">{texto}%</text>'
            '<text x="100" y="128" text-anchor="middle" font-size="11" fill="#475569">probabilidad de abandono</text>'
            '</svg>').format(r=radio, color=color, lleno=lleno, total=circunferencia,
                             texto=formato_numero(probabilidad * 100, 1))


def valor_para_mostrar(nombre, valor):
    """Devuelve el valor de una entrada como texto legible para el cliente (por ejemplo, 'Sí' en vez de 1)."""
    if nombre == "quejas":
        return "Sí" if int(valor) == 1 else "No"
    if nombre == "plan_tarifario":
        return "Contractual" if int(valor) == 2 else "Pago por uso"
    if nombre == "grupo_edad":
        return "Grupo " + str(int(valor))
    return str(int(round(valor))) if float(valor).is_integer() else formato_numero(valor, 1)


def factores_de_riesgo(fila, probabilidad):
    """
    Calcula qué entradas empujan el riesgo de este cliente. Para cada entrada sustituye su valor por el valor
    típico de los clientes de entrenamiento (la mediana) y mira cuánto cambia la probabilidad: si baja, esa
    entrada estaba empujando el riesgo hacia arriba. El grupo de edad y la edad se cambian juntos.
    Devuelve una lista de (nombre, cambio en puntos de probabilidad) ordenada de mayor a menor efecto.
    """
    nombres = []
    variantes = []
    for nombre in NOMBRES_CAMPOS:
        variante = fila.copy()
        variante[nombre] = MEDIANAS[nombre]
        if nombre == "grupo_edad":
            variante["edad"] = MEDIANAS["edad"]
        nombres.append(nombre)
        variantes.append(variante)
    probabilidades_variantes = predecir_probabilidad(pd.DataFrame(variantes))

    efectos = []
    for nombre, probabilidad_variante in zip(nombres, probabilidades_variantes):
        efectos.append((nombre, float(probabilidad - probabilidad_variante)))
    efectos.sort(key=abs_segundo, reverse=True)
    return efectos


def abs_segundo(pareja):
    """Devuelve el valor absoluto del segundo elemento de una pareja; sirve para ordenar los efectos."""
    return abs(pareja[1])


def html_factores(fila, probabilidad):
    """
    Arma el bloque "Qué empuja el riesgo de este cliente": las cuatro entradas con más efecto, con una barra
    que va a la derecha (sube el riesgo) o a la izquierda (lo baja) y el cambio en puntos porcentuales.
    """
    efectos = factores_de_riesgo(fila, probabilidad)
    filas = ""
    mayor = max(abs(efectos[0][1]), 0.01)
    for nombre, efecto in efectos[:4]:
        if abs(efecto) < 0.005:
            continue
        ancho = min(abs(efecto) / mayor, 1.0) * 50
        clase = "sube" if efecto > 0 else "baja"
        signo = "+" if efecto > 0 else "−"
        filas += ('<div class="factor"><span>{nombre}: <b>{valor}</b></span><div class="barra"><i class="{clase}" '
                  'style="width:{ancho:.0f}%"></i></div><span class="valor">{signo}{puntos}</span></div>').format(
                      nombre=ETIQUETAS[nombre], valor=valor_para_mostrar(nombre, fila[nombre]), clase=clase, ancho=ancho,
                      signo=signo, puntos=formato_numero(abs(efecto) * 100, 0))
    if filas == "":
        return ""
    return ('<div class="factores"><h4>Qué empuja el riesgo de este cliente</h4>' + filas
            + '<div class="nota-factores">Cambio en puntos porcentuales frente a un cliente con el valor típico de esa entrada '
            '(mediana de entrenamiento). Describe al modelo, no una causa: no prueba que cambiar ese dato cambie el comportamiento.</div></div>')


def aviso_fuera_de_rango(fila):
    """
    Revisa si alguna entrada queda fuera del rango visto en entrenamiento. Si es así devuelve un aviso en HTML
    (el modelo extrapola mal fuera de lo que vio); si no, devuelve un texto vacío.
    """
    fuera = []
    for nombre in NOMBRES_CAMPOS:
        minimo, maximo = RANGOS[nombre]
        if fila[nombre] < minimo or fila[nombre] > maximo:
            fuera.append(ETIQUETAS[nombre].lower())
    if len(fuera) == 0:
        return ""
    return ('<div class="aviso"><b>Valores fuera de lo que vio el modelo:</b> ' + ", ".join(fuera)
            + '. La probabilidad es menos confiable en esos casos.</div>')


def texto_resultado(fila, probabilidad):
    """
    Arma la tarjeta de resultado en HTML: nivel de riesgo, anillo con la probabilidad, acción sugerida, el umbral
    y el riesgo normal para dar contexto, los factores que empujan el riesgo y los avisos si hay datos atípicos.
    """
    nivel, color, accion = nivel_de_riesgo(probabilidad)
    nombre_nivel = {"bajo": "Riesgo bajo", "medio": "Riesgo medio", "alto": "Riesgo alto"}[nivel]
    icono = ICONO_BAJO if nivel == "bajo" else ICONO_ALERTA
    fichas = ('<div class="fichas"><span class="ficha">Umbral de alerta: {}%</span>'
              '<span class="ficha">Abandono promedio en los datos: {}%</span></div>').format(
                  formato_numero(UMBRAL * 100, 0), formato_numero(TRAIN["abandono"].mean() * 100, 1))
    return ('<div class="resultado"><span class="nivel {nivel}">{icono}{nombre}</span>{anillo}'
            '<div class="accion">{accion}</div>{fichas}</div>{factores}{aviso}').format(
                nivel=nivel, icono=icono, nombre=nombre_nivel, anillo=anillo_svg(probabilidad, color), accion=accion,
                fichas=fichas, factores=html_factores(fila, probabilidad), aviso=aviso_fuera_de_rango(fila))


def resultado_vacio():
    """Devuelve la tarjeta que se muestra antes de evaluar nada: invita a llenar el formulario o a cargar un cliente."""
    return ('<div class="vacio"><div class="punto"></div><div class="titulo-tarjeta">Aún no hay un cliente evaluado</div>'
            'Llena los datos y pulsa <b>Evaluar cliente</b>, o carga un cliente real de ejemplo para ver cómo responde el modelo.</div>')


def texto_valor_real(abandono):
    """Devuelve el aviso en HTML con lo que pasó de verdad con un cliente real de la muestra de prueba."""
    if int(abandono) == 1:
        return '<div class="real si"><b>Qué pasó de verdad:</b> este cliente abandonó al final del mes doce.</div>'
    return '<div class="real no"><b>Qué pasó de verdad:</b> este cliente permaneció al final del mes doce.</div>'


def convertir_valor(nombre, valor):
    """
    Convierte lo que entrega un campo del formulario a un número que entiende el modelo ('Sí' pasa a 1, 'Contractual'
    a 2, etc.). Si el campo está vacío lanza un gr.Error que dice cuál falta, en vez de dejar que la app se rompa.
    """
    if valor is None or (isinstance(valor, float) and math.isnan(valor)) or valor == "":
        raise gr.Error("Falta completar: " + ETIQUETAS[nombre] + ".")
    if nombre == "quejas":
        return OPCIONES_QUEJA[valor]
    if nombre == "plan_tarifario":
        return OPCIONES_PLAN[valor]
    numero = float(valor)
    if numero < 0:
        raise gr.Error(ETIQUETAS[nombre] + " no puede ser negativo.")
    return numero


def fila_desde_valores(valores):
    """
    Arma la fila de entradas del modelo a partir de los valores del formulario (en el orden de NOMBRES_CAMPOS).
    La edad no se pide: se calcula a partir del grupo de edad. Devuelve una Series con las columnas del modelo.
    """
    fila = {}
    for nombre, valor in zip(NOMBRES_CAMPOS, valores):
        fila[nombre] = convertir_valor(nombre, valor)
    if int(fila["grupo_edad"]) not in EDAD_POR_GRUPO:
        raise gr.Error("El grupo de edad debe ser un número del 1 al 5.")
    fila["edad"] = EDAD_POR_GRUPO[int(fila["grupo_edad"])]
    return pd.Series(fila)[COLUMNAS]


def evaluar_cliente(*valores):
    """
    Evalúa un cliente con los valores del formulario y devuelve la tarjeta de resultado en HTML.
    Usamos *valores porque Gradio entrega los campos del formulario como argumentos sueltos, en el mismo
    orden de NOMBRES_CAMPOS; es la única excepción a la regla de no usar *args.
    """
    fila = fila_desde_valores(valores)
    probabilidad = float(predecir_probabilidad(pd.DataFrame([fila]))[0])
    return texto_resultado(fila, probabilidad)


def valores_cliente(posicion):
    """
    Devuelve los valores de un cliente real de la muestra de prueba (el de la posición dada), en el orden del
    formulario y con las palabras que muestran los campos, más el aviso con lo que pasó de verdad con él.
    """
    cliente = EJEMPLOS.iloc[posicion]
    salida = []
    for nombre in NOMBRES_CAMPOS:
        valor = cliente[nombre]
        if nombre == "quejas":
            salida.append("Sí" if int(valor) == 1 else "No")
        elif nombre == "plan_tarifario":
            salida.append("Contractual" if int(valor) == 2 else "Pago por uso")
        elif nombre == "grupo_edad":
            salida.append(int(valor))
        else:
            salida.append(float(valor))
    salida.append(texto_valor_real(cliente["abandono"]))
    return salida


def cargar_cliente_al_azar():
    """Elige un cliente real al azar de la muestra de prueba y devuelve sus valores para llenar el formulario."""
    return valores_cliente(int(np.random.randint(0, len(EJEMPLOS))))


def cargar_cliente_tranquilo():
    """Carga un cliente real de riesgo bajo: el que está en el percentil 25 del riesgo de la muestra de prueba."""
    posicion = int(np.argsort(PROBABILIDADES_EJEMPLOS)[int(0.25 * len(EJEMPLOS))])
    return valores_cliente(posicion)


def cargar_cliente_limite():
    """Carga un cliente real cuyo riesgo queda justo en el umbral de alerta, el caso más difícil de decidir."""
    return valores_cliente(int(np.argmin(np.abs(PROBABILIDADES_EJEMPLOS - UMBRAL))))


def cargar_cliente_en_riesgo():
    """Carga un cliente real de riesgo alto: el que está en el percentil 95 del riesgo de la muestra de prueba."""
    posicion = int(np.argsort(PROBABILIDADES_EJEMPLOS)[int(0.95 * len(EJEMPLOS))])
    return valores_cliente(posicion)


def preparar_tabla_clientes(df):
    """
    Deja un CSV listo para el modelo: acepta los nombres en español o los originales en inglés, calcula la edad si
    solo viene el grupo de edad y revisa que estén todas las columnas y que no haya vacíos ni valores negativos.
    Si algo falla lanza un gr.Error que dice qué columnas faltan o cuáles tienen problemas.
    """
    df = df.rename(columns=TRADUCCION)
    if "edad" not in df.columns and "grupo_edad" in df.columns:
        df["edad"] = df["grupo_edad"].map(EDAD_POR_GRUPO)

    faltantes = []
    for columna in COLUMNAS:
        if columna not in df.columns:
            faltantes.append(columna)
    if len(faltantes) > 0:
        raise gr.Error("Al archivo le faltan estas columnas: " + ", ".join(faltantes)
                       + ". El modelo usa: " + ", ".join(COLUMNAS) + ".")

    tabla = df[COLUMNAS].apply(pd.to_numeric, errors="coerce")
    con_problemas = []
    for columna in COLUMNAS:
        if tabla[columna].isna().any() or (tabla[columna] < 0).any():
            con_problemas.append(columna)
    if len(con_problemas) > 0:
        raise gr.Error("Hay valores vacíos, no numéricos o negativos en: " + ", ".join(con_problemas) + ".")
    return tabla


def resumen_lista(probabilidades):
    """
    Arma el resumen en HTML de una lista de clientes: cuántos son, cuántos quedan en alerta y el riesgo
    promedio de la cartera.
    """
    alertas = int((probabilidades >= UMBRAL).sum())
    kpis = [("Clientes", str(len(probabilidades)), "evaluados"),
            ("En alerta", str(alertas), formato_numero(alertas / len(probabilidades) * 100, 1) + "% de la lista"),
            ("Riesgo promedio", formato_numero(probabilidades.mean() * 100, 1) + "%", "probabilidad media de abandono")]
    html = ""
    for nombre, valor, detalle in kpis:
        html += '<div class="kpi"><div class="nombre">{}</div><div class="valor">{}</div><div class="detalle">{}</div></div>'.format(
            nombre, valor, detalle)
    return '<div class="resumen-lista">' + html + "</div>"


def priorizar_csv(ruta):
    """
    Ordena a los clientes de un archivo CSV de mayor a menor riesgo de abandono. Devuelve el resumen en HTML, la
    tabla (posición, fila del archivo, probabilidad, nivel y acción sugerida) y la ruta de un CSV con la misma
    lista para descargar. Si no hay archivo lanza un gr.Error.
    """
    if ruta is None:
        raise gr.Error("Primero sube un archivo CSV.")
    df = pd.read_csv(ruta)
    tabla = preparar_tabla_clientes(df)
    probabilidades = predecir_probabilidad(tabla)

    orden = np.argsort(-probabilidades, kind="stable")
    niveles = []
    acciones = []
    for posicion in orden:
        nivel, color, accion = nivel_de_riesgo(probabilidades[posicion])
        niveles.append(nivel.capitalize())
        acciones.append(accion)
    ranking = pd.DataFrame({
        "Posición": range(1, len(orden) + 1),
        "Fila del archivo": orden + 2,
        "Probabilidad de abandono (%)": np.round(probabilidades[orden] * 100, 1),
        "Nivel de riesgo": niveles,
        "Acción sugerida": acciones,
    })
    carpeta = tempfile.mkdtemp()
    ruta_salida = os.path.join(carpeta, "ranking_riesgo_abandono.csv")
    ranking.to_csv(ruta_salida, index=False)
    return resumen_lista(probabilidades), ranking, ruta_salida


def resumen_modelo():
    """
    Arma el HTML de la pestaña "Sobre el modelo" con lo que guardó el notebook 04 en la metadata: modelo, estrategia,
    umbral, métricas en prueba con su intervalo, estimación de entrenamiento, qué ocurre al contactar solo a los de
    mayor riesgo, y cómo leer todo con prudencia.
    """
    m = METADATA
    fichas = ""
    for texto in [m["modelo"], "Balanceo: " + m["estrategia_balanceo"], "Probabilidades calibradas (sigmoide)",
                  "Preprocesamiento: " + m["preprocesamiento"], "Umbral de alerta: " + formato_numero(m["umbral"], 2)]:
        fichas += "<span>" + texto + "</span>"

    prueba = m["metricas_test"]
    intervalos = m["intervalos_bootstrap_test_95"]
    tarjetas = ""
    for clave, nombre in [("F1", "F1"), ("Recall", "Recall"), ("Precision", "Precisión"), ("ROC_AUC", "ROC AUC"),
                          ("Precision_promedio", "Precisión promedio"), ("Brier", "Brier")]:
        detalle = ""
        if clave in intervalos:
            detalle = "IC 95 %: " + formato_numero(intervalos[clave][0], 2) + " a " + formato_numero(intervalos[clave][1], 2)
        tarjetas += ('<div class="kpi"><div class="nombre">{}</div><div class="valor">{}</div><div class="detalle">{}</div></div>').format(
            nombre, formato_numero(prueba[clave], 3), detalle)

    anidadas = m["metricas_anidadas_entrenamiento"]
    prioridad = pd.read_csv(RUTA_RESULTADOS + "04_priorizacion_test.csv")
    filas_prioridad = ""
    for i in range(len(prioridad)):
        filas_prioridad += ("<tr><td>{}%</td><td>{}</td><td>{}%</td><td>{}%</td><td>{}</td></tr>").format(
            int(round(prioridad.loc[i, "Fraccion_contactada"] * 100)), int(prioridad.loc[i, "Clientes"]),
            formato_numero(prioridad.loc[i, "Precision"] * 100, 1), formato_numero(prioridad.loc[i, "Recall"] * 100, 1),
            formato_numero(prioridad.loc[i, "Lift"], 1))

    estres = ""
    ruta_estres = RUTA_RESULTADOS + "05_prueba_estres_grupos.csv"
    if os.path.exists(ruta_estres):
        tabla_estres = pd.read_csv(ruta_estres)
        estres = (" Con una agrupación más estricta (clientes parecidos juntos) el F1 baja a " +
                  formato_numero(tabla_estres.loc[1, "F1_umbral_0_5"], 2) + "; una lectura prudente es entre " +
                  formato_numero(tabla_estres.loc[1, "F1_umbral_0_5"], 2) + " y " + formato_numero(prueba["F1"], 2) + ".")

    return ('<div class="titulo-tarjeta">Modelo</div><div class="fichas-modelo">' + fichas + "</div>"
            '<div class="titulo-tarjeta">Resultados en la muestra de prueba</div>'
            '<p class="sub-tarjeta">Clientes que el modelo no vio al ajustarse. Abandono como clase positiva; IC = intervalo de confianza por bootstrap de perfiles.</p>'
            '<div class="resumen-lista">' + tarjetas + "</div>"
            '<p class="sub-tarjeta">Estimación solo con entrenamiento (validación anidada por grupos): F1 de '
            + formato_numero(anidadas["F1"], 3) + " ± " + formato_numero(anidadas["F1_desviacion"], 3) + ".</p>"
            '<div class="titulo-tarjeta" style="margin-top:14px">Si solo se contacta a los de mayor riesgo</div>'
            '<table><thead><tr><th>Se contacta al</th><th>Clientes</th><th>Aciertan</th><th>Abandonos capturados</th><th>Veces mejor que al azar</th></tr></thead>'
            "<tbody>" + filas_prioridad + "</tbody></table>"
            '<div class="nota-modelo"><b>Cómo leerlo.</b> El modelo ordena bien a los clientes por riesgo, pero es una herramienta para '
            "priorizar contactos: no estima si una oferta evitará el abandono ni promete ahorros." + estres
            + " Los datos no traen fechas, así que no se puede comprobar si alguna entrada (por ejemplo, el uso igual a cero) ya reflejaba "
            "un abandono ocurrido, ni cómo se comportaría en otro periodo o mercado.<br><br><b>Entrenado el</b> "
            + m["fecha_entrenamiento"] + " con scikit-learn " + m["version_sklearn"] + ". Proyecto académico.</div>")


def crear_campo(nombre):
    """
    Devuelve el campo del formulario para una entrada: un grupo de botones para las opciones de sí o no (queja y plan),
    un menú para el grupo de edad o un campo numérico para el resto. Todos llevan su descripción y empiezan vacíos.
    """
    descripcion = DESCRIPCIONES[nombre]
    if nombre == "quejas":
        return gr.Radio(choices=list(OPCIONES_QUEJA.keys()), value=None, label=ETIQUETAS[nombre], info=descripcion)
    if nombre == "plan_tarifario":
        return gr.Radio(choices=list(OPCIONES_PLAN.keys()), value=None, label=ETIQUETAS[nombre], info=descripcion)
    if nombre == "grupo_edad":
        return gr.Dropdown(choices=[1, 2, 3, 4, 5], value=None, label=ETIQUETAS[nombre], info=descripcion)
    return gr.Number(value=None, minimum=0, label=ETIQUETAS[nombre], info=descripcion)


def encabezado_html():
    """Arma el encabezado de la app: barras de señal animadas, título y los datos clave del modelo."""
    prueba = METADATA["metricas_test"]
    barras = ""
    for i in range(5):
        alto = 14 + i * 9
        barras += '<rect x="{}" y="{}" width="9" height="{}" rx="2"/>'.format(i * 13, 52 - alto, alto)
    datos = [("F1 en prueba", formato_numero(prueba["F1"], 2)), ("Recall", formato_numero(prueba["Recall"], 2)),
             ("ROC AUC", formato_numero(prueba["ROC_AUC"], 2))]
    chips = ""
    for nombre, valor in datos:
        chips += '<div class="dato"><b>{}</b><span>{}</span></div>'.format(valor, nombre)
    return ('<div class="appbar"><div class="marca"><svg class="senal" viewBox="0 0 62 52" aria-hidden="true">' + barras + "</svg>"
            "<div><h1>Riesgo de abandono de clientes</h1><p>Estima la probabilidad de que un cliente de telecomunicaciones abandone "
            "al final del mes doce, a partir de sus primeros nueve meses, para decidir a quién contactar primero.</p></div></div>"
            '<div class="datos">' + chips + "</div></div>")


def crear_tema():
    """
    Crea el tema de Gradio con los colores del proyecto. Definimos también las variables del modo oscuro con los
    mismos valores para que la app se vea igual aunque el navegador esté en modo oscuro.
    """
    tema = gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Fira Sans"), "Segoe UI", "Arial", "sans-serif"],
        radius_size=gr.themes.sizes.radius_lg,
    )
    return tema.set(
        body_background_fill="#F4F7FB", body_background_fill_dark="#F4F7FB",
        body_text_color="#0F172A", body_text_color_dark="#0F172A",
        block_background_fill="#FFFFFF", block_background_fill_dark="#FFFFFF",
        block_border_color="#DCE4EF", block_border_color_dark="#DCE4EF",
        block_label_text_color="#334155", block_label_text_color_dark="#334155",
        input_background_fill="#F8FAFC", input_background_fill_dark="#F8FAFC",
        button_primary_background_fill="#F59E0B", button_primary_background_fill_dark="#F59E0B",
        button_primary_text_color="#0B1B33", button_primary_text_color_dark="#0B1B33",
    )


def construir_interfaz():
    """
    Arma la interfaz con gr.Blocks y sus tres pestañas (evaluar un cliente, priorizar contactos con un CSV y sobre el
    modelo) y conecta los botones con las funciones. Devuelve la app lista para lanzar con el tema y los estilos.
    """
    with gr.Blocks(title="Riesgo de abandono de clientes") as interfaz:
        gr.HTML(encabezado_html())

        with gr.Tab("Evaluar un cliente"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, elem_classes="tarjeta"):
                    gr.HTML('<div class="titulo-tarjeta">Datos del cliente</div>'
                            '<p class="sub-tarjeta">Primeros nueve meses de servicio. Escribe los valores o carga un cliente real de ejemplo.</p>')
                    campos = []
                    gr.HTML('<div class="grupo-form">Uso del servicio</div>')
                    with gr.Row():
                        campos.append(crear_campo("segundos_uso"))
                        campos.append(crear_campo("frecuencia_uso"))
                    with gr.Row():
                        campos.append(crear_campo("frecuencia_sms"))
                        campos.append(crear_campo("numeros_distintos"))
                        campos.append(crear_campo("fallas_llamadas"))
                    gr.HTML('<div class="grupo-form">Cuenta</div>')
                    with gr.Row():
                        campos.append(crear_campo("antiguedad"))
                        campos.append(crear_campo("nivel_cargo"))
                    with gr.Row():
                        campos.append(crear_campo("plan_tarifario"))
                        campos.append(crear_campo("quejas"))
                    gr.HTML('<div class="grupo-form">Perfil</div>')
                    campos.append(crear_campo("grupo_edad"))
                    boton_evaluar = gr.Button("Evaluar cliente", variant="primary", size="lg")
                    gr.HTML('<p class="sub-tarjeta" style="margin-top:12px">Clientes reales de ejemplo (de la muestra de prueba)</p>')
                    with gr.Row():
                        boton_tranquilo = gr.Button("Cliente tranquilo", variant="secondary")
                        boton_limite = gr.Button("Cliente en el límite", variant="secondary")
                        boton_riesgo = gr.Button("Cliente en riesgo", variant="secondary")
                    boton_azar = gr.Button("Cargar un cliente real al azar", variant="secondary")
                with gr.Column(scale=5, elem_classes="tarjeta"):
                    resultado = gr.HTML(resultado_vacio())
                    valor_real = gr.HTML()

            boton_evaluar.click(evaluar_cliente, inputs=campos, outputs=resultado)
            # Primero llenamos el formulario con el cliente real y, cuando termina, lo evaluamos (por eso el .then)
            for boton, funcion in [(boton_tranquilo, cargar_cliente_tranquilo), (boton_limite, cargar_cliente_limite),
                                   (boton_riesgo, cargar_cliente_en_riesgo), (boton_azar, cargar_cliente_al_azar)]:
                boton.click(funcion, inputs=None, outputs=campos + [valor_real]).then(
                    evaluar_cliente, inputs=campos, outputs=resultado)

        with gr.Tab("Priorizar contactos"):
            with gr.Column(elem_classes="tarjeta"):
                gr.HTML('<div class="titulo-tarjeta">Lista de clientes ordenada por riesgo</div>'
                        '<p class="sub-tarjeta">Sube un CSV con un cliente por fila. Sirven los nombres en español o los originales en inglés. '
                        'El modelo usa estas columnas: <b>' + ", ".join(COLUMNAS) + '</b> (si falta la edad se calcula desde el grupo de edad).</p>')
                archivo = gr.File(label="Archivo CSV", file_types=[".csv"], type="filepath")
                boton_csv = gr.Button("Ordenar por riesgo", variant="primary", size="lg")
                resumen = gr.HTML()
                tabla = gr.Dataframe(label="Clientes ordenados de mayor a menor riesgo", interactive=False)
                descarga = gr.File(label="Descargar la lista", interactive=False)
            boton_csv.click(priorizar_csv, inputs=archivo, outputs=[resumen, tabla, descarga])

        with gr.Tab("Sobre el modelo"):
            with gr.Column(elem_classes="tarjeta"):
                gr.HTML(resumen_modelo())

        gr.HTML('<div class="pie">Fundamentos de Inteligencia Artificial · Universidad EIA · 2026-2 · '
                'Datos: Iranian Churn (UCI 563)</div>')
    return interfaz


if __name__ == "__main__":
    # En Gradio 6 el tema, los estilos y la cabecera se pasan al lanzar la app
    construir_interfaz().launch(share=EN_COLAB, theme=crear_tema(), css=ESTILOS_CSS, head=CABECERA_HTML)
