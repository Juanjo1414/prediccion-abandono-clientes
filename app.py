"""
App Gradio: riesgo de abandono de clientes (proyecto de Fundamentos de IA, Universidad EIA).

Carga el modelo que dejó Churn_Paso_a_Paso.ipynb (models/modelo_final.joblib y models/metadata_modelo.json):
un único objeto calibrado (Random Forest con peso por clase, calibración sigmoide) que recibe doce entradas
-las once del dataset original menos la edad, que se quitó por ser idéntica al grupo de edad- y entrega una
probabilidad de abandono. La app reconstruye la misma partición 80/20 que usó el notebook (misma semilla,
mismo test_size) para tener los 630 clientes de prueba reales y los valores típicos de los 2.520 de
entrenamiento, sin depender de archivos intermedios.

Tres pestañas:
  1. Evaluar un cliente: llenar sus datos a mano (o cargar un cliente real de la muestra de prueba) y ver su
     probabilidad de abandono, su nivel de riesgo y qué factores empujan ese riesgo.
  2. Priorizar contactos: subir un CSV con varios clientes y obtener la lista ordenada de mayor a menor riesgo.
  3. Sobre el modelo: qué modelo es, con qué umbral, qué tan bien funcionó en prueba, en qué se apoya más y
     qué pasa si solo se contacta a los clientes de mayor riesgo (con un deslizador en vivo).

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
from sklearn.model_selection import train_test_split

# Si corremos en Google Colab ponemos True; en el computador (VS Code), False
EN_COLAB = False

if EN_COLAB:
    # En Colab los archivos viven en Drive, así que primero lo montamos
    from google.colab import drive
    drive.mount("/content/drive")
    RUTA_BASE = "/content/drive/MyDrive/Prediccion_Abandono_Clientes/"
else:
    # Así la app funciona sin importar desde qué carpeta se lance
    RUTA_BASE = os.path.dirname(os.path.abspath(__file__)) + "/"

# Todas las rutas de la app salen de aquí, para no tener rutas regadas por el código
RUTA_DATOS = RUTA_BASE + "data/"
RUTA_MODELOS = RUTA_BASE + "models/"
RUTA_ORIGINAL = RUTA_DATOS + "original/Customer Churn.csv"

# Mismo diccionario que usa el notebook para pasar los nombres de UCI (con espacios dobles) a español.
# Lo usamos tanto para reconstruir la partición como para aceptar un CSV con los nombres originales.
TRADUCCION = {
    "Call  Failure": "fallas_llamadas", "Complains": "quejas", "Subscription  Length": "antiguedad",
    "Charge  Amount": "nivel_cargo", "Seconds of Use": "segundos_uso", "Frequency of use": "frecuencia_uso",
    "Frequency of SMS": "frecuencia_sms", "Distinct Called Numbers": "numeros_distintos",
    "Age Group": "grupo_edad", "Tariff Plan": "plan_tarifario", "Status": "estado", "Age": "edad",
    "Customer Value": "valor_cliente", "Churn": "abandono",
}

# Campos del formulario, en el orden en que se muestran (no es el orden que espera el modelo; eso lo maneja
# COLUMNAS). La edad no se pide: el notebook la quitó por ser una función exacta del grupo de edad.
NOMBRES_CAMPOS = ["segundos_uso", "frecuencia_uso", "frecuencia_sms", "numeros_distintos", "fallas_llamadas",
                  "antiguedad", "nivel_cargo", "plan_tarifario", "estado",
                  "quejas", "grupo_edad", "valor_cliente"]

ETIQUETAS = {
    "segundos_uso": "Segundos de uso",
    "frecuencia_uso": "Número de llamadas",
    "frecuencia_sms": "Número de mensajes",
    "numeros_distintos": "Números distintos llamados",
    "fallas_llamadas": "Llamadas fallidas",
    "antiguedad": "Antigüedad (meses)",
    "nivel_cargo": "Nivel de cargo",
    "plan_tarifario": "Plan tarifario",
    "estado": "Estado de la línea",
    "quejas": "Queja presentada",
    "grupo_edad": "Grupo de edad",
    "valor_cliente": "Valor del cliente",
}

DESCRIPCIONES = {
    "segundos_uso": "Segundos de llamadas en los primeros nueve meses",
    "frecuencia_uso": "Cantidad de llamadas realizadas",
    "frecuencia_sms": "Cantidad de mensajes de texto enviados",
    "numeros_distintos": "Cantidad de números distintos a los que llamó",
    "fallas_llamadas": "Llamadas que fallaron",
    "antiguedad": "Meses desde que se suscribió",
    "nivel_cargo": "Nivel de cargo, documentado de 0 a 9 (hay algunos casos de 10); no son unidades monetarias",
    "plan_tarifario": "Tipo de plan contratado",
    "estado": "Si la línea está activa o no en el momento de la medición. Es la variable que más usa el modelo, "
              "pero avisa tarde: suele cambiar cuando el cliente ya dejó de usar el servicio",
    "quejas": "Si el cliente presentó alguna queja (no cuenta cuántas)",
    "grupo_edad": "Grupo de edad del cliente, de 1 a 5",
    "valor_cliente": "Un valor calculado por la empresa a partir de otras variables; UCI no publica su fórmula "
                     "ni su unidad, así que no se lee como una cifra en dinero",
}

OPCIONES_QUEJA = {"No": 0, "Sí": 1}
OPCIONES_PLAN = {"Pago por uso": 1, "Contractual": 2}
OPCIONES_ESTADO = {"Activo": 1, "No activo": 2}

# Dominios válidos de las entradas categóricas, para avisar de valores raros al subir un CSV
DOMINIOS = {"quejas": [0, 1], "plan_tarifario": [1, 2], "estado": [1, 2]}

# Por debajo de este riesgo el cliente es de riesgo bajo. Es una convención de la app (no del modelo) para
# separar a los clientes sin alerta en dos grupos; el umbral de alerta sí es el del modelo.
RIESGO_BAJO_HASTA = 0.10

COLOR_BAJO = "#0F8B7A"
COLOR_MEDIO = "#D97706"
COLOR_ALTO = "#DC2626"

# En qué se apoya el modelo, calculado en la sección 12.1 de Churn_Paso_a_Paso.ipynb (importancia de Random
# Forest). Se deja como tabla fija porque viene de un modelo ya calibrado, cuyos árboles no son triviales de
# recorrer desde aquí; el número que importa (el riesgo de cada cliente) sí se recalcula siempre en vivo.
IMPORTANCIAS = [("estado", 0.1687), ("segundos_uso", 0.1411), ("frecuencia_uso", 0.1392), ("quejas", 0.1244),
                ("antiguedad", 0.1007), ("valor_cliente", 0.0949), ("numeros_distintos", 0.0677),
                ("frecuencia_sms", 0.0626), ("fallas_llamadas", 0.0420), ("grupo_edad", 0.0395),
                ("nivel_cargo", 0.0177), ("plan_tarifario", 0.0015)]

CABECERA_HTML = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
                 '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                 '<link href="https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&'
                 'family=Fira+Sans+Condensed:wght@500;600;700&family=Fira+Code:wght@500;600&display=swap" rel="stylesheet">')

# Gradio inserta `head=` con innerHTML, así que una etiqueta <script> ahí dentro queda inerte y nunca se
# ejecuta (es una regla del navegador, no un detalle de Gradio). Por eso la animación de conteo de los números
# clave se dispara con interfaz.load(..., js=JS_CONTADORES): ese sí es código que Gradio evalúa de verdad.
# Un observador de mutaciones no basta porque Gradio reemplaza nodos enteros al actualizar un componente (por
# ejemplo, al mover el deslizador de priorización); un sondeo simple es menos elegante pero nunca pierde el
# rastro, y el costo es insignificante.
JS_CONTADORES = """
() => {
  if (window.__contadoresListos) { return; }
  window.__contadoresListos = true;
  var sinMovimiento = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function formatear(valor, decimales, sufijo) {
    return valor.toFixed(decimales).replace(".", ",") + sufijo;
  }
  function animar(el) {
    el.setAttribute("data-animado", "1");
    var destino = parseFloat(el.getAttribute("data-valor"));
    var decimales = parseInt(el.getAttribute("data-decimales") || "0", 10);
    var sufijo = el.getAttribute("data-sufijo") || "";
    if (sinMovimiento || isNaN(destino)) { el.textContent = formatear(destino, decimales, sufijo); return; }
    var inicio = null;
    var duracion = 900;
    function paso(marca) {
      if (!inicio) { inicio = marca; }
      var avance = Math.min((marca - inicio) / duracion, 1);
      var suave = 1 - Math.pow(1 - avance, 3);
      el.textContent = formatear(destino * suave, decimales, sufijo);
      if (avance < 1) { requestAnimationFrame(paso); }
    }
    requestAnimationFrame(paso);
  }
  function recorrer() {
    var elementos = document.querySelectorAll(".contar:not([data-animado])");
    for (var i = 0; i < elementos.length; i++) { animar(elementos[i]); }
  }
  setInterval(recorrer, 400);
  recorrer();
}
"""

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
/* El grupo de botones de sí/no (queja, plan, estado) pone el nombre del campo en un <span> aparte del
   <label>, con un gris casi blanco por defecto; lo oscurecemos para que se lea igual que los demás campos.
   La descripción de cada campo (.info-text) también venía muy clara; la bajamos a un gris legible. */
.tarjeta span.has-info, .tarjeta legend { color: var(--texto) !important; opacity: 1 !important; }
.tarjeta .info-text, .gradio-container .info-text { color: var(--suave) !important; }
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
.factor .barra i { position: absolute; top: 0; bottom: 0; border-radius: 5px; transition: width .6s cubic-bezier(.2,.8,.2,1); }
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
.kpi { background: var(--papel); border-radius: 12px; padding: 12px 14px; transition: transform .2s; }
.kpi .nombre { color: var(--suave); font-size: 13px; }
.kpi .valor { font-family: var(--numero); font-size: 26px; font-weight: 600; color: var(--tinta); line-height: 1.15; }
.kpi .detalle { color: var(--suave); font-size: 12.5px; }
.fichas-modelo { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 16px; }
.fichas-modelo span { background: var(--tinta); color: #E8EEF8; border-radius: 8px; padding: 5px 12px; font-size: 14px; }
.nota-modelo { border-left: 4px solid var(--ambar); background: #FEF9E7; border-radius: 0 10px 10px 0; padding: 12px 16px; margin-top: 14px; font-size: 15px; line-height: 1.45; }
.gradio-container table { font-variant-numeric: tabular-nums; }
/* El encabezado de la tabla de resultados, el menú de cada columna ("Ordenar ascendente", "Filter"...) y los
   avisos emergentes (toasts) traen de Gradio un fondo oscuro con nuestro color de letra general (oscuro)
   encima: texto invisible. Esas celdas internas ya usan "!important" con dos clases (especificidad 0,2,x),
   así que para ganarles hace falta más especificidad, no solo "!important". Repetimos ".gradio-container"
   tres veces seguidas y sin espacio: sigue siendo la misma clase sobre el mismo elemento (el contenedor de
   la app, que ya existe), pero cuenta tres veces para la especificidad. */
.gradio-container.gradio-container.gradio-container table thead,
.gradio-container.gradio-container.gradio-container table th {
  background: var(--tinta) !important;
  color: #fff !important;
}
.gradio-container.gradio-container.gradio-container .cell-menu,
.gradio-container.gradio-container.gradio-container .cell-menu button {
  background: var(--tinta) !important;
  color: #fff !important;
}
.gradio-container.gradio-container.gradio-container .cell-menu button:hover { background: var(--azul) !important; }
.gradio-container.gradio-container.gradio-container .toast-body,
.gradio-container.gradio-container.gradio-container .toast-body * {
  color: #fff !important;
}
.pie { text-align: center; color: var(--suave); font-size: 13px; padding: 14px 0 4px; }

/* Importancia de cada variable */
.importancias { margin: 8px 0 4px; }
.importancia { display: grid; grid-template-columns: 170px 1fr 54px; gap: 10px; align-items: center; font-size: 14px; margin: 7px 0; }
.importancia .nombre-imp { color: var(--texto); }
.importancia .pista { height: 12px; background: var(--papel); border-radius: 6px; overflow: hidden; }
.importancia .relleno { height: 100%; background: var(--azul); border-radius: 6px; transition: width .7s cubic-bezier(.2,.8,.2,1); }
.importancia .valor-imp { font-family: var(--numero); font-size: 13px; text-align: right; color: var(--suave); }

/* Priorización en vivo */
.priorizacion { margin-top: 8px; }
.priorizacion .deslizador-info { color: var(--suave); font-size: 14px; margin: 6px 0 2px; }
.barra-cobertura { height: 22px; background: var(--papel); border-radius: 11px; overflow: hidden; margin: 10px 0; display: flex; }
.barra-cobertura .capturado { background: var(--bajo); transition: width .4s cubic-bezier(.2,.8,.2,1); }
.barra-cobertura .restante { background: var(--borde); transition: width .4s cubic-bezier(.2,.8,.2,1); }

@media (max-width: 760px) { .appbar { padding: 20px; } .appbar h1 { font-size: 30px; } .factor { grid-template-columns: 1fr 80px 48px; } .importancia { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { .appbar .senal rect, .resultado .anillo .progreso { animation: none; } }
"""

ICONO_BAJO = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" '
              'stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>')
ICONO_ALERTA = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" '
                'stroke-linejoin="round" aria-hidden="true"><path d="M12 9v4M12 17h.01"/>'
                '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>')


def reconstruir_particion():
    """
    Reconstruye la misma partición 80/20 que usó Churn_Paso_a_Paso.ipynb: lee el archivo original de UCI, le
    pone los mismos nombres en español, separa entrenamiento y prueba con la misma semilla y el mismo tamaño,
    y quita la edad (se eliminó por ser idéntica al grupo de edad). No depende de ningún CSV intermedio, así
    que no se desactualiza si otro cuaderno sobrescribe data/train.csv. Devuelve X_train, X_test y y_test.
    """
    datos = pd.read_csv(RUTA_ORIGINAL).rename(columns=TRADUCCION)
    X = datos.drop(columns="abandono")
    y = datos["abandono"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train = X_train.drop(columns="edad").reset_index(drop=True)
    X_test = X_test.drop(columns="edad").reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, y_train, X_test, y_test


def cargar_artefactos():
    """
    Lee lo que dejó el notebook: el modelo final (un único objeto calibrado, el umbral y las columnas que
    espera) y la metadata con sus métricas. Reconstruye además la partición de entrenamiento y prueba.
    Devuelve, en ese orden: el artefacto, la metadata, el entrenamiento y los clientes de prueba (con su
    abandono real) que sirven de ejemplo.
    """
    artefacto = joblib.load(RUTA_MODELOS + "modelo_final.joblib")
    with open(RUTA_MODELOS + "metadata_modelo.json", encoding="utf-8") as archivo:
        metadata = json.load(archivo)
    train, y_train, X_test, y_test = reconstruir_particion()
    clientes_ejemplo = X_test.copy()
    clientes_ejemplo["abandono"] = y_test
    return artefacto, metadata, train, clientes_ejemplo, float(y_train.mean())


# Se cargan una sola vez, al abrir la app. Los botones de Gradio no pueden recibir parámetros extra, así que
# las funciones de abajo leen estas variables globales (solo las leen, nunca las modifican)
ARTEFACTO, METADATA, TRAIN, EJEMPLOS, TASA_ABANDONO_TRAIN = cargar_artefactos()
COLUMNAS = ARTEFACTO["columnas_entrada"]
UMBRAL = ARTEFACTO["umbral"]


def predecir_probabilidad(X):
    """
    Recibe un DataFrame con las columnas de entrada del modelo y devuelve la probabilidad calibrada de
    abandono de cada fila. El modelo exportado ya es un único objeto (pipeline y calibración juntos), así
    que basta con llamar a predict_proba.
    """
    return ARTEFACTO["modelo"].predict_proba(X[COLUMNAS])[:, 1]


def calcular_referencias(train):
    """
    Calcula, con los datos de entrenamiento, el valor típico (mediana) y el rango de cada entrada. Devuelve
    dos diccionarios: medianas y rangos.
    """
    medianas = {}
    rangos = {}
    for columna in COLUMNAS:
        medianas[columna] = float(train[columna].median())
        rangos[columna] = (float(train[columna].min()), float(train[columna].max()))
    return medianas, rangos


MEDIANAS, RANGOS = calcular_referencias(TRAIN)

# Probabilidad de cada cliente de prueba, calculada una vez; sirve para elegir clientes de ejemplo según su
# riesgo y para el panel de priorización en vivo de la pestaña "Sobre el modelo"
PROBABILIDADES_EJEMPLOS = predecir_probabilidad(EJEMPLOS)
ORDEN_POR_RIESGO = np.argsort(-PROBABILIDADES_EJEMPLOS, kind="stable")
ABANDONOS_PRUEBA = int(EJEMPLOS["abandono"].sum())


def formato_numero(valor, decimales):
    """Devuelve un número como texto con coma decimal (6,6) y no con punto, como escribimos en todo el proyecto."""
    return ("{:." + str(decimales) + "f}").format(valor).replace(".", ",")


def nivel_de_riesgo(probabilidad):
    """
    Clasifica una probabilidad en bajo, medio o alto. Alto es desde el umbral del modelo (la alerta); medio,
    entre RIESGO_BAJO_HASTA y el umbral; bajo, por debajo. Devuelve el nombre del nivel, su color y la acción
    sugerida.
    """
    if probabilidad >= UMBRAL:
        return "alto", COLOR_ALTO, "Incluir en la lista de contacto prioritario de retención."
    if probabilidad >= RIESGO_BAJO_HASTA:
        return "medio", COLOR_MEDIO, "Mantener en seguimiento; contactar si hay capacidad de sobra."
    return "bajo", COLOR_BAJO, "Sin acción especial de retención por ahora."


def anillo_svg(probabilidad, color):
    """
    Dibuja el anillo del riesgo como SVG: una pista gris, un arco de progreso del color del nivel y el
    porcentaje en el centro. Lo hacemos a mano para poder animarlo con CSS sin depender de ninguna librería
    gráfica.
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
    if nombre == "estado":
        return "No activo" if int(valor) == 2 else "Activo"
    if nombre == "grupo_edad":
        return "Grupo " + str(int(valor))
    if nombre == "valor_cliente":
        return formato_numero(valor, 2)
    return str(int(round(valor))) if float(valor).is_integer() else formato_numero(valor, 1)


def factores_de_riesgo(fila, probabilidad):
    """
    Calcula qué entradas empujan el riesgo de este cliente. Para cada entrada sustituye su valor por el valor
    típico de los clientes de entrenamiento (la mediana) y mira cuánto cambia la probabilidad: si baja, esa
    entrada estaba empujando el riesgo hacia arriba. Devuelve una lista de (nombre, cambio en puntos de
    probabilidad) ordenada de mayor a menor efecto.
    """
    nombres = []
    variantes = []
    for nombre in COLUMNAS:
        variante = fila.copy()
        variante[nombre] = MEDIANAS[nombre]
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
    for nombre in COLUMNAS:
        minimo, maximo = RANGOS[nombre]
        if fila[nombre] < minimo or fila[nombre] > maximo:
            fuera.append(ETIQUETAS[nombre].lower())
    if len(fuera) == 0:
        return ""
    return ('<div class="aviso"><b>Valores fuera de lo que vio el modelo:</b> ' + ", ".join(fuera)
            + '. La probabilidad es menos confiable en esos casos.</div>')


def texto_resultado(fila, probabilidad):
    """
    Arma la tarjeta de resultado en HTML: nivel de riesgo, anillo con la probabilidad, acción sugerida, el
    umbral y el abandono promedio para dar contexto, los factores que empujan el riesgo y los avisos si hay
    datos atípicos.
    """
    nivel, color, accion = nivel_de_riesgo(probabilidad)
    nombre_nivel = {"bajo": "Riesgo bajo", "medio": "Riesgo medio", "alto": "Riesgo alto"}[nivel]
    icono = ICONO_BAJO if nivel == "bajo" else ICONO_ALERTA
    fichas = ('<div class="fichas"><span class="ficha">Umbral de alerta: {}%</span>'
              '<span class="ficha">Abandono promedio en los datos: {}%</span></div>').format(
                  formato_numero(UMBRAL * 100, 0), formato_numero(TASA_ABANDONO_TRAIN * 100, 1))
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
    Convierte lo que entrega un campo del formulario a un número que entiende el modelo ('Sí' pasa a 1,
    'Contractual' a 2, 'Activo' a 1, etc.). Si el campo está vacío lanza un gr.Error que dice cuál falta, en
    vez de dejar que la app se rompa.
    """
    if valor is None or (isinstance(valor, float) and math.isnan(valor)) or valor == "":
        raise gr.Error("Falta completar: " + ETIQUETAS[nombre] + ".")
    if nombre == "quejas":
        return OPCIONES_QUEJA[valor]
    if nombre == "plan_tarifario":
        return OPCIONES_PLAN[valor]
    if nombre == "estado":
        return OPCIONES_ESTADO[valor]
    numero = float(valor)
    if numero < 0:
        raise gr.Error(ETIQUETAS[nombre] + " no puede ser negativo.")
    return numero


def fila_desde_valores(valores):
    """
    Arma la fila de entradas del modelo a partir de los valores del formulario (en el orden de NOMBRES_CAMPOS).
    Devuelve una Series con las columnas en el orden que espera el modelo.
    """
    fila = {}
    for nombre, valor in zip(NOMBRES_CAMPOS, valores):
        fila[nombre] = convertir_valor(nombre, valor)
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
        elif nombre == "estado":
            salida.append("No activo" if int(valor) == 2 else "Activo")
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
    Deja un CSV listo para el modelo: acepta los nombres en español o los originales en inglés, revisa que
    estén las doce columnas y que no haya vacíos, valores negativos o categorías raras. Si algo falla lanza un
    gr.Error que dice qué columnas faltan o cuáles tienen problemas.
    """
    df = df.rename(columns=TRADUCCION)

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
        elif columna in DOMINIOS:
            valores_validos = tabla[columna].isin(DOMINIOS[columna])
            if not valores_validos.all():
                con_problemas.append(columna)
    if len(con_problemas) > 0:
        raise gr.Error("Hay valores vacíos, negativos o fuera de lo esperado en: " + ", ".join(con_problemas) + ".")
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
    Ordena a los clientes de un archivo CSV de mayor a menor riesgo de abandono. Devuelve el resumen en HTML,
    la tabla (posición, fila del archivo, probabilidad, nivel y acción sugerida) y la ruta de un CSV con la
    misma lista para descargar. Si no hay archivo lanza un gr.Error.
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


def html_importancias():
    """
    Arma el bloque "En qué se apoya el modelo": una barra por entrada con la importancia que calculó Random
    Forest, de mayor a menor. Viene de la sección 12.1 del notebook (ver comentario de IMPORTANCIAS).
    """
    filas = ""
    mayor = IMPORTANCIAS[0][1]
    for nombre, importancia in IMPORTANCIAS:
        ancho = importancia / mayor * 100
        filas += ('<div class="importancia"><span class="nombre-imp">{nombre}</span>'
                  '<div class="pista"><div class="relleno" style="width:{ancho:.0f}%"></div></div>'
                  '<span class="valor-imp">{valor}</span></div>').format(
                      nombre=ETIQUETAS[nombre], ancho=ancho, valor=formato_numero(importancia, 3))
    return ('<div class="titulo-tarjeta" style="margin-top:18px">En qué se apoya el modelo</div>'
            '<p class="sub-tarjeta">Cuánto usó cada entrada Random Forest para separar a los clientes, de mayor a menor. '
            'No dice si empuja hacia el abandono o en contra, solo cuánto se apoyó en esa entrada.</p>'
            '<div class="importancias">' + filas + '</div>'
            '<p class="sub-tarjeta">El estado de la línea encabeza la lista, pero es un indicador tardío: suele cambiar '
            'cuando el cliente ya dejó de usar el servicio, así que avisa con poco margen de maniobra.</p>')


def calcular_priorizacion_dinamica(cantidad):
    """
    Recalcula en vivo qué pasaría si solo se contactara a los "cantidad" clientes de mayor riesgo de los 630
    de la muestra de prueba: cuántos de ellos abandonaron de verdad, qué porcentaje de aciertos es y qué
    porcentaje de todos los abandonos reales queda cubierto. Reproduce la sección 11.6 del notebook, pero para
    cualquier cantidad que se mueva en el deslizador.
    """
    cantidad = int(cantidad)
    primeros = ORDEN_POR_RIESGO[:cantidad]
    capturados = int(EJEMPLOS.iloc[primeros]["abandono"].sum())
    aciertos = capturados / cantidad if cantidad > 0 else 0.0
    cobertura = capturados / ABANDONOS_PRUEBA if ABANDONOS_PRUEBA > 0 else 0.0
    veces_mejor_que_azar = aciertos / (ABANDONOS_PRUEBA / len(EJEMPLOS))
    capturado_pct = cobertura * 100

    return ('<p class="deslizador-info">Contactando a los <b>{cantidad}</b> clientes de mayor riesgo (de 630 de prueba):</p>'
            '<div class="resumen-lista">'
            '<div class="kpi"><div class="nombre">Abandonos capturados</div>'
            '<div class="valor contar" data-valor="{capturados}" data-decimales="0">0</div>'
            '<div class="detalle">de {total} abandonos reales en la prueba</div></div>'
            '<div class="kpi"><div class="nombre">Aciertan</div>'
            '<div class="valor contar" data-valor="{aciertos_pct:.1f}" data-decimales="1" data-sufijo="%">0%</div>'
            '<div class="detalle">de cada llamada hecha</div></div>'
            '<div class="kpi"><div class="nombre">Veces mejor que al azar</div>'
            '<div class="valor contar" data-valor="{veces:.1f}" data-decimales="1" data-sufijo="×">0×</div>'
            '<div class="detalle">frente a contactar sin ordenar</div></div></div>'
            '<div class="barra-cobertura"><div class="capturado" style="width:{capturado_pct:.1f}%"></div>'
            '<div class="restante" style="width:{restante_pct:.1f}%"></div></div>'
            '<p class="sub-tarjeta">Cubre el {capturado_pct_texto}% de todos los abandonos de la prueba.</p>').format(
                cantidad=cantidad, capturados=capturados, total=ABANDONOS_PRUEBA,
                aciertos_pct=aciertos * 100, veces=veces_mejor_que_azar, capturado_pct=capturado_pct,
                restante_pct=max(0.0, 100.0 - capturado_pct), capturado_pct_texto=formato_numero(capturado_pct, 1))


def resumen_modelo():
    """
    Arma el HTML de la pestaña "Sobre el modelo" con lo que guardó el notebook en la metadata: modelo,
    estrategia de balanceo, calibración, umbral, métricas en prueba frente a la estimación de validación, en
    qué se apoya el modelo y el panel de priorización (su valor inicial; el deslizador lo recalcula después).
    """
    m = METADATA
    fichas = ""
    for texto in [m["modelo"], "Balanceo: " + m["estrategia_balanceo"], "Calibración: " + m["calibracion"],
                  "Umbral de alerta: " + formato_numero(m["umbral"], 2)]:
        fichas += "<span>" + texto + "</span>"

    prueba = m["metricas_test"]
    tarjetas = ""
    for clave in ["F1 del abandono", "Recall", "Precisión", "Exactitud", "ROC AUC", "Precisión promedio", "Brier"]:
        tarjetas += ('<div class="kpi"><div class="nombre">{}</div>'
                     '<div class="valor contar" data-valor="{}" data-decimales="3">0</div></div>').format(
                         clave, prueba[clave])

    return ('<div class="titulo-tarjeta">Modelo</div><div class="fichas-modelo">' + fichas + "</div>"
            '<div class="titulo-tarjeta">Resultados en la muestra de prueba</div>'
            '<p class="sub-tarjeta">630 clientes reservados desde el inicio, con 99 abandonos; el modelo nunca los vio '
            'durante el entrenamiento ni para elegir el umbral.</p>'
            '<div class="resumen-lista">' + tarjetas + "</div>"
            '<p class="sub-tarjeta">En validación cruzada (solo entrenamiento) el F1 fue ' + formato_numero(m["f1_validacion"], 3)
            + '. ' + m["advertencia"] + '</p>'
            + html_importancias()
            + '<div class="nota-modelo"><b>Cómo leerlo.</b> El modelo ordena bien a los clientes por riesgo, pero es una '
            'herramienta para priorizar contactos: no estima si una oferta evitará el abandono ni promete ahorros. '
            'Los datos no traen fechas ni identificador de cliente, así que no se puede comprobar el orden de los hechos '
            'cliente por cliente ni cómo se comportaría el modelo en otro periodo o mercado. Las filas repetidas se '
            'trataron como clientes distintos, porque el archivo no trae identificador.'
            '<br><br><b>Entrenado el</b> ' + m["fecha_entrenamiento"][:10] + " con scikit-learn " + m["version_sklearn"]
            + " e imbalanced-learn " + m["version_imblearn"] + ". Proyecto académico.</div>")


def crear_campo(nombre):
    """
    Devuelve el campo del formulario para una entrada: un grupo de botones para las opciones de sí/no o
    activo/no activo, un menú para el grupo de edad o un campo numérico para el resto. Todos llevan su
    descripción y empiezan vacíos.
    """
    descripcion = DESCRIPCIONES[nombre]
    if nombre == "quejas":
        return gr.Radio(choices=list(OPCIONES_QUEJA.keys()), value=None, label=ETIQUETAS[nombre], info=descripcion)
    if nombre == "plan_tarifario":
        return gr.Radio(choices=list(OPCIONES_PLAN.keys()), value=None, label=ETIQUETAS[nombre], info=descripcion)
    if nombre == "estado":
        return gr.Radio(choices=list(OPCIONES_ESTADO.keys()), value=None, label=ETIQUETAS[nombre], info=descripcion)
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
    datos = [("F1 del abandono", prueba["F1 del abandono"]), ("Recall", prueba["Recall"]), ("ROC AUC", prueba["ROC AUC"])]
    chips = ""
    for nombre, valor in datos:
        chips += ('<div class="dato"><b class="contar" data-valor="{}" data-decimales="2">0</b>'
                  '<span>{}</span></div>').format(valor, nombre)
    return ('<div class="appbar"><div class="marca"><svg class="senal" viewBox="0 0 62 52" aria-hidden="true">' + barras + "</svg>"
            "<div><h1>Riesgo de abandono de clientes</h1><p>Estima la probabilidad de que un cliente de telecomunicaciones abandone "
            "al final del mes doce, a partir de sus primeros nueve meses, para decidir a quién contactar primero.</p></div></div>"
            '<div class="datos">' + chips + "</div></div>")


def crear_tema():
    """
    Crea el tema de Gradio con los colores del proyecto. Definimos también las variables del modo oscuro con
    los mismos valores para que la app se vea igual aunque el navegador esté en modo oscuro.
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
        block_label_background_fill="#F4F7FB", block_label_background_fill_dark="#F4F7FB",
        block_label_border_color="#DCE4EF", block_label_border_color_dark="#DCE4EF",
        input_background_fill="#F8FAFC", input_background_fill_dark="#F8FAFC",
        button_primary_background_fill="#F59E0B", button_primary_background_fill_dark="#F59E0B",
        button_primary_text_color="#0B1B33", button_primary_text_color_dark="#0B1B33",
        # Las opciones de los campos de sí/no (quejas, plan, estado) usaban el gris medio del tema (neutral_hue),
        # que con letra oscura encima quedaba casi ilegible. Las dejamos claras cuando no están elegidas y con
        # acento azul, bien marcado, cuando sí.
        checkbox_label_background_fill="#FFFFFF", checkbox_label_background_fill_dark="#FFFFFF",
        checkbox_label_background_fill_hover="#EFF4FB", checkbox_label_background_fill_hover_dark="#EFF4FB",
        checkbox_label_background_fill_selected="#1E40AF", checkbox_label_background_fill_selected_dark="#1E40AF",
        checkbox_label_text_color="#0F172A", checkbox_label_text_color_dark="#0F172A",
        checkbox_label_text_color_selected="#FFFFFF", checkbox_label_text_color_selected_dark="#FFFFFF",
        checkbox_label_border_color="#DCE4EF", checkbox_label_border_color_dark="#DCE4EF",
        checkbox_label_border_color_hover="#1E40AF", checkbox_label_border_color_hover_dark="#1E40AF",
        checkbox_label_border_color_selected="#1E40AF", checkbox_label_border_color_selected_dark="#1E40AF",
        checkbox_background_color_selected="#1E40AF", checkbox_background_color_selected_dark="#1E40AF",
        checkbox_border_color_selected="#1E40AF", checkbox_border_color_selected_dark="#1E40AF",
        table_text_color="#0F172A", table_text_color_dark="#0F172A",
        table_even_background_fill="#F8FAFC", table_even_background_fill_dark="#F8FAFC",
        table_odd_background_fill="#FFFFFF", table_odd_background_fill_dark="#FFFFFF",
    )


def construir_interfaz():
    """
    Arma la interfaz con gr.Blocks y sus tres pestañas (evaluar un cliente, priorizar contactos con un CSV y
    sobre el modelo) y conecta los botones con las funciones. Devuelve la app lista para lanzar con el tema y
    los estilos.
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
                    gr.HTML('<div class="grupo-form">Cuenta y estado de la línea</div>')
                    with gr.Row():
                        campos.append(crear_campo("antiguedad"))
                        campos.append(crear_campo("nivel_cargo"))
                    with gr.Row():
                        campos.append(crear_campo("plan_tarifario"))
                        campos.append(crear_campo("estado"))
                    gr.HTML('<div class="grupo-form">Quejas, perfil y valor</div>')
                    with gr.Row():
                        campos.append(crear_campo("quejas"))
                        campos.append(crear_campo("grupo_edad"))
                        campos.append(crear_campo("valor_cliente"))
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
                        '<p class="sub-tarjeta">Sube un CSV con un cliente por fila. Sirven los nombres en español o los originales en inglés de UCI. '
                        'El modelo usa estas columnas: <b>' + ", ".join(COLUMNAS) + '</b>.</p>')
                archivo = gr.File(label="Archivo CSV", file_types=[".csv"], type="filepath")
                boton_csv = gr.Button("Ordenar por riesgo", variant="primary", size="lg")
                resumen = gr.HTML()
                tabla = gr.Dataframe(label="Clientes ordenados de mayor a menor riesgo", interactive=False)
                descarga = gr.File(label="Descargar la lista", interactive=False)
            boton_csv.click(priorizar_csv, inputs=archivo, outputs=[resumen, tabla, descarga])

        with gr.Tab("Sobre el modelo"):
            with gr.Column(elem_classes="tarjeta"):
                gr.HTML(resumen_modelo())
                gr.HTML('<div class="titulo-tarjeta" style="margin-top:18px">Si solo se contacta a los de mayor riesgo</div>'
                        '<p class="sub-tarjeta">Mueve el deslizador: así se vería priorizar las llamadas de retención con '
                        'este modelo, usando los 630 clientes reales de la muestra de prueba.</p>')
                deslizador_prioridad = gr.Slider(minimum=10, maximum=len(EJEMPLOS), step=5, value=100,
                                                 label="Clientes a contactar")
                panel_prioridad = gr.HTML(calcular_priorizacion_dinamica(100), elem_classes="priorizacion")
            deslizador_prioridad.change(calcular_priorizacion_dinamica, inputs=deslizador_prioridad, outputs=panel_prioridad)

        gr.HTML('<div class="pie">Fundamentos de Inteligencia Artificial · Universidad EIA · 2026-2 · '
                'Datos: Iranian Churn (UCI 563)</div>')

        # Arranca la animación de conteo de los números clave; el sondeo que deja activo sigue atrapando
        # los números nuevos que aparezcan después (al evaluar un cliente o mover el deslizador).
        interfaz.load(None, None, None, js=JS_CONTADORES)
    return interfaz


if __name__ == "__main__":
    # En Gradio 6 el tema, los estilos y la cabecera se pasan al lanzar la app
    construir_interfaz().launch(share=EN_COLAB, theme=crear_tema(), css=ESTILOS_CSS, head=CABECERA_HTML)
