<div align="center">

# 📉 Predicción de Abandono de Clientes de Telecomunicaciones

**¿Qué cliente va a irse? Un modelo que ordena a los clientes por riesgo de abandono para priorizar las llamadas de retención.**

Proceso completo de Machine Learning sobre el dataset Iranian Churn, en un solo notebook organizado como un
proyecto real de analítica: del entendimiento del negocio a la exportación del modelo. Comparación de 12
clasificadores con 4 estrategias de balanceo, ajuste de hiperparámetros, calibración de probabilidades y
publicación del modelo final en una app web.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTENC-1F6FEB)
![pandas](https://img.shields.io/badge/pandas-datos-150458?logo=pandas&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-app-F97316?logo=gradio&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-compatible-F9AB00?logo=googlecolab&logoColor=white)
![uv](https://img.shields.io/badge/uv-entorno-DE5FE9?logo=uv&logoColor=white)

</div>

---

## Tabla de contenido

1. [Descripción del proyecto](#descripción-del-proyecto)
2. [Equipo](#equipo)
3. [El problema](#el-problema)
4. [Dataset](#dataset)
5. [El notebook, de principio a fin](#el-notebook-de-principio-a-fin)
6. [Modelos y estrategias de balanceo](#modelos-y-estrategias-de-balanceo)
7. [Estructura del repositorio](#estructura-del-repositorio)
8. [Stack tecnológico](#stack-tecnológico)
9. [Decisiones técnicas](#decisiones-técnicas)
10. [Puesta en marcha en local](#puesta-en-marcha-en-local)
11. [Ejecución en Google Colab](#ejecución-en-google-colab)
12. [App web](#app-web)
13. [Resultados](#resultados)
14. [Informe y presentación](#informe-y-presentación)
15. [Solución de problemas](#solución-de-problemas)
16. [Limitaciones conocidas](#limitaciones-conocidas)
17. [Mejoras futuras](#mejoras-futuras)
18. [Referencias](#referencias)

---

## Descripción del proyecto

Una empresa de telecomunicaciones no puede llamar a todos sus clientes para ofrecerles algo que los haga
quedarse: necesita saber a quién llamar primero. Este proyecto usa el uso del servicio durante los primeros
**nueve meses** de la suscripción para predecir si un cliente **abandonará en el mes doce**, con el fin de
priorizar los contactos de retención hacia los clientes de mayor riesgo.

Todo el análisis vive en un único notebook, **`Churn_Paso_a_Paso.ipynb`**, organizado como las doce etapas de
un proyecto de analítica (entendimiento del negocio, enfoque analítico, requerimiento de datos, recolección,
separación de entrenamiento y prueba, entendimiento de los datos, preparación, modelado, ajuste de
hiperparámetros, calibración y umbral, evaluación final, despliegue) en vez de como secciones técnicas sueltas.

Proyecto final de **Fundamentos de Inteligencia Artificial** — Universidad EIA, 2026-2.

## Equipo

| Integrante | GitHub |
| :--- | :--- |
| Juan Jose Jaramillo Mora | [@Juanjo1414](https://github.com/Juanjo1414) |
| Sebastian Giraldo Franco | [@sebasgiraldo69](https://github.com/sebasgiraldo69) |
| Martin Restrepo | [@martinrestrepoc](https://github.com/martinrestrepoc) |
| Miguel Angel Zuleta | [@miguel142434](https://github.com/miguel142434) |
| Julian Mora | [@julian95mora25](https://github.com/julian95mora25) |
| Santiago Zuluaga | [@santiago-zuluaga](https://github.com/santiago-zuluaga) |

## El problema

| Aspecto | Detalle |
| :--- | :--- |
| Tipo de tarea | Clasificación binaria supervisada |
| Entrada | 12 variables de los primeros 9 meses: quejas, antigüedad, nivel de cargo, uso (segundos, llamadas, SMS, números distintos), llamadas fallidas, plan tarifario, estado de la línea, grupo de edad y valor del cliente |
| Salida | `abandono`: 1 = el cliente abandonó en el mes doce, 0 = permaneció |
| Desbalance | ~15,7% de abandono (495 de 3.150 clientes) |
| Métrica principal | F1 de la clase que abandona |
| Métricas secundarias | Recall, precisión, precisión promedio, ROC AUC y priorización (la exactitud nunca decide) |

## Dataset

**Iranian Churn Dataset** — datos de una empresa de telecomunicaciones iraní, publicados en el UCI Machine
Learning Repository ([id 563](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)). Cada fila
resume el uso del servicio de un cliente durante nueve meses y si abandonó o permaneció en el mes doce.

| Columna | Descripción | Tipo |
| :--- | :--- | :--- |
| `fallas_llamadas` | Llamadas fallidas | Numérica (conteo) |
| `quejas` | 0 sin queja; 1 con queja (no cuenta cuántas) | Binaria |
| `antiguedad` | Meses de suscripción | Numérica |
| `nivel_cargo` | Nivel de cargo, documentado de 0 a 9 (hay algunos casos de 10) | Ordinal |
| `segundos_uso` | Segundos de llamadas | Numérica |
| `frecuencia_uso` | Número de llamadas | Numérica (conteo) |
| `frecuencia_sms` | Número de mensajes | Numérica (conteo) |
| `numeros_distintos` | Números distintos llamados | Numérica (conteo) |
| `grupo_edad` | Grupo de edad, de 1 a 5 | Ordinal |
| `plan_tarifario` | 1 = pago por uso; 2 = contractual | Nominal binaria |
| `estado` | 1 = línea activa; 2 = no activa | Nominal binaria |
| `valor_cliente` | Valor calculado por la empresa a partir de otras variables; UCI no publica su fórmula, así que no se lee como una cifra en dinero | Numérica derivada |
| `abandono` | 1 = el cliente abandonó en el mes doce | Objetivo |

`edad` viene en el archivo original pero se elimina en el notebook: coincide exactamente con `grupo_edad`
(coeficiente phik de 1,000 entre las dos), así que no aporta información nueva. Es la única variable que se
elimina en todo el proyecto.

## El notebook, de principio a fin

`Churn_Paso_a_Paso.ipynb` sigue doce etapas, cada una con su propio índice interno dentro del archivo:

| Etapa | Qué hace |
| :--- | :--- |
| 1–3. Entendimiento del negocio, enfoque analítico, requerimiento de datos | Por qué importa priorizar contactos, por qué el F1 es la métrica y qué le falta al dataset (sin identificador de cliente ni fechas) |
| 4. Recolección de los datos | Carga el CSV original, revisa tamaño, tipos y filas repetidas, y traduce los nombres de columna al español |
| 5. Separación de entrenamiento y prueba | `train_test_split` 80/20 estratificado (`random_state=42`): 2.520 filas de entrenamiento y 630 de prueba |
| 6. Entendimiento de los datos | Desbalance, variables categóricas y numéricas, asociación con el abandono (phik), y la decisión de eliminar `edad` |
| 7. Preparación | Escalado con `MinMaxScaler`, comparación de ANOVA contra PCA para seleccionar variables |
| 8. Modelado | 12 modelos × 4 estrategias de balanceo = 41 combinaciones, con validación cruzada de 10 particiones |
| 9. Ajuste de hiperparámetros | `GridSearchCV` sobre los 3 mejores candidatos, dentro de un `Pipeline` de imbalanced-learn |
| 10. Calibración y umbral | Probabilidades calibradas con una curva sigmoide y elección del umbral de alerta con entrenamiento |
| 11. Evaluación final | Una sola medición sobre los 630 clientes de prueba, nunca antes consultados |
| 12. Despliegue | Importancia de variables, exportación del modelo (`.joblib`) y comprobación de que al recargarlo predice exactamente igual |

El notebook se ejecuta de principio a fin sin depender de ningún archivo intermedio: solo necesita el CSV
original de UCI.

## Modelos y estrategias de balanceo

| Modelos individuales (8) | Ensambles (4) | Estrategias de balanceo (4) |
| :--- | :--- | :--- |
| KNN | Bagging | Sin balanceo |
| Naive Bayes | AdaBoost | Peso por clase (`class_weight="balanced"`) * |
| Árbol de decisión | Gradient Boosting | SMOTENC |
| Regresión logística | Stacking | Submuestreo aleatorio |
| SVM | | |
| SVM lineal | | |
| Red neuronal (MLP) | | |
| Random Forest | | |

\* El peso por clase solo aplica a los modelos que tienen ese parámetro: árbol de decisión, regresión
logística, SVM, SVM lineal y Random Forest. Con los demás modelos se evalúan sin balanceo, con SMOTENC y con
submuestreo → 12 × 3 + 5 = **41 combinaciones**.

## Estructura del repositorio

```text
prediccion-abandono-clientes/
├── data/
│   ├── original/                          # Customer Churn.csv, archivo original de UCI, sin modificar
│   ├── iranian_churn_uci.zip               # Descarga original comprimida, de respaldo
│   ├── train.csv                          # Generado por el notebook (2.520 filas, 12 entradas + abandono)
│   └── test.csv                           # Generado por el notebook (630 filas)
├── models/
│   ├── modelo_final.joblib                # Pipeline, calibrador, umbral y columnas (no se versiona, ver abajo)
│   └── metadata_modelo.json               # Modelo, hiperparámetros, umbral, métricas y fecha de entrenamiento
├── resultados/
│   └── priorizacion_test.csv              # Qué pasa si solo se contacta al 5 %, 10 %, 15 %... de mayor riesgo
├── informe/
│   ├── informe.html                       # Fuente del informe final (se imprime a PDF desde el navegador)
│   └── Informe_Final.pdf                  # Informe final
├── presentacion/
│   └── presentacion.html                  # Presentación interactiva (se abre en el navegador, sin servidor)
├── referencia/                            # Notebooks de clase usados como guía de estilo
├── Churn_Paso_a_Paso.ipynb                # El notebook: todo el análisis, de principio a fin
├── app.py                                 # App web en Gradio
├── pyproject.toml
├── uv.lock
├── .python-version
└── .gitignore
```

**Sobre `models/modelo_final.joblib`:** el modelo entrenado **sí vive en esa carpeta en local**, pero **no se
sube a GitHub** — está en `.gitignore` (`models/*.joblib`) porque pesa cerca de 50 MB y `joblib` no produce
archivos binarios idénticos entre una ejecución y otra, así que versionarlo no sirve de mucho. Quien clona el
repositorio lo regenera ejecutando `Churn_Paso_a_Paso.ipynb` completo; la app lo busca automáticamente en esa
ruta.

## Stack tecnológico

| Capa | Tecnología |
| :--- | :--- |
| Lenguaje | Python 3.11 |
| Datos y análisis | pandas, NumPy, SciPy |
| Visualización | Matplotlib, Seaborn, phik (correlación con variables categóricas) |
| Machine Learning | scikit-learn (12 clasificadores, `GridSearchCV`, `Pipeline`, PCA, ANOVA, `CalibratedClassifierCV`) |
| Balanceo de clases | imbalanced-learn (SMOTENC, `RandomUnderSampler`, `Pipeline` compatible con remuestreo) |
| Persistencia del modelo | joblib, JSON |
| App web | Gradio |
| Entorno | uv (`pyproject.toml` + `uv.lock`), Jupyter en VS Code, Google Colab |

## Decisiones técnicas

- **F1 de la clase que abandona como métrica principal.** Con 15,7% de positivos, un modelo que siempre
  responda "permanece" saca 84,3% de exactitud sin detectar a nadie; por eso la exactitud nunca decide.
- **`estado` y `valor_cliente` se conservan, con evidencia.** Son las dos variables con una definición menos
  clara, así que se revisaron aparte: `estado` separa el abandono incluso entre los clientes que nunca se
  quejaron, y `valor_cliente`, aunque las demás variables explican el 98% de su variación, no mete información
  de fuera del periodo medido. Ninguna de las dos se descarta sin comprobarlo primero.
- **PCA se probó y se descartó.** Resumir las 12 variables en componentes necesitaría 7 de 12 para conservar
  el 90% de la varianza — casi no comprime, y además cada componente mezcla las 12 variables originales, lo
  que haría imposible explicarle al equipo de retención *por qué* un cliente está en riesgo. Se prefirió
  conservar las variables originales, interpretables.
- **Selección de variables con ANOVA, medida con los modelos reales.** Dos variables (`grupo_edad` y
  `fallas_llamadas`) parecían poco útiles por separado, pero quitarlas empeoró a los tres candidatos en más de
  5 centésimas de F1: aportan combinadas con las demás, algo que una prueba variable por variable no puede ver.
  Se conservan las 12 variables completas.
- **Balanceo dentro de un `Pipeline`.** SMOTENC y el submuestreo se ajustan solo con la parte de entrenamiento
  de cada partición, nunca con la que se evalúa — ni directamente ni a través del escalado.
- **Calibración de probabilidades.** El peso por clase sesga las probabilidades hacia arriba (promedio de
  18,65% cuando el abandono real es 15,71%); se corrige con una calibración sigmoide (`CalibratedClassifierCV`)
  antes de elegir el umbral.
- **Modelo elegido entre un empate con una regla explícita.** Los tres candidatos ajustados quedaron a
  milésimas de F1, muy por debajo de la desviación entre particiones. En vez de elegir el número más alto (que
  sería casi azar), se aplicó la regla de "un error estándar": entre las configuraciones empatadas dentro de
  ese margen, se eligió la más simple y rápida de entrenar — Random Forest.
- **Set de prueba reservado.** Los 630 clientes de prueba se separan en la etapa 5 y no se vuelven a tocar
  hasta la etapa 11, una sola vez, después de congelar modelo, calibración y umbral.

## Puesta en marcha en local

Requisitos: Git, Visual Studio Code con las extensiones [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
y [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), y [uv](https://docs.astral.sh/uv/).
No es necesario instalar Python manualmente: uv descarga la versión 3.11 indicada en `.python-version`.

### 1. Instalar uv

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Cierra y abre la terminal, y verifica con `uv --version`.

### 2. Clonar el repositorio e instalar las dependencias

```bash
git clone <url-del-repositorio>
cd prediccion-abandono-clientes
uv sync
```

`uv sync` crea el entorno virtual `.venv` con Python 3.11 e instala exactamente las versiones registradas en
`uv.lock`. El dataset ya viene incluido en el repositorio (`data/original/`).

### 3. Seleccionar el kernel en VS Code

Abre la carpeta del proyecto en VS Code, abre `Churn_Paso_a_Paso.ipynb` y en **Select Kernel → Python
Environments** elige el intérprete de `.venv`:

- Windows: `.venv\Scripts\python.exe`
- macOS / Linux: `.venv/bin/python`

### 4. Ejecutar el notebook

Verifica que la primera celda tenga `EN_COLAB = False` y ejecuta con **Run All**. La sección 8 (41
combinaciones con validación cruzada) es la más pesada del notebook y tarda varios minutos.

También se puede ejecutar desde la terminal:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace Churn_Paso_a_Paso.ipynb --ExecutePreprocessor.timeout=-1
```

## Ejecución en Google Colab

1. Sube la carpeta del proyecto a Google Drive con el nombre `Prediccion_Abandono_Clientes`.
2. Abre el notebook en Colab y cambia `EN_COLAB = True` en la primera celda de configuración. La celda monta
   Drive, instala lo que falte e instala las rutas automáticamente.

```python
# La primera celda ya incluye esto cuando EN_COLAB = True
%pip install phik imbalanced-learn -q
```

## App web

```bash
uv run python app.py
```

La app se abre en `http://127.0.0.1:7860` y tiene tres pestañas:

| Pestaña | Qué hace |
| :--- | :--- |
| **Evaluar un cliente** | Formulario con las doce entradas del modelo, con su descripción. Muestra el riesgo estimado en %, un veredicto según el umbral y los factores que más empujan el riesgo de ese cliente. Botones de ejemplo cargan un cliente tranquilo, uno en el límite del umbral, uno en riesgo alto o uno real al azar del set de prueba, mostrando qué ocurrió realmente |
| **Priorizar contactos** | Recibe un CSV con un cliente por fila (en español o con los nombres originales en inglés de UCI) y devuelve una tabla ordenada por riesgo, con resumen y archivo descargable |
| **Sobre el modelo** | Modelo, estrategia de balanceo, calibración, umbral, métricas de validación y de prueba, en qué se apoya el modelo (importancia de variables) y un deslizador en vivo que recalcula cuántos abandonos se cubren según cuántos clientes se contacten |

La app reconstruye ella misma la partición 80/20 del notebook (misma semilla) a partir de
`data/original/Customer Churn.csv`, así que no depende de que `data/train.csv` o `data/test.csv` existan.

En Colab, cambia `EN_COLAB = True` dentro de `app.py` y ejecútalo desde la carpeta del proyecto. Gradio genera
un enlace público temporal (`*.gradio.live`).

> ⚠️ Proyecto académico. La app no estima si una oferta concreta evitará el abandono ni cuánto ahorraría
> hacerlo, y no reemplaza el criterio del equipo de retención.

## Resultados

**Preprocesamiento elegido:** `MinMaxScaler`, sin PCA ni selección de variables — las 12 entradas completas
superaron a cualquier subconjunto probado con ANOVA.

**Modelo final:** Random Forest (400 árboles, sin límite de profundidad, hoja mínima 1) con
`class_weight="balanced"`, calibración sigmoide de las probabilidades y umbral de decisión de **0,35**.

| Métrica | Validación (entrenamiento) | Set de prueba |
| :--- | :--- | :--- |
| F1 clase que abandona | 0,8722 | 0,9029 |
| Recall | — | 0,9394 |
| Precisión | — | 0,8692 |
| ROC AUC | — | 0,9861 |
| Precisión promedio | — | 0,9209 |
| Brier | — | 0,0278 |
| Exactitud | — | 0,9683 |

En el set de prueba el modelo detecta **93 de los 99 abandonos reales** y da 14 falsas alarmas entre 531
clientes que permanecen. Un modelo que siempre dijera "permanece" acertaría el 84,3% de las veces sin detectar
ninguno (F1 = 0).

**Priorizar contactos:** contactando solo al 15% de los clientes de prueba con mayor riesgo (94 clientes), se
cubre el 85,9% de todos los abandonos reales con un 90,4% de aciertos — 5,75 veces mejor que contactar al azar.

> **Nota de transparencia.** F1 de prueba (0,9029) queda por encima del F1 de validación (0,8722), y no debe
> leerse como que el modelo resultó mejor de lo esperado: 630 clientes con 99 abandonos son pocos, así que esta
> cifra tiene su propio margen, de unas tres centésimas. Lo razonable es reportar el desempeño como un rango,
> **entre 0,87 y 0,90**, no como una cifra exacta.

## Informe y presentación

- **Informe final:** [`informe/Informe_Final.pdf`](informe/Informe_Final.pdf). Su fuente es
  `informe/informe.html`; para regenerar el PDF basta con abrirlo en un navegador basado en Chromium e
  imprimir a PDF en tamaño A4 sin encabezados.
- **Presentación:** abre `presentacion/presentacion.html` en el navegador. Flechas o espacio para avanzar, `O`
  para la vista general, `N` para las notas del orador, `T` para el cronómetro y `F` para pantalla completa.
  Incluye gráficas interactivas con los resultados reales y un deslizador de priorización de contactos.

## Solución de problemas

| Problema | Solución |
| :--- | :--- |
| VS Code no muestra `.venv` como kernel | Ejecuta `uv sync` desde la raíz, verifica que las extensiones Python y Jupyter estén habilitadas y usa **Python: Select Interpreter** para elegir `.venv` manualmente. Si no aparece, ejecuta **Developer: Reload Window** |
| `FileNotFoundError` al cargar los datos | Verifica que `data/original/Customer Churn.csv` exista y que `EN_COLAB = False` en local |
| `ModuleNotFoundError: imblearn` | El paquete se llama `imbalanced-learn`. En local ejecuta `uv sync`; en Colab, `!pip install imbalanced-learn -q` |
| La app no encuentra el modelo | Ejecuta `Churn_Paso_a_Paso.ipynb` completo antes de `app.py`, para que genere `models/modelo_final.joblib` |
| Falta una dependencia | No uses `pip install` en local. Agrégala con `uv add nombre-del-paquete` y haz commit de `pyproject.toml` y `uv.lock` |

## Limitaciones conocidas

- Los datos no traen fecha de medición ni identificador de cliente, así que no se puede comprobar el orden
  exacto de los hechos cliente por cliente, ni descartar que alguna entrada (por ejemplo, uso igual a cero) ya
  reflejara un abandono ocurrido.
- `estado`, la variable más usada por el modelo, es un indicador tardío: suele cambiar cuando el cliente ya
  dejó de usar el servicio, justo cuando queda menos margen para retenerlo.
- Los datos son de una sola empresa de telecomunicaciones; no se sabe cómo se comportaría el modelo en otro
  periodo o mercado.
- El umbral de 0,35 maximiza el F1, pero el umbral adecuado en la práctica depende de cuánto cueste contactar
  a un cliente frente a perderlo; ese costo no se estimó en este trabajo.
- El modelo ordena a los clientes por riesgo de abandono; no estima si una oferta de retención cambiaría ese
  riesgo ni cuánto ahorraría.

## Mejoras futuras

- Conseguir datos con fecha para validar con una partición temporal.
- Costos asimétricos entre contactar de más y perder un cliente, y selección del umbral a partir de esos
  costos.
- Un diseño experimental (A/B) que mida si contactar a los clientes de mayor riesgo reduce el abandono real.
- Datos de otra empresa o periodo, y despliegue permanente de la app.

## Referencias

- *Iranian Churn Dataset* (2020) [Dataset]. UCI Machine Learning Repository, id 563. https://doi.org/10.24432/C5JW3Z
- [scikit-learn: documentación oficial](https://scikit-learn.org/stable/)
- [imbalanced-learn: documentación oficial](https://imbalanced-learn.org/stable/)
- [Gradio: documentación oficial](https://www.gradio.app/docs)
- [uv: documentación oficial](https://docs.astral.sh/uv/)
