<div align="center">

# Predicción de abandono de clientes de telecomunicaciones

**¿Qué cliente va a irse? Modelos que ordenan a los clientes por riesgo de abandono para decidir a quién llamar primero.**

Proceso completo de Machine Learning sobre el dataset Iranian Churn, en dos cuadernos: el análisis original,
que optimiza el F1, y su corrección con Recall y piso de precisión, que ajusta los doce modelos antes de probar
la selección de variables, el PCA y SHAP. Incluye una aplicación web, un informe y una presentación.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTENC-1F6FEB)
![SHAP](https://img.shields.io/badge/SHAP-explicabilidad-8B5CF6)
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
5. [Los dos cuadernos](#los-dos-cuadernos)
6. [Cuaderno de corrección, sección por sección](#cuaderno-de-corrección-sección-por-sección)
7. [Cuaderno original, sección por sección](#cuaderno-original-sección-por-sección)
8. [Modelos, balanceo y métodos de selección](#modelos-balanceo-y-métodos-de-selección)
9. [Estructura del repositorio](#estructura-del-repositorio)
10. [Stack tecnológico](#stack-tecnológico)
11. [Decisiones técnicas](#decisiones-técnicas)
12. [Puesta en marcha en local](#puesta-en-marcha-en-local)
13. [Ejecución en Google Colab](#ejecución-en-google-colab)
14. [App web](#app-web)
15. [Resultados](#resultados)
16. [Informe y presentación](#informe-y-presentación)
17. [Solución de problemas](#solución-de-problemas)
18. [Limitaciones conocidas](#limitaciones-conocidas)
19. [Mejoras futuras](#mejoras-futuras)
20. [Referencias](#referencias)

---

## Descripción del proyecto

Una empresa de telecomunicaciones no puede llamar a todos sus clientes para ofrecerles algo que los haga
quedarse: necesita saber a quién llamar primero. Este proyecto usa el uso del servicio durante los primeros
**nueve meses** de la suscripción para predecir si un cliente **abandonará en el mes doce**, con el fin de
priorizar los contactos de retención hacia los clientes de mayor riesgo.

El análisis está en dos cuadernos que se pueden ejecutar por separado:

- **`Prediccion_abandono_clientes_Original.ipynb`** es el análisis original. Compara 12 clasificadores con 4
  estrategias de balanceo (41 combinaciones), ajusta los tres mejores y elige un Random Forest con umbral de
  0,50. Optimiza el F1.
- **`Prediccion_abandono_clientes_Correccion.ipynb`** es la corrección de lo que nos señaló la profesora: primero
  se ajustan los hiperparámetros de los doce modelos con los datos en crudo; después se prueban, sobre cada modelo
  ya ajustado, todos los métodos de selección de variables, el PCA y SHAP; la métrica es el Recall con un piso de
  precisión; y se elige el mejor de todas las combinaciones y se valida.

Los dos siguen las etapas de un proyecto de analítica, del entendimiento del negocio al despliegue.

Proyecto de **Fundamentos de Inteligencia Artificial** — Universidad EIA, 2026-2.

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
| Desbalance | ~15,7 % de abandono (495 de 3.150 clientes) |
| Métrica del cuaderno original | F1 de la clase que abandona |
| Métrica del cuaderno de corrección | Recall con un piso de precisión de 0,60 (el F1 desempata) |
| Métricas secundarias | Precisión, recall, ROC AUC y priorización de contactos; la exactitud nunca decide |

El Recall solo se puede engañar: un modelo que dijera que todos los clientes abandonan tendría un Recall de 1,00
y una precisión de 0,157. Por eso la corrección lo acompaña de un piso de precisión.

## Dataset

**Iranian Churn Dataset** — datos de una empresa de telecomunicaciones iraní, publicados en el UCI Machine
Learning Repository ([id 563](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset), licencia CC BY 4.0).
Cada fila resume el uso del servicio de un cliente durante nueve meses y si abandonó o permaneció en el mes doce.

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

`edad` viene en el archivo original pero se elimina en los dos cuadernos: coincide exactamente con `grupo_edad`
(coeficiente phik de 1,000 entre las dos), así que no aporta información nueva. Es la única variable que se
elimina en todo el proyecto. El archivo original no se modifica; `data/procedencia.json` guarda su origen,
licencia y huellas de verificación.

## Los dos cuadernos

| Aspecto | Cuaderno original | Cuaderno de corrección |
| :--- | :--- | :--- |
| Archivo | `Prediccion_abandono_clientes_Original.ipynb` | `Prediccion_abandono_clientes_Correccion.ipynb` |
| Métrica que se optimiza | F1 de la clase que abandona | Recall con piso de precisión de 0,60 |
| Orden del trabajo | Compara 41 combinaciones con valores por defecto, elige los tres mejores y solo a ellos les ajusta los hiperparámetros | Ajusta primero los doce modelos con una búsqueda y después los cruza con los métodos de selección, PCA y SHAP |
| Hiperparámetros | `GridSearchCV` con rejillas pequeñas, para tres candidatos | `RandomizedSearchCV` en rangos amplios, 60 combinaciones por modelo (30 en Stacking), para los doce |
| Selección de variables | ANOVA con tres candidatos | 15 métodos por 12 modelos: 180 combinaciones, con PCA y SHAP |
| Matriz de confusión | Del modelo final y de otros dos candidatos | De los doce modelos ajustados |
| Elección del mejor | Regla del error estándar: el más rápido de los empatados | Mayor Recall con precisión sobre el piso, validación anidada |
| Umbral | 0,50 (el de mayor F1) | 0,34 (el de mayor Recall con el piso) |
| Modelo elegido | Random Forest con peso por clase | Red neuronal con las doce variables |
| Modelo guardado | `models/modelo_final.joblib` | `models/modelo_recall.joblib` |
| Relación con la app | La app y la presentación usan este modelo | Documenta las correcciones pedidas |

Los dos usan la misma partición 80/20 con `random_state=42` y las mismas secciones de exploración. Se pueden
ejecutar de forma independiente: ninguno necesita al otro, solo el CSV original de UCI.

## Cuaderno de corrección, sección por sección

| Sección | Qué hace |
| :--- | :--- |
| 1–3. Negocio, enfoque, datos | Por qué importa priorizar contactos, por qué el Recall necesita un piso de precisión y qué le falta al dataset (sin identificador de cliente ni fechas) |
| 4. Recolección | Carga el CSV original, revisa tamaño, tipos y filas repetidas, y traduce los nombres de columna al español |
| 5. Separación | `train_test_split` 80/20 estratificado: 2.520 filas de entrenamiento y 630 de prueba |
| 6. Entendimiento de los datos | Desbalance, variables categóricas y numéricas, asociación con el abandono (phik) y la decisión de eliminar `edad` |
| 7. Preparación | Escalado, tratamiento del desbalance y por qué todo va dentro de un `Pipeline`; se dejan como opciones de la búsqueda |
| 8. Modelado | Los doce modelos con valores por defecto: la línea base, medida con Recall y piso de precisión |
| 9. Ajuste de hiperparámetros | Búsqueda aleatoria para los doce modelos, comparación con la línea base, sobreajuste de KNN y árbol, y matriz de confusión de cada modelo |
| 10. Selección de variables, PCA y SHAP | La matriz de 12 modelos por 15 métodos (180 combinaciones), el mejor de todos, su reajuste, SHAP, validación anidada y PCA |
| 11. Selección del umbral | Probabilidades fuera de pliegue y umbral de mayor Recall con precisión sobre el piso |
| 12. Evaluación final | Una sola medición sobre los 630 clientes de prueba: métricas, matriz de confusión, curvas ROC y Precision-Recall, lista de prioridad y comparación con el original |
| 13. Despliegue y conclusiones | Variables que pesan, referencia para el PSI, modelo guardado y recarga, función `predecir_abandono`, plan de monitoreo y conclusiones |

## Cuaderno original, sección por sección

| Sección | Qué hace |
| :--- | :--- |
| 1–6 | Igual que en la corrección: negocio, enfoque, datos, recolección, separación y exploración |
| 7. Preparación | Escalado con `MinMaxScaler` y comparación de ANOVA contra PCA para seleccionar variables |
| 8. Modelado | 12 modelos × 4 estrategias de balanceo = 41 combinaciones, con validación cruzada de 10 particiones |
| 9. Ajuste de hiperparámetros | `GridSearchCV` sobre los 3 mejores candidatos, dentro de un `Pipeline` de imbalanced-learn |
| 10. Selección del umbral | Probabilidades fuera de pliegue y umbral de mayor F1 |
| 11. Evaluación final | Una sola medición sobre los 630 clientes de prueba, con las matrices de los otros dos candidatos y la lista de prioridad |
| 12. Despliegue y conclusiones | Importancia de variables, exportación del modelo y de sus datos, y comprobación de que al recargarlo predice exactamente igual |

## Modelos, balanceo y métodos de selección

| Modelos individuales (8) | Ensambles (4) | Estrategias de balanceo |
| :--- | :--- | :--- |
| KNN | Bagging | Sin balanceo |
| Naive Bayes | AdaBoost | Peso por clase (`class_weight="balanced"`) * |
| Árbol de decisión | Gradient Boosting | SMOTENC |
| Regresión logística | Stacking | Submuestreo aleatorio |
| SVM | | |
| SVM lineal | | |
| Red neuronal (MLP) | | |
| Random Forest | | |

\* El peso por clase solo aplica a los modelos que tienen ese parámetro: árbol de decisión, regresión logística,
SVM, SVM lineal y Random Forest. En el cuaderno original eso da 12 × 3 + 5 = **41 combinaciones**; en el de
corrección el balanceo y el peso por clase son parámetros de la búsqueda de cada modelo.

**Métodos de selección del cuaderno de corrección (15):** todas las variables, ANOVA, chi cuadrado, información
mutua, varianza baja, selección hacia adelante, RFECV, regularización L1, importancia de Gradient Boosting,
SHAP (k mejores y 95 % acumulado), correlación entre variables, y PCA (95 % de la varianza, k componentes y dos
componentes). Los que son transformadores de scikit-learn se reajustan dentro del `Pipeline` en cada partición;
los demás entregan una lista de variables armada con una parte interna del entrenamiento.

## Estructura del repositorio

```text
prediccion-abandono-clientes/
├── data/
│   ├── original/                          # Customer Churn.csv, archivo original de UCI, sin modificar
│   ├── iranian_churn_uci.zip              # Descarga original comprimida, de respaldo
│   ├── procedencia.json                   # Origen, licencia y huellas del archivo original
│   ├── train.csv                          # Lo genera el cuaderno original (2.520 filas)
│   └── test.csv                           # Lo genera el cuaderno original (630 filas)
├── models/
│   ├── metadata_modelo.json               # Metadatos del modelo del cuaderno original
│   ├── metadata_recall.json               # Metadatos del modelo del cuaderno de corrección
│   ├── modelo_final.joblib                # Modelo del cuaderno original (no se versiona, ver abajo)
│   └── modelo_recall.joblib               # Modelo del cuaderno de corrección (no se versiona)
├── resultados/
│   ├── priorizacion_test.csv              # Contactar al 5 %, 10 %... de mayor riesgo (original)
│   ├── priorizacion_recall.csv            # Lo mismo para el cuaderno de corrección
│   ├── ajuste_hiperparametros.csv         # Mejor configuración de cada uno de los doce modelos
│   ├── matriz_combinaciones.csv           # Las 180 combinaciones de modelo y método
│   └── curvas_k.csv                       # Recall en validación interna según el número de variables
├── informe/
│   ├── informe.html                       # Fuente del informe (se imprime a PDF desde el navegador)
│   └── Informe_Final.pdf                  # Informe final con los dos cuadernos
├── presentacion/
│   └── presentacion.html                  # Presentación interactiva (se abre en el navegador)
├── pruebas/
│   └── prueba_clientes.csv                # Clientes de ejemplo para la pestaña "Priorizar contactos"
├── referencia/                            # Cuadernos de clase usados como guía
├── Prediccion_abandono_clientes_Original.ipynb
├── Prediccion_abandono_clientes_Correccion.ipynb
├── app.py                                 # App web en Gradio
├── pyproject.toml
├── uv.lock
├── .python-version
└── .gitignore
```

**Sobre los archivos `models/*.joblib`:** los modelos entrenados **sí viven en esa carpeta en local**, pero **no se
suben a GitHub** — están en `.gitignore` porque pesan cerca de 50 MB y `joblib` no produce archivos binarios
idénticos entre una ejecución y otra. Quien clona el repositorio los regenera ejecutando el cuaderno
correspondiente; la app busca `models/modelo_final.joblib`, que genera el cuaderno original.

Los archivos de `resultados/` los escriben los cuadernos. El de corrección guarda ahí el avance de la búsqueda y
de la matriz: si los archivos existen, los reutiliza y no repite lo que ya está calculado (con `REANUDAR` en `True`);
para recalcular desde cero basta con borrarlos o poner esa bandera en `False`.

## Stack tecnológico

| Capa | Tecnología |
| :--- | :--- |
| Lenguaje | Python 3.11 |
| Datos y análisis | pandas, NumPy, SciPy |
| Visualización | Matplotlib, Seaborn, phik (asociación entre variables numéricas y categóricas) |
| Machine Learning | scikit-learn (12 clasificadores, `GridSearchCV`, `RandomizedSearchCV`, `Pipeline`, PCA, selección de variables) |
| Balanceo de clases | imbalanced-learn (SMOTENC, `RandomUnderSampler`, `Pipeline` compatible con remuestreo) |
| Explicabilidad | SHAP |
| Persistencia del modelo | joblib, JSON |
| App web | Gradio |
| Entorno | uv (`pyproject.toml` + `uv.lock`), Jupyter en VS Code, Google Colab |

## Decisiones técnicas

**En los dos cuadernos**

- **Exactitud nunca decide.** Con 15,7 % de positivos, un modelo que siempre responda «permanece» saca 84,3 % de
  exactitud sin detectar a nadie.
- **`estado` y `valor_cliente` se conservan, con evidencia.** `estado` separa el abandono incluso entre los
  clientes que nunca se quejaron, y `valor_cliente`, aunque las demás variables explican el 98 % de su variación,
  no mete información de fuera del periodo medido. Ninguna se descarta sin comprobarlo primero.
- **Balanceo y escalado dentro de un `Pipeline`.** SMOTENC, el submuestreo y el escalador se ajustan solo con la
  parte de entrenamiento de cada partición, nunca con la que se evalúa.
- **Conjunto de prueba reservado.** Los 630 clientes de prueba se separan en la etapa 5 y no se vuelven a tocar
  hasta la evaluación final, una sola vez. Ese conjunto ya se había consultado en una versión anterior del
  trabajo, así que no es una prueba nueva intacta: no se usa para elegir nada.
- **Quitar variables no ayuda.** En el original, ANOVA empeora a los tres candidatos; en la corrección, ninguna de
  las 180 combinaciones supera al mejor modelo con las doce variables.

**Cuaderno original**

- **PCA se probó y se descartó.** Hacen falta 7 de 12 componentes para conservar el 90 % de la varianza, casi no
  comprime, y cada componente mezcla las doce variables, lo que haría imposible explicar por qué un cliente está en
  riesgo.
- **Modelo elegido entre un empate con una regla explícita.** Los tres candidatos ajustados quedaron a milésimas
  de F1, muy por debajo de la desviación entre particiones. Se aplicó la regla del error estándar: entre los
  empatados dentro de ese margen, el más rápido de entrenar, que es Random Forest.
- **Umbral con predicciones fuera de pliegue.** Se recorren cortes de 0,05 a 0,95 con probabilidades que el
  modelo dio sin haber visto al cliente; el de mayor F1 es 0,50.

**Cuaderno de corrección**

- **Recall con piso de precisión.** Entre las configuraciones con precisión de al menos 0,60 se elige la de mayor
  Recall, con el F1 como desempate.
- **Hiperparámetros por búsqueda, no a mano.** Rangos amplios con distribuciones log-uniformes; el escalador, el
  balanceo y el peso por clase son parámetros de la búsqueda. Solo van fijos los topes de cómputo.
- **Cada selector se ajusta dentro de cada partición** cuando es un transformador de scikit-learn; las listas
  fijas se marcan en el cuaderno como algo optimistas.
- **Elección del mejor sin PCA.** Las combinaciones de PCA no compiten por el primer puesto: sus componentes no
  se pueden explicar al equipo de retención. En un empate exacto de Recall gana la de mayor precisión.
- **Validación anidada.** La cifra honesta del modelo elegido se mide con diez particiones externas y cinco
  internas.

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
cd <carpeta-del-repositorio>
uv sync
```

`uv sync` crea el entorno virtual `.venv` con Python 3.11 e instala exactamente las versiones registradas en
`uv.lock`. El dataset ya viene incluido en el repositorio (`data/original/`).

### 3. Seleccionar el kernel en VS Code

Abre la carpeta del proyecto en VS Code, abre el cuaderno que quieras revisar y en **Select Kernel → Python
Environments** elige el intérprete de `.venv`:

- Windows: `.venv\Scripts\python.exe`
- macOS / Linux: `.venv/bin/python`

### 4. Ejecutar un cuaderno

Verifica que la primera celda tenga `EN_COLAB = False` y ejecuta con **Run All**. También se puede ejecutar desde
la terminal (con `--inplace`, el cuaderno queda con los resultados guardados):

```bash
uv run python -m nbconvert --to notebook --execute --inplace Prediccion_abandono_clientes_Original.ipynb --ExecutePreprocessor.timeout=-1
uv run python -m nbconvert --to notebook --execute --inplace Prediccion_abandono_clientes_Correccion.ipynb --ExecutePreprocessor.timeout=-1
```

**Cuánto tardan, aproximadamente.** El cuaderno original corre en unos cuatro minutos; la celda más pesada es la
comparación de las 41 combinaciones. El de corrección tarda bastante más: desde cero, más de una hora (la búsqueda
de los doce modelos, la matriz de 180 combinaciones y la validación anidada son las partes largas); con los
archivos de `resultados/` ya guardados, del orden de media hora. En ambos conviene dejar el equipo sin otras
tareas pesadas.

## Ejecución en Google Colab

1. Sube la carpeta del proyecto a Google Drive con el nombre `churn` (la ruta que usan los cuadernos es
   `/content/drive/MyDrive/churn/`).
2. Abre el cuaderno en Colab y cambia `EN_COLAB = True` en la primera celda. La celda monta Drive, instala lo que
   falte y ajusta las rutas automáticamente.

```python
# La primera celda ya incluye esto cuando EN_COLAB = True (el cuaderno original no instala shap)
%pip install phik imbalanced-learn shap -q
```

## App web

La app carga el modelo del **cuaderno original**, así que primero hay que generarlo ejecutando ese cuaderno
(crea `models/modelo_final.joblib`). Después:

```bash
uv run python app.py
```

La app se abre en `http://127.0.0.1:7860` y tiene tres pestañas:

| Pestaña | Qué hace |
| :--- | :--- |
| **Evaluar un cliente** | Formulario con las doce entradas del modelo, con su descripción. Muestra el riesgo estimado en %, un veredicto según el umbral y los factores que más empujan el riesgo de ese cliente. Botones de ejemplo cargan un cliente tranquilo, uno en el límite del umbral, uno en riesgo alto o uno real al azar del set de prueba, mostrando qué ocurrió realmente |
| **Priorizar contactos** | Recibe un CSV con un cliente por fila (en español o con los nombres originales en inglés de UCI; `pruebas/prueba_clientes.csv` sirve de ejemplo) y devuelve una tabla ordenada por riesgo, con resumen y archivo descargable |
| **Sobre el modelo** | Modelo, estrategia de balanceo, calibración (ninguna), umbral, métricas de validación y de prueba, en qué se apoya el modelo (importancia de variables) y un deslizador en vivo que recalcula cuántos abandonos se cubren según cuántos clientes se contacten |

La app reconstruye ella misma la partición 80/20 de los cuadernos (misma semilla) a partir de
`data/original/Customer Churn.csv`, así que no depende de que `data/train.csv` o `data/test.csv` existan.

En Colab, cambia `EN_COLAB = True` dentro de `app.py` y ejecútalo desde la carpeta del proyecto. Gradio genera
un enlace público temporal (`*.gradio.live`).

> ⚠️ Proyecto académico. La app no estima si una oferta concreta evitará el abandono ni cuánto ahorraría
> hacerlo, y no reemplaza el criterio del equipo de retención.

## Resultados

Los dos modelos se miden sobre los mismos 630 clientes de prueba (99 abandonos).

| | Cuaderno original | Cuaderno de corrección |
| :--- | :--- | :--- |
| Modelo | Random Forest, 400 árboles, `MinMaxScaler`, peso por clase | Red neuronal de dos capas (32 y 16), `MinMaxScaler`, submuestreo, doce variables |
| Umbral | 0,50 | 0,34 |
| Validación (entrenamiento) | F1 0,8722 en validación cruzada; F1 0,8742 con el umbral elegido | Recall 0,9546 y precisión 0,6684 en validación anidada |
| Recall en prueba | 0,9495 | 0,9596 |
| Precisión en prueba | 0,8624 | 0,6835 |
| F1 en prueba | 0,9038 | 0,7983 |
| Exactitud en prueba | 0,9683 | 0,9238 |
| Abandonos detectados | 94 de 99 | 95 de 99 |
| Falsas alarmas | 15 entre 531 | 44 entre 531 |

**Qué se ve.** El cuaderno de corrección detecta un abandono más, pero con casi el triple de falsas alarmas; con 99
abandonos en la prueba, esa ganancia está dentro del ruido y el costo no. El original queda mejor en precisión y en
F1. La elección entre los dos depende de cuánto cueste una llamada de más frente a un cliente que se pierde, algo
que este trabajo no estimó. En el cuaderno de corrección, 145 de las 180 combinaciones cumplen el piso de
precisión y ninguna supera a la red neuronal con las doce variables.

**Priorizar contactos.** En el original, contactar solo al 15 % de los clientes de prueba con mayor riesgo (94
clientes) cubre el 85,9 % de los abandonos con un 90,4 % de aciertos, 5,75 veces mejor que contactar al azar. En la
corrección, contactar al 20 % (126 clientes) cubre el 93,9 % de los abandonos con un 73,8 % de aciertos.

> **Nota de transparencia.** Con 99 abandonos en la prueba, cada cliente pesa un punto de Recall y las cifras de
> prueba tienen su propio margen. En el original, el F1 de prueba (0,9038) queda por encima del de validación
> (0,8722) y no debe leerse como que el modelo resultó mejor de lo esperado; lo razonable es reportar el
> desempeño como un rango, entre 0,87 y 0,90.

## Informe y presentación

- **Informe final:** [`informe/Informe_Final.pdf`](informe/Informe_Final.pdf). Describe los dos cuadernos, deja
  claro que uno es el original y el otro la corrección de lo que pidió la profesora, y los compara. Su fuente es
  `informe/informe.html`; para regenerar el PDF basta con abrirlo en un navegador basado en Chromium e imprimir a
  PDF en tamaño A4 sin encabezados ni pies de página.
- **Presentación:** abre `presentacion/presentacion.html` en el navegador. Flechas o espacio para avanzar, `O`
  para la vista general, `N` para las notas del orador, `T` para el cronómetro y `F` para pantalla completa.
  Incluye gráficas interactivas con los resultados del cuaderno original y un deslizador de priorización de
  contactos.

## Solución de problemas

| Problema | Solución |
| :--- | :--- |
| VS Code no muestra `.venv` como kernel | Ejecuta `uv sync` desde la raíz, verifica que las extensiones Python y Jupyter estén habilitadas y usa **Python: Select Interpreter** para elegir `.venv` manualmente. Si no aparece, ejecuta **Developer: Reload Window** |
| `FileNotFoundError` al cargar los datos | Verifica que `data/original/Customer Churn.csv` exista y que `EN_COLAB = False` en local |
| `ModuleNotFoundError: imblearn` o `shap` | El primero es el paquete `imbalanced-learn`. En local ejecuta `uv sync`; en Colab, `%pip install imbalanced-learn shap -q` |
| La app no encuentra el modelo | Ejecuta `Prediccion_abandono_clientes_Original.ipynb` completo antes de `app.py`, para que genere `models/modelo_final.joblib` |
| El cuaderno de corrección tarda mucho | Si ya ejecutaste la búsqueda, conserva los archivos de `resultados/`: con `REANUDAR` en `True` no se repite. Para recalcular, borra esos archivos |
| Falta una dependencia | No uses `pip install` en local. Agrégala con `uv add nombre-del-paquete` y haz commit de `pyproject.toml` y `uv.lock` |

## Limitaciones conocidas

- Los datos no traen fecha de medición ni identificador de cliente, así que no se puede comprobar el orden
  exacto de los hechos cliente por cliente, ni descartar que alguna entrada (por ejemplo, uso igual a cero) ya
  reflejara un abandono ocurrido.
- `estado`, una de las variables más usadas, es un indicador tardío: suele cambiar cuando el cliente ya dejó de
  usar el servicio, justo cuando queda menos margen para retenerlo.
- Los datos son de una sola empresa de telecomunicaciones; no se sabe cómo se comportarían los modelos en otro
  periodo o mercado.
- La partición no agrupa perfiles idénticos, así que puede haber el mismo perfil repartido entre entrenamiento y
  prueba, y no se midió ese solapamiento.
- En el cuaderno de corrección, el método de selección y el número de variables se eligen con todo el
  entrenamiento y la validación anidada no los corrige.
- Los umbrales (0,50 y 0,34) salen de optimizar una métrica; el adecuado en la práctica depende de cuánto cueste
  contactar a un cliente frente a perderlo, y ese costo no se estimó.
- Los modelos ordenan a los clientes por riesgo de abandono; no estiman si una oferta de retención cambiaría ese
  riesgo ni cuánto ahorraría.

## Mejoras futuras

- Conseguir datos con fecha para validar con una partición temporal.
- Costos asimétricos entre contactar de más y perder un cliente, y selección del umbral a partir de esos costos;
  con ellos, decidir entre los dos modelos.
- Un diseño experimental (A/B) que mida si contactar a los clientes de mayor riesgo reduce el abandono real.
- Datos de otra empresa o periodo, y despliegue permanente de la app.

## Referencias

- *Iranian Churn Dataset* (2020) [Dataset]. UCI Machine Learning Repository, id 563. https://doi.org/10.24432/C5JW3Z
- [scikit-learn: documentación oficial](https://scikit-learn.org/stable/)
- [imbalanced-learn: documentación oficial](https://imbalanced-learn.org/stable/)
- [SHAP: documentación oficial](https://shap.readthedocs.io/)
- [Gradio: documentación oficial](https://www.gradio.app/docs)
- [uv: documentación oficial](https://docs.astral.sh/uv/)
