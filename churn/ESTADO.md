# Estado del proyecto

Actualizado: 2026-10-10.

- [x] Paso 1: alcance acordado.
- [x] Paso 2: estructura y entorno verificados.
- [x] Paso 3: descargar y auditar dataset; notebook ejecutado en local.
- [x] Paso 4: partición por grupos reservada y verificada.
- [x] Paso 6: notebook 01 completado y ejecutado en local; Colab pendiente.
- Paso 5: omitido por decisión del usuario; se avanzó directamente al notebook 02.
- [x] Paso 7: notebook 02 simplificado (regresión logística y ANOVA), ejecutado en local; Colab pendiente.
- [x] Paso 8: notebook 03 completado y ejecutado en local; Colab pendiente.
- [x] Paso 9: notebook 04 completado y ejecutado en local (unos 26 minutos); Colab pendiente.
- [x] Paso 9b (adicional, no numerado en PLAN.md): notebook 05 de controles de rigor, ejecutado en local; Colab pendiente.
- [x] Paso 10: app Gradio (`app.py`) creada y probada en local.
- [x] Paso 11: informe (`informe/informe.html` e `Informe_Final.pdf`) y presentación (`presentacion/presentacion.html`) terminados.

## Decisiones vigentes

- Proyecto independiente dentro del espacio de trabajo actual; sismos intacto.
- Dataset previsto: Iranian Churn, UCI 563; objetivo binario de abandono.
- Conservar repeticiones y separar perfiles idénticos por grupos, autorizado por el usuario.
- Status y Customer Value excluidas inicialmente, según acuerdo en chat. Se conservan las once entradas restantes, incluidos edad y grupo de edad; su redundancia se revisará en el notebook 02.
- Se conservan las siete filas con cargo de nivel 10, sin corregirlas ni eliminarlas sin evidencia.
- Partición existente reutilizada: 2.511 filas de entrenamiento y 639 de prueba, sin perfiles compartidos. Grupos definidos por las once entradas, nunca por abandono. Primera partición de StratifiedGroupKFold de cinco pliegues, semilla 42.
- EDA y comparaciones realizados únicamente sobre entrenamiento. Se entrenó regresión logística de referencia en CV; no hay clasificador final ni evaluación predictiva de prueba.
- Notebook 02: StandardScaler y las once entradas, sin PCA ni selección ANOVA. F1 CV 0,5972 ± 0,0632, precisión 0,4544 y recall 0,8804. Diferencia pequeña frente a no escalar (0,5948); decisión operativa, no superioridad demostrada.
- Alcance del 02: tres escalados, PCA 90/95 % y ANOVA k=1..10 con regresión logística balanceada (liblinear). No incluye los demás selectores ni SVM de la versión de minas. La plantilla exportada está sin ajustar; sus pasos se concatenan con los del clasificador, sin anidar Pipeline dentro de imblearn.
- Notebook 03: 12 clasificadores con cuatro estrategias (sin balanceo, class_weight, SMOTENC y submuestreo; 41 combinaciones), con las diez particiones por grupos del 02 leídas de `02_particiones_cv.csv`. Sobremuestreo con SMOTENC y `quejas` y `plan_tarifario` como categóricas (SMOTE inventó 12 valores inválidos de `plan_tarifario` en la primera partición; SMOTENC ninguno). Candidatos: Stacking+SMOTENC (F1 0,862), Random Forest+class_weight (0,858) y red neuronal sin balancear (0,838). Control de fuga entre particiones sin evidencia de fuga.
- Notebook 04: hiperparámetros por precisión promedio, calibración sigmoide con puntajes fuera de pliegue, umbral con probabilidades calibradas de entrenamiento y estimación **anidada** por grupos (5 externas, 5 internas). Modelo final: Random Forest con class_weight (400 árboles, profundidad 15, hojas mínimas 1), umbral 0,37. F1 anidado 0,851 ± 0,076; **F1 en prueba 0,857** (IC 95 % por bootstrap de grupos: 0,800 a 0,914), recall 0,874, precisión 0,841, ROC AUC 0,982, Brier 0,035; 90 de 103 abandonos detectados y 17 falsas alarmas entre 536. Empata en la práctica con Stacking y con la red neuronal (diferencia de 0,0024 en F1 anidado). Prueba abierta una sola vez, con todo congelado.
- El F1 de 0,59 del notebook 02 era de una regresión logística (modelo lineal); la regresión logística vuelve a dar 0,57–0,60 en el 03. Los modelos de árboles llegan a 0,79–0,86 por interacciones (quejas, uso bajo), no por fuga. Cautela abierta: sin fechas no se puede descartar que alguna entrada (uso cero) refleje un abandono ya ocurrido.
- Cada paso requiere indicación del usuario. No crear notebooks o app durante la preparación del entorno.
- Carpeta `churn/`: nombre corto para evitar el límite de rutas de Windows. No se modificó la configuración del sistema.

## Bitácora

- 2026-10-10: terminados los pasos 10 y 11 a petición del usuario, después de revisar que los cinco notebooks corrieran sin fallos. Notebook 00 tenía un error al reejecutarse (comprobaciones que exigían que `train.csv`/`test.csv` no existieran): se quitaron esas comprobaciones y se reejecutaron 00, 01 y 02 completos sin errores. Se creó `05_Controles_Rigor_Churn.ipynb` (integridad de archivos y de la partición por perfiles, etiquetas mezcladas al azar, agrupación más estricta de perfiles parecidos y revisión de vecinos cercanos): sin evidencia de fuga mecánica (ROC AUC con etiquetas mezcladas 0,492), pero con grupos más estrictos el F1 baja de 0,859 a 0,811, por lo que el F1 final se reporta como rango (0,81–0,86) y no como cifra única. Se creó `app.py` (formulario, priorización por CSV y pestaña del modelo) y se probó en local contra los 639 clientes de prueba (probabilidades idénticas a las del notebook 04). Se armaron `presentacion/presentacion.html` (14 diapositivas interactivas, mismo motor que la del proyecto de sismos con paleta y datos propios) y `informe/informe.html` con su `Informe_Final.pdf` (24 páginas), siguiendo la misma estructura que el informe del proyecto de sismos. Colab sigue sin probarse. Sin commits.

- 2026-10-10: creados y ejecutados 03_Balanceo_Comparacion_Modelos_Churn.ipynb y 04_Ajuste_Evaluacion_Exportacion_Churn.ipynb a petición del usuario. Generan `comparacion_modelos.csv`, `03_candidatos.csv`, controles de sobremuestreo y de fuga, `ajuste_hiperparametros.csv`, `04_priorizacion_test.csv`, `models/modelo_final.joblib` (diccionario con pipeline, calibrador, umbral y columnas) y `models/metadata_modelo.json`, más figuras `03_*` y `04_*`. Verificada la recarga del modelo (probabilidades y predicciones idénticas). El 04 tardó unos 26 minutos, por encima de los 20 de referencia de las reglas; no se consultó antes. Las conclusiones del 04 se rellenaron sobre el notebook ya ejecutado, sin repetir la ejecución. Colab no probado. Sin commits.

- 2026-10-09: revisión completa de 02_Escalado_PCA_Seleccion_Churn.ipynb. Completadas conclusiones e interpretaciones, guardadas asignaciones de diez particiones y comparación ANOVA, documentados alcance simplificado y sesgo de selección. Añadidas huellas de entradas y parámetros del modelo a decisiones_preprocesamiento.joblib; comprobadas plantilla sin ajustar e integración plana con imblearn. Ejecución completa sin errores de celda en local. StandardScaler gana con 0,5972; PCA 90 % 0,5733, PCA 95 % 0,5716, mejor ANOVA k=10 0,5729. No se consultó prueba. Colab pendiente.

- 2026-10-09: revisado 01_EDA_Limpieza_Churn.ipynb a petición del usuario. Completadas explicaciones y conclusiones con resultados, verificación del original, cobertura de la partición y coherencia de CSV existentes. Documentada correspondencia edad/grupo de edad y guardadas tablas del EDA. Ejecución completa con nbconvert en local; 392 abandonos en entrenamiento (15,61 %), phik de quejas 0,747, segundos de uso 0,464 y frecuencia de uso 0,435. Reserva y CSV originales conservados. Windows requirió ejecución fuera del sandbox para iniciar Jupyter. Colab no probado.

- 2026-10-09: paso 3 completado. Dataset oficial conservado con hashes; auditoría en 00_Auditoria_Datos.ipynb y resultados/AUDITORIA.md. 3.150 filas, 495 abandonos, cero nulos, 300 repeticiones adicionales, 14 perfiles con etiquetas distintas y 7 cargos de nivel 10 fuera de la descripción. Copiadas dos referencias de clase sin modificarlas. Status/Customer Value siguen sin trazabilidad suficiente: propuesta de exclusión inicial, pendiente de acuerdo. No se dividieron datos ni entrenaron modelos. Colab no probado.

- 2026-10-09: paso 2 terminado. Python 3.11.9, entorno `.venv` propio y `uv sync --locked` completado. Verificados los 13 imports principales (incluidos scikit-learn, imbalanced-learn, SHAP, Gradio, ipykernel y nbconvert), misma lista de dependencias que el proyecto anterior y ausencia de CSV/notebooks. El nombre largo impedía cargar un componente de scikit-learn; se recreó el entorno en `churn/` y los imports pasaron. Sin descargas de datos, entrenamiento ni modificaciones del proyecto sísmico. Colab todavía no probado.

Siguiente tarea: si se pide, probar en Google Colab y hacer commit. No avanzar sin indicación del usuario.
