# Estado del proyecto

Actualizado: 2026-10-13.

- [x] Paso 1: alcance acordado.
- [x] Paso 2: estructura y entorno verificados.
- [x] Paso 3: descargar y auditar dataset; notebook ejecutado en local.
- [x] Paso 4: partición estratificada 80/20 sin grupos (2.520 filas de train y 630 de test).
- [x] Paso 6: notebook 01 completado y ejecutado en local; Colab pendiente.
- Paso 5: omitido por decisión del usuario; se avanzó directamente al notebook 02.
- [x] Paso 7: notebook 02 simplificado (regresión logística y ANOVA), ejecutado en local; Colab pendiente.
- [x] Paso 8: notebook 03 completado y ejecutado en local; Colab pendiente.
- [x] Paso 9: notebook 04 completado y ejecutado en local (versión actual: unos 2 minutos, según la bitácora); Colab pendiente.
- [x] Paso 9b (adicional, no numerado en PLAN.md): notebook 05 de controles de rigor, ejecutado en local; Colab pendiente.
- [x] Paso 10: app Gradio (`app.py`) creada y probada en local.
- [x] Paso 11: informe (`informe/informe.html` e `Informe_Final.pdf`) y presentación (`presentacion/presentacion.html`) terminados.

## Decisiones vigentes

Estas decisiones describen los notebooks 01–04 de la propuesta actual. El notebook 05 y los entregables anteriores corresponden a otra versión y no validan esta partición.

- Único proyecto del repositorio (desde 2026-10-13); el problema de peligro sísmico que ocupaba la raíz se descartó por acuerdo con la profesora y se eliminó del repositorio.
- Dataset: Iranian Churn, UCI 563; objetivo binario de abandono.
- Conservar repeticiones sin agrupar perfiles, autorizado por el usuario el 2026-10-13 para mantener la metodología de las referencias.
- Status y Customer Value excluidas inicialmente, según acuerdo en chat. Se conservan las once entradas restantes, incluidos edad y grupo de edad; su redundancia se revisará en el notebook 02.
- Se conservan las siete filas con cargo de nivel 10, sin corregirlas ni eliminarlas sin evidencia.
- Partición actual: train_test_split(test_size=0.20, stratify=y, random_state=42); 2.520 filas de entrenamiento y 630 de prueba. CV: StratifiedKFold de 10 particiones, shuffle=True, random_state=42. No se mantienen perfiles idénticos juntos. La revisión encontró 86 perfiles compartidos (160 filas de train y 102 de test).
- EDA y comparaciones de alternativas sobre entrenamiento; evaluación predictiva final en el notebook 04.
- Notebook 02: exporta StandardScaler + ANOVA k=10, F1 CV aproximadamente 0,603. Las conclusiones del notebook todavía describen la alternativa anterior de once variables.
- Alcance del 02: tres escalados, PCA 90/95 % y ANOVA k=1..10 con regresión logística balanceada (liblinear). No incluye los demás selectores ni SVM de la versión de minas. La plantilla exportada está sin ajustar; sus pasos se concatenan con los del clasificador, sin anidar Pipeline dentro de imblearn.
- Notebook 03: 12 clasificadores y 41 combinaciones con CV estratificada sin grupos. Pipeline corregido: escalador → balanceador (si aplica) → ANOVA → modelo. Candidatos actualizados: Stacking + SMOTENC (F1 0,8641), Random Forest + class_weight (0,8548) y MLP + SMOTENC (0,8399). Control de categorías en los diez pliegues: cero valores inválidos para quejas y plan_tarifario. Algunas ejecuciones de MLP llegaron a max_iter=1000 sin converger.
- Notebook 04: GridSearchCV por precisión promedio, calibración sigmoide y umbral con train, sin validación anidada. Tras corregir el orden del pipeline selecciona MLP + SMOTENC, capas (100, 50), alpha=0,001, k_neighbors=5 y umbral 0,41. F1 de selección 0,86165 (Random Forest 0,86154, diferencia mínima); F1 test 0,8614, recall 0,8788, precisión 0,8447 y ROC AUC 0,9860; 87 de 99 abandonos detectados. Pendientes fuera de esta corrección: calibrador final ajustado con puntajes del propio entrenamiento y nombre de métricas anidadas. Conclusiones actualizadas mediante tabla calculada.
- Las diferencias de F1 entre modelos no demuestran ausencia de fuga. La partición actual admite perfiles compartidos; sin fechas tampoco se puede descartar que alguna entrada refleje un abandono ya ocurrido.
- Cada paso requiere indicación del usuario. No crear notebooks o app durante la preparación del entorno.

## Bitácora

- 2026-10-13: completada la instalación con uv sync --locked (sin nuevas dependencias ni cambios en pyproject.toml/uv.lock). Ejecutados completos con nbconvert los notebooks 03 y 04 tras corregir el orden SMOTENC–ANOVA; 13 celdas de código en cada uno, cero errores de celda. Funciones armar_pipeline y crear_balanceador idénticas en ambos. Verificación de categorías guardada en resultados/03_control_categorias_smotenc.csv; cero valores inválidos en diez pliegues. Regenerados comparaciones, candidatos, figuras, modelo final y metadatos. Recarga del modelo con probabilidades y predicciones idénticas. Huellas de original, train y test sin cambios. La evaluación de test se repitió por esta corrección autorizada, sin usarla para elegir parámetros. Advertencias de convergencia de MLP registradas; no se modificaron sus límites. Colab no probado.

- 2026-10-13: comparación de la referencia de clasificadores con `clasificacion.ipynb` de Fundamentos de analítica, solicitada por el usuario. Este último propone buscar hiperparámetros para nueve familias (solo cuatro búsquedas tienen salidas guardadas), pero imputa incluyendo la etiqueta y balancea/transforma antes de separar datos; además, selecciona con el conjunto llamado test y reconstruye otro pipeline para despliegue. Recomendación: conservar los cuatro notebooks y pipelines, incorporar búsqueda acotada por familia antes de descartar modelos y no imponer a todos la selección elegida con regresión logística. Solo análisis; sin modificar notebooks ni entrenar modelos.

- 2026-10-13: matizadas las afirmaciones absolutas sobre ausencia de fuga en el markdown de los notebooks 01–03. Ahora explican que test se reserva para evaluación final, que las transformaciones y el remuestreo aprenden dentro de cada pliegue y que pueden existir perfiles repetidos entre conjuntos porque no se usan grupos. Verificado que código y salidas no cambiaron y que los archivos siguen siendo JSON válido. Se intentó ejecutar nbconvert con uv, pero el entorno local carece de dependencias en caché (llvmlite) y la ejecución offline no pudo iniciarse; no se afirma reejecución.

Las entradas anteriores documentan decisiones históricas; las menciones a agrupación de perfiles y sus resultados no describen la propuesta vigente.

- 2026-10-13: por petición del usuario, actualizadas las reglas, el plan y la documentación para describir la partición estratificada 80/20 y CV de 10 pliegues sin grupos. Se retira la exigencia de mantener perfiles idénticos juntos. Sin cambios en datos, entrenamiento ni notebooks; resultados y controles de versiones anteriores identificados como históricos.

- 2026-10-13: refactorización de notebooks 01 a 04 para alinearlos al estilo estándar de clase (referencia/):
  - Notebook 01: eliminación de assert, hashlib y StratifiedGroupKFold por grupos idénticos; partición 80/20 estándar con train_test_split estratificado.
  - Notebook 02: eliminación de comprobaciones rígidas y partición de grupos; StratifiedKFold de 10 particiones estándar, ganando StandardScaler y ANOVA k=10 (F1 0,603).
  - Notebook 03: evaluación limpia de 41 combinaciones (12 modelos × 4 estrategias) en 10 pliegues estratificados dentro de Pipeline; candidatos: Random Forest (class_weight, F1 0,855), Stacking (SMOTENC, F1 0,850) y Red Neuronal MLP (sin balanceo, F1 0,840).
  - Notebook 04: GridSearchCV con 10 pliegues y scoring average_precision, calibración sigmoide y umbral óptimo de 0,43. Ganador: Red Neuronal MLP sin balanceo con F1 en test de 0,8738 (Recall 0,9091, Precisión 0,8411, ROC AUC 0,9905). Tiempo de ejecución reducido de 26 a ~2 minutos. Compatible con app.py.

- 2026-10-13: tras hablar con la profesora, el equipo se queda solo con este problema (abandono de clientes). Se eliminó del repositorio todo lo del proyecto de peligro sísmico (notebooks, `data/`, `models/`, `resultados/`, `figuras/`, `informe/`, `presentacion/`, `app.py`, `referencia/` y sus `AGENTS.md`/`PLAN.md`/`ESTADO.md`/`pyproject.toml`/`uv.lock`/`.gitignore`/`.python-version`, además de `graphify-out/`, que describía ambos proyectos) y se movió todo el contenido de `churn/` a la raíz del repositorio, que ahora es el único proyecto. Se actualizaron `AGENTS.md`, `CLAUDE.md`, `ESTADO.md` y `README.md` para quitar las referencias al proyecto padre. Los nombres de archivo (notebooks con sufijo `_Churn`, por ejemplo) se dejaron igual para no romper referencias internas. Sin commit todavía, a la espera de que el usuario lo revise.

- 2026-10-10: terminados los pasos 10 y 11 a petición del usuario, después de revisar que los cinco notebooks corrieran sin fallos. Notebook 00 tenía un error al reejecutarse (comprobaciones que exigían que `train.csv`/`test.csv` no existieran): se quitaron esas comprobaciones y se reejecutaron 00, 01 y 02 completos sin errores. Se creó `05_Controles_Rigor_Churn.ipynb` (integridad de archivos y de la partición por perfiles, etiquetas mezcladas al azar, agrupación más estricta de perfiles parecidos y revisión de vecinos cercanos): sin evidencia de fuga mecánica (ROC AUC con etiquetas mezcladas 0,492), pero con grupos más estrictos el F1 baja de 0,859 a 0,811, por lo que el F1 final se reporta como rango (0,81–0,86) y no como cifra única. Se creó `app.py` (formulario, priorización por CSV y pestaña del modelo) y se probó en local contra los 639 clientes de prueba (probabilidades idénticas a las del notebook 04). Se armaron `presentacion/presentacion.html` (14 diapositivas interactivas, mismo motor que la del proyecto de sismos con paleta y datos propios) y `informe/informe.html` con su `Informe_Final.pdf` (24 páginas), siguiendo la misma estructura que el informe del proyecto de sismos. Colab sigue sin probarse. Sin commits.

- 2026-10-10: creados y ejecutados 03_Balanceo_Comparacion_Modelos_Churn.ipynb y 04_Ajuste_Evaluacion_Exportacion_Churn.ipynb a petición del usuario. Generan `comparacion_modelos.csv`, `03_candidatos.csv`, controles de sobremuestreo y de fuga, `ajuste_hiperparametros.csv`, `04_priorizacion_test.csv`, `models/modelo_final.joblib` (diccionario con pipeline, calibrador, umbral y columnas) y `models/metadata_modelo.json`, más figuras `03_*` y `04_*`. Verificada la recarga del modelo (probabilidades y predicciones idénticas). El 04 tardó unos 26 minutos, por encima de los 20 de referencia de las reglas; no se consultó antes. Las conclusiones del 04 se rellenaron sobre el notebook ya ejecutado, sin repetir la ejecución. Colab no probado. Sin commits.

- 2026-10-09: revisión completa de 02_Escalado_PCA_Seleccion_Churn.ipynb. Completadas conclusiones e interpretaciones, guardadas asignaciones de diez particiones y comparación ANOVA, documentados alcance simplificado y sesgo de selección. Añadidas huellas de entradas y parámetros del modelo a decisiones_preprocesamiento.joblib; comprobadas plantilla sin ajustar e integración plana con imblearn. Ejecución completa sin errores de celda en local. StandardScaler gana con 0,5972; PCA 90 % 0,5733, PCA 95 % 0,5716, mejor ANOVA k=10 0,5729. No se consultó prueba. Colab pendiente.

- 2026-10-09: revisado 01_EDA_Limpieza_Churn.ipynb a petición del usuario. Completadas explicaciones y conclusiones con resultados, verificación del original, cobertura de la partición y coherencia de CSV existentes. Documentada correspondencia edad/grupo de edad y guardadas tablas del EDA. Ejecución completa con nbconvert en local; 392 abandonos en entrenamiento (15,61 %), phik de quejas 0,747, segundos de uso 0,464 y frecuencia de uso 0,435. Reserva y CSV originales conservados. Windows requirió ejecución fuera del sandbox para iniciar Jupyter. Colab no probado.

- 2026-10-09: paso 3 completado. Dataset oficial conservado con hashes; auditoría en 00_Auditoria_Datos.ipynb y resultados/AUDITORIA.md. 3.150 filas, 495 abandonos, cero nulos, 300 repeticiones adicionales, 14 perfiles con etiquetas distintas y 7 cargos de nivel 10 fuera de la descripción. Copiadas dos referencias de clase sin modificarlas. Status/Customer Value siguen sin trazabilidad suficiente: propuesta de exclusión inicial, pendiente de acuerdo. No se dividieron datos ni entrenaron modelos. Colab no probado.

- 2026-10-09: paso 2 terminado. Python 3.11.9, entorno `.venv` propio y `uv sync --locked` completado. Verificados los 13 imports principales (incluidos scikit-learn, imbalanced-learn, SHAP, Gradio, ipykernel y nbconvert), misma lista de dependencias que el proyecto anterior y ausencia de CSV/notebooks. El nombre largo impedía cargar un componente de scikit-learn; se recreó el entorno en `churn/` y los imports pasaron. Sin descargas de datos, entrenamiento ni modificaciones del proyecto sísmico. Colab todavía no probado.

Siguiente tarea: si se pide, probar en Google Colab y hacer commit. No avanzar sin indicación del usuario.
