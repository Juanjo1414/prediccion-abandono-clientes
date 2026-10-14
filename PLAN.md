# Plan por pasos

Una tarea por conversación. Consultar ESTADO.md para no repetir trabajo ni resultados. El alcance no incluye ejecutar todos los pasos automáticamente.

1. **Alcance:** anticipar abandono para priorizar contactos; etiqueta binaria y F1 principal. Acordado en chat.
2. **Carpeta y entorno:** documentación mínima, carpetas de salida, Python 3.11 y uv con dependencias existentes. Verificar imports. No descargar datos ni crear notebooks.
3. **Auditar dataset:** descargar original UCI 563, registrar fuente y checksum, revisar esquema, clases, nulos, duplicados y disponibilidad temporal. Aclarar Status y Customer Value. Copiar las referencias de clase necesarias antes de crear código de análisis.
4. **Reservar test:** usar train_test_split con test_size=0.20, stratify=y y random_state=42, conservando repeticiones y sin agrupar perfiles. Guardar train y test; no cambiar la reserva para mejorar resultados.
5. **Viabilidad (opcional, omitida por decisión del usuario):** se avanzó directamente al notebook 02. Las comparaciones iniciales se realizan allí; no se crea un notebook 01b.
6. **Notebook 01:** completar EDA y preparación reutilizando la partición y los resultados anteriores. Explorar train; no volver a elegir la reserva.
7. **Notebook 02:** comparar escaladores, PCA y selección dentro de la validación. No imponer un mínimo de rasgos por estética. Usar StratifiedKFold de 10 particiones sin grupos. La versión implementada usa regresión logística de referencia y selección ANOVA; documenta su cobertura simplificada (menos selectores que un flujo completo tipo SHAP/RFECV). Guarda una plantilla sin ajustar para el notebook 03.
8. **Notebook 03:** comparar 12 clasificadores y balanceos compatibles. Determinar el tratamiento correcto de categorías para sobremuestreo. Seleccionar tres candidatos con train.
9. **Notebook 04:** ajustar, calibrar si corresponde y elegir umbral con train. Usar CV estratificada de 10 particiones sin grupos; distinguir las métricas usadas para seleccionar del resultado final en test. Congelar decisiones, evaluar test al final y exportar.
10. **App:** Gradio con formulario y CSV; ranking de riesgo y métricas. No presentar puntuaciones sin calibrar como probabilidades confiables.
11. **Entregables:** informe y presentación a partir de resultados guardados, sin reentrenamientos innecesarios.

Referencias previstas: IA_ML_Example_Dif_Classifiers_CV para evaluación; ML_Normalización para escalado; PCA_Wine y ML_PCA para PCA; ML_FeatureSelection_comparacion_metodos_SHAP para selección. Aún no copiadas: se hará en el paso que las necesite.
