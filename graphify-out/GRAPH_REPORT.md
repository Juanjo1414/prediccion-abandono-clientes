# Graph Report - Proyecto ML - Prediccion Sismica Minas  (2026-10-10)

## Corpus Check
- 21 files · ~166,288 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 80 file(s) not represented in the graph (top: .csv 41, .ipynb 19, (none) 11)

## Summary
- 204 nodes · 315 edges · 15 communities (11 shown, 4 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fe7aaf34`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- AGENTS.md — Predicción de peligro sísmico en minas de carbón
- PLAN.md (plan de implementación)
- construir_interfaz
- formato_numero
- evaluar_turno
- construir_interfaz
- Auditoría del dataset — paso 3
- Reglas del proyecto de abandono
- Estado del proyecto
- Predicción de abandono de clientes
- PLAN.md
- prediccion-abandono-clientes
- prediccion-sismica-minas
- churn/app.py

## God Nodes (most connected - your core abstractions)
1. `PLAN.md (plan de implementación)` - 28 edges
2. `construir_interfaz()` - 12 edges
3. `construir_interfaz()` - 12 edges
4. `AGENTS.md — Predicción de peligro sísmico en minas de carbón` - 11 edges
5. `02_PCA_Seleccion_Rasgos.ipynb` - 11 edges
6. `01_EDA_Preprocesamiento.ipynb` - 10 edges
7. `03_Modelos_Evaluacion.ipynb` - 10 edges
8. `formato_numero()` - 9 edges
9. `texto_resultado()` - 8 edges
10. `app.py (app Gradio)` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Bitácora` --references--> `evaluar_turno()`  [INFERRED]
  ESTADO.md → app.py
- `6.2 Una responsabilidad por función` --references--> `codificar_categoricas()`  [INFERRED]
  AGENTS.md → app.py
- `Librerías del proyecto (gestionadas con uv)` --semantically_similar_to--> `Stack tecnológico (README)`  [INFERRED] [semantically similar]
  PLAN.md → README.md
- `Limitaciones conocidas (README)` --semantically_similar_to--> `Riesgo de validación cruzada demasiado optimista`  [INFERRED] [semantically similar]
  README.md → PLAN.md
- `PLAN.md (plan de implementación)` --references--> `Forma de trabajar (CLAUDE.md)`  [EXTRACTED]
  PLAN.md → CLAUDE.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Notebooks de clase usados como guía de estilo para los tres notebooks del proyecto** — plan_ia_ml_example_dif_classifiers_cv, plan_ml_normalizacion, plan_pca_wine, plan_ml_featureselection_shap [EXTRACTED 0.85]
- **Artefactos .joblib persistidos que alimentan la app** — plan_label_encoder_joblib, plan_rasgos_usados_joblib, plan_modelo_final_joblib [EXTRACTED 0.90]
- **Flujo secuencial del pipeline: EDA -> PCA/selección -> modelos -> app** — plan_notebook_01, plan_notebook_02, plan_notebook_03, plan_app_py [EXTRACTED 0.90]

## Communities (15 total, 4 thin omitted)

### Community 0 - "AGENTS.md — Predicción de peligro sísmico en minas de carbón"
Cohesion: 0.09
Nodes (18): 10. Definición de terminado (por notebook), 1. El proyecto en pocas palabras, 2. Archivos clave, 3. Flujo de notebooks y artefactos, 4. Decisiones ya tomadas (no cambiarlas sin preguntar), 5. Reglas de datos (para que los resultados sean válidos), 6.1 Prima el estilo de la profesora, 6.2 Una responsabilidad por función (+10 more)

### Community 1 - "PLAN.md (plan de implementación)"
Cohesion: 0.14
Nodes (25): CLAUDE.md (instrucciones del proyecto), app.py (app Gradio), Cronograma de 12 días, Dataset UJIIndoorLoc (UCI id 310), Celda de configuración EN_COLAB / RUTA_BASE, Función eval_cv (StratifiedKFold + cross_validate), IA_ML_Example_Dif_Classifiers_CV.ipynb (notebook de referencia), label_encoder.joblib (+17 more)

### Community 2 - "construir_interfaz"
Cohesion: 0.10
Nodes (11): cargar_turno_al_azar(), cargar_turno_intenso(), cargar_turno_limite(), cargar_turno_tranquilo(), construir_interfaz(), crear_campo(), encabezado_html(), resultado_vacio() (+3 more)

### Community 3 - "formato_numero"
Cohesion: 0.11
Nodes (11): anillo_svg(), aviso_fuera_de_rango(), encabezado_html(), formato_numero(), html_factores(), nivel_de_riesgo(), preparar_tabla_clientes(), priorizar_csv() (+3 more)

### Community 4 - "evaluar_turno"
Cohesion: 0.10
Nodes (15): evaluar_turno(), formato_numero(), resumen_modelo(), svg_medidor(), texto_veredicto(), Avance, Bitácora, Decisiones tomadas (+7 more)

### Community 5 - "construir_interfaz"
Cohesion: 0.12
Nodes (10): cargar_cliente_al_azar(), cargar_cliente_en_riesgo(), cargar_cliente_limite(), cargar_cliente_tranquilo(), construir_interfaz(), crear_campo(), resultado_vacio(), resumen_modelo() (+2 more)

### Community 6 - "Auditoría del dataset — paso 3"
Cohesion: 0.29
Nodes (6): Artefactos, Auditoría del dataset — paso 3, Disponibilidad de variables y filtración, Límites de esta revisión, Procedencia, Resultados comprobados

### Community 7 - "Reglas del proyecto de abandono"
Cohesion: 0.50
Nodes (3): Alcance acordado, Reglas del proyecto de abandono, Trabajo y estilo

### Community 8 - "Estado del proyecto"
Cohesion: 0.50
Nodes (3): Bitácora, Decisiones vigentes, Estado del proyecto

### Community 9 - "Predicción de abandono de clientes"
Cohesion: 0.50
Nodes (4): Carpetas, Entorno local, Estado, Predicción de abandono de clientes

### Community 14 - "churn/app.py"
Cohesion: 0.07
Nodes (12): cargar_artefactos(), crear_tema(), abs_segundo(), calcular_referencias(), cargar_artefactos(), convertir_valor(), crear_tema(), evaluar_cliente() (+4 more)

## Knowledge Gaps
- **34 isolated node(s):** `prediccion-abandono-clientes`, `prediccion-sismica-minas`, `1. El proyecto en pocas palabras`, `2. Archivos clave`, `3. Flujo de notebooks y artefactos` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 97 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `codificar_categoricas()` connect `AGENTS.md — Predicción de peligro sísmico en minas de carbón` to `churn/app.py`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `construir_interfaz()` (e.g. with `cargar_turno_al_azar()` and `cargar_turno_intenso()`) actually correct?**
  _`construir_interfaz()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `construir_interfaz()` (e.g. with `cargar_cliente_al_azar()` and `cargar_cliente_en_riesgo()`) actually correct?**
  _`construir_interfaz()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `prediccion-abandono-clientes`, `prediccion-sismica-minas`, `1. El proyecto en pocas palabras` to the rest of the system?**
  _34 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AGENTS.md — Predicción de peligro sísmico en minas de carbón` be split into smaller, more focused modules?**
  _Cohesion score 0.08695652173913043 - nodes in this community are weakly interconnected._
- **Should `PLAN.md (plan de implementación)` be split into smaller, more focused modules?**
  _Cohesion score 0.13949579831932774 - nodes in this community are weakly interconnected._
- **Should `construir_interfaz` be split into smaller, more focused modules?**
  _Cohesion score 0.1038961038961039 - nodes in this community are weakly interconnected._