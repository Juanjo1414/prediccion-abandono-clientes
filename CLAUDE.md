# CLAUDE.md

Las reglas de este proyecto están en `AGENTS.md`, que compartimos con Codex y Antigravity. No las dupliques aquí: si algo cambia, se cambia en `AGENTS.md`.

@AGENTS.md

## Solo para Claude Code

- Al iniciar una sesión, lee `ESTADO.md` y dime en una o dos líneas en qué va el proyecto antes de empezar.
- Para ejecutar notebooks usa siempre `uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb --ExecutePreprocessor.timeout=-1` (nunca `pip install` local; ver "Trabajo y estilo" en `AGENTS.md`).
- Cuando termines una tarea, actualiza `ESTADO.md` antes de darme el resumen.
