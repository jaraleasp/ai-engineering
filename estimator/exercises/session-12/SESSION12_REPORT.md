# Sesión 12 — Agente de estimación (a mano, sin framework)

Entregable de la pre-work de la Sesión 12: un agente que recibe una transcripción,
la descompone en componentes, usa dos tools (`search_budgets`, `calculate_estimate`)
en un **bucle manual** (razona → actúa → observa → repite) sobre la **Responses API**
de OpenAI, y devuelve una estimación estructurada + una traza de su razonamiento.

## Cómo ejecutar

```bash
cd estimator
# Stub offline (sin BD — items históricos canónicos):
uv run python scripts/run_agent_s12.py exercises/session-12/sample_transcript_complex.txt --stub

# Pipeline real (necesita Docker + corpus ingestado):
uv run python scripts/run_agent_s12.py exercises/session-12/sample_transcript_complex.txt

# Modelo alternativo:
uv run python scripts/run_agent_s12.py exercises/session-12/sample_transcript_complex.txt --stub --model gpt-5-mini
```

Tests (sin API, todo con fakes): `uv run pytest tests/generation/agentic/ -v` → **10 passing**.

## Qué se construyó

| Pieza | Fichero |
| --- | --- |
| Schemas de las 2 tools (Responses API, formato plano, `strict: true`) | `app/generation/agentic/agent_schemas.py` |
| Tools: `calculate_estimate` (determinista) + `search_budgets` (envuelve el retrieval) | `app/generation/agentic/agent_tools.py` |
| Bucle manual + captura de traza + `render_trace` | `app/generation/agentic/agent_loop.py` |
| Wiring (retriever real inyectado) + config | `app/dependencies.py`, `app/config.py` |
| Runner CLI (`--stub`, `--model`) | `scripts/run_agent_s12.py` |
| Traza real de `gpt-5` sobre la transcripción compleja | `exercises/session-12/trace_complex_gpt5.txt` |

## Criterios de aceptación (con `gpt-5`, `sample_transcript_complex.txt`)

| Criterio | Resultado |
| --- | --- |
| Identifica >1 componente y hace >1 `search_budgets` | ✅ 4 componentes, **5 llamadas** a `search_budgets` |
| Llama a `calculate_estimate` | ✅ 1 llamada, `TOTAL: 3651.3h` |
| Termina por sí solo | ✅ `iterations=3, stopped=completed` |
| Estimación estructurada coherente | ✅ desglose por componente + total |
| Traza con razonamiento + acción + observación por paso | ✅ formato `STEP N` (ver `trace_complex_gpt5.txt`) |

**Comportamiento agéntico destacable** (STEP 5 de la traza): tras buscar SAP con
filtro `sectors=['logistics']` y **observar** un resultado flojo, el agente razonó
que no era satisfactorio y **re-buscó sin filtro de sector**, recuperando las
referencias correctas. Esa capacidad de decidir sobre la marcha —según lo que
devuelve cada observación— es justo lo que un pipeline fijo no tiene.

## Decisiones de diseño

- **Bucle a mano, no delegado.** Conducimos el ida-y-vuelta `function_call` →
  ejecutar → `function_call_output` (con el **mismo `call_id`**) →
  `previous_response_id`, para poder capturar cada paso. Salvaguarda
  `AGENT_MAX_ITERATIONS` además de la salida natural.
- **Responses API cruda** (`client.responses.create`, `reasoning={"effort":"medium",
  "summary":"auto"}`), no `LLMWrapper`: el resto del proyecto usa Instructor, pero
  aquí necesitamos ver y capturar las tool calls y los reasoning summaries.
- **Respeto de capas (el punto arquitectónico).** `agentic` no puede importar
  `generation/rag` (regla de `ARCHITECTURE.md` §3). Por eso `search_budgets` recibe
  el retrieval **inyectado** desde `dependencies.py` (el composition root): envuelve
  el pipeline híbrido + reranking de S9–S10 sin acoplar las capas. La misma costura
  permite el `--stub` sin cambiar el código del agente.
- **`calculate_estimate` determinista**: mediana de las referencias (robusta a un
  outlier) + 15% de contingencia; sin referencias → 0h marcado `unbudgeted` (nunca
  inventa un número).

## Nota sobre el corpus y el `--stub`

El corpus real ingestado es el de la S10 (presupuestos en **inglés**:
finance/ecommerce/healthcare/industrial…), sin proyectos de **logística**, que es el
dominio de `sample_transcript_complex.txt` (en español). Por eso la demo usa el
`--stub` (items canónicos de logística, la red de seguridad que provee el ejercicio):
el bucle del agente es idéntico en ambos modos gracias a la inyección. Con un corpus
que cubriera el dominio, `search_budgets` real luciría igual sin tocar el agente.
