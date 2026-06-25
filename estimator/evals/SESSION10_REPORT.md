# Sesión 10 — Búsqueda híbrida + reranking: medición y conclusiones

Entregable de la pre-work de la Sesión 10. Cubre **solo** búsqueda híbrida y
reranking (sin expansión de consultas, routing multi-índice ni filtrado por
metadatos — eso se trabaja en la sesión en vivo).

## Cómo reproducir

Stack levantado, corpus base ingestado y `OPENAI_API_KEY` en `.env`:

```bash
cd estimator
# Gate: el cross-encoder descarga, carga y rankea bien
docker compose exec estimator python -m app.generation.rag.retrieval.verify_reranker
# Medición de las 4 configuraciones contra el golden set
uv run python scripts/eval_retrieval_s10.py
```

## Qué se mide

- **Golden set** ([evals/golden_retrieval.json](golden_retrieval.json)): 5 descripciones
  de proyecto representativas del dominio, cada una anotada **a mano** con los
  `budget_id` históricos realmente relevantes. Los distractores son presupuestos
  "parecidos pero de otro sector" a propósito (p. ej. pagos de marketplace
  e-commerce vs. un gateway de pagos fintech) — el fallo exacto que el ejercicio ataca.
- **Métrica**: `precision@5` = (relevantes en el top-5) / 5, media sobre las 5 queries.
  Un chunk cuenta como relevante si su `budget_id` está en `relevant_budget_ids`.
- **Latencia**: media de la consulta de retrieval/rerank (el embedding de la query
  se excluye: es una llamada a OpenAI compartida por las 4 configs). 1 warm-up
  descartado + 3 runs medidos por query.
- **Umbral de distancia permisivo** (2.0) para no truncar el top-5 con el suelo de
  relevancia: medimos calidad de **ranking**, no el soft-fail.

## Las 4 configuraciones

| Config | Búsqueda  | Reranking | **Precision@5** | **Latencia (ms)** |
| ------ | --------- | --------- | --------------- | ----------------- |
| A      | Vectorial | No        | 0.92            | 46                |
| B      | Híbrida   | No        | 0.92            | 51                |
| C      | Vectorial | Sí        | 0.92            | 2047              |
| D      | Híbrida   | Sí        | 0.92            | 2171              |

### Precision@5 por query

| Query | A    | B    | C    | D    |
| ----- | ---- | ---- | ---- | ---- |
| Q1    | 1.00 | 1.00 | 0.80 | 0.80 |
| Q2    | 1.00 | 1.00 | 1.00 | 1.00 |
| Q3    | 0.80 | 0.80 | 0.80 | 0.80 |
| Q4    | 0.80 | 0.80 | 1.00 | 1.00 |
| Q5    | 1.00 | 1.00 | 1.00 | 1.00 |

## Lectura de los resultados

1. **La rama léxica no contribuyó nada en este corpus.** Los logs muestran
   `lexical_hits=0` en todas las corridas híbridas: `plainto_tsquery` une los
   lexemas con AND, y las queries son frases largas en lenguaje natural, así que
   ningún chunk contiene *todos* los términos a la vez. Resultado: **B ≡ A** y
   **D ≡ C** (la fusión RRF de "vector + lista vacía" es el vector intacto). La
   columna híbrida solo añade ~5 ms de una query léxica que no devuelve nada.

2. **El reranking no movió la precisión media** (0.92 → 0.92), pero **redistribuyó**
   qué query gana o pierde: en Q4 ayudó (0.80 → 1.00) y en Q1 perjudicó
   (1.00 → 0.80). En este corpus el cross-encoder no aporta señal neta.

3. **El reranking cuesta ~44× más latencia** (46 ms → 2047 ms): puntúa 50 pares
   `(query, doc)` en CPU (~2 s). El baseline vectorial ya es muy fuerte (0.92)
   porque el corpus es pequeño (60 chunks) y los dominios están bien separados en
   el espacio de embeddings; queda poco margen para mejorar.

## Conclusión

Para **este** caso de uso usaría la **configuración A (vectorial, sin reranking)**.
Da la misma precisión@5 (0.92) que cualquier otra configuración medida, con la
latencia más baja (~46 ms, 44× más rápida que el reranking) y la menor complejidad
operativa. La ganancia de relevancia del reranking **no justifica su latencia aquí**:
sobre este golden set no produjo ninguna mejora neta de precisión —solo barajó qué
queries acierta— a cambio de multiplicar por ~44 el tiempo de consulta. Con un
baseline denso que ya separa bien los dominios, el coste del cross-encoder no se paga.

**Matices honestos.** (a) La medición es ruidosa: con 5 queries y `k=5`, cada
precision se mueve a saltos de 0.2, así que diferencias pequeñas no son
significativas. (b) La rama **híbrida está, en la práctica, sin probar**: la léxica
no devolvió candidatos, de modo que estos números no permiten concluir que "híbrido
es inútil en general", solo que *esta* implementación léxica (`plainto_tsquery` con
AND) sobre *este* estilo de query no aporta; una `websearch_to_tsquery` (semántica
OR) probablemente recuperaría más y haría medible la fusión. (c) En un corpus mayor,
más ruidoso o con más solapamiento entre dominios, el reranking sí podría justificar
su coste — la recomendación es específica de este dataset y esta escala.

## Nota sobre la configuración de text search (`english` vs `spanish`)

El enunciado dice que el dataset está en español, pero el dataset entregado
(`data/budgets_sample.json`, `data/task_corpus.json`) y el golden set están en
**inglés** ("Mobile banking application with OAuth2…"). Por eso la columna generada
y la query léxica usan `to_tsvector('english', …)` / `plainto_tsquery('english', …)`:
el stemming y las stop-words inglesas son los que de verdad ayudan a la recall léxica
sobre este texto. Cambiar a `spanish` solo requeriría tocar el regconfig en dos
sitios (la migración `0003_session10_fts` y `repository.search_lexical`), mantenidos
consistentes a propósito.
