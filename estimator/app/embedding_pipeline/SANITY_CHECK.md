# Embedding Pipeline Sanity Check

Este archivo registra una verificacion minima end-to-end del script `scripts/compare.py` usando tres parejas de textos fijas.

## Resultados

### Pareja A - Textos semantica y tematicamente cercanos
- Texto 1: "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app"
- Texto 2: "Authorization service using JSON Web Tokens for a banking application"
- Cosine similarity: **0.5958**

### Pareja B - Textos no relacionados
- Texto 1: "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app"
- Texto 2: "Database migration from MySQL to PostgreSQL with zero downtime"
- Cosine similarity: **0.1920**

### Pareja C - Textos genericos y ambiguos
- Texto 1: "Backend services"
- Texto 2: "API development"
- Cosine similarity: **0.5407**

## Comentario breve
Los resultados son razonables en terminos generales: la Pareja A puntua mas alto que la Pareja B, y la B cae claramente en zona baja, como se esperaba. La Pareja A queda cerca del umbral orientativo de 0.6 pero no lo supera, lo que sugiere que aun siendo cercanas no son practicamente equivalentes en framing tecnico. La Pareja C da una similitud media-alta pese a ser corta y ambigua, algo esperable porque ambas frases comparten un espacio semantico muy general de desarrollo backend/API. Esto confirma que el pipeline funciona end-to-end y que discrimina bastante bien entre cercania y lejania, aunque los textos cortos tienden a agruparse mas de lo intuitivo.
