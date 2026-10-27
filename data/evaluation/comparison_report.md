# Informe de comparación — Recuperación vectorial vs. híbrida (CelsiaBot)

Fecha: 2026-10-27
Preguntas evaluadas: 20 (`data/evaluation/questions.json`)
Resultados generados con `k = 5` por pregunta.

## 1. Alcance y método

Se compararon dos estrategias de recuperación implementadas en `src/retrieval/retriever.py`:

- **Vector**: consulta directa a Chroma con `n_results = k`. Puntuación: similitud coseno = `1 - distance`.
- **Híbrida**: pool amplio (`k * CANDIDATE_MULTIPLIER = 15`, limitado por el total de chunks) → fusión por Reciprocal Rank Fusion (RRF, `RRF_K = 60`) con puntuación BM25 sobre el pool vectorial.

Los resultados analizados corresponden a las ejecuciones con el código corregido y se guardaron en:

- `data/evaluation/vector_results_final.jsonl`
- `data/evaluation/hybrid_results_final.jsonl`

Los archivos previos de evaluación (`data/evaluation/vector_results.jsonl` y `data/evaluation/hybrid_results.jsonl`) se mantuvieron sin modificar durante este trabajo.

## 2. Corrección aplicada a la puntuación híbrida

**Problema detectado:** la fusión por Reciprocal Rank Fusion calculaba el score fusionado (`fused_score`) y lo descartaba. Los resultados devueltos por la estrategia híbrida conservaban la puntuación vectorial original (`score = 1 - distance`), aunque el orden final se hubiera calculado con la fusión RRF.

**Corrección aplicada** en `src/retrieval/retriever.py`:

- `_reciprocal_rank_fusion` ahora devuelve los resultados con el score RRF real, usando `dataclasses.replace` sobre el `RetrievalResult` congelado: `replace(result, score=fused_score)`.
- La estrategia vectorial no pasa por fusión, por lo que conserva `score = 1 - distance`.

**Pruebas actualizadas/creadas** en `tests/test_retrieval.py`:

- `test_retrieve_returns_most_relevant_chunk_with_metadata` actualizado a `strategy="vector"` (la aserción de similitud `score > 0.99` aplica solo a similitud vectorial).
- `test_retriever_hybrid_returns_fused_rrf_score` (nuevo): verifica que la estrategia híbrida devuelve el score RRF calculado y que no es la similitud vectorial.
- `test_retriever_hybrid_returns_deterministic_ranking` (nuevo): verifica orden determinista e idéntico en ejecuciones consecutivas.
- Se mantiene la verificación de que la estrategia vectorial conserva `score = 1 - distance`.

## 3. Tabla comparativa — Pregunta, resultado principal vectorial vs. híbrido

| ID | Pregunta | Vector — top 1 (chunk / título / categoría / score) | Híbrida — top 1 (chunk / título / categoría / score RRF) | Conclusión |
|---|---|---|---|---|
| q01 | ¿Con qué frecuencia realiza Celsia mantenimientos de la red eléctrica? | `source_07-chunk-002` / Granjas solares / energia_solar / 0.4688 | `source_01-chunk-001` / Sitio web oficial de Celsia / corporativo / 0.0320 | **Híbrida mejor.** Vector devolvió un fragmento de granjas solares (off-topic). Híbrida recuperó contenido de mantenimiento/cortes. |
| q02 | ¿Cuáles son los horarios de los cortes de energía programados? | `source_01-chunk-001` / Sitio web oficial / corporativo / 0.4076 | `source_01-chunk-001` / Sitio web oficial / corporativo / 0.0320 | Ambas estrategias recuperaron el mismo fragmento; relevante en contexto (cortes por mantenimiento) pero sin horario concreto. |
| q03 | ¿Qué proyectos de mejora del factor de potencia desarrolla Celsia? | `source_01-chunk-002` / Sitio web oficial / corporativo / 0.6734 | `source_01-chunk-002` / Sitio web oficial / corporativo / 0.0328 | Ambas recuperaron un top-1 relevante (proyectos de mejora de factor de potencia). |
| q04 | ¿Cuántos clientes atiende Celsia actualmente? | `source_02-chunk-001` / Sobre nosotros / corporativo / 0.6247 | `source_02-chunk-001` / Sobre nosotros / corporativo / 0.0328 | Ambas recuperaron un top-1 relevante (1,3 millones de clientes). |
| q05 | ¿Cuántos colaboradores tiene Celsia? | `source_08-chunk-008` / Oportunidades laborales / empleo / 0.6580 | `source_08-chunk-008` / Oportunidades laborales / empleo / 0.0325 | Ambas recuperaron un top-1 parcialmente relevante (contenido de empleo que menciona colaboradores). |
| q06 | ¿En qué países opera Celsia? | `source_03-chunk-015` / Gobierno corporativo / gobierno_corporativo / 0.5174 | `source_08-chunk-006` / Oportunidades laborales / empleo / 0.0323 | **Híbrida mejor.** Vector top-1 off-topic (gobierno corporativo). Híbrida top-1 menciona países; top-2 de vector también era relevante. |
| q07 | ¿Cuál es la política de riesgos de Celsia? | `source_03-chunk-017` / Gobierno corporativo / gobierno_corporativo / 0.7829 | `source_03-chunk-017` / Gobierno corporativo / gobierno_corporativo / 0.0328 | Ambas recuperaron un top-1 relevante (política de riesgos). |
| q08 | ¿Qué política de ciberseguridad tiene Celsia? | `source_03-chunk-010` / Gobierno corporativo / gobierno_corporativo / 0.7095 | `source_03-chunk-010` / Gobierno corporativo / gobierno_corporativo / 0.0328 | Ambas recuperaron un top-1 relevante (política de ciberseguridad). |
| q09 | ¿Qué metas de sostenibilidad tiene Celsia? | `source_04-chunk-000` / Sostenibilidad / sostenibilidad / 0.8007 | `source_04-chunk-003` / Sostenibilidad / sostenibilidad / 0.0323 | Ambas recuperaron un top-1 relevante (sostenibilidad). |
| q10 | ¿Qué indicadores de sostenibilidad reporta Celsia? | `source_04-chunk-000` / Sostenibilidad / sostenibilidad / 0.7419 | `source_04-chunk-004` / Sostenibilidad / sostenibilidad / 0.0325 | Ambas recuperaron un top-1 relevante (indicadores / ReverdeC / CO2). |
| q11 | ¿Cuántas toneladas de CO2 ha evitado Celsia con sus proyectos renovables? | `source_07-chunk-000` / Granjas solares / energia_solar / 0.7042 | `source_07-chunk-000` / Granjas solares / energia_solar / 0.0320 | Ambas recuperaron un top-1 relevante (97.595 tCO2e evitadas). |
| q12 | ¿Cuántas granjas solares tiene Celsia en operación? | `source_07-chunk-005` / Granjas solares / energia_solar / 0.6437 | `source_07-chunk-000` / Granjas solares / energia_solar / 0.0325 | Ambas recuperaron un top-1 relevante (25 granjas / ~480 MWp). |
| q13 | ¿Cuál es la capacidad instalada de las granjas solares de Celsia? | `source_07-chunk-000` / Granjas solares / energia_solar / 0.6830 | `source_07-chunk-000` / Granjas solares / energia_solar / 0.0328 | Ambas recuperaron un top-1 relevante (capacidad instalada). |
| q14 | ¿Qué es la granja solar Celsia Solar Escobal? | `source_07-chunk-005` / Granjas solares / energia_solar / 0.5242 | `source_07-chunk-004` / Granjas solares / energia_solar / 0.0320 | **Recuperación dudosa en ambas.** Top-1 se refiere a otras granjas solares, no a Celsia Solar Escobal. |
| q15 | ¿Qué beneficios ofrece Celsia a sus colaboradores? | `source_08-chunk-008` / Oportunidades laborales / empleo / 0.5249 | `source_08-chunk-005` / Oportunidades laborales / empleo / 0.0315 | **Híbrida mejor.** Vector top-1 genérico de empleo; Híbrida top-1 específico de beneficios. |
| q16 | ¿Cómo puedo contactar a Celsia? | `source_08-chunk-008` / Oportunidades laborales / empleo / 0.4712 | `source_10-chunk-001` / Información para inversionistas / inversionistas / 0.0325 | **Recuperación incorrecta en ambas.** Ninguna recuperó fragmentos de contacto (`source_09-chunk-000` no aparece en el top-5). |
| q17 | ¿Cuál es la calificación crediticia de Celsia? | `source_10-chunk-000` / Información para inversionistas / inversionistas / 0.5610 | `source_10-chunk-000` / Información para inversionistas / inversionistas / 0.0311 | Ambas recuperaron un top-1 relevante (calificaciones BRC/Fitch). |
| q18 | ¿Cuándo fue la recompra de acciones de Celsia? | `source_08-chunk-008` / Oportunidades laborales / empleo / 0.4639 | `source_10-chunk-001` / Información para inversionistas / inversionistas / 0.0323 | **Híbrida mejor.** Vector top-1 off-topic (empleo); Híbrida top-1 relevante (recompra de acciones). |
| q19 | ¿Qué es el programa ReverdeC de Celsia? | `source_08-chunk-008` / Oportunidades laborales / empleo / 0.5223 | `source_08-chunk-008` / Oportunidades laborales / empleo / 0.0328 | Ambas recuperaron un top-1 con mención parcial de ReverbeC. |
| q20 | ¿Qué son los autogeneradores de Celsia? | `source_01-chunk-002` / Sitio web oficial / corporativo / 0.4763 | `source_01-chunk-002` / Sitio web oficial / corporativo / 0.0328 | Ambas recuperaron un top-1 relevante (conexión para autogeneradores). |

## 4. Casos de recuperación incorrecta o dudosa

- **Mantenimiento y cortes (q01, q02):** el vector devolvió fragmentos de granjas solares para la pregunta de frecuencia de mantenimientos (q01), mientras que la híbrida recuperó contenido de mantenimiento/cortes. La respuesta a "frecuencia" no está en los fragmentos revisados; el caso requiere validación humana sobre fuentes de mantenimiento.
- **Sostenibilidad (q09–q11):** ambas estrategias recuperaron fragmentos de sostenibilidad relevantes; el contenido de ReverdeC y CO2 apareció en los top-3.
- **Vacantes / colaboradores (q05, q06):** el top-1 se ubicó en la categoría de empleo. Para q06 (países) el contenido de países apareció en el top-2, lo que sugiere solapamiento semántico entre empleo y operación.
- **ReverbeC (q19) y autogeneradores (q20):** se recuperó contenido con mención parcial del tema; no constituye una respuesta completa.

Casos claramente fallidos en ambas estrategias: q14 (Escobal) y q16 (contacto).

## 5. Hit@1 y Hit@3

Se calculó Hit@1 y Hit@3 tras revisión manual de pertinencia de los fragmentos recuperados (texto + metadatos), sin usar la puntuación como criterio.

- **Vector:** Hit@1 = 0.75 (15/20); Hit@3 = 0.90 (18/20).
- **Híbrida:** Hit@1 = 0.90 (18/20); Hit@3 = 0.90 (18/20).

Estas cifras se interpretan con cautela: los criterios de relevancia fueron aplicados por revisión de contenido y los casos borderline (p. ej. q01, q05, q14, q16, q19) requieren validación humana.

## 6. Limitaciones

- Las puntuaciones de ambas estrategias no son directamente comparables: vector usa similitud (1 - distance) en escala 0–1; híbrida usa RRF (escala propia, valores cercanos a 0.03 para rankings altos). La comparación se hizo por ranking/id, no por valor numérico.
- BM25 se construyó solo sobre el pool vectorial (`CANDIDATE_MULTIPLIER = 3`), por lo que no explora toda la base de datos.
- Las preguntas q01 y q02 probablemente no tengan respuesta completa en los datos procesados.

## 7. Conclusión y recomendación provisional

Con la corrección aplicada, la estrategia híbrida ya devuelve el score RRF real de forma coherente con el ordenamiento resultante, y la estrategia vectorial conserva la similitud como puntuación.

En la comparación de este informe, la estrategia híbrida obtuvo un mejor resultado en el top-1 para las preguntas de mantenimiento/cortes (q01), países (q06), beneficios (q15) y recompra de acciones (q18), mientras que la estrategia vectorial tuvo mejores top-1 en preguntas de gobierno/sostenibilidad (q07–q11).

Recomendación provisional: mantener la estrategia híbrida como configuración por defecto (ya era la default), ya que tras la corrección sus resultados no distorsionan la interpretación de la puntuación. No se recomienda cambiar el modelo de embeddings ni reconstruir el índice en este momento. Se recomienda mejorar la recuperación para las preguntas cuyo contenido no se recuperó bien (p. ej. horarios de cortes, contacto, granja Escobal) y validar manualmente las preguntas cuya respuesta no está en los datos procesados.

## 8. Verificación final del estado del repositorio

- `uv run pytest -q`: 65 passed.
- `uv run ruff check src/retrieval src/generation tests`: All checks passed.
