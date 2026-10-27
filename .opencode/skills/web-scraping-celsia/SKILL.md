---
name: web-scraping-celsia
description: Extraer, validar y organizar información pública de las fuentes web de Celsia para alimentar la base de conocimiento de CelsiaBot.
compatibility: opencode
---

# Skill: Web Scraping para CelsiaBot

## Objetivo

Guiar al agente en la extracción responsable, verificable y organizada de información pública de Celsia para el proyecto académico CelsiaBot.

## Cuándo utilizar esta skill

Utilizar esta skill cuando sea necesario:
- Consultar las fuentes públicas registradas en el proyecto.
- Extraer el texto principal de páginas web.
- Verificar los resultados de una extracción.
- Identificar errores de conexión o páginas sin contenido útil.
- Preparar documentos para su posterior limpieza y procesamiento.

## Herramientas del proyecto

Antes de modificar archivos, revisar el código existente.

- `src/scraping/sources.py`: inventario de fuentes.
- `src/scraping/extractor.py`: lógica de extracción.
- `data/raw/`: destino de los documentos originales extraídos.
- `data/processed/`: destino previsto para los documentos limpios.
- `data/chunks/`: destino previsto para los fragmentos de texto.

No asumir que estos archivos están completos o que la extracción ya funciona. Verificar su contenido y los resultados reales.

## Procedimiento de trabajo

### 1. Revisar las fuentes

- Leer el inventario de fuentes existente.
- Priorizar páginas oficiales de Celsia.
- Comprobar que las URL sean accesibles.
- No inventar direcciones web ni asumir que una página sigue disponible.
- Conservar la URL de origen de cada documento.

### 2. Verificar las condiciones de acceso

- Consultar las reglas aplicables de `robots.txt`.
- Respetar las condiciones de uso del sitio web y las restricciones de acceso.
- Si no se pueden verificar las condiciones, detener la extracción de esa fuente y registrar el motivo.
- No intentar evadir bloqueos, autenticación ni controles de acceso.
- Evitar solicitudes excesivas e introducir pausas entre peticiones.

### 3. Extraer el contenido

- Utilizar primero las herramientas y dependencias existentes en el proyecto.
- Descargar únicamente contenido público cuya extracción esté permitida.
- Extraer el texto principal, evitando menús, navegación y elementos repetitivos cuando sea posible.
- Verificar el tipo de contenido de la respuesta.
- Manejar errores de conexión, códigos HTTP inesperados y páginas sin texto útil.

### 4. Guardar los resultados

- Guardar los documentos originales en `data/raw/`.
- Utilizar JSON con codificación UTF-8 cuando corresponda.
- Conservar título, URL de origen, categoría, fecha de extracción y texto.
- Registrar el estado de cada fuente en un manifiesto.
- No reemplazar datos existentes sin comprobar las consecuencias.

### 5. Validar la extracción

Comprobar:

- Cuántas fuentes se intentaron procesar.
- Cuántas se extrajeron correctamente.
- Cuántas fueron omitidas o produjeron errores.
- Si los documentos contienen texto útil.
- Si las URL y los títulos corresponden a las fuentes esperadas.
- Si el manifiesto coincide con los archivos generados.

No declarar una extracción exitosa sin comprobar sus resultados.

### 6. Preparar la siguiente etapa

Una vez validados los documentos originales, identificar los pasos necesarios para:

- Limpiar el texto.
- Eliminar duplicados y contenido irrelevante.
- Conservar los metadatos de procedencia.
- Dividir los documentos en fragmentos.
- Preparar los datos para la futura base de conocimiento.

No mezclar la extracción web con la generación de respuestas del modelo de lenguaje.

## Restricciones

- No inventar información ni resultados de extracción.
- No ignorar errores para aparentar que el proceso terminó correctamente.
- No recopilar información privada ni acceder a contenido restringido.
- No modificar dependencias o archivos ajenos a la tarea sin necesidad.
- Antes de ejecutar una extracción completa, revisar el código y explicar cualquier cambio importante.
- Si una página no se puede extraer de manera autorizada, registrar el problema y proponer una alternativa legítima, como utilizar un documento público descargable.

## Resultado esperado

Al terminar una ejecución, proporcionar un resumen con:

1. Fuentes procesadas.
2. Documentos guardados.
3. Fuentes omitidas y sus motivos.
4. Errores encontrados.
5. Ubicación de los archivos generados.
6. Recomendaciones para continuar con el procesamiento de los datos.
