# Definición del proyecto
Herramienta para revisar documentos ya etiquetados y obtener el feedback de los LLMs.

# ¿Cómo funciona?
1 - La herramienta recibe un batch de muestras.
2 - Carga la primera muestra.
3 - En base al contexto se le pregunta si es correcto la entidad asignada a la palabra.
4 - Capturar el output del LLM. Si el LLM cambia la entidad se le reasignará esa nueva entidad a la palabra.
5 - Se devuelve un nuevo archivo con las muestras correctas y corregidas.


# ¿Qué formatos vamos a manejar?
- Datasets: Dado que tenemos los datasets etiquetados con Prodigy comenzaremos con jsonl
- Prompts: yaml/jinja

# ¿Qué modelos vamos a usar?
- OpenAI
- Gemini
- Ollama

# Etapas
1 - Notebook con langchain llamando al LLM
2 - Probar con una muestra
3 - Cargar lote de muestras
4 - Llevar el código a un script

# Extras
Considerar la opción de implementar RAG cargando el documento de la muestra para darle más contexto al LLM y que pueda comprender mejor en caso de duda.


