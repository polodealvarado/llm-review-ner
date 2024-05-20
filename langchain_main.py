import json
import argparse
from typing import List, Union
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from llm_models import get_llm
import sys
from tqdm import tqdm

logger.remove()
logger.add(sys.stderr, level="INFO")

load_dotenv(find_dotenv())

# System message prompt
system_message = """Eres una herramienta que revisa el etiquetado de entidades en documentos.
Recibes una entrada JSON con los siguientes campos:
- Text: El texto que ha sido etiquetado.
- Spans: Lista de diccionarios. Cada diccionario es una palabra etiquetada con su entidad.

Aquí está la lista de entidades:
- ORG_JUR: Organizaciones legales (Tribunales, juzgados, poder judicial, sala de lo contencioso ...).
- ORG_PRI: Organizaciones privadas o empresas.
- ORG_PUB: Organizaciones públicas (ayuntamientos, empresas públicas, Estado, administración ...).
- PER: Nombre de la persona.
- ART: Artículos legales.
- DAT: Fechas.
- EXP: Números de expediente. Aparece indicatado por "expediente" o "número de expediente" o "EXP".
- NUM_VOT: Número de voto.
- NUM_SENT: Número de sentencia.
- ADD: Direcciones.
- IBAN: Número de cuenta bancaria.
- PHO: Número de teléfono o fax.
- CUR: Dinero.
- CED: Número de cédula de identidad (letras o números).
- DNI: DNI.
- NIF: NIF o CIF,
- LOC: Ubicación para una Ciudad, País, Pueblo ... etc.

Pasos que debes seguir:
1. Haz la revisión del etiquetado:
* Si es otra entidad, puedes cambiarla.
* Si está mal etiquetada, bórrala.
2. Revisa si hay entidades no etiquetadas:
* Si encuentras entidades no etiquetadas etiquétalas al final del JSON.

Asegúrate de usar solo las etiquetas de entidades proporcionadas y mantener el orden (puede estar vacía si no hay entidades).
Devuelve el JSON con el campo "spans" con cada "word" y "label". 
No des explicaciones y no devuelvas nada más."""


# Spans data structure
class Span(BaseModel):
    word: str = Field(description="Palabra etiquetada del texto")
    label: str = Field(description="Entidad")


class Spans(BaseModel):
    spans: List[Union[Span, None]] = Field(description="Lista de spans", default=[])


# Inicializa el cliente de OpenAI a través de LangChain y el prompt
def load_llm(model: str):
    llm = get_llm(model)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_message), ("human", "{input_llm}")]
    )
    parser = JsonOutputParser(pydantic_object=Spans)
    chain = prompt | llm | parser
    return chain


# Función para cargar el archivo JSONL
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


# Función para guardar el archivo JSONL
def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Función para procesar las muestras
def process_samples(chain, samples):
    updated_samples = []

    for sample in tqdm(samples):
        text = sample.get("text", "")
        spans = sample.get("spans", [])

        # Preprocessing to get the text of each entity
        input_spans_llm = []
        for span in spans:
            # Chage "text" key by word to make it easier to the LLM
            if "text" in span:
                span["word"] = span["text"]
                del span["text"]
            else:
                start = span.get("start")
                end = span.get("end")
                if start is not None and end is not None and text is not None:
                    span["word"] = text[start:end]
            input_spans_llm.append({"word": span["word"], "label": span["label"]})
        input_llm = {"text": text, "spans": input_spans_llm}
        with get_openai_callback() as cb:
            input_llm_str = json.dumps(input_llm)
            try:
                spans_updated = chain.invoke({"input_llm": input_llm_str})
            except Exception as e:
                logger.error(f"Error processing spans: {e}")
                logger.error(f"LLM Output: {input_llm_str}")
                spans_updated = {"spans": []}
            logger.info(
                f"\n------\nInput LLM: {input_llm}\n------\nOutput LLM: {spans_updated}\n------\n{cb}"
            )

        if not "spans" in spans_updated:
            spans_updated = {"spans": spans_updated}

        updated_sample = {"text": text, "spans": spans_updated["spans"]}
        updated_samples.append(updated_sample)

    return updated_samples


def main(llm, input_file_path, output_file_path):
    # Load chain with a specific llm
    chain = load_llm(llm)

    # Carga las muestras del archivo JSONL
    samples = load_jsonl(input_file_path)

    # Procesa las muestras y actualiza los spans
    updated_samples = process_samples(chain, samples)

    # Guarda las muestras actualizadas en un nuevo archivo JSONL
    save_jsonl(updated_samples, output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procesar un archivo JSONL de Prodigy y actualizar los spans usando OpenAI."
    )
    parser.add_argument("input_file", help="Ruta del archivo JSONL de entrada")
    parser.add_argument(
        "--llm",
        default="vertexai",
        choices=["openai", "azure", "vertexai"],
        help="Choose one LLM",
    )
    parser.add_argument(
        "--output_file",
        default="reviewed_dataset.jsonl",
        help="Ruta del archivo JSONL de salida",
    )

    args = parser.parse_args()

    main(args.llm, args.input_file, args.output_file)
