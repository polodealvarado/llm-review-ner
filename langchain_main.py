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
from postprocessing import get_postprocessed_sample
from tqdm import tqdm

logger.remove()
logger.add(sys.stderr, level="INFO")

load_dotenv(find_dotenv())

# System message prompt
system_message = """Eres una herramienta de revisión de etiquetado de entidades en documentos. Recibirás una entrada en formato JSON con los siguientes campos:
- Text: El texto etiquetado.
- Spans: Listado de spans. Cada uno contiene un span etiquetado con su entidad correspondiente.
Lista de entidades:
- ORG_JUR: Organizaciones legales (Tribunales, juzgados, poder judicial, sala de lo contencioso, etc.).
- ORG_PRI: Organizaciones o empresas privadas.
- ORG_PUB: Organizaciones públicas (ayuntamientos, empresas públicas, Estado, administración, etc.).
- PER: Nombre de la persona.
- ART: Artículos legales.
- DAT: Fechas.
- EXP: Números de expediente. Indicados por "expediente", "número de expediente" o "EXP".
- NUM_VOT: Número de voto.
- NUM_SENT: Número de sentencia.
- ADD: Direcciones de calle, avenida ... etc. No incluir correos electrónicos.
- IBAN: Número de cuenta bancaria.
- PHO: Número de teléfono o fax.
- CUR: Cantidades de dinero.
- CED: Número de cédula de identidad (letras o números).
- DNI: DNI.
- NIF: NIF o CIF.
- LOC: Ubicación (Ciudad, País, Pueblo, etc.).
Importante:
* Usar solo las etiquetas de entidades proporcionadas y de mantener el orden.
* Nunca devuelvas el campo 'text'.
Instrucciones:
1. Revisa el etiquetado:
   * Si un span está etiquetado con la entidad incorrecta, corrígela.
   * Si un span está mal etiquetado, elimina la etiqueta.
2. Busca entidades no etiquetadas:
   * Si encuentras spans no etiquetados añádelos al final de la lista de spans.
Devolver solo el JSON solo con el campo "spans", donde cada entrada contiene "span" y "label"."""


# Spans data structure
class Span(BaseModel):
    span: str = Field(description="Palabra etiquetada del texto")
    label: str = Field(description="Entidad")


class Spans(BaseModel):
    spans: List[Union[Span, None]] = Field(description="Lista de spans", default=[])


# Inicializa el cliente de OpenAI a través de LangChain y el prompt
def load_llm(model: str):
    llm = get_llm(model)
    final_prompt = ChatPromptTemplate.from_messages(
        [("system", system_message), ("human", "{input_llm}")]
    )
    parser = JsonOutputParser(pydantic_object=Spans)
    chain = final_prompt | llm | parser
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
            # Chage "text" key by "span"
            if "text" in span:
                span["span"] = span["text"]
                del span["text"]
            else:
                start = span.get("start")
                end = span.get("end")
                if start is not None and end is not None and text is not None:
                    span["span"] = text[start:end]

            input_spans_llm.append({"span": span["span"], "label": span["label"]})

        input_llm = {"text": text, "spans": input_spans_llm}
        with get_openai_callback() as cb:
            input_llm_str = json.dumps(input_llm)
            try:
                output_spans_llm = chain.invoke({"input_llm": input_llm_str})
            except Exception as e:
                logger.error(f"Error processing: {e}")
                logger.error(f"LLM Output: {output_spans_llm}")
                output_spans_llm = {
                    "spans": (
                        output_spans_llm["spans"] if "spans" in output_spans_llm else []
                    )
                }
            logger.info(
                f"\n------\nInput LLM: {input_llm}\n------\nOutput LLM: {output_spans_llm}\n------\n{cb}"
            )

        if not "spans" in output_spans_llm:
            output_spans_llm = {"spans": output_spans_llm}

        # Post-processing
        updated_sample = get_postprocessed_sample(text, output_spans_llm)
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
        description="Procesar un archivo JSONL de Prodigy y actualizar los spans usando LLMs."
    )
    parser.add_argument("input_file", help="Ruta del archivo JSONL de entrada")
    parser.add_argument(
        "--llm",
        default="azure",
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
