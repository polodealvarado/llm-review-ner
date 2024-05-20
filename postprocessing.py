from typing import Dict


def get_postprocessed_sample(sample_text: str, output_spans_llms: Dict) -> Dict:
    text = sample_text
    llm_spans = output_spans_llms["spans"]
    updated_spans = []
    for llm_span in llm_spans:
        llm_span_text = llm_span["span"]
        llm_span_start = sample_text.find(llm_span_text)
        llm_span_end = sample_text.find(llm_span_text) + len(llm_span_text)
        updated_spans.append(
            {
                "span": sample_text[llm_span_start:llm_span_end],
                "label": llm_span["label"],
                "start": llm_span_start,
                "end": llm_span_end,
            }
        )
        text = text.replace(llm_span_text, "*" * len(llm_span_text))

    return {"text": sample_text, "spans": updated_spans}
