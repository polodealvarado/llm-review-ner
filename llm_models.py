from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())


def get_openai_llm(
    model: str = os.getenv("OPENAI_MODEL"),
    temperature: int = 0,
    max_tokens: int = 256,
    top_p=1,
    **kwargs,
):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        **kwargs,
    )


def get_vertexai_llm(
    model: str = os.getenv("VERTEXAI_MODEL"),
    temperature: int = 0,
    max_output_tokens: int = 8192,
    top_p: int = 1,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    return ChatVertexAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        safety_settings=safety_settings,
    )


def get_azure_llm(
    model: str = os.getenv("AZURE_OPENAI_MODEL"),
    temperature: int = 0,
    top_p: int = 1,
    **kwargs,
):

    return AzureChatOpenAI(
        model_name=model,
        temperature=temperature,
        top_p=top_p,
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        **kwargs,
    )


_GET_MODEL = {
    "vertexai": get_vertexai_llm,
    "openai": get_openai_llm,
    "azure": get_azure_llm,
}


def get_llm(model_name: str, **kwargs):
    model_fn = _GET_MODEL[model_name]
    return model_fn(**kwargs)
