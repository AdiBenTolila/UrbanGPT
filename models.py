
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import LlamaCpp
from langchain_community.llms import Bedrock
from langchain_aws import ChatBedrock
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
import os
from langchain_core.language_models.llms import LLM
from langchain_core.language_models import BaseChatModel
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.messages import HumanMessage
# from langchain_cohere import ChatCohere
# from langchain_cohere.llms import Cohere
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import BitsAndBytesConfig
from langchain.prompts import PromptTemplate
import tiktoken

import torch
import time
import dotenv
import logging
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define model names
# embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embedding_model_name = "sentence-transformers/LaBSE"

class RateLimitLLM(LLM):
    def __init__(self, llm, rate_limit=60):
        self.llm = llm
        self.rate_limit = rate_limit
        self.last_request_time = None

    def _call(self, *args, **kwargs):
        if self.last_request_time is not None:
            elapsed_time = time.time() - self.last_request_time
            if elapsed_time < self.rate_limit:
                time.sleep(self.rate_limit - elapsed_time)
        self.last_request_time = time.time()
        return self.llm(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)
    
    @property
    def _llm_type(self):
        return self.llm._llm_type

class RateLimitChat(BaseChatModel):
    chat_model: BaseChatModel = None
    rate_limit: int = 60
    last_request_time: Optional[float] = None
    def __init__(self, model, rate_limit=60):
        super().__init__()
        self.chat_model = model
        self.rate_limit = rate_limit
        self.last_request_time = None
        
    def _generate(self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> ChatResult:
        if self.last_request_time is not None:
            elapsed_time = time.time() - self.last_request_time
            if elapsed_time < 60 / self.rate_limit:
                time.sleep((60 / self.rate_limit) - elapsed_time)

        self.last_request_time = time.time()
        return self.chat_model._generate(messages, stop, run_manager, **kwargs)
    
    def _stream(self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> Iterator[ChatGenerationChunk]:
        if self.last_request_time is not None:
            elapsed_time = time.time() - self.last_request_time
            print("sleeping for:", (60 / self.rate_limit) - elapsed_time)
            if elapsed_time < (60 / self.rate_limit):
                time.sleep((60 / self.rate_limit) - elapsed_time)
        self.last_request_time = time.time()
        return self.chat_model._stream(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.chat_model._llm_type
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return self.chat_model._identifying_params

def get_translation_pipe(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cacheModels")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="cacheModels")
    return pipeline("translation", model=model, tokenizer=tokenizer)


# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
def get_embedding_model(model_name=embedding_model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,                     # Provide the pre-trained model's path
        model_kwargs={'device':'cpu'},                  # Pass the model configuration options
        encode_kwargs={'normalize_embeddings': False}   # Pass the encoding options
    )

def get_gemini_llm(rate_limit=False):
    assert "GEMINI_API_KEY" in os.environ, "Please set the GEMINI_API_KEY environment variable"
    model = ChatGoogleGenerativeAI(model="gemini-pro", safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }, google_api_key=os.environ["GEMINI_API_KEY"])
    if rate_limit:
        return RateLimitChat(model, rate_limit=rate_limit)
    else:
        return model
    
def get_vertex_llm(rate_limit=False):
    assert "GOOGLE_API_KEY" in os.environ, "Please set the GOOGLE_API_KEY environment variable"
    model = ChatVertexAI(model="gemini-1.5-pro", temprature=0.0)
    if rate_limit:
        return RateLimitChat(model, rate_limit=rate_limit)
    else:
        return model

def get_claude_llm():
    # return Bedrock(model_id="anthropic.claude-instant-v1")
    return ChatBedrock(model_id="anthropic.claude-instant-v1")

def get_llamaCpp_llm(model_name, ):
    return LlamaCpp(
        model_path=model_name,
        temperature=0.0,
        max_tokens=1024,
        top_p=1,
        n_ctx=8192,  # Reduce the context size
        verbose=True,
    )
hf_model_map = {}
def get_huggingface_llm(model_name):
    if model_name in hf_model_map:
        return hf_model_map[model_name]
    
    logger.info(f"Loading HuggingFace model: {model_name}")
    assert os.environ.get("HUGGINGFACEHUB_API_TOKEN") is not None, "Please set the HUGGINGFACEHUB_API_TOKEN environment variable"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        # bnb_4bit_use_double_quant=True
        # load_in_8bit=True,
        # llm_int8_threshold = 6.
    )
    llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            device=None,
            pipeline_kwargs=dict(
                max_new_tokens=2048,
                repetition_penalty=1.03,
                return_full_text=False,
                token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
            ),
            model_kwargs={"quantization_config": quantization_config},
        )
    # chat_model = ChatHuggingFace(llm=llm)
    hf_model_map[model_name] = llm
    return llm

def get_huggingface_chat(model_name):
    llm = get_huggingface_llm(model_name)
    return ChatHuggingFace(llm=llm)

def get_openai_llm(model_name="gpt-4o-mini"):
    assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"
    return ChatOpenAI(model_name=model_name)

def get_mistral_llm():
    return get_llamaCpp_llm("dicta-il/dictalm2.0-instruct-GGUF")

# def get_commandR_llm(rate_limit=False):
#     assert "COHERE_API_KEY" in os.environ, "Please set the COHERE_API_KEY environment variable"
#     if rate_limit:
#         return RateLimitChat(ChatCohere(model="command-r-plus"), rate_limit=rate_limit)
#     return ChatCohere(model="command-r-plus")

model_map = {
    "gemini": get_gemini_llm,
    "vertex": get_vertex_llm,
    "claude": get_claude_llm,
    "openai-gpt-4o": lambda **kwargs: get_openai_llm("gpt-4o", **kwargs),
    "openai-gpt-4o-mini": lambda **kwargs: get_openai_llm("gpt-4o-mini", **kwargs),
    "hf-gemma-2-9b-it": lambda **kwargs: get_huggingface_chat("google/gemma-2-9b-it", **kwargs),
    "hf-gemma-2-2b-it": lambda **kwargs: get_huggingface_chat("google/gemma-2-2b-it", **kwargs),
    "hf-dictalm2.0-instruct": lambda **kwargs: get_huggingface_chat("dicta-il/dictalm2.0-instruct", **kwargs),
    "hf-Qwen2-7B-Instruct": lambda **kwargs: get_huggingface_chat("Qwen/Qwen2-7B-Instruct", **kwargs),
    "hf-Meta-Llama-3.1-8B-Instruct": lambda **kwargs: get_huggingface_chat("meta-llama/Meta-Llama-3.1-8B-Instruct", **kwargs),
}
def get_llm(name="openai-gpt-4o-mini", **kwargs):
    if name in model_map:
        return model_map[name](**kwargs)
    else:
        logger.error(f"Model {name} not found in model_map, using default model")
        return get_openai_llm("gpt-4o-mini")


def openai_count_tokens(string: str, encoding_name: str) -> int:
    """counts number of tokens in a string using the specified encoding from openai

    Args:
        string (str): string to tokenize
        encoding_name (str): encoding name, for gpt-4o use o200k_base and for any other model use cl100k_base

    Returns:
        int: number of tokens in the string
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == '__main__':
    llm = get_huggingface_chat("dicta-il/dictalm2.0-instruct")
    prompt_template = "תכתוב לנו שיר על המרצה {name} הגדול מכולם."
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    print(chain.invoke(dict(name="אסף")).content)
    
