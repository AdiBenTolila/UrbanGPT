
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.llms import Bedrock
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
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
from langchain_cohere import ChatCohere
import torch
import time
import dotenv
dotenv.load_dotenv(".environment")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define model names
he_en_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
en_he_model_name = "Helsinki-NLP/opus-mt-en-he"
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

    

# # Create tokenizer and model instances
# he_en_tokenizer = AutoTokenizer.from_pretrained(he_en_model_name,cache_dir="cacheModels")
# he_en_model = AutoModelForSeq2SeqLM.from_pretrained(he_en_model_name, cache_dir="cacheModels")
# # Create translation pipeline using the tokenizer and model
# he_en_translate_pipe = pipeline("translation", model=he_en_model, tokenizer=he_en_tokenizer)

# # Create tokenizer and model instances
# en_he_tokenizer = AutoTokenizer.from_pretrained(en_he_model_name, cache_dir="cacheModels")
# en_he_model = AutoModelForSeq2SeqLM.from_pretrained(en_he_model_name, cache_dir="cacheModels")
# # Create translation pipeline using the tokenizer and model
# en_he_translate_pipe = pipeline("translation", model=en_he_model, tokenizer=en_he_tokenizer)

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
    assert "GOOGLE_API_KEY" in os.environ, "Please set the GOOGLE_API_KEY environment variable"
    model = ChatGoogleGenerativeAI(model="gemini-pro", safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        })
    if rate_limit:
        return RateLimitChat(model, rate_limit=rate_limit)
    else:
        return model

def get_claude_llm():
    return Bedrock(model_id="anthropic.claude-instant-v1")

def get_mistral_llm():
    return LlamaCpp(
        model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.0,
        max_tokens=1024,
        top_p=1,
        n_ctx=8192,  # Reduce the context size
        verbose=True,
    )

def get_commandR_llm(rate_limit=False):
    assert "COHERE_API_KEY" in os.environ, "Please set the COHERE_API_KEY environment variable"
    if rate_limit:
        return RateLimitChat(ChatCohere(model="command-r-plus"), rate_limit=rate_limit)
    return ChatCohere(model="command-r-plus")

def get_llm():
    return get_gemini_llm()

if __name__ == '__main__':
    llm = get_llm()
    # stream llm query
    for chunk in llm.stream("תכתוב לנו שיר על המרצה אסף הגדול מכולם."):
        print(chunk.content, end="", flush=True)
    print()
