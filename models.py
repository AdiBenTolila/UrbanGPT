
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.llms import Bedrock
from langchain_google_genai import ChatGoogleGenerativeAI
import os

assert "GOOGLE_API_KEY" in os.environ
# Define model names
he_en_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
en_he_model_name = "Helsinki-NLP/opus-mt-en-he"
# embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embedding_model_name = "sentence-transformers/LaBSE"
language_model_name_debug = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
language_model_name = "anthropic.claude-instant-v1"
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
def get_embedding_model(model_name = embedding_model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,                     # Provide the pre-trained model's path
        model_kwargs={'device':'cpu'},                  # Pass the model configuration options
        encode_kwargs={'normalize_embeddings': False}   # Pass the encoding options
    )

def get_llm(model_name = language_model_name, debug = False):
    if debug:
        return LlamaCpp(
            model_path=language_model_name_debug,
            temperature=0.0,
            max_tokens=1024,
            top_p=1,
            n_ctx=8192,  # Reduce the context size
            verbose=True,
        )
    # return Bedrock(model_id=model_name)
    return ChatGoogleGenerativeAI(model="gemini-pro")


# embeddings_he = HuggingFaceEmbeddings(
#     model_name=embedding_model_he,                     # Provide the pre-trained model's path
#     model_kwargs={'device':'cpu'},                  # Pass the model configuration options
#     encode_kwargs={'normalize_embeddings': False}   # Pass the encoding options
# )



if __name__ == '__main__':
    llm = get_llm()
    # stream llm query
    for chunk in llm.stream("תכתוב לנו שיר על המרצה אסף הגדול מכולם."):
        print(chunk, end="", flush=True)

