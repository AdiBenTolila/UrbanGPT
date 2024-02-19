
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

# Define model names
he_en_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
en_he_model_name = "Helsinki-NLP/opus-mt-en-he"
# embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
language_model = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Create tokenizer and model instances
he_en_tokenizer = AutoTokenizer.from_pretrained(he_en_model_name,cache_dir="cacheModels")
he_en_model = AutoModelForSeq2SeqLM.from_pretrained(he_en_model_name, cache_dir="cacheModels")
# Create translation pipeline using the tokenizer and model
he_en_translate_pipe = pipeline("translation", model=he_en_model, tokenizer=he_en_tokenizer)

# Create tokenizer and model instances
en_he_tokenizer = AutoTokenizer.from_pretrained(en_he_model_name, cache_dir="cacheModels")
en_he_model = AutoModelForSeq2SeqLM.from_pretrained(en_he_model_name, cache_dir="cacheModels")
# Create translation pipeline using the tokenizer and model
en_he_translate_pipe = pipeline("translation", model=en_he_model, tokenizer=en_he_tokenizer)


# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,                     # Provide the pre-trained model's path
    model_kwargs={'device':'cpu'},                  # Pass the model configuration options
    encode_kwargs={'normalize_embeddings': False}   # Pass the encoding options
)


llm = LlamaCpp(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.0,
    max_tokens=1024,
    top_p=1,
    n_ctx=8192,  # Reduce the context size
    verbose=False,
)
