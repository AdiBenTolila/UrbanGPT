import pandas as pd
from download_program_doc import extract_xplan_number,program_doc_url
from urllib.parse import unquote
import os
import requests
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
import textract
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tqdm

# Define model names
he_en_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
en_he_model_name = "Helsinki-NLP/opus-mt-en-he"
# embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
language_model = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Create tokenizer and model instances
he_en_tokenizer = AutoTokenizer.from_pretrained(he_en_model_name)
he_en_model = AutoModelForSeq2SeqLM.from_pretrained(he_en_model_name)
# Create translation pipeline using the tokenizer and model
he_en_translate_pipe = pipeline("translation", model=he_en_model, tokenizer=he_en_tokenizer)

# Create tokenizer and model instances
en_he_tokenizer = AutoTokenizer.from_pretrained(en_he_model_name)
en_he_model = AutoModelForSeq2SeqLM.from_pretrained(en_he_model_name)
# Create translation pipeline using the tokenizer and model
en_he_translate_pipe = pipeline("translation", model=en_he_model, tokenizer=en_he_tokenizer)


# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,                     # Provide the pre-trained model's path
    model_kwargs={'device':'cpu'},                  # Pass the model configuration options
    encode_kwargs={'normalize_embeddings': False}   # Pass the encoding options
)

# Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


llm = LlamaCpp(
    model_path=language_model,
    temperature=0.0,
    max_tokens=1024,
    top_p=1,
    # callback_manager=callback_manager, 
    n_ctx=8192,  # Reduce the context size
    verbose=False,
)


def get_docs_for_plan(plan_number):
    xplan_number = extract_xplan_number(plan_number)
    doc_type = "rsPlanDocs"
    generated_urls = program_doc_url(xplan_number, doc_type , plan_number)
    try:
        os.makedirs(f"out/{plan_number}")
    except FileExistsError:
        pass
    for generated_url in generated_urls:
        url = unquote(generated_url)
        file_name = url.split('fn=')[-1].split('&')[0]
        if "הוראות התכנית" in file_name:
            response = requests.get(generated_url)
            destination = f"out/{plan_number}/{file_name}"
            with open(destination, 'wb') as file:
                file.write(response.content)
            yield destination

def pdf_to_text(filename):
    pdffileobj = open(filename, 'rb')
    pdfReader = PdfReader(pdffileobj)
    text = ""
    for page in pdfReader.pages:
        text += page.extract_text()
    if text == "":
        text = textract.process(filename, method='tesseract', language='eng')
    return text

def clean_and_split(text):
    # Define the pattern for the unwanted text
    unwanted_pattern = r'תכנון זמין\s+[0-9]\s+מונה\s+הדפסה'

    # Use re.split to split the text based on the unwanted pattern
    chunks = re.split(unwanted_pattern, text)

    # Remove empty lines in each chunk and filter out empty chunks
    chunks = [Document(page_content=re.sub(r'\n\s*\n', '\n', chunk).strip(),metadata={"source": "local"}) for chunk in chunks if chunk.strip()]

    return chunks

def load_and_split(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    return pages


def sentencewise_translate(text,translate_pipe):
    sentences = [t for t in text.split('.') if t]

    translates = translate_pipe(sentences)
    sentence_translations = [t['translation_text'] for t in translates]

    # Join translations for each original chunk using a dot
    translated_text = '.'.join(sentence_translations)

    return translated_text

if __name__ == '__main__':
    data = pd.read_csv('shpan.csv')
    for i in range(5):
        print(f"{i}s iteration:")
        # Get text from all docs
        files_text = (pdf_to_text(doc) for doc in get_docs_for_plan(data['pl_number'][i]))

        # for each doc, clean and split to chunks
        files_cleand_chunks = [clean_and_split(text) for text in files_text]

        # flatten files chunck to 1d list
        documents = [sentencewise_translate(chunk,he_en_translate_pipe) for cleand_chunks in files_cleand_chunks for chunk in cleand_chunks]

        db = FAISS.from_documents(documents, embeddings)

        question = "In the construction planning document provided, identify and extract the street or specific location where the construction is planned. If the street or location is Uruguay answer yes, if not answer no and if the document does not contain this information, please indicate that it does not exist"
        # he_question = "במסמך תכנון הבנייה שסופק, זהה וחלץ את הרחוב או המיקום הספציפי בו מתוכננת הבנייה. אם הרחוב או המיקום הם אורוגוואי השב כן, אם לא השב לא ואם המסמך אינו מכיל מידע זה, נא לציין שהוא אינו קיים"
        # he_retrived_docs = db.similarity_search(he_question)
        # question = sentencewise_translate(he_question,he_en_translate_pipe)
        # retrived_doc = sentencewise_translate(he_retrived_docs[0].page_content,he_en_translate_pipe)
        retrived_doc = db.similarity_search(question)[0].page_content
        print(retrived_doc)
        prompt_template = """
        <s>[INST]
        
        {question}
        Document:
        ```{document}```
        ​[/INST]
        """
        prompt = PromptTemplate.from_template(prompt_template)

        chain = LLMChain(llm=llm, prompt=prompt)

        output = chain.predict(question=question, document=retrived_doc)

        # maybe try this syntax?
        # chain = prompt | llm
        # output = chain.invoke({"question":question, "document":retrived_doc})

        he_output = sentencewise_translate(output,en_he_translate_pipe)
        print(he_output[::-1]) # print backward for readability


