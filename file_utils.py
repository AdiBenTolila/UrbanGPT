import pandas as pd
from download_program_doc import extract_xplan_number,program_doc_url
from urllib.parse import unquote
import os
import requests
from PyPDF2 import PdfReader
# from langchain_community.document_loaders import PyPDFLoader
import textract
import re
from langchain.docstore.document import Document


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
    # TODO is this the right way to read the text?
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

# def load_and_split(file):
#     loader = PyPDFLoader(file)
#     pages = loader.load_and_split()
#     return pages


def sentencewise_translate(text,translate_pipe):
    sentences = [t for t in text.split('.') if t]

    translates = translate_pipe(sentences)
    sentence_translations = [t['translation_text'] for t in translates]

    # Join translations for each original chunk using a dot
    translated_text = '.'.join(sentence_translations)

    return translated_text
