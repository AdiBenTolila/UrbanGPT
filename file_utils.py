import pandas as pd
from download_program_doc import extract_xplan_number,program_doc_url
from urllib.parse import unquote
import os
import requests
from pypdf import PdfReader
import textract
import re
from langchain.docstore.document import Document
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import time
import random
from itertools import chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
logger = logging.getLogger(__name__)

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

def temp_get_docs_for_plan(plan_number):
    if not os.path.exists(f"out/{plan_number}"):
        return
    files = os.listdir(f"out/{plan_number}")
    for file in files:
        if "הוראות" in file:
            yield f"out/{plan_number}/{file}"

def get_docs_for_plan_selenum(plan_number,redownload=False,retry=3):
    pwd = os.getcwd()
    xplan_number = extract_xplan_number(plan_number)

    # URL of the website containing the PDF
    url = f'https://mavat.iplan.gov.il/SV4/1/{xplan_number}/310'

    # Start a new Chrome browser session
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    os.makedirs(f'{pwd}/out/{plan_number}', exist_ok=True)
    # if file already exists, return without downloading
    if os.listdir(f'{pwd}/out/{plan_number}') and not redownload:
        return
    prefs = {'download.default_directory' : f'{pwd}/out/{plan_number}/'}
    chrome_options.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(options=chrome_options)

    for i in range(retry):
        try:
            # Open the webpage
            print(f'opening webpage {url}')
            driver.get(url)
            button = WebDriverWait(driver, 5).until(lambda x: x.find_element('xpath','//a[@title="הוראות התכנית "]'))
            if 'gray-disabled' in button.get_attribute('class'):
                print(f'button is disabled for {plan_number}')
                break

            # Download the PDF
            button.click()
            # Wait for a few seconds to ensure the PDF is downloaded
            time.sleep(5)
            break
        except:
            print(f'failed to open webpage, retrying {i+1} out of {retry}')
            amount = random.randint(1, 5)
            time.sleep(5 + amount)

    # if file was not downloaded, remove the folder
    if not os.listdir(f'{pwd}/out/{plan_number}'):
        os.rmdir(f'{pwd}/out/{plan_number}')
        print(f'no file for {plan_number}')
    else:
        print(f'downloaded {plan_number}')
    # Close the browser
    driver.quit()    

def pdf_to_text(filename):
    pdffileobj = open(filename, 'rb')
    pdfReader = PdfReader(pdffileobj)
    text = ""
    for page in pdfReader.pages:
        text += page.extract_text()
    # TODO
    # if text == "":
    #     text = textract.process(filename, method='tesseract', language='eng')
    return text

def pdf_bin_to_text(pdf_bin):
    pdfReader = PdfReader(pdf_bin)
    text = ""
    for page in pdfReader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    # Define the pattern for the unwanted text
    unwanted_pattern = r'תכנון זמין(\s|\n)*[0-9]*(\s|\n)*+מונה\s+הדפסה'
    cleaned_text = re.sub(unwanted_pattern, '', text)
    
    # if there's more then 2 newline, remove them
    cleaned_text = re.sub(r'\n\n+', '\n\n', cleaned_text)

    return cleaned_text.strip()

def clean_and_split(text, doc_id=None, chunk_size=2048, chunk_overlap=128):
    cleaned_text = clean_text(text)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n","."," ",""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([cleaned_text])
    for t in texts:
        t.metadata = {"doc_id": doc_id}

    return texts

def sentencewise_translate(text,translate_pipe):
    sentences = [t for t in text.split('.') if t]

    translates = translate_pipe(sentences)
    sentence_translations = [t['translation_text'] for t in translates]

    # Join translations for each original chunk using a dot
    translated_text = '.'.join(sentence_translations)

    return translated_text

def get_text_for_plan(plan_number):
    files_text = (pdf_to_text(doc) for doc in temp_get_docs_for_plan(plan_number))
    files_cleand = [clean_text(text) for text in files_text]
    docs_str = "\n".join(files_cleand)
    return docs_str

if __name__ == '__main__':
    data = pd.read_csv('shpan.csv')
    pl_nums = data['pl_number'].sort_values().unique()
    files_text = [(pdf_to_text(doc) for doc in temp_get_docs_for_plan(pl_num)) for pl_num in pl_nums]
    files_cleand = ["\n".join([clean_text(text) for text in texts]) for texts in files_text]
    filtered_docs = [doc for doc in files_cleand if len(doc.strip()) > 0]
    print(len(files_cleand))
    # for each doc, clean and split to chunks
    # files_cleand_chunks = [clean_and_split(text) for text in files_text]
    # flattened_documents = list(chain(*files_cleand_chunks))
    # print(type(flattened_documents[0]))

    # print(type(files_cleand_chunks[0]))

        


