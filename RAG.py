from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from file_utils import get_docs_for_plan, pdf_to_text, clean_and_split,sentencewise_translate, temp_get_docs_for_plan
from models import he_en_translate_pipe, en_he_translate_pipe, embeddings , llm
import pandas as pd
from itertools import chain

if __name__ == '__main__':
    data = pd.read_csv('shpan.csv')
    # Get text from all docs
    files_text = (pdf_to_text(doc) for doc in temp_get_docs_for_plan(data['pl_number'][2]))

    # for each doc, clean and split to chunks
    files_cleand_chunks = [clean_and_split(text) for text in files_text]
    flattened_documents = list(chain(*files_cleand_chunks))


    # flatten files chunck to 1d list
    print(type(flattened_documents[0]))
    db = FAISS.from_documents(flattened_documents, embeddings)

    question = "In the construction planning document provided, identify and extract the street or specific location where the construction is planned. If the street or location is Uruguay answer yes, if not answer no and if the document does not contain this information, please indicate that it does not exist"
    he_question = "במסמך תכנון הבנייה שסופק, זהה וחלץ את הרחוב או המיקום הספציפי בו מתוכננת הבנייה. אם הרחוב או המיקום הם אורוגוואי השב כן, אם לא השב לא ואם המסמך אינו מכיל מידע זה, נא לציין שהוא אינו קיים"
  
    retrived_doc = db.similarity_search(he_question)[0].page_content
    prompt_template = """
    <s>[INST]
    
    {question}
    Document:
    ```{document}```
    ​[/INST]
    """
    translate_chunck = sentencewise_translate(retrived_doc,he_en_translate_pipe)
    prompt = PromptTemplate.from_template(prompt_template)

    chain = LLMChain(llm=llm, prompt=prompt)

    output = chain.predict(question=question, document=translate_chunck)

    he_output = sentencewise_translate(output,en_he_translate_pipe)
    print(he_output[::-1]) # print backward for readability



