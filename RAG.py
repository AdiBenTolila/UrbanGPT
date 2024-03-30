from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from file_utils import pdf_to_text, clean_and_split, temp_get_docs_for_plan, pdf_bin_to_text
from models import get_embedding_model, get_llm
import pandas as pd
from itertools import chain
import os
import re

def diffrent_question_rephrasing(question):
    model = get_llm()
    prompt_template = """
    Human:  בהינתן שאלה, עליך לנסח אותה מחדש בעשרה דרכים שונות בעברית:
    {question}
    Assistent:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    rephrase_chain = LLMChain(llm=model, prompt=prompt)
    output = rephrase_chain.predict(question=question)
    output_parser_list = output.split("\n")
    # remove line number
    output_parser_list = [re.sub(r"^\d+\.\s", "", line) for line in output_parser_list]
    output_parser_list.append(question)
    return output_parser_list

def get_top_k_by_count(questions, db, pl_number=None, k=3):
    docs_counts = {}
    docs_scores = {}
    for q in questions:
        db_size = db.index.ntotal
        doc_filter = dict(doc_id=pl_number) if pl_number else None
        docs = db.similarity_search_with_score(q, filter=doc_filter, fetch_k=db_size, k=k)

        for doc,score in docs:
            if doc.page_content in docs_counts:
                docs_scores[doc.page_content] += score
                docs_counts[doc.page_content] += 1
            else:
                docs_scores[doc.page_content] = score
                docs_counts[doc.page_content] = 1
    docs_scores = {k: v / docs_counts[k] for k, v in docs_scores.items()}
    sorted_docs = sorted(docs_counts.items(), key=lambda x: x[1], reverse=True)
    print("counts:", [val for key,val in sorted_docs])
    return [key for key,val in sorted_docs[:k]]

def get_db(data,path):
    if(os.path.exists(path)):
        db = FAISS.load_local(path,get_embedding_model())
        return db
    documents = []
    for i,row in data.iterrows():
        print(f"{i}s document:")
        # Get text from all docs
        files_text = (pdf_to_text(doc) for doc in temp_get_docs_for_plan(row['pl_number']))
        print(row['pl_number'])

        # for each doc, clean and split to chunks
        files_cleand_chunks = [clean_and_split(text, row['pl_number']) for text in files_text]
        # flatten files chunck to 1d list
        flattened_documents = list(chain(*files_cleand_chunks))

        documents.extend(flattened_documents)
    embeddings = get_embedding_model()
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(path)
    return db

def create_db(file):
    documents = []
    text = pdf_bin_to_text(file)
    documents.extend(clean_and_split(text, file))
    embeddings = get_embedding_model()
    db = FAISS.from_documents(documents, embeddings)
    return db

def get_answer(question,chunks):
    if len(chunks) == 0:
        return "No document found"
    # the new text contain the first 3 of the retrived docs
    prompt_template = """
    Human: בהינתן מספר קטעים מתוך מסמך תכנון הבנייה ושאלה בנוגע לתוכנית שבמסמך, מצא את התשובה המדוייקת לשאלה על בסיס המסמך וענה עליה בקצרה ובמדויק. 
    שאלה:
    {question}
    קטע 1:
    ```{document_1}```
    קטע 2:
    ```{document_2}```
    קטע 3:
    ```{document_3}```
    Assistent:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    llm = get_llm()
    chain_llm = LLMChain(llm=llm, prompt=prompt)
    output = chain_llm.predict(question=question, document_1=chunks[0], document_2=chunks[1], document_3=chunks[2])
    return output

if __name__ == '__main__':
    data = pd.read_csv('shpan.csv') # read the data
    db = get_db(data,"faiss.index") 


    #creat csv file
    if os.path.exists('answers_top_3_rephrasing_2.csv'):
        df = pd.read_csv('answers_top_3_rephrasing_2.csv')
    else:
        df = pd.DataFrame(columns=['pl_number','answer','chunks'])
    count = 0

    question = "האם הבניה המתוכננת היא ברחוב אורוגוואי?"
    questions = diffrent_question_rephrasing(question)
    print(questions)
    for i,row in data.iterrows():
        chunks = get_top_k_by_count(questions, db, row['pl_number'], k=3)
        print(f"{i}s document:")
        output = get_answer(question,chunks)
        df1 = pd.DataFrame([[row['pl_number'],output,chunks]],columns=['pl_number','answer','chunks'])
        df = pd.concat([df,df1])
        df.to_csv('answers_top_3_rephrasing_2.csv',index=False)


    



