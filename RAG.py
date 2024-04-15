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
from multiprocessing.pool import ThreadPool
import tqdm

def diffrent_question_rephrasing(question, k=10, model=None):
    if k==0:
        return [question]
    if model is None:
        model = get_llm()
    prompt_template = """
    Human:  בהינתן שאלה, עליך לנסח אותה מחדש ב{n} דרכים שונות בעברית:
    {question}
    Assistent:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    rephrase_chain = LLMChain(llm=model, prompt=prompt)
    output = rephrase_chain.predict(question=question, n=k)
    output_parser_list = output.split("\n")
    # remove line number
    output_parser_list = [re.sub(r"^\d+\.\s", "", line) for line in output_parser_list]
    output_parser_list.append(question)
    return output_parser_list

def get_top_k_by_count(questions, db, pl_number=None, k=3, verbose=False):
    docs_counts = {}
    docs_scores = {}
    for q in questions:
        db_size = db.index.ntotal
        doc_filter = dict(doc_id=pl_number) if pl_number else None
        docs = db.similarity_search_with_score(q, filter=doc_filter, fetch_k=db_size, k=k)
        if verbose:
            print(f"len(docs): {len(docs)}")

        for doc,score in docs:
            if doc.page_content in docs_counts:
                docs_scores[doc.page_content] += score
                docs_counts[doc.page_content] += 1
            else:
                docs_scores[doc.page_content] = score
                docs_counts[doc.page_content] = 1
    docs_scores = {k: v / docs_counts[k] for k, v in docs_scores.items()}
    sorted_docs = sorted(docs_counts.items(), key=lambda x: x[1], reverse=True)
    if verbose:
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

def get_answer(question,chunks, model=None):
    if len(chunks) == 0:
        return "No document found"
    sources_template = ""
    for i, chunk in enumerate(chunks):
        sources_template += f"קטע {i+1}:\n```{chunk}```\n"
    
    # the new text contain the first 3 of the retrived docs
    prompt_template = """
    Human: בהינתן מספר קטעים מתוך מסמך תכנון הבנייה ושאלה בנוגע לתוכנית שבמסמך, מצא את התשובה המדוייקת לשאלה על בסיס המסמך וענה עליה בקצרה ובמדויק.
    תשובתך צריכה להסתיים באחת מהתשובות הבאות: כן, לא, אין ברשותי מספיק מידע.
    שאלה:
    {question}
    קטעים:
    {sources}
    Assistent:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    if model is None:
        model = get_llm()
    chain_llm = LLMChain(llm=model, prompt=prompt)
    output = chain_llm.predict(question=question, sources=sources_template)
    return output

def get_answer_foreach_doc(question, db, data, num_docs=3, num_rephrasings=10, model=None, verbose=False, multiprocess=False):
    if model is None:
        model = get_llm()
    questions = diffrent_question_rephrasing(question, k=num_rephrasings, model=model)
    if verbose:
        print(questions)
    answers = []
    
    docs_counts = {k:{} for k in data['pl_number']}
    docs_score_sum = {k:0 for k in data['pl_number']}
    for q in questions:
        res = db.similarity_search_with_score(q, k=db.index.ntotal, fetch_k=db.index.ntotal)
        res_dict = {}
        for doc, score in res:
            if len(res_dict.get(doc.metadata['doc_id'], [])) < num_docs:
                if doc.metadata['doc_id'] in docs_counts.keys():
                    res_dict[doc.metadata['doc_id']] = res_dict.get(doc.metadata['doc_id'], []) + [(doc, score)]
                    docs_counts[doc.metadata['doc_id']][doc.page_content] = docs_counts[doc.metadata['doc_id']].get(doc.page_content, 0) + 1
                    docs_score_sum[doc.metadata['doc_id']] += score
    # sorted_docs_count = {doc_id:{chnk:cnt for chnk,cnt in sorted(count_dict.items(), key=lambda x:x[1], reverse=True)[:num_docs]} for doc_id,count_dict in docs_counts.items()}
    docs_chunks = {doc_id:[chnk for chnk,cnt in sorted(count_dict.items(), key=lambda x:x[1], reverse=True)[:num_docs]] for doc_id,count_dict in docs_counts.items()}
    
    if multiprocess:
        pool = ThreadPool(multiprocess)
        def _get_answer(pl_num, question, chunks, model):
            ans = get_answer(question, chunks, model)
            if verbose:
                print(pl_num, "done")
            return {
                "pl_number": pl_num,
                "answer": ans,
                "chunks": chunks
            }
        answers = pool.starmap(_get_answer, [(pl_num, question, chunks, model) for pl_num,chunks in docs_chunks.items()])
    else:
        for i,pl_num in tqdm.tqdm(enumerate(data['pl_number'].values)):
            # chunks = get_top_k_by_count(questions, db, row['pl_number'], k=num_docs)
            # chunks = list(sorted_docs_count[row['pl_number']].keys())
            chunks = docs_chunks[pl_num]
            output = get_answer(question,chunks, model=model)
            answers.append({
                "pl_number": pl_num,
                "answer": output,
                "chunks": chunks
            })
            if verbose:
                print(f"{i}s document:",output)
    return answers

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


    



