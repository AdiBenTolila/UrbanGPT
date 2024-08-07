from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.output_parsers import OutputFixingParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.vectorstores import FAISS
from file_utils import pdf_to_text, clean_and_split, temp_get_docs_for_plan, pdf_bin_to_text, clean_text
from models import get_embedding_model, get_llm, openai_count_tokens
import pandas as pd
from itertools import chain
import os
import re
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import tqdm
import time
from uuid import UUID
from datetime import datetime
import dateutil.parser as parser
import logging
import torch
from dataclasses import Field, dataclass,field
from typing import List, Dict, Any, Callable, Union

logger = logging.getLogger(__name__)

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
    rephrase_chain = prompt | model
    output = rephrase_chain.invoke(dict(question=question, n=k)).content
    output_parser_list = output.split("\n")
    # remove line number
    output_parser_list = [re.sub(r"^\d+\.\s", "", line) for line in output_parser_list]
    output_parser_list.append(question)
    return output_parser_list

def question_from_description(description, model=None, answer_parser=None):
    if model is None:
        model = get_llm()
        
    parser = QuestionOutputParser()
    prompt_template = """
    בהינתן תיאור השדה הבא, נסח שאלה שתישאל על כל מסמך בכדי לקבל את המידע הרלוונטי לשדה זה.
    על השאלה לדרוש שדה המתאים לתיאור, לדוגמה, אם התיאור מתאר כמות יחידות דיור בתוכנית, נספק שאלה שתדרוש את מספר היחידות הדיור בתוכנית.
    אם התיאור מתאר את שכונת התוכנית, נספק שאלה שתדרוש את שכונת התוכנית.
    תיאור: "{description}"
    {format_description}
    """
    if answer_parser is not None:
    #     prompt_template += f"התשובות לשאלה צריכות להתאים לפורמט: \"{answer_parser.get_format_instructions()}\" אז נסח את השאלה בהתאם."
        prompt_template += f"נסח את השאלה כך שתדרוש את הערך הרלוונטי לשדה מסוג {answer_parser.get_type()} המתאים לתיאור זה."
    prompt = PromptTemplate.from_template(prompt_template)
    question_chain = prompt | model | parser
    output = question_chain.invoke(dict(description=description, format_description=parser.get_format_instructions()))
    if type(output) != str:
        output = output.content
    logger.info(f"from description: {description} got question: {output}")
    return output

def query_from_description(description, model=None):
    if model is None:
        model = get_llm()
        
    prompt_template = """
    בהינתן תיאור השדה הבא, נסח שאילתת חיפוש מתאימה.
    תיאור: "{description}"
    על השאילתה להיות בעברית, קצרה ומדוייקת ככל האפשר.
    על השאילתא לא לכלול יחידות מידה או מילות מפתח שעלולות לבלבל את החיפוש ולהביא תוצאות לא רלוונטיות.
    """
    prompt = PromptTemplate.from_template(prompt_template)
    question_chain = prompt | model
    output = question_chain.invoke(dict(description=description))
    if type(output) != str:
        output = output.content
    logger.info(f"from description: {description} got query: {output}")
    return output

def query_from_question(question, model=None):
    if model is None:
        model = get_llm()
        
    prompt_template = """
    בהינתן השאלה הבאה, נסח שאילתת חיפוש מתאימה.
    שאלה: "{question}"
    על השאילתה להיות בעברית, קצרה ומדוייקת ככל האפשר.
    על השאילתא לא לכלול יחידות מידה או מילות מפתח שעלולות לבלבל את החיפוש ולהביא תוצאות לא רלוונטיות.
    """
    prompt = PromptTemplate.from_template(prompt_template)
    question_chain = prompt | model
    output = question_chain.invoke(dict(question=question))
    if type(output) != str:
        output = output.content
    logger.info(f"from question: {question} got query: {output}")
    return output

def title_from_quary(query, model=None):
    if model is None:
        model = get_llm()
        
    prompt_template = """
    בהינתן השאילתא הבאה, נסח את כותרת מתאימה לשיחה.
    שאילתא: "{query}"
    על הכותרת להיות בעברית, קצרה ומדוייקת ככל האפשר.
    """
    prompt = PromptTemplate.from_template(prompt_template)
    question_chain = prompt | model
    output = question_chain.invoke(dict(query=query))
    if type(output) != str:
        output = output.content
    logger.info(f"from query: {query} got title: {output}")
    return output

def get_top_k(question, db, pl_number=None, k=3):
    doc_filter = dict(doc_id=pl_number) if pl_number else None
    docs = db.similarity_search(question, filter=doc_filter, fetch_k=db.index.ntotal, k=k)
    return [doc.page_content for doc in docs]

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

def get_top_k_by_count_foreach(questions, db, k=3):
    db_size = db.index.ntotal
    chunk_counts_by_doc = {}
    for q in questions:
        # get all documents and their scores
        similarities = db.similarity_search_with_score(q, fetch_k=db_size, k=db_size)
        assert len(similarities) == db_size

        # for each document, get all chunks and their scores
        doc_chunks_by_score = {}
        for chunk, score in similarities:
            doc_id = chunk.metadata["doc_id"]
            if doc_id not in doc_chunks_by_score:
                doc_chunks_by_score[doc_id] = []
            doc_chunks_by_score[doc_id].append((chunk, score))

        # for each document, get top k chunks by score and count them
        for doc_id, chunks in doc_chunks_by_score.items():
            chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
            doc_chunks_by_score[doc_id] = chunks[:k]
            chunk_counts_by_doc[doc_id] = {}
            for chunk, score in chunks:
                if chunk.page_content in chunk_counts_by_doc[doc_id]:
                    chunk_counts_by_doc[doc_id][chunk.page_content] += 1
                else:
                    chunk_counts_by_doc[doc_id][chunk.page_content] = 1
        
    # get top k chunks by count for each document
    top_cunks_by_doc = {}
    for doc_id, chunks_counts in chunk_counts_by_doc.items():
        sorted_chunks = sorted(chunks_counts.items(), key=lambda x: x[1], reverse=True)
        top_cunks_by_doc[doc_id] = [key for key,val in sorted_chunks[:k]]
        
    return top_cunks_by_doc

def get_db(data,path, chunk_size=512, chunk_overlap=128):
    if(os.path.exists(path)):
        db = FAISS.load_local(path,get_embedding_model(), allow_dangerous_deserialization=True)
        return db
    documents = []
    for i,row in data.iterrows():
        # Get text from all docs
        files_text = (pdf_to_text(doc) for doc in temp_get_docs_for_plan(row['pl_number']))

        # for each doc, clean and split to chunks
        files_cleand_chunks = [clean_and_split(text, row['pl_number'], chunk_size=chunk_size, chunk_overlap=chunk_overlap) for text in files_text]
        # flatten files chunck to 1d list
        flattened_documents = list(chain(*files_cleand_chunks))
        logger.info(f"loaded plan {row['pl_number']} with {len(flattened_documents)} docs")
        documents.extend(flattened_documents)
    embeddings = get_embedding_model()
    assert len(documents) > 0, "No documents found for indexing."
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

def create_vector_from_df(plans_df, chunk_size=512, chunk_overlap=128):
    documents = []
    logger.info(f"plans_df: {plans_df}")
    for i,row in plans_df.iterrows():
        # Get text from all docs
        files_text = row['content']

        # for each doc, clean and split to chunks
        files_cleand_chunks = clean_and_split(files_text, row['id'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info(f"loaded plan {row['id']} with {len(files_cleand_chunks)} docs")
        documents.extend(files_cleand_chunks)
    embeddings = get_embedding_model()
    assert len(documents) > 0, "No documents found for indexing."
    db = FAISS.from_documents(documents, embeddings)
    return db

def get_answer(question,chunks, model=None, instructions=None, parser=None):
    if parser is None:
        parser = TextualOutputParser()
    if len(chunks) == 0 or "".join(chunks) == "":
        # return "No document found"
        logger.info(f"document not found for question: {question}")
        return None
    sources_template = ""
    for i, chunk in enumerate(chunks):
        sources_template += f"קטע {i+1}:\n```{chunk}```\n"
    # answer_format_description = "תשובתך צריכה להסתיים באחת מהתשובות הבאות: כן, לא, אין ברשותי מספיק מידע." if binary_answer else ""
    # answer_format_description = "" if format_description is None else format_description
    answer_format_description = parser.get_format_instructions()
    prompt_message = "בהינתן מספר קטעים מתוך מסמך תכנון הבנייה ושאלה בנוגע לתוכנית שבמסמך, מצא את התשובה המדוייקת לשאלה על בסיס המסמך וענה עליה בקצרה ובמדויק." if instructions is None else instructions
    # the new text contain the first 3 of the retrived docs 
    prompt_template = """
    {prompt_message}
    קטעים:
    {sources}
    שאלה:
    {question}
    {answer_format}
    """
    prompt = PromptTemplate.from_template(prompt_template)
    if model is None:
        model = get_llm()
    chain_llm = prompt | model | parser
    try:
        output = chain_llm.invoke(dict(question=question, sources=sources_template, answer_format=answer_format_description, prompt_message=prompt_message))
    except Exception as e:
        logger.error(f"error: {e}")
        if e.__class__.__name__ == "OutOfMemoryError":
            # print cuda memory usage
            logger.error(torch.cuda.memory_summary())
            raise e

        return None
    return output
    
def get_answers(questions, chunks_list, model=None, instructions=None, parser=None, batch_size=50, callbacks=[]):
    if parser is None:
        parser = TextualOutputParser()
    if model is None:
        model = get_llm()
    if type(questions) == str:
        questions = [questions] * len(chunks_list)

    sources_tamplates = ["".join([f"קטע {i+1}:\n```{chunk}```\n" for i, chunk in enumerate(chunks)]) for chunks in chunks_list]
    answer_format_description = parser.get_format_instructions()
    prompt_message = "בהינתן מספר קטעים מתוך מסמך תכנון הבנייה ושאלה בנוגע לתוכנית שבמסמך, מצא את התשובה המדוייקת לשאלה על בסיס המסמך וענה עליה בקצרה ובמדויק." if instructions is None else instructions
    # the new text contain the first 3 of the retrived docs 
    prompt_template = """
    {prompt_message}
    קטעים:
    {sources}
    שאלה:
    {question}
    {answer_format}
    """
    prompt = PromptTemplate.from_template(prompt_template)
    fixing_parser = OutputFixingParser.from_llm(llm=model, parser=parser, max_retries=3)
    chain_llm = prompt | model | fixing_parser
    try:
        outputs = chain_llm.batch([dict(question=question,
                                        sources=sources_template,
                                        answer_format=answer_format_description,
                                        prompt_message=prompt_message) for question, sources_template in zip(questions,sources_tamplates)],
                                    config={"max_concurrency": batch_size, "callbacks": callbacks})
    except Exception as e:
        logger.error(f"error: {e}")
        if e.__class__.__name__ == "OutOfMemoryError":
            # print cuda memory usage
            logger.error(torch.cuda.memory_summary())
            raise e
        return None
    return outputs

def get_answer_foreach_doc(question, db, doc_ids, num_docs=3, num_rephrasings=0, model=None, verbose=False, multiprocess=False, parser=None, full_doc=False, instructions=None, query=None, batch_size=5, callbacks=[]):
    if model is None:
        model = get_llm()
    if full_doc:
        docs_chunks = {doc_id:[clean_text(pdf_to_text(doc)) for doc in temp_get_docs_for_plan(doc_id)] for doc_id in doc_ids}
    else:
        if query:
            questions = [query]
        else:
            questions = diffrent_question_rephrasing(question, k=num_rephrasings, model=model)
        if verbose:
            print(questions)
        docs_counts = {k:{} for k in doc_ids}
        docs_score_sum = {k:0 for k in doc_ids}
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
    answers = []
    if multiprocess:
        pool = ThreadPool(multiprocess)
        def _get_answer(pl_num, question, chunks, model):
            ans = get_answer(question, chunks, model, parser=parser, instructions=instructions)
            if verbose:
                print(pl_num, "done")
            return {
                "id": pl_num,
                "value": ans,
                "chunks": chunks
            }
        answers = pool.starmap(_get_answer, [(pl_num, question, chunks, model) for pl_num,chunks in docs_chunks.items()])
        return answers
    elif True:
        outputs = get_answers(question, [chunks for pl_num,chunks in docs_chunks.items()], model=model, instructions=instructions, parser=parser, batch_size=batch_size, callbacks=callbacks)
        assert doc_ids is not None, "doc_ids must be provided"
        assert outputs is not None, "outputs must be provided"
        assert docs_chunks is not None, "docs_chunks must be provided"
        answers = [{
            "id": pl_num,
            "value": output,
            "chunks": chunks
        } for pl_num,output,chunks in zip(doc_ids, outputs, [chunks for pl_num,chunks in docs_chunks.items()])]
        return answers

    else:
        for i,pl_num in tqdm.tqdm(enumerate(doc_ids[1:]), total=len(doc_ids[1:])):
            # chunks = get_top_k_by_count(questions, db, row['pl_number'], k=num_docs)
            # chunks = list(sorted_docs_count[row['pl_number']].keys())
            chunks = docs_chunks[pl_num]
            output = get_answer(question,chunks, model=model, parser=parser, instructions=instructions)
            logger.info(f"{pl_num} answer:{output}")
            if verbose:
                print(f"{i}s document:",output)
            # yield {
            #     "id": pl_num,
            #     "value": output,
            #     "chunks": chunks
            # }

def multiquery_retriver(question, db):
    llm = get_llm()
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
    )
    unique_docs = retriever_from_llm.get_relevant_documents(query=question)
    print("len_docs:",len(unique_docs))

    return unique_docs

def get_answers_for_all_docs_return_them(question):
    question_individual = """ :"בהינתן שאלה הנשאלת ברבים תהפוך אותה שאלה ליחיד והשאלה חייבת להתחיל ב- "האם"""
    llm_model = get_llm()
    prompt_template = """
    Human: {question_individual}
    {question}
    Assistent:
    """
    # chain_llm = LLMChain(llm=llm_model, prompt=PromptTemplate.from_template(prompt_template))
    chain_llm = PromptTemplate.from_template(prompt_template) | llm_model
    output = chain_llm.invoke(dict(question=question, question_individual=question_individual))
    print("output:",output[::-1])
    return output[::-1]
    # if model is None:
    #     model = get_llm()
    # answers = get_answer_foreach_doc(question, db, data, num_docs=num_docs, num_rephrasings=num_rephrasings, model=model, verbose=verbose, multiprocess=multiprocess)
    # yes_answers = [ans for ans in answers if "כן" in ans["answer"]]
    # print("yes_answers:",yes_answers)
    # return answers

def stream_chat(data, messages, model=None):
    if model is None:
        model = get_llm()
    
    docs = "\n".join([f"### document {doc['id']}:\n```{doc['content']}```" for doc in data.to_dict(orient='records')])
    system_message = f"You are given a set of urban planning documents. Your task is to answer questions based on the content of the documents.\n{docs}"
    messages = [SystemMessage(system_message)] + messages
    # count number of tokens in the messages
    num_tokens = openai_count_tokens("\n".join([msg.content for msg in messages]), "cl100k_base")
    logger.info(f"num_tokens in request: {num_tokens}")
    for msg in model.stream(messages):
        logger.info(f"got message: {msg}")
        yield msg

class FilteredRetriever(VectorStoreRetriever):
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = {}
    doc_id: str
    k: int = 7
    
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        logger.info(f"quering document {self.doc_id} with query: {query}")
        results = self.vectorstore.get_relevant_documents(query=query,metadata={"doc_id":self.doc_id}, **kwargs, **self.search_kwargs)
        results = [doc for doc in results if doc.metadata['doc_id'] == self.doc_id][:self.k]
        logger.info(f"got {len(results)} results from retrval")
        return results

class BatchCallback(BaseCallbackHandler):
    def __init__(self, callable: Callable):
        super().__init__()
        self.func = callable

    def on_chain_end(self, outputs: Dict[str, Any], run_id: UUID, parent_run_id: Union[UUID, None], tags: List[str], **kwargs):
        """Run when chain ends running."""
        self.func(outputs, run_id=run_id, tags=tags, **kwargs)

class BooleanOutputParser(BaseOutputParser[bool]):
    """Boolean boolean parser."""

    true_val: str = "כן"
    false_val: str = "לא"
    unknown_val: str = "אין ברשותי מספיק מידע"
    def __init__(self, true_val: str = "כן", false_val: str = "לא", unknown_val: str = "אין ברשותי מספיק מידע"):
        super().__init__()
        self.true_val = true_val
        self.false_val = false_val
        self.unknown_val = unknown_val

    def parse(self, text: str) -> bool:
        logger.info(f"parsing boolean: {text}")
        # clean text and remove punctuation
        cleaned_text = text.strip().lower()
        cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text)
        
        if cleaned_text not in (self.true_val.lower(), self.false_val.lower(), self.unknown_val.lower()):
            raise OutputParserException(
                f"BooleanOutputParser expected output value to either be "
                f"{self.true_val}, {self.false_val}, or {self.unknown_val}, but "
                f"Received {cleaned_text}."
            )
        return True if cleaned_text == self.true_val.lower() else False if cleaned_text == self.false_val.lower() else None

    @property
    def _type(self) -> str:
        return "boolean_output_parser"
    
    def get_format_instructions(self):
        return 'תשובתך צריכה להיות בפורמט כזה: "כן", "לא" או "אין ברשותי מספיק מידע" בלבד, אין להשיב שום פרט נוסף.'
    
    def get_type(self):
        return 'BOOLEAN'

punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
class NumericOutputParser(BaseOutputParser[int]):
    """Numeric output parser."""

    unknown_val = "אין ברשותי מספיק מידע"
    def parse(self, text: str) -> int:
        logger.info(f"parsing numeric: {text}")
        # clean text and remove punctuation
        cleaned_text = text.strip().replace(",", "")
        
        # cleaned_text = re.sub(rf"[{punctuation}]", "", cleaned_text)
        
        if self.unknown_val in cleaned_text:
            return None
        try:
            return float(cleaned_text)
        except ValueError:
            raise OutputParserException(
                f"NumericOutputParser expected output value to be an numeric value, but "
                f"Received {cleaned_text}."
            )

    @property
    def _type(self) -> str:
        return "numeric_output_parser"
    
    def get_format_instructions(self):
        return f"תשובתך צריכה להיות מספר, אם אין מספיק מידע כדי לספק תשובה תשיב: \"{self.unknown_val}\" בלבד."
    
    def get_type(self):
        return 'FLOAT'

class TextualOutputParser(BaseOutputParser[str]):
    """Textual output parser."""

    unknown_val = "אין ברשותי מספיק מידע"
    def parse(self, text: str) -> str:
        logger.info(f"parsing text: {text}")
        if self.unknown_val in text:
            return None
        return text.strip()

    @property
    def _type(self) -> str:
        return "textual_output_parser"
    
    def get_format_instructions(self)->str:
        # return f"תשובתך צריכה להכיל את התשובה הסופית לשאלה שנשאלה, אם אין מספיק מידע כדי לספק תשובה תשיב: \"{self.unknown_val}\" בלבד."
        return f"תשובתך צריכה להכיל את התשובה הסופית לשאלה שנשאלה בלבד. שים לב כי התשובה נכנסת ישירות לבסיס נתונים אז אינה צריכה להכיל נימוקים או הקדמות כגון \"התשובה היא:\" וכו', אם אין מספיק מידע כדי לספק תשובה תשיב: \"{self.unknown_val}\" בלבד."
    
    def get_type(self):
        return 'TEXT'

class DateOutputParser(BaseOutputParser[str]):
    """Date output parser."""

    unknown_val = "אין ברשותי מספיק מידע"
    def parse(self, text: str) -> str:
        text = text.strip()
        logger.info(f"parsin date: {text}")
        if self.unknown_val in text:
            return None
        try:
            # date = datetime.strptime(text, '%d/%m/%Y')
            date = parser.parse(text, dayfirst = True)
            # return date.strftime('%Y-%m-%d %H:%M:%S.%f')
            return date
        except ValueError as e:
            raise OutputParserException(
                f"DateOutputParser expected output value to be a date, but "
                f"Received {text}."
                f"Error: {e}"
            )
    
    @property
    def _type(self) -> str:
        return "date_output_parser"
    
    def get_format_instructions(self):
        return f"תשובתך צריכה להיות תאריך בפורמט: יום/חודש/שנה, אם אין מספיק מידע כדי לספק תשובה תשיב: \"{self.unknown_val}\" בלבד."

    def get_type(self):
        return 'TIMESTAMP'
        # return 'TEXT'

class QuestionOutputParser(BaseOutputParser[str]):
    """Question output parser."""

    def parse(self, text: str) -> str:
        logger.info(f"parsing question: {text}")
        if not text.endswith("?") and not '\n' in text:
            raise OutputParserException(
                f"QuestionOutputParser expected output value to be a question, but "
                f"Received {text}."
            )
        return text

    @property
    def _type(self) -> str:
        return "question_output_parser"
    
    def get_format_instructions(self)->str:
        return f"תשובתך צריכה להיות שאלה בעברית, והשאלה חייבת להסתיים בסימן שאלה. אין צורך בעוד פירוט על התשובה."
    
    def get_type(self):
        return 'TEXT'

if __name__ == '__main__':
    data = pd.read_csv('shpan.csv') # read the data
    db = get_db(data,"faiss.index") 
    #creat csv with pl_number and answer
    # if os.path.exists('answers_top_3_rephrasing_2.csv'):
    #     df = pd.read_csv('answers_top_3_rephrasing_2.csv')
    # else:
    #     df = pd.DataFrame(columns=['pl_number','answer','chunks'])
    count = 0
    question = get_answers_for_all_docs_return_them("מהן הבניות המתוכננות ברחוב אורוגוואי?")
    # exit()
    # question = "האם הבניה המתוכננת היא ברחוב אורוגוואי?"
    questions = diffrent_question_rephrasing(question)
    print(questions)
    chunks_for_docs={}
    t_start = time.time()
    for i,row in data.iterrows():
        chunks = get_top_k_by_count(questions, db, row['pl_number'], k=3)
        print(f"{i}s document:")
        chunks_for_docs[row['pl_number']] = chunks
    t_end = time.time()
    print(f"retrive time: {t_end-t_start}")
    # t_start = time.time()
    # pool = Pool(4)
    # results = pool.starmap(get_answer, [(question, chunks) for chunks in chunks_for_docs.values()])
    # pool.close()
    # pool.join()
    # t_end = time.time()
    # print(f"generate time: {t_end-t_start}")
    # for i, (pl_number, answer) in enumerate(zip(chunks_for_docs.keys(), results)):
    #     df = pd.DataFrame([[pl_number, answer, chunks_for_docs[pl_number]]], columns=['pl_number', 'answer', 'chunks'])
    #     # df = pd.concat([df, df1])
    #     df.to_csv('answers_top_3_rephrasing_multiprocess.csv', index=False)
    #     # df1 = pd.DataFrame([[row['pl_number'],output,chunks]],columns=['pl_number','answer','chunks'])
    #     # df = pd.concat([df,df1])
    #     # df.to_csv('answers_top_3_rephrasing_2.csv',index=False)



    # import time
    # start = time.time()
    # question = "האם הבניה המתוכננת היא ברחוב אורוגוואי?"
    # questions = diffrent_question_rephrasing(question)
    # t_refarshing = time.time()
    # print(f"refarshing time: {t_refarshing-start}")
    # chunks_per_doc = get_top_k_by_count_foreach(questions, db, k=3)
    # t_retrive = time.time()
    # print(f"retrive time: {t_retrive-t_refarshing}")
    # # save chunks per doc to csv
    # chank_list = []
    # for pl_number, chunks in chunks_per_doc.items():
    #     chank_list.append({
    #         "pl_number": pl_number,
    #         "chunk_0": chunks[0],
    #         "chunk_1": chunks[1],
    #         "chunk_2": chunks[2]
    #     })
    # df = pd.DataFrame(chank_list)
    # df.to_csv('chunks_top_3_rephrasing.csv', index=False)
    # # pool all docs and get anskwers
    # pool = Pool(4)
    # results = pool.starmap(get_answer, [(question, chunks) for chunks in chunks_per_doc.values()])
    # pool.close()
    # pool.join()
    # t_answer = time.time()
    # print(f"answer time: {t_answer-t_retrive}")
    # for i, (pl_number, answer) in enumerate(zip(chunks_per_doc.keys(), results)):
    #     df = pd.DataFrame([[pl_number, answer, chunks_per_doc[pl_number]]], columns=['pl_number', 'answer', 'chunks'])
    #     df.to_csv('answers_top_3_rephrasing_multiprocess.csv', index=False)
    # t_end = time.time()
    # print(f"end time: {t_end-start}")

    # print(questions)
    # for i,row in data.iterrows():
    #     chunks = get_top_k_by_count(questions, db, row['pl_number'], k=3)
    #     print(f"{i}s document:")
    #     output = get_answer(question,chunks)
    #     df1 = pd.DataFrame([[row['pl_number'],output,chunks]],columns=['pl_number','answer','chunks'])
    #     df = pd.concat([df,df1])
    #     df.to_csv('answers_top_3_rephrasing_2.csv',index=False)

    # multiquery_retriver("כמה דירות אשרו מאז שנת 2020?",db)
    exit()

    #creat csv file
    if os.path.exists('answers_top_3_rephrasing_2.csv'):
        df = pd.read_csv('answers_top_3_rephrasing_2.csv')
    else:
        df = pd.DataFrame(columns=['pl_number','answer','chunks'])
    count = 0

    question = "כמה דירות אשרו מאז שנת 2020 ?"
    questions = diffrent_question_rephrasing(question)
    print(questions)
    for i,row in data.iterrows():
        chunks = get_top_k_by_count(questions, db, row['pl_number'], k=3)
        print(f"{i}s document:")
        output = get_answer(question,chunks)
        df1 = pd.DataFrame([[row['pl_number'],output,chunks]],columns=['pl_number','answer','chunks'])
        df = pd.concat([df,df1])
        df.to_csv('answers_top_3_rephrasing_2.csv',index=False)


    



