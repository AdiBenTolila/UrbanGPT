import pytest
import pandas as pd
import numpy as np
import os
import logging
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from datetime import datetime
from test_utils import init_data, llm_verify_chunk, llm_verify_answer

from RAG import get_top_k, query_from_question, get_answer, question_from_description, query_from_description, stream_chat
from file_utils import get_docs_for_plan_selenum
from models import get_embedding_model, get_llm, get_openai_llm,openai_count_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(f'test/{date}', exist_ok=True)


@pytest.mark.parametrize("pl_number", [
    "101-1192871",
    "101-1194000",
    "101-1205152",
])
def test_download_pdf(pl_number):
    pdf_dir = get_docs_for_plan_selenum(pl_number, path=f"test/{date}")
    pdfs_in_dir = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    assert len(pdfs_in_dir) > 0, "No pdf files were downloaded."

@pytest.mark.parametrize("llm", [
    "hf-gemma-2-9b-it",
])
def test_ask_llm(llm):
    llm = get_llm(llm)
    prompt = """{question}"""
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm
    res = chain.invoke({"question": "מה יותר מהיר או צ'יטה?"})
    assert res.content

def test_chunk_to_vector():
    embeading_model = get_embedding_model()
    chunk1 = """elephant is a large mammal"""
    chunk2 = """pinguin is a bird"""
    chunk3 = """once upon a time there was a king"""
    vector1 = embeading_model.embed_query(chunk1)
    vector2 = embeading_model.embed_query(chunk2)
    vector3 = embeading_model.embed_query(chunk3)
    assert vector1 is not None, "The vector is None."
        
    def euclidean_distance(v1, v2):
        return np.linalg.norm(np.subtract(v1,v2))
        
    assert euclidean_distance(vector1, vector2) < euclidean_distance(vector1, vector3), "The euclidean distance is not correct."


@pytest.mark.parametrize("question, ground_truth, doc_id", [
    ("מהי השכונה בה מתוכננת התוכנית?", "קריית היובל", "101-1192871"),
    ('כמה מ"ר שצ"פ אקטיבי מאושר יש בתוכנית?', '3,639 מ"ר', '101-1192871'),
    ("מהי השכונה בה מתוכננת התוכנית?", "עין כרם", "101-1194000"),
])
def test_retrival_and_generation(init_data, question, ground_truth, doc_id):
    llm = get_llm("hf-gemma-2-9b-it")
    # retrive by query and check if the retrived document helps to answer the question
    search_quary = query_from_question(question, llm)
    chunks = get_top_k(search_quary, init_data['vector_store'], k=3, pl_number=doc_id)
    # assert cunks contains relevant information
    assert any([llm_verify_chunk(question, chunk, llm) for chunk in chunks]), "The retrived documents does not contain relevant information."
    
    # generate answer from the retrived document
    answer = get_answer(question, chunks, llm)
    # assert the answer is correct
    assert llm_verify_answer(question, ground_truth, answer, llm), f"The model answer is {answer} which is incompetable with {ground_truth}."

@pytest.mark.parametrize("description, ground_truth, doc_id", [
    ("שכונת התוכנית", "קריית היובל", "101-1192871"),
    ("שכונת התוכנית", "עין כרם", "101-1194000"),
    ("שצפ אקטיבי מאושר", "3,639 מ\"ר", "101-1192871"),
])
def test_get_answer_from_description(init_data, description, ground_truth, doc_id):
    llm = get_llm("hf-gemma-2-9b-it")
    #  generate a question from the description
    question = question_from_description(description, llm)
    # TODO assert the question is correct
    
    # generate query from the description and retrive the document
    query = query_from_description(description, llm)
    chunks = get_top_k(query, init_data['vector_store'], k=3, pl_number=doc_id)
    # assert the retrived document contains relevant information
    assert any([llm_verify_chunk(question, chunk, llm) for chunk in chunks]), "The retrived documents does not contain relevant information."

    # generate answer from the retrived document
    answer = get_answer(question, chunks, llm)
    # assert the answer is correct
    assert llm_verify_answer(question, ground_truth, answer, llm), f"The model answer is {answer} which is incompetable with {ground_truth}."

# def test_full_context_chat(init_data):
#     llm = get_llm("openai-gpt-4o-mini")
#     data = init_data['df'][['id', 'content']]
    
#     res = stream_chat(data,[HumanMessage("כמה דירות חדשות תוכננו בשנת 2021?")], llm)
#     assert res==0, f"The context length is {res}."

if __name__ == '__main__':
    pytest.main([__file__, '--log-level=INFO', '--log-format="%(asctime)s - %(levelname)s - %(message)s"', f'--log-file=out/{date}/log.txt'])