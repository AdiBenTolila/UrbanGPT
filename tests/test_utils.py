import pytest
from agent import get_agent, get_tools, stream_agent, system_message, create_plans_index,agentic_question_ansewr
from RAG import get_top_k, create_vector_from_df, BooleanOutputParser, query_from_question, get_answer, question_from_description, query_from_description
from file_utils import get_docs_for_plan, pdf_bin_to_text, get_docs_for_plan_selenum
from models import get_embedding_model, get_llm, get_openai_llm
import pandas as pd
import os
import logging
from langchain.prompts import PromptTemplate
from datetime import datetime
import sqlite3
from langchain_community.vectorstores import FAISS

HERE = os.path.dirname(os.path.abspath(__file__))
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

@pytest.fixture
def init_data():
    # Configure logging to write to a file
    os.makedirs(f'out/{date}', exist_ok=True)
    logging.info("Initializing test data")
    sql_db_name = f"out/{date}/indexis.db"
    vec_db_name = f"out/{date}/faiss_c512_o128.index"
    data = pd.read_csv('shpan.csv')
    os.environ["SQL_DB"] = sql_db_name
    os.environ["VEC_DB"] = vec_db_name

    test_plans = [
        "101-1192871",
        "101-1194000",
        "101-1205152",
        "101-1206572",
        "101-1213339",
        "101-1214527",
        "101-1218858",
        "101-1219609",
        "101-1227206",
        "101-1230291",
        "101-1277680",
        "151-0241281",
        "151-0387068",
        "151-0439083",
        "151-0607630",
        "151-0697219",
        "151-0840710",
        "151-1028653",
        "151-1088400"
    ]
    test_data = data[data['pl_number'].isin(test_plans)]
    create_plans_index(sql_db_name, test_data)
    conn = sqlite3.connect(sql_db_name)
    df = pd.read_sql_query("SELECT * FROM plans", conn)
    vector_store = create_vector_from_df(df)
    vector_store.save_local(vec_db_name)
    # vector_store = FAISS.load_local(vec_db_name,get_embedding_model(), allow_dangerous_deserialization=True)
    return {
        "sql_db_name": sql_db_name,
        "vec_db_name": vec_db_name,
        "data": data,
        "test_plans": test_plans,
        "test_data": test_data,
        "df": df,
        "vector_store": vector_store
    }

def llm_verify_answer(question, ground_truth, answer, llm):
    prompt = """
        given the following question:
        {question}
        and the ground truth answer:
        {ground_truth}
        the model answer is:
        {answer}
        is the model answer matches the ground truth? answer with yes or no only, no other answers are allowed.
    """
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm | BooleanOutputParser(true_val="yes", false_val="no", unknown_val="unknown")
    return chain.invoke({"question": question, "ground_truth": ground_truth, "answer": answer})

def llm_verify_chunk(question, chunk, llm):
    prompt = """
        given the following question:
        {question}
        and the following chunk:
        {chunk}
        is the chunk relevant to the question? answer with yes or no only, no other answers are allowed.
    """
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm | BooleanOutputParser(true_val="yes", false_val="no", unknown_val="unknown")
    return chain.invoke({"question": question, "chunk": chunk})
