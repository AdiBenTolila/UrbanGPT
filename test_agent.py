import pytest
from agent import get_agent, get_tools, stream_agent, system_message, create_plans_index,agentic_question_ansewr
from RAG import get_top_k, create_vector_from_df, BooleanOutputParser, query_from_question, get_answer, question_from_description, query_from_description
from file_utils import get_docs_for_plan, pdf_bin_to_text
from models import get_embedding_model, get_llm, get_openai_llm
import pandas as pd
import os
import logging
from langchain.prompts import PromptTemplate
from datetime import datetime
import sqlite3
from langchain_community.vectorstores import FAISS

HERE = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(f'test/{date}', exist_ok=True)

@pytest.fixture
def init_data():
    # Configure logging to write to a file

    logging.info("Initializing test data")
    sql_db_name = f"test/{date}/indexis.db"
    vec_db_name = f"test/{date}/faiss_c512_o128.index"
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

@pytest.mark.parametrize("question, ground_truth, doc_id", [
    ("מהי השכונה בה מתוכננת התוכנית?", "קריית היובל", "101-1192871"),
    ('כמה שצ"פ אקטיבי מאושר יש בתוכנית?', '3,639 מ"ר', '101-1192871'),
    ("מהי השכונה בה מתוכננת התוכנית?", "עין כרם", "101-1194000"),
])
def test_agentic_document_ask(init_data, question, ground_truth, doc_id):
    llm = get_openai_llm('gpt-4o')
    result = agentic_question_ansewr(question, init_data['vector_store'], doc_id, llm)
    
    assert llm_verify_answer(question, ground_truth, result, llm), f"The model answer is {result.content} which is incorrect."


@pytest.mark.parametrize("question, ground_truth", [
    ('כמה שטח שצ"פ אקטיבי מאושר יש בשכונת אורה?', '3,639 מ"ר'),
    ("מהי השכונה שבה יש הכי הרבה תוכניות?", "אורה"),
    # ("מה הם תוכניות הבינוי שאושרו ברחוב שטרן ב10 שנים האחרונות?", "101-1192871, 101-1194000, 101-1205152, 101-1206572"),
    # ("כמה דירות אשרו במאז 2010?", "1,000"),
    # ("כמה דירות מעל 100 מטר אושרו מאז 2010?", "1,000"),
    # ("כמה בניינים מעל 10 קומות אשרו?", "1,000"),
    # ("יצר גרף ציר x שנים וציר y כמות דירות שאשרו בכל שנה.", "כן"),
])
def test_ask_agent(init_data, question, ground_truth):
    llm = get_openai_llm('gpt-4o')
    agent = get_agent(llm, get_tools())
    agent_generator = stream_agent(agent, system_message, question, "1")
    messages = [m for m in agent_generator]
    for m in messages:
        logger.info(m)
    result = messages[-1]
    
    assert llm_verify_answer(question, ground_truth, result, llm), f"The model answer is {result.content} which is incorrect."

if __name__ == '__main__':
    pytest.main([__file__, '--log-level=INFO', '--log-format="%(asctime)s - %(levelname)s - %(message)s"', f'--log-file=test/{date}/test.log'])