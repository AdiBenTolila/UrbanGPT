from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessageGraph, StateGraph
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.config import RunnableConfig
from langchain_community.vectorstores import FAISS

from models import get_embedding_model, get_llm,get_vertex_llm, get_gemini_llm,get_claude_llm, get_huggingface_llm,get_llamaCpp_llm,get_huggingface_chat, get_openai_llm
from file_utils import get_text_for_plan, pdf_bin_to_text, clean_and_split
from RAG import get_answer_foreach_doc, TextualOutputParser, NumericOutputParser, BooleanOutputParser, DateOutputParser, question_from_description, get_db, queries_from_description, FilteredRetriever, get_top_k, get_answer

from download_program_doc import extract_xplan_attributes

import pandas as pd
import sqlite3
import dotenv
import os
import logging
from datetime import datetime
import torch
import re
import json
from multiprocessing import Lock

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
dotenv.load_dotenv()
create_col_mutex = Lock()

system_message = """
            you are a professional agent that can answer questions about urban planning based on documents from the database.
            the documents in the database are urban planning documents and can contain information about various topics such as building permits, zoning plans, and more. so when you get a question, be sure to filter the relevant documents and provide the information needed.
            you can use the tools at your disposal to achieve your goals.
            in order to filter documents you can see the existing columns in the database and create new columns if needed.
            you can also query the database to get the relevant information.
            be sure to check the available columns in the documents table and create new columns if needed.
            the database contains documents table with urban planning documents and you could add new columns to the table to get more information.
            the questions and answers should be in hebrew.
            you can access any information you may need by creating new columns in the database based on the description you provide. 
            if you encounter any error regarding missing columns, you can create new columns based on the description you provide.
            only call one tool at a time and wait for the response before calling the next tool.
            base your final answer on the information you got from the tools you called.
        """


def create_plans_index(db_name, data, limit=None):
    conn = sqlite3.connect(db_name)
    
    indexes = data[['pl_number', 'pl_name', 'receiving_date', 'open_date', 'station_desc']].copy()
    indexes.columns = ['id', 'name', 'receiving_date', 'approval_date', 'status']
    # convert date to ISO8601 notation: YYYY-MM-DD HH:MM:SS.SSS
    # indexes['receiving_date'] = pd.to_datetime(indexes['receiving_date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    # indexes['receiving_date'] = pd.to_datetime(indexes['receiving_date'], errors='coerce').timestamp()
    indexes['content'] = indexes['id'].apply(lambda id: get_text_for_plan(id))
    #remove empty documents whene content is ""
    indexes = indexes[indexes['content'] != ""]
    if limit:
        indexes = indexes.head(limit)
    logger.info(f"create_plans_index with {len(indexes)} documents")
    # indexes.to_sql("plans", conn, index=False, if_exists='replace')
    conn.execute("DROP TABLE IF EXISTS plans")
    conn.execute("CREATE TABLE plans (id TEXT PRIMARY_KRY, name TEXT, receiving_date TIMESTAMP, approval_date  TIMESTAMP, status TEXT, content TEXT)")
    conn.executemany("INSERT INTO plans (id, name, receiving_date, approval_date, status, content) VALUES (?, ?, ?, ?, ?, ?)", indexes.to_records(index=False))    
    
    # drop the columns table if exists
    conn.execute("DROP TABLE IF EXISTS columns")
    conn.execute("CREATE TABLE columns (name TEXT, type TEXT, description TEXT, question TEXT, search_query TEXT)")
    conn.executemany("INSERT INTO columns (name, type, description) VALUES (?, ?, ?)", [
        ('id', 'TEXT', 'מספר מזהה של התוכנית'),
        ('name', 'TEXT', 'שם התוכנית'),
        ('receiving_date', 'TIMESTAMP', 'תאריך קבלת התוכנית'),
        ('approval_date', 'TIMESTAMP', 'תאריך אישור התוכנית'),
        ('status', 'TEXT', 'סטטוס התוכנית'),
        # ('content', 'TEXT', 'מסמך התוכנית המלא, מכיל את כל המידע הקיים במסמך')
        ])
    conn.commit()
    conn.close()

def add_plans_index(db_name, documents, id):
    conn = sqlite3.connect(db_name)
    docs_str = ",".join([f"'{doc}'" for doc in documents])
    conn.execute(f"CREATE TABLE IF NOT EXISTS plans_{id} AS SELECT * FROM plans WHERE id IN ({docs_str})")
    conn.execute(f"CREATE TABLE IF NOT EXISTS columns_{id} AS SELECT * FROM columns")
    conn.commit()
    
    # verify that the plans_{id} table was created and contains the documents
    res = conn.execute(f"SELECT COUNT(*) FROM plans_{id}").fetchone()
    logger.info(f"add_plans_index: {res[0]} documents added to plans_{id} out of {len(documents)}")
    
    conn.close()

def insert_new_column(column_name, column_type, column_description, data, question=None, search_queries=[], db_name='indexis.db', table_name='plans', col_table_name='columns'):
    conn = sqlite3.connect(db_name)
    if column_name in [name for _, name, _, _, _, _ in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]:
        conn.execute(f"ALTER TABLE {table_name} DROP COLUMN {column_name}")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    
    conn.executemany(f"UPDATE {table_name} SET {column_name}=? WHERE id=?", [(r['value'], r['id']) for r in data])
    conn.execute(f"INSERT OR REPLACE INTO {col_table_name} (name, description, type, question, search_query) VALUES (?, ?, ?, ?, ?)", (column_name, column_description, column_type, question, json.dumps(search_queries)))
    conn.commit()
    conn.close()

def add_new_document(doc_id, doc_bin, db_name='indexis.db', vec_db_name='faiss_c512_o128.index', chunk_size=512, chunk_overlap=128, num_docs=3, model=None):
    if model is None:
        get_llm()
    
    # get the document id and other metadata
    doc_txt = pdf_bin_to_text(doc_bin)
    doc_id = re.search(r'תכנית מס\' ([0-9]{3}-[0-9]{7}|[^\n]+)', doc_txt.split('\n')[2]).group(1)
    doc_metadata = extract_xplan_attributes(doc_id)
    os.makedirs(f'{ROOT_PATH}/out/{doc_id}', exist_ok=True)
    doc_bin.save(f'{ROOT_PATH}/out/{doc_id}/{doc_bin.filename}')
    
    # create a new vector db for the document
    embedding_model = get_embedding_model()
    documents = clean_and_split(doc_txt, doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pl_vec_db = FAISS.from_documents(documents, embedding_model)
    doc_values = {}
    with sqlite3.connect(f"{ROOT_PATH}/{db_name}") as conn:
        required_columns = conn.execute("SELECT * FROM columns WHERE search_query IS NOT NULL").fetchall()
        
        for name, type, desc, question, queries in required_columns:
            # get the answer for the question
            parser = {
                'TEXT': TextualOutputParser,
                'FLOAT': NumericOutputParser,
                'BOOLEAN': BooleanOutputParser,
                'TIMESTAMP': DateOutputParser
            }[type]()
            
            
            chunks = get_top_k(json.loads(queries), pl_vec_db, doc_id, k=num_docs)
            res = get_answer(question, chunks, model, parser)
            converted_res = {
                'TEXT': str,
                'FLOAT': float,
                'BOOLEAN': bool,
                'TIMESTAMP': datetime.fromtimestamp
            }[type](res)
            doc_values[name] = converted_res
        cols_titles = ['id', 'name', 'receiving_date', 'approval_date', 'status'] + list(doc_values.keys())

        cols_values = [doc_id, doc_metadata['pl_name'], doc_txt, doc_metadata['receiving_date']*1e6] + list(doc_values.values())
        vals_placeholder = ', '.join(['?']*len(cols_titles))
        conn.execute(f"INSERT INTO plans ({', '.join(cols_titles)}) VALUES ({vals_placeholder})", cols_values)
        conn.commit()
        # marge vector store with the main vector store
    vec_db = FAISS.load_local(f"{ROOT_PATH}/{vec_db_name}")
    vec_db.merge_from(pl_vec_db)
    vec_db.save_local(f"{ROOT_PATH}/{vec_db_name}")

# Define the function that determines whether to continue or not
def should_continue(messages):
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    else:
        return "action"

@tool
def query_document(document_id:str, query:str)->str:
    """
    Query a document with a given question.
    שאל שאלה מסויימת על מסמך מסויים
    """
    
    pass

@tool
def get_available_columns(config: RunnableConfig)->str:
    """
    Get all available columns in the database.
    קבל את כל העמודות הזמינות במסד הנתונים
    """
    # read the index
    sql_db_name = config.get("configurable", {}).get("sql_db", "indexis.db")
    conversation_id = config.get("configurable", {}).get("conversation_id", "1")
    conn = sqlite3.connect(sql_db_name)
    columns = pd.read_sql(f"SELECT name, type, description FROM columns_{conversation_id}", conn)
    conn.close()
    res = "#### Table `plans` with the folowing columns:\n" + "\n".join([f"- **{name}** ({type}): {description}" for name, type, description in columns.values])
    logger.info(f"get_available_columns: {res}")
    return res

@tool(parse_docstring=True)
def query_database(query:str, config: RunnableConfig)->str:
    """
    send SQL query to the database to get relevant information. be shure to check available columns and create new column if needed. the quary should be on 'plans' table. be sure to verify that the needed columns are created successfully. give your quary columns meaningful names so you can understand the results.
    שלח שאילתת SQL למסד הנתונים כדי לקבל מידע רלוונטי. וודא שיש לך עמודות זמינות וצור עמודה חדשה אם נדרש. השאילתא צריכה להיות על טבלת 'plans'. וודא שהעמודות הנדרשות נוצרו בהצלחה. תן לעמודות שלך שמות משמעותיים כך שתוכל להבין את התוצאות.
    
    Args:
        query: The SQL query to be executed on the 'plans' table.
    """
    # if query contains the \\[0-9]{3} pattern, return an error
    if re.search(r"\\[0-9]{3}", query):
        return "the query should not contain the pattern \\[0-9]{3}"
    sql_db_name = config.get("configurable", {}).get("sql_db", "indexis.db")
    conversation_id = config.get("configurable", {}).get("conversation_id", "1")
    query = query.replace("plans", f"plans_{conversation_id}")
    logger.info(f"query_database: {query}")
    
    conn = sqlite3.connect(sql_db_name)
    try:
        # df = pd.read_sql(query, conn)
        # replace the table name with the filtered table name
        res = conn.execute(query)
        if res.description is None:
            return "the query to the database returned no results"
        df = pd.DataFrame(res.fetchall(), columns=[col[0] for col in res.description])
        
    except Exception as e:
        return str(e)
    finally:
        conn.close()
    logger.info(f"query_database: {df}")
    # format the results to be shown as text
    results = df.to_dict(orient='records')
    # results = f"# the query to the database returned:\n" + "\n".join([", ".join([f"{k}: {v}" for k, v in d.items()]) for d in results])
    results = f"#### the query to the database returned:\n" + df.to_markdown(index=False)
    return results

@tool
def display_svg(svg:str)->str:
    """
    Display an SVG image in the chat.
    הצג תמונת SVG בצ'אט
    
    Args:
        svg: The SVG image as a string.
    """
    
    return svg.replace("\n", "")

def create_column(column_name, column_description, parser, config: RunnableConfig):
    # column name should be in english and without spaces, and the description should be in hebrew and describe the column intended value.
    if " " in column_name or not column_name.isascii():
        return "column name should be in english and without spaces"

    if re.search(r"\\[0-9]{3}", column_description):
        return "column description should not contain the pattern \\[0-9]{3}"

    with create_col_mutex:
        model_name = config.get("configurable", {}).get("model_name", "hf-gemma-2-9b-it")
        sql_db_name = config.get("configurable", {}).get("sql_db", "indexis.db")
        vec_db_name = config.get("configurable", {}).get("vec_db", "faiss_c512_o128.index")
        num_docs = config.get("configurable", {}).get("num_docs", 3)
        num_queries = config.get("configurable", {}).get("num_queries", 1)
        conversation_id = config.get("configurable", {}).get("conversation_id", "1")
        llm = get_llm(model_name)
        conn = sqlite3.connect(sql_db_name)
        data = pd.read_csv('shpan.csv') # read the data
        db = get_db(data, vec_db_name) # generate a question from the column description
        question = question_from_description(column_description, answer_parser=parser, model=llm)

        search_queries = queries_from_description(column_description, model=llm, num_queries=num_queries)

        logger.info(f"create_column: {column_name}, description: {column_description}, question: {question}")
        
        index = pd.read_sql(f"SELECT id FROM plans_{conversation_id}", conn)
        res = pd.DataFrame(get_answer_foreach_doc(question, db, index['id'],num_docs=num_docs, model=llm, multiprocess=False, parser=parser, full_doc=False, queries=search_queries))[['id', 'value']]

        # remove rows with None values
        # remove rows with null values
        res = res[res['value'].notnull()].astype(str)
        res = res[res['value'] != None]
        logger.info(f"inserting: {res[['value', 'id']]}")
        
        if res.empty:
            logger.info("no results found for the question: " + question)
            return "could not create a column for the given description, try a different description or different strategy"
        
        # insert the results into the new column in plans table
        try:
            insert_new_column(column_name, parser.get_type(), column_description, res.to_dict(orient='records'), question, search_queries, db_name=sql_db_name, table_name=f"plans_{conversation_id}", col_table_name=f"columns_{conversation_id}")
            # conn.execute(f"ALTER TABLE plans_{conversation_id} ADD COLUMN {column_name} {parser.get_type()}")
            # conn.execute(f"INSERT INTO columns_{conversation_id} (name, type, description) VALUES ('{column_name}', '{parser.get_type()}', '{column_description}')")
            # conn.executemany(f"UPDATE plans_{conversation_id} SET {column_name} = ? WHERE id = ?", res[['value', 'id']].to_records(index=False))
        except Exception as e:
            logger.error(f"could not create column: {e}")
            return str(e)
        finally:
            conn.commit()
            conn.close()
    return f"column created with {res.shape[0]} out of {index.shape[0]} valid values"

@tool(parse_docstring=True)
def create_numeric_column(column_name:str, column_description:str, config: RunnableConfig)->str:
    """
    Create a new numeric column for each urban planning document in the database by giving a description of the column. the column name should be in english and without spaces, and the description should be in hebrew and describe the column intended numeric value, also be sure to describe the units of the value if needed.
    צור עמודה מספרית חדשה לכל תוכנית בינוי עירוני במסד הנתונים על ידי תיאור לעמודה. שם העמודה צריך להיות באנגלית ובלי רווחים, והתיאור צריך להיות בעברית ולתאר את הערך המספרי הנדרש לעמודה, וגם לוודא שהיחידות של הערך מוגדרות בתיאור אם נדרש.
    
    Args:
        column_name: The name of the new column.
        column_description: The description of the new column.
    """
    return create_column(column_name, column_description, NumericOutputParser(), config)

@tool(parse_docstring=True)
def create_boolean_column(column_name:str, column_description:str, config: RunnableConfig)->str:
    """
    Create a new boolean column for each urban planning document in the database by giving a description of the column. the column name should be in english and without spaces, and the description should be in hebrew and describe the column intended boolean value.
    צור עמודה בוליאנית חדשה לכל תוכנית בינוי עירוני במסד הנתונים על ידי תיאור לעמודה. שם העמודה צריך להיות באנגלית ובלי רווחים, והתיאור צריך להיות בעברית ולתאר את הערך הבוליאני הנדרש לעמודה.
    
    Args:
        column_name: The name of the new column.
        column_description: The description of the new column.
    """
    return create_column(column_name, column_description, BooleanOutputParser(), config)

@tool(parse_docstring=True)
def create_textual_column(column_name:str, column_description:str, config: RunnableConfig)->str:
    """
    Create a new textual column for each urban planning document in the database by giving a description of the column. the column name should be in english and without spaces, and the description should be in hebrew and describe the column intended textual value.
    צור עמודה טקסטואלית חדשה לכל תוכנית בינוי עירוני במסד הנתונים על ידי תיאור לעמודה. שם העמודה צריך להיות באנגלית ובלי רווחים, והתיאור צריך להיות בעברית ולתאר את הערך הטקסטואלי הנדרש לעמודה.
    
    Args:
        column_name: The name of the new column.
        column_description: The description of the new column.

    """
    return create_column(column_name, column_description, TextualOutputParser(), config)

@tool(parse_docstring=True)
def create_date_column(column_name:str, column_description:str, config: RunnableConfig)->str:
    """
    Create a new date column for each urban planning document in the database by giving a description of the column. the column name should be in english and without spaces, and the description should be in hebrew and describe the column intended date value.
    צור עמודת תאריך חדשה לכל תוכנית בינוי עירוני במסד הנתונים על ידי תיאור לעמודה. שם העמודה צריך להיות באנגלית ובלי רווחים, והתיאור צריך להיות בעברית ולתאר את הערך התאריכי הנדרש לעמודה.
    
    Args:
        column_name: The name of the new column.
        column_description: The description of the new column.
    """
    return create_column(column_name, column_description, DateOutputParser(), config)

@tool()
def filter_by_index(query:str, index:str)->str:
    """
    Filter documents by index.
    סנן מסמכים על פי אינדקס מסויים
    """
    logger.info(f"filter_by_index: {query}, {index}")
    pass

def get_agent(llm, tools, use_memory=True):
    model = llm.bind_tools(tools, )
    def call_model(message, config):
        sys_msg = config.get("configurable", {}).get("system_message", system_message)
        history = config.get("configurable", {}).get("history", [])
        # if llm supports system messages, add it to the messages, otherwise, ignore it
        messages = [SystemMessage(content=sys_msg)] + history + message
        response = model.invoke(messages)
        return response
    workflow = MessageGraph()
    workflow.add_node("agent", call_model)
    workflow.add_node("action", ToolNode(tools))

    workflow.set_entry_point("agent")
    
    # Conditional agent -> action OR agent -> END
    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )

    # Always transition `action` -> `agent`
    workflow.add_edge("action", "agent")
    if use_memory:
        memory = MemorySaver() # Here we only save in-memory

        # Setting the interrupt means that any time an action is called, the machine will stop
        app = workflow.compile(checkpointer=memory)
    else:
        app = workflow.compile()

    return app


def get_map_reduce_agent(llm, tools):
    model = llm.bind_tools(tools)
    
    class Strategy(BaseModel):
        actions: list[str]

    def call_model(messages, config):
        if "system_message" in config["configurable"]:
            messages = [SystemMessage(content=config["configurable"]["system_message"])] + messages
        if "history" in config["configurable"]:
            messages = config["configurable"]["history"] + messages
        response = model.invoke(messages)
        return response
    
    def define_strategy(state):
        query = state["query"]
        prompt_template = """
        define a strategy to answer the question: {query}
        your strategy should be a sequence of actions that will help you to answer the question.
        each action should be a tool that you have at your disposal.
        you can use the tools to filter the documents, query the database, or create new columns.
        your strategy should be a comma separated list of actions to be executed in order to answer the question.
        """
        prompt = prompt_template.format(query=query)
        response = llm.with_structured_output(Strategy).invoke(prompt)
        return {"actions": response.actions}
        
    def select_documents(state):
        action = state["action"]
        prompt_template = """
        given an action: {action} that you defined in the strategy, get the documents that are relevant to the action.
        you need use sql query that will filter the documents based on the action you defined.
        """
        prompt = prompt_template.format(action=action)
        response = llm.invoke(prompt)
        conn = sqlite3.connect("indexis.db")
        res = conn.execute(response).fetchall()
        
        conn.close()
        return {"response": res}
    
    def query_document(state):
        document = state["document"]
        action = state["action"]
        prompt_template = """
        given the documents: {document}, and the action: {action}, query the documents to get the relevant information.
        you need to use the action you defined in the strategy to query the documents.
        """
        prompt = prompt_template.format(document=document, action=action)
        response = llm.invoke(prompt)
        return {"response": response}
        
    graph = StateGraph()
    
    graph.add_node("agent", define_strategy)
    graph.add_node("document_selection", select_documents)
    # graph.add_node("action", ToolNode(tools))
    

def agentic_question_ansewr(question, db, doc_id, llm):
    # create filtered retriever    
    db_size = db.index.ntotal
    retriever = db.as_retriever(search_kwargs={"k": db_size, "fetch_k": db_size})
    filtered_retriever = FilteredRetriever(vectorstore=retriever, doc_id=doc_id, k=3)

    retriever_tool = create_retriever_tool(
        filtered_retriever,
        "search_in_documents",
        """Using semantic similarity, retrieves some chuncks from the construction plan document that have the closest embeddings to the input query. The query should be semantically close to your target documents. Use the affirmative form rather than a question.
        תוך שימוש בדמיון סמנטי, מחזיר קטעים ממסמך תוכנית הבנייה שיש להן דמיון סמנטי קרוב ביותר לשאילתת הקלט. השאילתה חייבת להיות קרובה סמנטית למסמכי המטרה שלך. יש להשתמש בצורת הפועל ולא בשאלה."""
    )
    tools = [retriever_tool]
    app = get_agent(llm, tools)
    
    thread = {
        "configurable": {
            "thread_id": doc_id,
            "system_message": """
                Using the information contained in your knowledge base about urban construction plan, which you can access with the 'retriever' tool,
                give an answer to the question below. 
                Respond only to the question asked, response should be concise and relevant to the question without any additional information.
                Do not answer if your'e not sure about the answer, and do not provide information that could be misleading.
                If you cannot find information, do not give up and try calling your retriever again with different arguments!
                Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries.
                Your queries should not be questions but affirmative form sentences: e.g. rather than "מהו שטח הדירה?", query should be "שטח הדירה".
            """
            }
        }
    
    result = list(app.stream(question, thread, stream_mode="values"))
    
    # log the results
    res_str = "\n================\n".join([f"{r[-1].content}{r[-1].additional_kwargs.get('function_call', '')}" for r in result])
    logger.info(f"agentic_question_ansewr: {res_str}")
    return result[-1][-1]


def stream_agent(agent, system_message, question, thread_id, history=[], model_name="hf-gemma-2-9b-it", sql_db="indexis.db", vec_db="faiss_c512_o128.index", documents=None, **kwargs):
    if documents is None:
        conn = sqlite3.connect(sql_db)
        documents = pd.read_sql("SELECT id FROM plans", conn)['id'].tolist()
        conn.close()
    thread = {
        "configurable": {
            "thread_id": thread_id,
            "conversation_id": thread_id,
            "system_message": system_message,
            "history": history,
            "model_name": model_name,
            "sql_db": sql_db,
            "vec_db": vec_db,
            "documents": documents,
            **kwargs
            }
        }
    
    # create a new table in database named 'plans_{thread_id}' with the same columns as the original table and only the relevant documents
    add_plans_index(sql_db, documents, thread_id)
    for event in agent.stream(question, thread, stream_mode="values"):
        logger.info(f"stream_agent: {event[-1].content}{event[-1].additional_kwargs.get('function_call', '')}")
        yield event[-1]

def get_tools():
    tools = [get_available_columns, query_database, create_numeric_column, create_boolean_column, create_textual_column, create_date_column ,display_svg]
    return tools

if __name__ == '__main__':
    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename=f'logs/{curr_time}_agent.log', level=logging.INFO)
    
    #if db does not exist, create it
    sql_db_name = os.environ.get("SQL_DB", "indexis.db")
    conn = sqlite3.connect(sql_db_name)
    if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='plans'").fetchone():
        data = pd.read_csv('shpan.csv')
        create_plans_index(sql_db_name, data, limit=50)
    
    # embedding_model = get_embedding_model()
    # data = pd.read_csv('shpan.csv') # read the data
    # db = get_db(data,"faiss_c512_o128.index") 
    # retriever = db.as_retriever()
    # retriever_tool = create_retriever_tool(
    #     retriever,
    #     "search_in_documents",
    #     "מחפש במשמכי תוכניות הבינוי לפי שאילתת משתמש",
    # )

    # Define a new graph
    
    tools = get_tools()
    app = get_agent(get_vertex_llm(), tools)
    
    # plot the graph
    app.get_graph()
    
    # Run the graph
    # for msg in stream_agent(app, system_message, "מהו שטח השצ\"פ האקטיבי הכולל בשכונת בית הכרם?", "1"):
    #     # msg.pretty_print()
    #     print({
    #         'type': type(msg).__name__,
    #         'content': msg.content,
    #         'tool_calls': [] if type(msg).__name__ != 'AIMessage' else msg.tool_calls
    #     })
    # thread = {
    #     "configurable": {
    #         "thread_id": "1",
    #         # "system_message": "You are a very smart agent designed to help users with urban planning related questions, the user will ask you a question and you will use your tools to find the answer. your answer will be in hebrew."
    #         # "system_message": "אתה סוכן חכם אשר יודע להשיב על שאלות בנושא בינוי עירוני בהיסתמך על מסמכים מבסיס הנתונים, אתה יכול להשתמש בכלים הנתונים לך על מנת להשיג את מטרותיך. בעזרת הכלים אתה יכול לבצע סינון על המסמכים ולקבל את המסמכים הרלוונטיים לך וכך לתת מענה אמין לשאלות"
    #         "system_message": """
    #             you are a professional agent that can answer questions about urban planning based on documents from the database.
    #             the documents in the database are urban planning documents and can contain information about various topics such as building permits, zoning plans, and more. so when you get a question, be sure to filter the relevant documents and provide the information needed.
    #             you can use the tools at your disposal to achieve your goals.
    #             in order to filter documents you can see the existing columns in the database and create new columns if needed.
    #             you can also query the database to get the relevant information.
    #             be sure to check the available columns in the plans table and create new columns if needed.
    #             the database contains plans table with urban planning documents and you could add new columns to the table to get more information.
    #             the questions and answers should be in hebrew.
    #             you can access any information you may need by creating new columns in the database based on the description you provide. 
    #             if you encounter any error regarding missing columns, you can create new columns based on the description you provide.
    #             only call one tool at a time and wait for the response before calling the next tool.
    #         """
    #         }
    #     }
    
    # for event in app.stream("מהו שטח השצ\"פ האקטיבי הכולל בשכונת בית הכרם?", thread, stream_mode="values"):
    #     # for v in event:
    #     event[-1].pretty_print()

    # llm = get_vertex_llm()
    # llm = get_huggingface_chat("google/gemma-2-9b-it")
    # llm = get_huggingface_chat("dicta-il/dictalm2.0-instruct")
    # llm = get_huggingface_chat("Qwen/Qwen2-7B-Instruct")
    # answer = agentic_question_ansewr("מהי הכתובת המדוברת בתוכנית?",db, "101-0977165",llm)[-1][-1].content
    # print(answer)
