from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
import pandas as pd
from models import get_embedding_model, get_llm,get_vertex_llm, get_gemini_llm,get_claude_llm, get_huggingface_llm,get_llamaCpp_llm,get_huggingface_chat, get_openai_llm
from langchain_community.vectorstores import FAISS
from file_utils import get_text_for_plan
from RAG import get_answer_foreach_doc, TextualOutputParser, NumericOutputParser, BooleanOutputParser, DateOutputParser, question_from_description, get_db, query_from_description, FilteredRetriever
import sqlite3
import dotenv
import os
import logging
from datetime import datetime
import torch
import re
from multiprocessing import Lock


logger = logging.getLogger(__name__)
dotenv.load_dotenv()

system_message = """
            you are a professional agent that can answer questions about urban planning based on documents from the database.
            the documents in the database are urban planning documents and can contain information about various topics such as building permits, zoning plans, and more. so when you get a question, be sure to filter the relevant documents and provide the information needed.
            you can use the tools at your disposal to achieve your goals.
            in order to filter documents you can see the existing columns in the database and create new columns if needed.
            you can also query the database to get the relevant information.
            be sure to check the available columns in the plans table and create new columns if needed.
            the database contains plans table with urban planning documents and you could add new columns to the table to get more information.
            the questions and answers should be in hebrew.
            you can access any information you may need by creating new columns in the database based on the description you provide. 
            if you encounter any error regarding missing columns, you can create new columns based on the description you provide.
            only call one tool at a time and wait for the response before calling the next tool.
            base your final answer on the information you got from the tools you called.
        """


def create_plans_index(db_name, data, limit=None):
    conn = sqlite3.connect(db_name)
    
    indexes = data[['pl_number', 'pl_name', 'receiving_date']].copy()
    indexes.columns = ['id', 'name', 'receiving_date']
    # convert date to ISO8601 notation: YYYY-MM-DD HH:MM:SS.SSS
    indexes['receiving_date'] = pd.to_datetime(indexes['receiving_date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    indexes['content'] = indexes['id'].apply(lambda id: get_text_for_plan(id))
    #remove empty documents whene content is ""
    indexes = indexes[indexes['content'] != ""]
    if limit:
        indexes = indexes.head(limit)
    logger.info(f"create_plans_index with {len(indexes)} documents")
    indexes.to_sql("plans", conn, index=False, if_exists='replace')
    
    # drop the columns table if exists
    conn.execute("DROP TABLE IF EXISTS columns")
    conn.execute("CREATE TABLE columns (name TEXT, type TEXT, description TEXT)")
    conn.executemany("INSERT INTO columns (name, type, description) VALUES (?, ?, ?)", [
        ('id', 'TEXT', 'מספר מזהה של התוכנית'),
        ('name', 'TEXT', 'שם התוכנית'),
        ('receiving_date', 'TEXT', 'תאריך קבלת התוכנית'),
        # ('content', 'TEXT', 'מסמך התוכנית המלא, מכיל את כל המידע הקיים במסמך')
        ])
    conn.commit()
    conn.close()


# Define the function that determines whether to continue or not
def should_continue(messages):
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    else:
        return "action"

create_col_mutex = Lock()

@tool
def query_document(document_id:str, query:str)->str:
    """
    Query a document with a given question.
    שאל שאלה מסויימת על מסמך מסויים
    """
    
    pass

@tool
def get_available_columns()->str:
    """
    Get all available columns in the database.
    קבל את כל העמודות הזמינות במסד הנתונים
    """
    # read the index
    sql_db_name = os.environ.get("SQL_DB", "indexis.db")
    conn = sqlite3.connect(sql_db_name)
    columns = pd.read_sql("SELECT * FROM columns", conn)
    conn.close()
    res = "Table plans with the folowing columns:\n" + "\n".join([f"{name} ({type}): {description}" for name, type, description in columns.values])
    logger.info(f"get_available_columns: {res}")
    return res

@tool
def query_database(query:str)->str:
    """
    send SQL query to the database to get relevant information. be shure to check available columns and create new column if needed. the quary should be on 'plans' table. be sure to verify that the needed columns are created successfully. give your quary columns meaningful names so you can understand the results.
    שלח שאילתת SQL למסד הנתונים כדי לקבל מידע רלוונטי. וודא שיש לך עמודות זמינות וצור עמודה חדשה אם נדרש. השאילתא צריכה להיות על טבלת 'plans'. וודא שהעמודות הנדרשות נוצרו בהצלחה. תן לעמודות שלך שמות משמעותיים כך שתוכל להבין את התוצאות.
    """
    logger.info(f"query_database: {query}")
    # if query contains the \\[0-9]{3} pattern, return an error
    if re.search(r"\\[0-9]{3}", query):
        return "the query should not contain the pattern \\[0-9]{3}"
    sql_db_name = os.environ.get("SQL_DB", "indexis.db")
    conn = sqlite3.connect(sql_db_name)
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        return str(e)
    finally:
        conn.close()
    logger.info(f"query_database: {df}")
    # format the results to be shown as text
    results = df.to_dict(orient='records')
    results = f"the query to the database returned:\n" + "\n".join([", ".join([f"{k}: {v}" for k, v in d.items()]) for d in results])
    return results

@tool
def display_svg(svg:str)->str:
    """
    Display an SVG image in the chat.
    הצג תמונת SVG בצ'אט
    
    Args:
        svg (str): The SVG image as a string.
    """
    
    return svg.replace("\n", "")

def create_column(column_name, column_description, parser):
    # column name should be in english and without spaces, and the description should be in hebrew and describe the column intended value.
    if " " in column_name or not column_name.isascii():
        return "column name should be in english and without spaces"

    if re.search(r"\\[0-9]{3}", column_description):
        return "column description should not contain the pattern \\[0-9]{3}"

    with create_col_mutex:
        # llm = get_gemini_llm(rate_limit=50)
        # llm = get_openai_llm()
        # llm = get_huggingface_chat("dicta-il/dictalm2.0-instruct")
        # llm = get_llamaCpp_llm("models/dictalm2.0-instruct.Q4_K_M.gguf")
        # llm = get_huggingface_llm("Qwen/Qwen2-7B-Instruct")
        llm = get_huggingface_chat("google/gemma-2-9b-it")

        sql_db_name = os.environ.get("SQL_DB", "indexis.db")
        vec_db_name = os.environ.get("VEC_DB", "faiss_c512_o128.index")
        conn = sqlite3.connect(sql_db_name)
        data = pd.read_csv('shpan.csv') # read the data
        db = get_db(data, vec_db_name) # generate a question from the column description
        question = question_from_description(column_description, answer_parser=parser, model=llm)

        search_query = query_from_description(column_description, model=llm)

        logger.info(f"create_column: {column_name}, description: {column_description}, question: {question}")
        
        index = pd.read_sql("SELECT * FROM plans", conn)
        res = pd.DataFrame(get_answer_foreach_doc(question, db, index['id'], model=llm, multiprocess=False, parser=parser, full_doc=False, query=search_query))[['pl_number', 'answer']]
        res.columns = ['id', 'value']
        logger.info(f"inserting: {res[['value', 'id']]}")

        # remove rows with null values
        res = res[res['value'].notnull()].astype(str)
        
        if res.empty:
            logger.info("no results found for the question: " + question)
            return "could not create a column for the given description, try a different description or different strategy"
        
        # insert the results into the new column in plans table
        try:
            conn.execute(f"ALTER TABLE plans ADD COLUMN {column_name} {parser.get_type()}")
            conn.execute(f"INSERT INTO columns (name, type, description) VALUES ('{column_name}', '{parser.get_type()}', '{column_description}')")
            conn.executemany(f"UPDATE plans SET {column_name} = ? WHERE id = ?", res[['value', 'id']].to_records(index=False))
        except Exception as e:
            return str(e)
        finally:
            conn.commit()
            conn.close()
    return "column created successfully"

@tool
def create_numeric_column(column_name:str, column_description:str)->str:
    """
    Create a new numeric column for each urban planning document in the database by giving a description of the column. the column name should be in english and without spaces, and the description should be in hebrew and describe the column intended numeric value, also be sure to describe the units of the value if needed.
    צור עמודה מספרית חדשה לכל תוכנית בינוי עירוני במסד הנתונים על ידי תיאור לעמודה. שם העמודה צריך להיות באנגלית ובלי רווחים, והתיאור צריך להיות בעברית ולתאר את הערך המספרי הנדרש לעמודה, וגם לוודא שהיחידות של הערך מוגדרות בתיאור אם נדרש.
    """
    return create_column(column_name, column_description, NumericOutputParser())

@tool
def create_boolean_column(column_name:str, column_description:str)->str:
    """
    Create a new boolean column for each urban planning document in the database by giving a description of the column. the column name should be in english and without spaces, and the description should be in hebrew and describe the column intended boolean value.
    צור עמודה בוליאנית חדשה לכל תוכנית בינוי עירוני במסד הנתונים על ידי תיאור לעמודה. שם העמודה צריך להיות באנגלית ובלי רווחים, והתיאור צריך להיות בעברית ולתאר את הערך הבוליאני הנדרש לעמודה.
    """
    return create_column(column_name, column_description, BooleanOutputParser())

@tool
def create_textual_column(column_name:str, column_description:str)->str:
    """
    Create a new textual column for each urban planning document in the database by giving a description of the column. the column name should be in english and without spaces, and the description should be in hebrew and describe the column intended textual value.
    צור עמודה טקסטואלית חדשה לכל תוכנית בינוי עירוני במסד הנתונים על ידי תיאור לעמודה. שם העמודה צריך להיות באנגלית ובלי רווחים, והתיאור צריך להיות בעברית ולתאר את הערך הטקסטואלי הנדרש לעמודה.
    """
    return create_column(column_name, column_description, TextualOutputParser())

@tool
def create_date_column(column_name:str, column_description:str)->str:
    """
    Create a new date column for each urban planning document in the database by giving a description of the column. the column name should be in english and without spaces, and the description should be in hebrew and describe the column intended date value.
    צור עמודת תאריך חדשה לכל תוכנית בינוי עירוני במסד הנתונים על ידי תיאור לעמודה. שם העמודה צריך להיות באנגלית ובלי רווחים, והתיאור צריך להיות בעברית ולתאר את הערך התאריכי הנדרש לעמודה.
    """
    return create_column(column_name, column_description, DateOutputParser())

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
    def call_model(messages, config):
        
        if "system_message" in config["configurable"]:
            # messages.prepend(SystemMessage(content=config["configurable"]["system_message"]))
            messages = [SystemMessage(content=config["configurable"]["system_message"])] + messages
        if "history" in config["configurable"]:
            messages = config["configurable"]["history"] + messages
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
        memory = SqliteSaver.from_conn_string(":memory:") # Here we only save in-memory

        # Setting the interrupt means that any time an action is called, the machine will stop
        app = workflow.compile(checkpointer=memory)
    else:
        app = workflow.compile()

    return app

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
            "thread_id": "2",
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


def stream_agent(agent, system_message, question, thread_id, history=[]):
    thread = {
        "configurable": {
            "thread_id": thread_id,
            "system_message": system_message,
            "history": history
            }
        }
    for event in agent.stream(question, thread, stream_mode="values"):
        # event[-1].pretty_print()
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
    # Run the graph
    
    for msg in stream_agent(app, system_message, "מהו שטח השצ\"פ האקטיבי הכולל בשכונת בית הכרם?", "1"):
        # msg.pretty_print()
        print({
            'type': type(msg).__name__,
            'content': msg.content,
            'tool_calls': [] if type(msg).__name__ != 'AIMessage' else msg.tool_calls
        })
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
