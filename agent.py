from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
import pandas as pd
from models import get_embedding_model, get_llm,get_vertex_llm, get_gemini_llm, get_commandR_llm,get_claude_llm, get_huggingface_llm,get_llamaCpp_llm
from langchain_community.vectorstores import FAISS
from file_utils import get_text_for_plan
from RAG import get_answer_foreach_doc, TextualOutputParser, NumericOutputParser, BooleanOutputParser, DateOutputParser, question_from_description
import sqlite3
import dotenv
import logging
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

db_name = "indexis_partial.db"

def create_plans_index():
    conn = sqlite3.connect(db_name)
    df = pd.read_csv("shpan.csv")[:10]
    indexes = df[['pl_number', 'pl_name']]
    indexes.columns = ['id', 'name']
    indexes['content'] = indexes['id'].apply(lambda id: get_text_for_plan(id))
    indexes.to_sql("plans", conn, index=False)
    
    conn.execute("CREATE TABLE columns (name TEXT, type TEXT, description TEXT)")
    conn.executemany("INSERT INTO columns (name, type, description) VALUES (?, ?, ?)", [
        ('id', 'TEXT', 'מספר מזהה של התוכנית'),
        ('name', 'TEXT', 'שם התוכנית'),
        ('content', 'TEXT', 'מסמך התוכנית המלא, מכיל את כל המידע הקיים במסמך')])
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

def create_column(column_name, column_description, parser):    
    llm = get_gemini_llm(rate_limit=50)
    # llm = get_huggingface_llm("Qwen/Qwen2-1.5B-Instruct")
    # llm = get_llamaCpp_llm("models/dictalm2.0-instruct.Q4_K_M.gguf")
    conn = sqlite3.connect(db_name)

    # generate a question from the column description
    question = question_from_description(column_description, model=llm)
    
    logger.info(f"create_column: {column_name}, description: {column_description}, question: {question}")
    
    index = pd.read_sql("SELECT * FROM plans", conn)
    res = pd.DataFrame(get_answer_foreach_doc(question, db, index['id'], model=llm, multiprocess=False, parser=parser, full_doc=True))[['pl_number', 'answer']]
    res.columns = ['id', 'value']
    # insert the results into the new column in plans table
    try:
        conn.execute(f"ALTER TABLE plans ADD COLUMN {column_name} {parser.get_type()}")
        conn.execute(f"INSERT INTO columns (name, type, description) VALUES ('{column_name}', '{parser.get_type()}', '{column_description}')")
        conn.executemany(f"UPDATE plans SET {column_name} = ? WHERE id = ?", res[['value', 'id']].values)
    except Exception as e:
        return str(e)
    return "column created successfully"
    
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
    conn = sqlite3.connect(db_name)
    columns = pd.read_sql("SELECT * FROM columns", conn)
    res = "Table plans woth the folowing columns:\n" + "\n".join([f"{name} ({type}): {description}" for name, type, description in columns.values])
    logger.info(f"get_available_columns: {res}")
    return res

@tool
def query_database(query:str)->str:
    """
    send SQL query to the database to get relevant information. be shure to check available columns and create new column if needed. the quary should be on 'plans' table. be sure to verify that the needed columns are created successfully.
    שלח שאילתת SQL למסד הנתונים כדי לקבל מידע רלוונטי. וודא שיש לך עמודות זמינות וצור עמודה חדשה אם נדרש. השאילתא צריכה להיות על טבלת 'plans'. וודא שהעמודות הנדרשות נוצרו בהצלחה.
    """
    logger.info(f"query_database: {query}")
    conn = sqlite3.connect(db_name)
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        return str(e)
    
    # format the results to be human readable
    results = df.to_string()
    return results

# @tool
# def create_table(table_name:str, question:str)->str:
#     """
#     Create a new table in the database by asking a question on each document separately and saving the results in the table. your question should be in hebrew and request a specified format for the answer in order for the table to be usable.
#     צור טבלה חדשה במסד הנתונים על ידי שאלת שאלה על כל מסמך בנפרד ושמירת התוצאות בטבלה. השאלה שלך צריכה להיות בעברית ולבקש פורמט מסויים לתשובה כדי שהטבלה תהיה שימושית.
#     """
#     # print(YELLOW_BACKGROUND, "Creating table:", table_name, question, END_COLOR)
    
#     llm = get_gemini_llm(rate_limit=50)

#     # read the index
#     conn = sqlite3.connect(db_name)

#     index = pd.read_sql("SELECT * FROM plans", conn)
#     # create index name
#     res = pd.DataFrame(get_answer_foreach_doc(question, db, index['id'], model=llm, multiprocess=False))[['pl_number', 'answer']]

#     res.columns = ['id', 'value']
#     print(YELLOW_BACKGROUND, res.head(), END_COLOR)
#     try:
#         res.to_sql(table_name, conn, index=False)
#     except Exception as e:
#         return str(e)
#     return "table created successfully"

@tool
def create_numeric_column(column_name:str, column_description:str)->str:
    """
    Create a new numeric column for each urban planning document in the database by giving a description of the column. the column name should be in english and without spaces, and the description should be in hebrew and describe the column intended numeric value.
    צור עמודה מספרית חדשה לכל תוכנית בינוי עירוני במסד הנתונים על ידי תיאור לעמודה. שם העמודה צריך להיות באנגלית ובלי רווחים, והתיאור צריך להיות בעברית ולתאר את הערך המספרי הנדרש לעמודה.
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

@tool
def filter_by_index(query:str, index:str)->str:
    """
    Filter documents by index.
    סנן מסמכים על פי אינדקס מסויים
    """
    logger.info(f"filter_by_index: {query}, {index}")
    pass


def get_agent(llm, tools):
    model = llm.bind_tools(tools)
    def call_model(messages, config):
        if "system_message" in config["configurable"]:
            messages = [
                SystemMessage(content=config["configurable"]["system_message"])
            ] + messages
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
    
    # memory = SqliteSaver.from_conn_string(":memory:") # Here we only save in-memory

    # Setting the interrupt means that any time an action is called, the machine will stop
    # app = workflow.compile(checkpointer=memory)
    app = workflow.compile()

    return app

if __name__ == '__main__':
    logging.basicConfig(filename='agent.log', level=logging.INFO)
    
    #if db does not exist, create it
    conn = sqlite3.connect(db_name)
    if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='plans'").fetchone():
        create_plans_index()
    
    embedding_model = get_embedding_model()
    db = FAISS.load_local("faiss_index", embedding_model)
    retriever = db.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "search_in_documents",
        "מחפש במשמכי תוכניות הבינוי לפי שאילתת משתמש",
    )

    # Define a new graph
    
    tools = [get_available_columns, query_database, create_numeric_column, create_boolean_column, create_textual_column, create_date_column]
    # tools = [retriever_tool]
    app = get_agent(get_vertex_llm(), tools)
    # Run the graph
    thread = {
        "configurable": {
            "thread_id": "1",
            # "system_message": "You are a very smart agent designed to help users with urban planning related questions, the user will ask you a question and you will use your tools to find the answer. your answer will be in hebrew."
            # "system_message": "אתה סוכן חכם אשר יודע להשיב על שאלות בנושא בינוי עירוני בהיסתמך על מסמכים מבסיס הנתונים, אתה יכול להשתמש בכלים הנתונים לך על מנת להשיג את מטרותיך. בעזרת הכלים אתה יכול לבצע סינון על המסמכים ולקבל את המסמכים הרלוונטיים לך וכך לתת מענה אמין לשאלות"
            "system_message": """
                you are a professional agent that can answer questions about urban planning based on documents from the database.
                you can use the tools at your disposal to achieve your goals.
                in order to filter documents you can see the existing columns in the database and create new columns if needed.
                you can also query the database to get the relevant information.
                be sure to check the available columns in the plans table and create new columns if needed.
                the database contains plans table with urban planning documents and you could add new columns to the table to get more information.
                the questions and answers should be in hebrew.
                you can access any information you may need by creating new columns in the database based on the description you provide. 
                if you encounter any error regarding missing columns, you can create new columns based on the description you provide.
            """
            }
        }
    
    for event in app.stream("כמה דירות מעל 100 מטר אושרו מאז 2010?", thread, stream_mode="values"):
        for v in event:
            v.pretty_print()
