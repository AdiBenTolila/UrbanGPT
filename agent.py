from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
import pandas as pd
from models import get_embedding_model, get_llm,get_vertex_llm, get_gemini_llm, get_commandR_llm,get_claude_llm
from langchain_community.vectorstores import FAISS
from file_utils import get_text_for_plan
from RAG import get_answer_foreach_doc, TextualOutputParser, NumericOutputParser, BooleanOutputParser
import sqlite3
import dotenv
dotenv.load_dotenv()

db_name = "indexis_partial.db"

# conn = sqlite3.connect(db_name)


YELLOW_BACKGROUND = "\033[43m"
END_COLOR = "\033[0m"

def create_plans_index():
    conn = sqlite3.connect(db_name)
    df = pd.read_csv("shpan.csv")[:10]
    indexes = df[['pl_number', 'pl_name']]
    indexes.columns = ['id', 'name']
    indexes['content'] = indexes['id'].apply(lambda id: get_text_for_plan(id))
    indexes.to_sql("plans", conn, index=False)


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
    print(YELLOW_BACKGROUND,"Querying document with id:", document_id, "and query:", query, END_COLOR)
    
    return 


@tool
def get_available_tables()->str:
    """
    Get all available tables in the database.
    קבל את כל הטבלאות הזמינות במסד הנתונים
    """
    # read the index
    conn = sqlite3.connect(db_name)
    indexes = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    schemas = [pd.read_sql(f"PRAGMA table_info({table})", conn)[['name', 'type']].values for table in indexes["name"]]
    schemas = [[f"{name} ({type})" for name, type in schema] for schema in schemas]
    print(YELLOW_BACKGROUND, schemas, END_COLOR)
    res = "\n".join([f"{table} with columns: {', '.join(schema)}" for table, schema in zip(indexes["name"], schemas)])
    
    print(YELLOW_BACKGROUND,"Available indexes:", res, END_COLOR)
    return res

@tool
def query_database(query:str)->str:
    """
    send SQL query to the database to get relevant information. be shure to check available tables and create new table if needed.
    שאילתת מסד נתונים על ידי שאילתה מסויימת בכדי לקבל את התוכניות הרלוונטיות. יש לוודא קיום הטבלאות וליצור טבלאות חדשות במידת הצורך.
    """
    print(YELLOW_BACKGROUND,"Querying database with query:", query, END_COLOR)
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
def create_numeric_table(table_name:str, question:str)->str:
    """
    Create a new table in the database by asking a question on each document separately and saving the results in the table. your question should be in hebrew and request numeric value.
    צור טבלה חדשה במסד הנתונים על ידי שאלת שאלה על כל מסמך בנפרד ושמירת התוצאות בטבלה. השאלה שלך צריכה להיות בעברית ולבקש ערך מספרי.
    """
    print(YELLOW_BACKGROUND, "create_numeric_table:", table_name, question, END_COLOR)

    llm = get_gemini_llm(rate_limit=50)

    conn = sqlite3.connect(db_name)
    index = pd.read_sql("SELECT * FROM plans", conn)
    res = pd.DataFrame(get_answer_foreach_doc(question, db, index['id'], model=llm, multiprocess=False, parser=NumericOutputParser(), full_doc=True))[['pl_number', 'answer']]
    res.columns = ['id', 'value']
    
    print(YELLOW_BACKGROUND, res.head(), END_COLOR)
    try:
        res.to_sql(table_name, conn, index=False)
    except Exception as e:
        return str(e)
    return "table created successfully"

@tool
def create_boolean_table(table_name:str, question:str)->str:
    """
    Create a new table in the database by asking a question on each document separately and saving the results in the table. your question should be in hebrew and request yes or no answer.
    צור טבלה חדשה במסד הנתונים על ידי שאלת שאלה על כל מסמך בנפרד ושמירת התוצאות בטבלה. השאלה שלך צריכה להיות בעברית ולבקש ערך כן או לא.
    """
    print(YELLOW_BACKGROUND, "create_boolean_table:", table_name, question, END_COLOR)
    llm = get_gemini_llm(rate_limit=50)
    
    # read the index
    conn = sqlite3.connect(db_name)

    index = pd.read_sql("SELECT * FROM plans", conn)
    # create index name
    res = pd.DataFrame(get_answer_foreach_doc(question, db, index['id'], model=llm, multiprocess=False, parser=BooleanOutputParser()))[['pl_number', 'answer']]

    res.columns = ['id', 'value']
    print(YELLOW_BACKGROUND, res.head(), END_COLOR)
    try:
        res.to_sql(table_name, conn, index=False)
    except Exception as e:
        return str(e)
    return "table created successfully"

@tool
def create_textual_table(table_name:str, question:str)->str:
    """
    Create a new table in the database by asking a question on each document separately and saving the results in the table. your question should be in hebrew and request textual answer.
    צור טבלה חדשה במסד הנתונים על ידי שאלת שאלה על כל מסמך בנפרד ושמירת התוצאות בטבלה. השאלה שלך צריכה להיות בעברית ולבקש תשובה טקסטואלית.
    """
    print(YELLOW_BACKGROUND, "create_textual_table:", table_name, question, END_COLOR)

    llm = get_gemini_llm(rate_limit=50)

    conn = sqlite3.connect(db_name)
    index = pd.read_sql("SELECT * FROM plans", conn)
    res = pd.DataFrame(get_answer_foreach_doc(question, db, index['id'], model=llm, multiprocess=False, parser=TextualOutputParser()))[['pl_number', 'answer']]
    res.columns = ['id', 'value']
    
    print(YELLOW_BACKGROUND, res.head(), END_COLOR)
    try:
        res.to_sql(table_name, conn, index=False)
    except Exception as e:
        return str(e)
    return "table created successfully"

@tool
def filter_by_index(query:str, index:str)->str:
    """
    Filter documents by index.
    סנן מסמכים על פי אינדקס מסויים
    """
    print(YELLOW_BACKGROUND, "Filtering by index:", query, index, END_COLOR)
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
    
    tools = [get_available_tables, query_database, create_numeric_table, create_boolean_table, create_textual_table]
    # tools = [retriever_tool]
    app = get_agent(get_claude_llm(), tools)
    # Run the graph
    thread = {
        "configurable": {
            "thread_id": "1",
            # "system_message": "You are a very smart agent designed to help users with urban planning related questions, the user will ask you a question and you will use your tools to find the answer. your answer will be in hebrew."
            # "system_message": "אתה סוכן חכם אשר יודע להשיב על שאלות בנושא בינוי עירוני בהיסתמך על מסמכים מבסיס הנתונים, אתה יכול להשתמש בכלים הנתונים לך על מנת להשיג את מטרותיך. בעזרת הכלים אתה יכול לבצע סינון על המסמכים ולקבל את המסמכים הרלוונטיים לך וכך לתת מענה אמין לשאלות"
            "system_message": """
                you are a professional agent that can answer questions about urban planning based on documents from the database.
                you can use the tools at your disposal to achieve your goals.
                in order to filter documents you can see the existing tables in the database and create new tables if needed.
                you can also query the database to get the relevant information.
                be sure to check the available tables and fields before querying the database.
                the questions and answers should be in hebrew.
                you have the ability to answer any question that is asked about urban planning.
            """
            }
        }
    
    log = []
    for event in app.stream("כמה דירות מעל 100 מטר אושרו מאז 2010?", thread, stream_mode="values"):
        for v in event:
            
            v.pretty_print()
            # print(v.pretty_print())
            log.append(v)

    # # save log to file
    # with open("log.txt", "w") as f:
    #     for v in log:
    #         f.write(v.pretty_print())
    #         f.write("\n")

    # # prompt = "You are a very smart agent designed to help users with urban planning related questions."
    # prompt = ChatPromptTemplate.from_messages(
    # [
    #     (
    #         "system",
    #         "You are a very smart agent designed to help users with urban planning related questions, the user will ask you a question and you will use your tools to find the answer.",
    #     ),
    #     ("user", "{query}"),
    #     MessagesPlaceholder(variable_name="agent_scratchpad"),
    # ]
    # )


    # agent = create_tool_calling_agent(llm, tools, prompt)
    # # model_with_tools = llm.bind_tools(tools)


    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # res= agent_executor.invoke({"query": "give me all the documents related to urban planning in the street of uruguay"})
    # # res = model_with_tools.invoke("give me all the documents related to urban planning in the street of uruguay")


    # print(res)
