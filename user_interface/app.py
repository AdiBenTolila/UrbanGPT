from flask import Flask, render_template, request, jsonify, session, send_from_directory,Response, stream_with_context
from flask_cors import CORS, cross_origin
# from flask_session import Session
# from redis import Redis
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from RAG import *
from agent import get_agent, get_tools, stream_agent, system_message as default_system_message, create_plans_index
from file_utils import get_docs_for_plan, pdf_bin_to_text
from models import get_embedding_model, get_llm, model_map
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import uuid
import sqlite3
import dotenv
import json
import logging
dotenv.load_dotenv()
logger = logging.getLogger(__name__)
curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
logging.basicConfig(filename=f'logs/{curr_time}_flask.log', level=logging.DEBUG)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('APP_SECRET_KEY')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SESSION_TYPE'] = 'filesystem'

embedding_model = get_embedding_model()
db = FAISS.load_local(f"{root_path}/faiss_c512_o128.index", embedding_model, allow_dangerous_deserialization=True)
logger.info("vector db loaded")
# create plans database if it doesn't exist
sql_db_name = os.environ.get("SQL_DB", "indexis.db")
conn = sqlite3.connect(sql_db_name)
if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='plans'").fetchone():
    create_plans_index(sql_db_name, data=pd.read_csv(f"{root_path}/shpan.csv"), limit=50)
logger.info("plans db loaded")

def message_to_json(message):
    return json.dumps(message, default=lambda o: o.__dict__)

def json_to_message(json_message):
    obj_map = {
        'human': HumanMessage,
        'ai': AIMessage,
        'tool': ToolMessage,
        'system': SystemMessage
    }
    obj = obj_map[json_message['type']]
    return obj(**json_message)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/available_plans')
def available_plans():
    plans = []
    # get all available plans from the database
    plans = {doc.metadata['doc_id'] for k,doc in db.docstore._dict.items()}
    return jsonify(list(plans))

@app.route('/available_models')
def available_models():
    return jsonify(list(model_map.keys()))

@app.route('/ask/<doc_id>', methods=['POST'])
def ask(doc_id=None):
    # get db subset for the requested doc_id
    question = request.get_json()['question']
    num_rephrasings = request.get_json().get('num_rephrasings', 0)
    num_chunks = request.get_json().get('num_chunks', 3)
    questions = diffrent_question_rephrasing(question, k=num_rephrasings)
    chunks = get_top_k_by_count(questions, db, k=num_chunks, pl_number=doc_id)
    output = get_answer(question,chunks)
    return jsonify({'message': output, 'sources': chunks})

@app.route('/ask_foreach', methods=['POST'])
def ask_foreach():
    question = request.get_json()['question']
    num_rephrasings = request.get_json().get('num_rephrasings', 0)
    num_chunks = request.get_json().get('num_chunks', 3)
    # question = get_answers_for_all_docs_return_them(question)
    questions = diffrent_question_rephrasing(question, k=num_rephrasings)
    data = pd.read_csv(f'{root_path}/shpan.csv')

    answers = pd.DataFrame(get_answer_foreach_doc(question, db, data, num_rephrasings=num_rephrasings, num_docs=num_chunks))
    df = pd.DataFrame(columns=['pl_number', 'answer', 'chunks'])
    for i, row in answers.iterrows():
        #save just if the answer is yes else delete from answers
        if row['answer'] == 'כן':
            df = pd.concat([df, pd.DataFrame([[row['pl_number'], row['answer'], row['chunks']]], columns=['pl_number', 'answer', 'chunks'])])
        else:
            answers.drop(i, inplace=True)
    print(df)
    df.to_csv('answers_top_3_rephrasing_multiprocess_chat.csv', index=False)
    
    return jsonify({'columns': answers.columns.tolist(), 'data': answers.to_dict(orient='records')})
    # # db = get_db(data,f"{root_path}/faiss_index") 

    # chunks_for_docs={}
    # t_start = time.time()
    # count = 0
    # if os.path.exists("answers_top_3_rephrasing_multiprocess_chat.csv"):
    #     os.remove("answers_top_3_rephrasing_multiprocess_chat.csv")
    # df = pd.DataFrame(columns=['pl_number', 'answer', 'chunks'])

    # for i,row in data.iterrows():
    #     if count > 3:
    #         break
    #     count += 1
    #     chunks = get_top_k_by_count(questions, db, row['pl_number'], k=3)
    #     print(f"{i}s document:")
    #     print(chunks)
    #     chunks_for_docs[row['pl_number']] = chunks
    # # t_end = time.time()
    # # print(f"retrive time: {t_end-t_start}")
    # # t_end = time.time()
    # # print(f"generate time: {t_end-t_start}")
    # # t_start = time.time()
    # # results = get_answer(question, chunks_for_docs, binary_answer=True)
    
    #     results = get_answer(question, chunks_for_docs, binary_answer=True)
    #     print("results: ", results[::-1])
    #     for i, (pl_number, answer) in enumerate(zip(chunks_for_docs.keys(), results)):

    #         df = pd.DataFrame([[pl_number, answer, chunks_for_docs[pl_number]]], columns=['pl_number', 'answer', 'chunks'])
    #         # df = pd.concat([df, df1])
    #         df.to_csv('answers_top_3_rephrasing_multiprocess_chat.csv', index=False)

    df['pl_name'] = data['pl_name']
    df_filtered = df[df['answer'] == 'כן']
    print(df_filtered.to_dict(orient='records'))
    return jsonify({'columns': df_filtered.columns.tolist(), 'data': df_filtered.to_dict(orient='records')})

@app.route('/ask_agent')
def ask_agent():
    question = request.args.get('question')
    model_name = session.get('attributes', {}).get('model')
    llm = get_llm(model_name)
    agent = get_agent(llm, get_tools())
    logger.info(f"historical messages sequence: {' -> '.join([type(m).__name__ for m in session.get('history', [])])}")
    @stream_with_context
    def stream_agent_json(agent, system_message, question, agent_id, history):
        for message in stream_agent(agent, system_message, question, agent_id, history):
            serialized_message = message_to_json(message)
            yield f"event: message\ndata: {serialized_message}\n\n"
        yield f"event: end\n\n"
    deserialised_history = [json_to_message(m) for m in session.get('history', [])]
    system_message = session.get('attributes', {}).get('system_message', default_system_message)
    return Response(stream_agent_json(agent, system_message, question, "1", deserialised_history), content_type='text/event-stream')

@app.route('/submit_history', methods=['POST'])
def submit_history():
    history = request.get_json()['history']
    session['history'] = history
    session.modified = True
    return jsonify({'status': 'ok'})

@app.route('/clear_data')
def clear_database():
    sql_db_name = os.environ.get("SQL_DB", "indexis.db")
    conn = sqlite3.connect(sql_db_name)
    # delete all columns in plans that are not in ['id', 'name', 'content', 'receiving_date']
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE plans_backup AS SELECT * FROM plans")
    conn.execute("DROP TABLE plans")
    conn.execute("CREATE TABLE plans (id TEXT PRIMARY KEY, name TEXT, content TEXT, receiving_date TEXT)")
    conn.execute("INSERT INTO plans (id, name, content, receiving_date) SELECT id, name, content, receiving_date FROM plans_backup")
    conn.execute("DROP TABLE plans_backup")
    conn.execute("DELETE FROM columns WHERE name NOT IN ('id', 'name', 'receiving_date')")
    conn.commit()
    return jsonify({'status': 'ok'})

@app.route('/get_attributes')
def get_attributes():
    return jsonify(session.get('attributes', {
        "system_message": default_system_message,
        "model": "openai-gpt-4o-mini",
        "num_chunks": 3
    }))
    

@app.route('/set_attributes', methods=['POST'])
def set_attributes():
    # update session attributes with the new attributes
    attributes = request.get_json()
    session['attributes'] = attributes
    session.modified = True
    return jsonify({'status': 'ok'})

# support static files in the 'static' directory
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    logger.info("app started")
    # start redis server
    # os.system("redis-server")
    app.run(debug=True, host='0.0.0.0', port=5000)
