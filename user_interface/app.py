from flask import Flask, render_template, request, jsonify, session, send_from_directory, Response, stream_with_context, redirect, url_for, flash
from flask_cors import CORS, cross_origin
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import sqlite3
import dotenv
import json
import logging
import sys
import os
import re
import pandas as pd
from datetime import datetime
from sql_models import db as app_db, User, Conversation, UserConfig, ConversationMessage, AnonymousUser, ConversationConfig, ContactMessage,SystemConfig

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from RAG import (get_db,
                get_top_k_by_count,
                get_top_k,
                get_answer,
                get_answer_foreach_doc,
                title_from_quary,
                diffrent_question_rephrasing,
                stream_chat,
                question_from_description,
                query_from_description,
                queries_from_description,
                BatchCallback,
                TextualOutputParser,
                NumericOutputParser, 
                BooleanOutputParser, 
                DateOutputParser)
from agent import get_agent, get_tools, stream_agent, system_message as default_system_message, create_plans_index, insert_new_column
from file_utils import get_docs_for_plan, pdf_bin_to_text, clean_and_split
from models import get_embedding_model, get_llm, model_map
from download_program_doc import extract_xplan_attributes

dotenv.load_dotenv()
logger = logging.getLogger(__name__)
curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(filename=f'logs/{curr_time}_flask.log', level=logging.DEBUG)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('APP_SECRET_KEY')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app_db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.anonymous_user = AnonymousUser
sql_db_name = os.environ.get("SQL_DB", "indexis.db")
vec_db_name = os.environ.get("VEC_DB", "faiss_c512_o128.index")

data = pd.read_csv("shpan.csv")
vec_db = get_db(data, f"{root_path}/{vec_db_name}", chunk_size=app.config.get('CHUNK_SIZE', 512), chunk_overlap=app.config.get('CHUNK_OVERLAP', 128))
logger.info("vector db loaded")
# create plans database if it doesn't exist
with sqlite3.connect(f"{root_path}/{sql_db_name}") as conn:
    if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='plans'").fetchone():
        create_plans_index(f"{root_path}/{sql_db_name}", data=pd.read_csv(f"{root_path}/shpan.csv"))
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
    msg_dict = json.loads(json_message)
    obj = obj_map[msg_dict['type']]
    return obj(**msg_dict)

@login_manager.user_loader
def load_user(user_id):
    return app_db.session.get(User, user_id)

@app.route('/')
def index():
    # create admin user if it doesn't exist
    if not User.query.filter_by(username='admin').first():
        hashed_password = generate_password_hash('admin', method="scrypt")
        new_user = User(username='admin', email='admin@admin.com', password=hashed_password, permission='admin')
        app_db.session.add(new_user)
        app_db.session.commit()
        print('Admin user created')

    conn = sqlite3.connect(f"{root_path}/{sql_db_name}")
    # cols = conn.execute("PRAGMA table_info(plans)").fetchall()
    cols = conn.execute("SELECT name, type FROM columns").fetchall()
    cols = {n: {'INTEGER': 'int', 'TEXT': 'str', 'FLOAT': 'float', 'BOOLEAN': 'boolean', 'TIMESTAMP': 'datetime64[ns]'}[t] for n,t in cols}
    data = pd.read_sql("SELECT * FROM plans", conn, dtype=cols).drop(columns=['content']).to_dict(orient='records')

    columns = pd.read_sql("SELECT * FROM columns", conn).to_dict(orient='records')
    models = list(model_map.keys())
    return render_template('index.html', user=current_user, documents=data, models=models, columns=columns)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password, method="scrypt")
        new_user = User(username=username, email=email, password=hashed_password)
        app_db.session.add(new_user)
        app_db.session.commit()
        login_user(new_user)
        flash('Registration Successful', 'success')
        return redirect(url_for('index'))
    return render_template('register.html', user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', user=current_user)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/about', methods=['GET', 'POST'])
def about():
    about_config = SystemConfig.query.filter_by(key='about').first()
    if request.method == 'POST':
        about_text = request.form.get('about_text')
        if about_config:
            about_config.value = about_text
        else:
            about_config = SystemConfig(key='about', value=about_text)
            app_db.session.add(about_config)
        app_db.session.commit()
        return render_template('about.html', user=current_user, about_text=about_text)
    if about_config:
        about_text = about_config.value
    else:
        about_text = "UrbanGPT הוא כלי מהפכני שבא לתת מענה לבעית תכנון הבינוי העירוני."

    return render_template('about.html', user=current_user, about_text=about_text)

@app.route('/contact', methods=['GET', 'POST', 'DELETE'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        # send email to the admin
        new_message = ContactMessage(name=name, email=email, content=message)
        app_db.session.add(new_message)
        app_db.session.commit()
        flash('Your message has been sent', 'success')
        return redirect(url_for('contact'))
    elif request.method == 'DELETE':
        id = request.json.get('id')
        ContactMessage.query.filter_by(id=id).delete()
        app_db.session.commit()
        return jsonify({'status': 'ok'})
    elif request.method == 'GET':
        if current_user.is_authenticated and current_user.permission == 'admin':
            messages = ContactMessage.query.all()
            return render_template('contact.html', user=current_user, messages=messages)
        return render_template('contact.html', user=current_user)

@app.route('/available_plans')
def available_plans():
    plans = []
    # get all available plans from the database
    plans = {doc.metadata['doc_id'] for k,doc in vec_db.docstore._dict.items()}
    return jsonify(list(plans))

@app.route('/available_models')
def available_models():
    return jsonify(list(model_map.keys()))

@stream_with_context
def json_stream_agent(llm, system_message, question, agent_id, history, conversation, **kwargs):
    agent = get_agent(llm, get_tools())
    for message in stream_agent(agent, system_message, question, agent_id, history, **kwargs):
        serialized_message = message_to_json(message)
        if message.type != 'human':
            conversation_message = ConversationMessage(conversation_id=conversation.id, sender=message.type, content=serialized_message)
            app_db.session.add(conversation_message)
            app_db.session.commit()
        yield f"event: message\ndata: {json.dumps({'conversation_id':conversation.id, 'title':conversation.title, 'message':serialized_message, 'mode': 'agent'})}\n\n"
    yield f"event: end\n\n"

@stream_with_context
def json_stream_chat(data, messages, llm, conversation):
    whole_message = ""
    for message in stream_chat(data, messages, llm):
        whole_message += message.content
        msg = {'conversation_id':conversation.id,
                'title':conversation.title,
                'message_id':message.id,
                'message':message_to_json(message),
                'mode': 'chat'}
        yield f"event: message\ndata: {json.dumps(msg)}\n\n"
    ai_message = AIMessage(whole_message)
    conversation_message = ConversationMessage(conversation_id=conversation.id, sender=ai_message.type, content=message_to_json(ai_message))
    app_db.session.add(conversation_message)
    app_db.session.commit()
    yield f"event: end\n\n"

@app.route('/chat', methods=['GET', 'POST', 'DELETE'])
def chat():
    if request.method == 'POST':
        # create a new conversation with the given documents and model
        conversation = Conversation(user_id=current_user.id)
        app_db.session.add(conversation)
        app_db.session.commit()
        for key, value in request.get_json().items():
            config = ConversationConfig(conversation_id=conversation.id, key=key, value=json.dumps(value))
            app_db.session.add(config)
        app_db.session.commit()
        return jsonify({'conversation_id': conversation.id})
    elif request.method == 'GET':
        conversation_id = request.args.get('conversation_id')
        conversation = Conversation.query.filter_by(id=conversation_id).first()
        attributes =  {a.key: json.loads(a.value) for a in ConversationConfig.query.filter_by(conversation_id=conversation_id).all()}
        if current_user.id != Conversation.query.filter_by(id=conversation_id).first().user_id:
            return jsonify({'error': 'conversation not found'})
        
        # add the question to the conversation
        question = request.args.get('question')
        new_message = ConversationMessage(conversation_id=conversation.id, sender="human", content=message_to_json(HumanMessage(question)))
        app_db.session.add(new_message)
        app_db.session.commit()
        
        history = ConversationMessage.query.filter_by(conversation_id=conversation_id).order_by(ConversationMessage.timestamp).all()
        history = [json_to_message(m.content) for m in history]
        llm = get_llm(attributes.get('model', _get_attr('model')))
        document_list = attributes.get('documents')
        mode = attributes.get('mode', 'agent')
        # if conversation has no title, set it according to the question
        if conversation.title is None:
            conversation.title = title_from_quary(question, llm)
            app_db.session.add(conversation)
            app_db.session.commit()
        
        if mode == 'agent':
            system_message = attributes.get('system_message', _get_attr('system_message'))
            doc_model = attributes.get('doc_llm', _get_attr('doc_llm'))
            num_chunks = attributes.get('num_chunks', _get_attr('num_chunks'))
            return Response(json_stream_agent(llm,
                                            system_message,
                                            question,
                                            conversation_id,
                                            history,
                                            conversation,
                                            model_name=doc_model,
                                            num_chunks=num_chunks,
                                            documents=document_list), content_type='text/event-stream')
        elif mode == 'context_chat':
            conn = sqlite3.connect(f"{root_path}/{sql_db_name}")
            docs_str = ", ".join([f'"{doc}"' for doc in document_list])
            data = pd.read_sql(f"SELECT id, content FROM plans WHERE id IN ({docs_str})", conn)
            return Response(json_stream_chat(data, history, llm, conversation), content_type='text/event-stream')
        else:
            print("mode", mode, "is not supported")
            return jsonify({'error': 'mode not supported'})
    elif request.method == 'DELETE':
        conversation_id = request.args.get('conversation_id')
        Conversation.query.filter_by(id=conversation_id).delete()
        ConversationMessage.query.filter_by(conversation_id=conversation_id).delete()
        ConversationConfig.query.filter_by(conversation_id=conversation_id).delete()
        app_db.session.commit()
        return jsonify({'status': 'ok'})

@app.route('/clear_data')
@login_required
def clear_database():
    if current_user.permission != 'admin':
        return jsonify({'status': 'error', 'message': 'permission denied'})
    sql_db_name = os.environ.get("SQL_DB", "indexis.db")
    with sqlite3.connect(f"{root_path}/{sql_db_name}") as conn:
        # delete all columns in plans that are not in ['id', 'name', 'content', 'receiving_date']
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN TRANSACTION")
        conn.execute("CREATE TABLE plans_backup AS SELECT * FROM plans")
        conn.execute("DROP TABLE plans")
        conn.execute("CREATE TABLE plans (id TEXT PRIMARY KEY, name TEXT, content TEXT, receiving_date TEXT)")
        conn.execute("INSERT INTO plans (id, name, content, receiving_date) SELECT id, name, content, receiving_date FROM plans_backup")
        conn.execute("DROP TABLE plans_backup")
        conn.execute("DELETE FROM columns WHERE name NOT IN ('id', 'name', 'receiving_date')")
        
        # list all tables in the database
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        for table in tables:
            if table[0] not in ['plans', 'columns']:
                conn.execute(f"DROP TABLE {table[0]}")
                logger.info(f"dropped table {table[0]}")
        conn.commit()
    return jsonify({'status': 'ok'})

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.permission != 'admin':
        return jsonify({'status': 'error', 'message': 'permission denied'})
    
    with sqlite3.connect(f"{root_path}/{sql_db_name}") as conn:
        cols = conn.execute("SELECT name, type FROM columns").fetchall()
        cols = {col[0]: {'INTEGER': 'int', 'TEXT': 'str', 'FLOAT': 'float', 'BOOLEAN': 'boolean',  'TIMESTAMP': 'datetime64[ns]'}[col[1]] for col in cols}
        doc_df = pd.read_sql("SELECT * FROM plans", conn, dtype=cols).drop(columns=['content'])
        columns = pd.read_sql("SELECT * FROM columns", conn).to_dict(orient='records')
    documents = doc_df.to_dict(orient='records')
    
    return render_template('dashboard.html', user=current_user, documents=documents, columns=columns, models=list(model_map.keys()))

@app.route('/document/<path:doc_id>', methods=['GET', 'DELETE', 'PUT'])
@app.route('/document', methods=['POST'])
@login_required
def document(doc_id=None):
    if current_user.permission != 'admin':
        return jsonify({'status': 'error', 'message': 'permission denied'})
    if request.method == 'GET':
        # return the document pdf file
        files = os.listdir(f'out/{doc_id}')
        with open(f'out/{doc_id}/{files[0]}', 'rb') as file:
            return Response(file.read(), content_type='application/pdf')
    elif request.method == 'POST':
        # upload the document pdf file
        doc = request.files['document']
        llm_name = request.form.get('model', _get_attr('model'))
        num_docs = request.form.get('num_docs', 3)
        
        # get the document id and other metadata
        doc_txt = pdf_bin_to_text(doc)
        doc_id = re.search(r'תכנית מס\' ([0-9]{3}-[0-9]{7}|[^\n]+)', doc_txt.split('\n')[2]).group(1)
        doc_metadata = extract_xplan_attributes(doc_id)
        os.makedirs(f'{root_path}/out/{doc_id}', exist_ok=True)
        doc.save(f'{root_path}/out/{doc_id}/{doc.filename}', )
        
        # create a new vector db for the document
        embedding_model = get_embedding_model()
        documents = clean_and_split(doc_txt, doc_id, chunk_size=app.config.get('CHUNK_SIZE', 512), chunk_overlap=app.config.get('CHUNK_OVERLAP', 128))
        pl_vec_db = FAISS.from_documents(documents, embedding_model)
        doc_values = {}
        with sqlite3.connect(f"{root_path}/{sql_db_name}") as conn:
            required_columns = conn.execute("SELECT * FROM columns WHERE name NOT IN ('id', 'name', 'content', 'receiving_date')").fetchall()
            
            for name, type, desc, question, queries in required_columns:
                # get the answer for the question
                parser = {
                    'TEXT': TextualOutputParser,
                    'FLOAT': NumericOutputParser,
                    'BOOLEAN': BooleanOutputParser,
                    'TIMESTAMP': DateOutputParser
                }[type]()
                
                
                chunks = get_top_k(queries, pl_vec_db, doc_id, k=num_docs)
                res = get_answer(question, chunks, get_llm(llm_name), parser)
                converted_res = {
                    'TEXT': str,
                    'FLOAT': float,
                    'BOOLEAN': bool,
                    'TIMESTAMP': datetime.fromtimestamp
                }[type](res)
                doc_values[name] = converted_res
            cols_titles = ['id', 'name', 'content', 'receiving_date'] + list(doc_values.keys())

            cols_values = [doc_id, doc_metadata['pl_name'], doc_txt, doc_metadata['receiving_date']*1e6] + list(doc_values.values())
            vals_placeholder = ', '.join(['?']*len(cols_titles))
            conn.execute(f"INSERT INTO plans ({', '.join(cols_titles)}) VALUES ({vals_placeholder})", cols_values)
            conn.commit()
            # marge vector store with the main vector store
        vec_db.merge_from(pl_vec_db)
        vec_db.save_local(f"{root_path}/{vec_db_name}")
        return jsonify({'status': 'ok','success': True, 'doc_id': doc_id, 'data': {t:v for t,v in zip(cols_titles, cols_values)}})
    elif request.method == 'DELETE':
        # delete the document from the database
        with sqlite3.connect(f"{root_path}/{sql_db_name}") as conn:
            conn.execute("DELETE FROM plans WHERE id=?", (doc_id,))
            conn.commit()
    elif request.method == 'PUT':
        # update the document in the database
        conn = sqlite3.connect(f"{root_path}/{sql_db_name}")
        attr = request.get_json()
        print(attr)
        key_str = ", ".join([f"{key}=?" for key, _ in attr.items()])
        conn.execute(f"UPDATE plans SET {key_str} WHERE id=?", (*attr.values(), doc_id))
        conn.commit()
        
    
    return jsonify({'status': 'ok', 'success': True})
    # doc_id = request.form.get('doc_id')
    # doc_name = request.form.get('doc_name')
    # doc_content = request.form.get('doc_content')
    # receiving_date = request.form.get('receiving_date')
    # conn = sqlite3.connect(sql_db_name)
    # conn.execute("INSERT INTO plans (id, name, content, receiving_date) VALUES (?, ?, ?, ?)", (doc_id, doc_name, doc_content, receiving_date))
    # conn.commit()
    # return jsonify({'status': 'ok'})

@app.route('/column', methods=['POST', 'DELETE'])
@login_required
def column():
    if current_user.permission != 'admin':
        return jsonify({'status': 'error', 'message': 'permission denied'})
    if request.method == 'POST':
        attributes = request.get_json()
        col_name = attributes['name']
        col_desc = attributes['description']
        col_type = attributes['type']
        llm = get_llm(attributes.get('model', _get_attr('model')))
        num_docs = int(attributes.get('num_retrieval_docs', 3))
        num_queries = int(attributes.get('num_queries', 1))
        is_full_doc = attributes.get('is_full_doc', False)
        question = attributes.get('question', None)
        search_query = attributes.get('search_query', None)
        parser = {
            'text': TextualOutputParser,
            'numeric': NumericOutputParser,
            'boolean': BooleanOutputParser,
            'date': DateOutputParser
        }[col_type]()
        if question is None:
            question = question_from_description(col_desc, answer_parser=parser, model=llm)
        if search_query is None:
            search_queries = queries_from_description(col_desc, model=llm, num_queries=num_queries)
        else:
            search_queries = [search_query]
        # conn = sqlite3.connect(sql_db_name)
        with sqlite3.connect(f"{root_path}/{sql_db_name}") as conn:
            index = pd.read_sql(f"SELECT id FROM plans", conn)
        try:
            res = get_answer_foreach_doc(question, vec_db, index['id'],num_docs=num_docs, model=llm, multiprocess=False, parser=parser, full_doc=is_full_doc, queries=search_queries)
            # add the column to the database
            insert_new_column(col_name, parser.get_type(), col_desc, res, question=question, search_queries=search_queries, db_name=f"{root_path}/{sql_db_name}", table_name='plans', col_table_name='columns')
            
            return jsonify({'status': 'ok', 'success': True, 'name': col_name, 'description': col_desc, 'type': col_type, 'data': res})
        except Exception as e:
            return jsonify({'status': 'error', 'success': False, 'message':str(e)})
            
            
    elif request.method == 'DELETE':
        col_name = request.get_json()['name']
        with sqlite3.connect(f"{root_path}/{sql_db_name}") as conn:
            conn.execute(f"ALTER TABLE plans DROP COLUMN {col_name}")
            conn.execute("DELETE FROM columns WHERE name=?", (col_name,))
            conn.commit()
        return jsonify({'status': 'ok'})

def _get_attr(key=None, default_value=None):
    default_attributes = {
        "system_message": default_system_message,
        "model": "openai-gpt-4o",
        "doc_llm": "openai-gpt-4o-mini",
        "num_chunks": 3
    }
    if current_user.is_authenticated:
        config = UserConfig.query.filter_by(user_id=current_user.id).all()
        attr = {c.key: c.value for c in config}
    attr = session.get('attributes', {})
    
    return {**default_attributes, **attr} if key is None else {**default_attributes, **attr}.get(key, default_value)

@app.route('/get_attributes')
def get_attributes():
    return jsonify(_get_attr())

@app.route('/set_attributes', methods=['POST'])
def set_attributes():
    # update session attributes with the new attributes
    attributes = request.get_json()
    if current_user.is_authenticated:
        for key, value in attributes.items():
            user_config = UserConfig.query.filter_by(user_id=current_user.id, key=key).first()
            if user_config:
                user_config.value = value
            else:
                user_config = UserConfig(user_id=current_user.id, key=key, value=value)
            app_db.session.add(user_config)
        app_db.session.commit()
    
    session['attributes'] = attributes
    session.modified = True
    return jsonify({'status': 'ok'})

@app.route('/get_conversations')
def get_conversations():
    if current_user.is_authenticated:
        conversations = Conversation.query.filter_by(user_id=current_user.id).all()
        messages = ConversationMessage.query.filter(ConversationMessage.conversation_id.in_([c.id for c in conversations])).group_by(ConversationMessage.conversation_id).all()
        return jsonify([{'id': c.id,
                        'title': c.title,
                        'messages': [m.content for m in ConversationMessage.query.filter_by(conversation_id=c.id).order_by(ConversationMessage.timestamp)]
                        } for c in conversations])
    return jsonify([])

# support static files in the 'static' directory
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    with app.app_context():
        app_db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
