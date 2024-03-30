from flask import Flask, render_template, request, jsonify, session
import sys
sys.path.append('..')
from RAG import create_db, get_answer, get_top_k_by_count, diffrent_question_rephrasing
from file_utils import get_docs_for_plan, pdf_bin_to_text
from models import get_embedding_model, get_llm
from langchain_community.vectorstores import FAISS
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'c6a8b2f9424d5303fd77f516ad2e763e'

databases = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    db = create_db(file)
    # if user_id not is session generate new id and save the db
    if session.get('user_id') is None:
        session['user_id'] = str(uuid.uuid4())

    databases[session['user_id']] = db

    # Code to process the uploaded document
    # For demonstration purposes, let's assume the document is processed successfully
    return jsonify({'message': 'Document uploaded successfully.'})

@app.route('/ask', methods=['POST'])
def ask():
    if session.get('user_id') is None:
        return jsonify({'message': 'Please upload a document first.'})
    
    db = databases.get(session['user_id'])
    question = request.get_json()['question']
    questions = diffrent_question_rephrasing(question)
    chunks = get_top_k_by_count(questions, db, k=3)
    output = get_answer(question,chunks)
    return jsonify({'message': output, 'sources': chunks})

if __name__ == '__main__':
    app.run(debug=True)
