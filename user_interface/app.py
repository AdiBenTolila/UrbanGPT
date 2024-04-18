from flask import Flask, render_template, request, jsonify, session
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from RAG import create_db, get_answer, get_top_k_by_count, diffrent_question_rephrasing
from file_utils import get_docs_for_plan, pdf_bin_to_text
from models import get_embedding_model, get_llm
from langchain_community.vectorstores import FAISS
import uuid
import dotenv
dotenv.load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('APP_SECRET_KEY')
embedding_model = get_embedding_model()
db = FAISS.load_local("../faiss_index", embedding_model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/available_plans')
def available_plans():
    plans = []
    # get all available plans from the database
    plans = {doc.metadata['doc_id'] for k,doc in db.docstore._dict.items()}
    return jsonify(list(plans))

@app.route('/ask/<doc_id>', methods=['POST'])
def ask(doc_id=None):
    # get db subset for the requested doc_id
    question = request.get_json()['question']
    questions = diffrent_question_rephrasing(question, k=0)
    chunks = get_top_k_by_count(questions, db, k=3, pl_number=doc_id)
    output = get_answer(question,chunks, binary_answer=False)
    return jsonify({'message': output, 'sources': chunks})

if __name__ == '__main__':
    app.run(debug=True)
