from flask import Flask, render_template, request, jsonify, session, send_from_directory
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from RAG import *
from file_utils import get_docs_for_plan, pdf_bin_to_text
from models import get_embedding_model, get_llm
from langchain_community.vectorstores import FAISS
import uuid
import dotenv
dotenv.load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('APP_SECRET_KEY')
embedding_model = get_embedding_model()
db = FAISS.load_local(f"../faiss_index", embedding_model)

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

@app.route('/ask_general', methods=['POST'])
def ask_general():
    question = request.get_json()['question']
    # question = get_answers_for_all_docs_return_them(question)
    questions = diffrent_question_rephrasing(question, k=0)
    data = pd.read_csv(f'{root_path}/shpan.csv')
    print(question )
    print(questions)

    answers = pd.DataFrame(get_answer_foreach_doc(question, db, data, num_rephrasings=0))
    print("answers coul: ", answers)
    print("answers: ", answers["answer"])
    print(answers.columns.tolist())
    print(answers.to_dict(orient='records'))
    print(answers.to_csv('answers_top_3_rephrasing_multiprocess_chat.csv', index=False))
    #save the answers to a csv file
    #creat data frame with the answers
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

# support static files in the 'static' directory
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
