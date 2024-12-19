from flask import Flask, request, jsonify, render_template, session
from retrieval import retrieve_documents
from generation import generate_response
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session management

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/retrieve', methods=['POST'])
def retrieve():
    query = request.json.get('query')
    documents = retrieve_documents(query)
    return jsonify(documents)

@app.route('/generate', methods=['POST'])
def generate():
    context = request.json.get('context')
    retrieved_data = request.json.get('retrieved_data', [])

    # Get the conversation history from the session
    conversation_history = session.get('conversation_history', [])
    conversation_history.append(context)
    session['conversation_history'] = conversation_history

    response = generate_response(context, retrieved_data, conversation_history)
    return jsonify(response)

@app.route('/reset', methods=['POST'])
def reset():
    session.pop('conversation_history', None)
    return jsonify({"message": "Conversation history reset."})

if __name__ == '__main__':
    app.run(debug=True)