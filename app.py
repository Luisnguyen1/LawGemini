from flask import Flask, request, jsonify, render_template
from retrieval import retrieve_documents
from generation import generate_response
import os
from pyngrok import ngrok

# Set your ngrok authentication token
ngrok.set_auth_token("2q69bqUTOpmclLDzfLIijJC6k4b_3CKLcWti7j55Xahu4Xw6h")

app = Flask(__name__)

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
    response = generate_response(context, retrieved_data)
    return jsonify(response)

if __name__ == '__main__':
    port = 5000
    public_url = ngrok.connect(port)
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
    app.run(debug=True, port=port)
