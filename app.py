import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import PyPDF2
import chromadb
from chromadb.config import Settings
import requests
import re

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL = 'llama-3.3-70b-versatile'

chroma_client = chromadb.PersistentClient(path='chroma_db', settings=Settings(anonymized_telemetry=False))
collection = None

def get_or_create_collection():
    global collection
    try:
        collection = chroma_client.get_collection('documents')
    except:
        collection = chroma_client.create_collection('documents')

def extract_text_from_pdf(filepath):
    text = ''
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk) < 50:
            if chunks:
                chunks[-1] += ' ' + chunk
            else:
                chunks.append(chunk)
            break
        chunks.append(chunk)
        start = end - overlap
    return chunks

GREETINGS = {
    'hi', 'hello', 'hey', 'greetings', 'howdy', 'hola',
    'salam', 'assalamualaikum', 'adaab',
    'thank you', 'thanks', 'thankyou',
    'ok', 'okay', 'sure', 'alright',
    'good morning', 'good afternoon', 'good evening',
    'good night', 'good day',
}

def detect_greeting(text):
    text_lower = text.lower().strip()
    if text_lower in GREETINGS or text_lower.rstrip('.!?') in GREETINGS:
        return True
    first_word = text_lower.split()[0] if text_lower.split() else ''
    if first_word in {'hi', 'hello', 'hey', 'thanks', 'thank', 'greetings'}:
        return True
    short_phrases = [
        'how are you', 'how do you do', 'whats up', "what's up",
        'nice to meet', 'how is it going', "how's it going",
        'good to see', 'great to see', 'how have you been',
        'long time no see', 'pleasure', 'whats going on',
        'how are things', 'how you doing', 'how are you doing',
        'how is everything', 'how are you today',
    ]
    for phrase in short_phrases:
        if phrase in text_lower:
            return True
    return False

ROMAN_URDU_WORDS = {
    'kya', 'kyun', 'kaise', 'kahan', 'kisko', 'kisne', 'kaisa', 'kaun',
    'aap', 'aapka', 'aapki', 'aapko', 'aapne',
    'tum', 'tumhara', 'tumhari', 'tumhe', 'tumne',
    'hai', 'hain', 'nahi', 'nahin',
    'mera', 'meri', 'tere', 'teri', 'apna', 'apni', 'apne',
    'yeh', 'woh', 'chahiye', 'batao', 'bata', 'karo',
    'sakta', 'sakti', 'sakte',
    'bahut', 'thoda', 'thodi', 'thode', 'kuch', 'koi', 'sab',
    'shayad', 'zaroor', 'maloom', 'pata',
    'yaha', 'waha', 'idhar', 'udhar', 'abhi', 'phir', 'jab', 'toh',
    'achha', 'accha', 'sahi', 'galat',
    'baat', 'baare', 'samajh',
    'haan', 'hoga', 'hogi', 'honge',
    'chahiye', 'batana', 'bataye',
    'aata', 'aati', 'aate', 'jaata', 'jaati', 'jaate',
    'karta', 'karti', 'karte', 'kiya', 'kiye',
    'diya', 'diye', 'liya', 'liye',
    'hua', 'hui', 'huwe',
    'raha', 'rahi', 'rahe',
    'milta', 'milti', 'milte',
    'rakhta', 'rakhti', 'rakhte',
    'samajhte', 'samajhta', 'samajhti',
    'bolo', 'bolta', 'bolti', 'bolte',
    'jaana', 'dena', 'lena', 'karna',
    'hota', 'hoti', 'hote',
    'kabhi', 'kab', 'kitna', 'kitni', 'kitne',
    'kaun', 'konsa', 'konsi', 'kin',
    'unka', 'unki', 'inka', 'inki',
    'tumhara', 'tumhari', 'hamara', 'hamari', 'humara', 'humari',
    'inke', 'unke', 'unhe', 'inhe',
    'nikla', 'nikli', 'nikle', 'nikalta', 'nikalti', 'nikalte',
    'rakha', 'rakhi', 'rakhe',
    'kaho', 'kahi', 'kahte', 'kah', 'keh',
    'aaya', 'aaye', 'gaya', 'gaye', 'gayi',
}

def detect_language(text):
    if re.search(r'[\u0600-\u06FF\u0750-\u077F]', text):
        return 'urdu'
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    if words & ROMAN_URDU_WORDS:
        return 'roman-urdu'
    return 'english'

def query_groq(system_prompt, user_prompt):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Set it as an environment variable or in app.py")

    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': GROQ_MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'temperature': 0.3,
        'max_tokens': 1024
    }
    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")
    return resp.json()['choices'][0]['message']['content']

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'success': False, 'error': 'Only PDF files are supported'}), 400

    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        text = extract_text_from_pdf(filepath)
        if not text:
            return jsonify({'success': False, 'error': 'No text could be extracted from the PDF'}), 400

        chunks = chunk_text(text)

        get_or_create_collection()

        existing = collection.get(ids=[file_id])
        if existing and existing['ids']:
            collection.delete(ids=[file_id])

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({'file_id': file_id, 'filename': file.filename, 'chunk_index': i})

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        return jsonify({'success': True, 'file_id': file_id, 'filename': file.filename, 'chunks': len(chunks)})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'question' not in data or 'file_id' not in data:
        return jsonify({'success': False, 'error': 'Missing question or file_id'}), 400

    question = data['question']
    file_id = data['file_id']

    try:
        lang = detect_language(question)
        lang_suffix = '' if lang == 'english' else ' Answer in Roman Urdu.'

        if detect_greeting(question):
            greeting_prompt = (
                "You are a warm, friendly document assistant. The user is greeting you "
                "or making casual conversation. Respond with genuine warmth and enthusiasm. "
                "Politely mention you're here to help with their document and invite them "
                "to ask any questions about it. Keep it concise and natural."
                + lang_suffix
            )
            answer = query_groq(greeting_prompt, f"User said: {question}")
            return jsonify({'success': True, 'answer': answer, 'filename': ''})

        get_or_create_collection()

        results = collection.query(
            query_texts=[question],
            n_results=5,
            where={'file_id': file_id}
        )

        if not results or not results['documents'] or not results['documents'][0]:
            no_answer_prompt = (
                "You are a friendly document assistant. The user asked a question, but no "
                "relevant information was found in their uploaded document. Respond politely "
                "and warmly, letting them know you couldn't find an answer in the document "
                "and suggest they try rephrasing or asking something else. Be empathetic, "
                "not robotic."
                + lang_suffix
            )
            answer = query_groq(no_answer_prompt, f"User asked: {question}")
            return jsonify({'success': True, 'answer': answer, 'filename': ''})

        context = '\n\n'.join(results['documents'][0])
        filename = results['metadatas'][0][0]['filename'] if results['metadatas'] else 'document'

        lang_instruction = (
            "Answer in English."
            if lang == 'english'
            else "Answer in Roman Urdu."
        )

        system_prompt = (
            "You are a warm and helpful document analysis assistant. Answer the user's "
            "question using ONLY the provided context from the document. If the answer "
            "cannot be found in the context, politely say you couldn't find it and "
            "suggest rephrasing. Be friendly and natural, not robotic. Do not make up "
            "information or use external knowledge. "
            + lang_instruction
        )

        user_prompt = f"Document: {filename}\n\nContext from document:\n{context}\n\nQuestion: {question}"

        answer = query_groq(system_prompt, user_prompt)

        return jsonify({'success': True, 'answer': answer, 'filename': filename})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
