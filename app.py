import os
import uuid
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import PyPDF2
import chromadb
from chromadb.config import Settings
import requests
import re
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = 'users.db'
FREE_QUESTION_LIMIT = 3
RESET_TOKEN_EXPIRY_HOURS = 1

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                token TEXT,
                created_at TEXT NOT NULL,
                reset_token TEXT,
                reset_token_expiry TEXT
            )
        ''')
        try:
            conn.execute('ALTER TABLE users ADD COLUMN reset_token TEXT')
        except:
            pass
        try:
            conn.execute('ALTER TABLE users ADD COLUMN reset_token_expiry TEXT')
        except:
            pass
        conn.commit()
init_db()

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
    page_texts = []
    page_count = 0
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        page_count = len(reader.pages)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                page_text = page_text.strip()
                page_texts.append((page_text, i + 1))
                text += page_text + '\n'
    return text.strip(), page_count, page_texts

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

def chunk_with_page_numbers(page_texts, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    pages = []
    for page_text, page_num in page_texts:
        start = 0
        while start < len(page_text):
            end = start + chunk_size
            chunk = page_text[start:end]
            if len(chunk) < 50:
                if chunks:
                    chunks[-1] += ' ' + chunk
                else:
                    chunks.append(chunk)
                    pages.append(page_num)
                break
            chunks.append(chunk)
            pages.append(page_num)
            start = end - overlap
    return chunks, pages

QUERY_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can',
    'could', 'shall', 'should', 'may', 'might', 'must', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'and', 'but',
    'or', 'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
    'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'because',
    'if', 'then', 'else', 'when', 'where', 'why', 'how', 'which', 'who',
    'whom', 'what', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
    'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
    'it', 'its', 'they', 'them', 'their', 'about', 'up', 'down', 'out',
    'over', 'under', 'again', 'further', 'once', 'here', 'there',
}

def extract_query_terms(query):
    words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
    return [w for w in words if w not in QUERY_STOP_WORDS and len(w) > 1]

def term_overlap_score(query_terms, doc_text):
    if not query_terms:
        return 0
    doc_lower = doc_text.lower()
    matches = sum(1 for t in query_terms if t in doc_lower)
    return matches / len(query_terms)

def jaccard_similarity(a, b):
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0
    return len(words_a & words_b) / len(words_a | words_b)

def mmr_rerank(documents, distances, query_terms=None, lambda_mult=0.7, top_k=12):
    selected = []
    candidates = list(range(len(documents)))

    while len(selected) < top_k and candidates:
        best_idx = None
        best_score = -float('inf')

        for i in candidates:
            dense_rel = 1 - distances[i]
            if query_terms:
                term_rel = term_overlap_score(query_terms, documents[i])
                relevance = 0.7 * dense_rel + 0.3 * term_rel
            else:
                relevance = dense_rel

            if selected:
                similarities = [jaccard_similarity(documents[i], documents[j]) for j in selected]
                max_sim = max(similarities)
            else:
                max_sim = 0
            score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            selected.append(best_idx)
            candidates.remove(best_idx)

    return selected

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
    'batana', 'bataye',
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
    'mein', 'hum', 'aur', 'lekin', 'magar', 'kyo',
    'ji', 'theek', 'waise', 'aisa', 'waisa', 'jaisa',
    'dikh', 'dekh', 'dekho', 'dekhte', 'dekhta', 'dekhti',
    'kar',
    'chal', 'chalo', 'chalta', 'chalti', 'chalte',
    'aana', 'jaana', 'jana', 'kahna', 'kehna',
    'sun', 'suno', 'sunta', 'sunti', 'sunte',
    'puch', 'pucho', 'puchta', 'puchtay',
    'dil', 'dilchasp', 'maza', 'mazedaar',
    'waqt', 'wqt', 'time',
    'kam', 'zyada', 'kum',
    'sath', 'saath', 'pasand',
    'sirf', 'srf', 'bas',
    'warna', 'vrna', 'na',
    'han', 'ha', 'acha', 'accha', 'theek',
    'iska', 'uska', 'jiska', 'kiska',
    'idhr', 'udhr', 'jaha', 'waha',
    'kabi', 'kabhi', 'kbhi',
    'phr', 'phir', 'fer',
}

def detect_language(text):
    if re.search(r'[\u0600-\u06FF\u0750-\u077F]', text):
        return 'urdu'
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    if words & ROMAN_URDU_WORDS:
        return 'roman-urdu'
    return 'english'

def extract_page_query(text):
    m = re.search(r'page\s*(?:number|no|#|\.)?\s*(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def generate_search_queries(question):
    prompt = (
        "You are a search query generator. Given a user's question, generate 3 "
        "different phrasings that capture the same intent using different words. "
        "This helps find relevant information in documents. Return exactly 3 "
        "queries, one per line. Do not number them or add any extra text."
    )
    try:
        text = query_groq(prompt, question)
        queries = [q.strip().strip('"').strip("'") for q in text.strip().split('\n') if q.strip()]
        return queries[:3]
    except:
        return []

MAX_HISTORY = 6

def query_groq(system_prompt, user_prompt, history=None):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")

    messages = [{'role': 'system', 'content': system_prompt}]
    if history:
        messages.extend(history[-MAX_HISTORY:])
    messages.append({'role': 'user', 'content': user_prompt})

    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': GROQ_MODEL,
        'messages': messages,
        'temperature': 0.6,
        'max_tokens': 1024
    }
    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")
    return resp.json()['choices'][0]['message']['content']

def query_groq_stream(system_prompt, user_prompt, history=None):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")

    messages = [{'role': 'system', 'content': system_prompt}]
    if history:
        messages.extend(history[-MAX_HISTORY:])
    messages.append({'role': 'user', 'content': user_prompt})

    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': GROQ_MODEL,
        'messages': messages,
        'temperature': 0.6,
        'max_tokens': 1024,
        'stream': True
    }
    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, stream=True, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")

    for line in resp.iter_lines():
        if line:
            line = line.decode('utf-8').strip()
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    token = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if token:
                        yield token
                except:
                    continue

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
        text, page_count, page_texts = extract_text_from_pdf(filepath)
        if not text:
            return jsonify({'success': False, 'error': 'No text could be extracted from the PDF'}), 400

        chunks, pages = chunk_with_page_numbers(page_texts)

        get_or_create_collection()

        existing = collection.get(ids=[file_id])
        if existing and existing['ids']:
            collection.delete(ids=[file_id])

        ids = []
        documents = []
        metadatas = []

        for i, (chunk, page) in enumerate(zip(chunks, pages)):
            chunk_id = f"{file_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({'file_id': file_id, 'filename': file.filename, 'chunk_index': i, 'page': page})

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        summary, topics = '', []
        try:
            summary_text = text[:4000]
            summary_resp = query_groq(
                "Summarize the following document text in 2-3 concise sentences. "
                "Then list 3-5 key topics as a comma-separated list. "
                "Format your response exactly as:\n"
                "SUMMARY: <summary>\n"
                "TOPICS: topic1, topic2, topic3",
                summary_text, None
            )
            parts = summary_resp.split('TOPICS:')
            if len(parts) > 1:
                summary = parts[0].replace('SUMMARY:', '').strip()
                topics = [t.strip() for t in parts[1].split(',')][:5]
            else:
                summary = summary_resp.strip()
        except:
            pass

        return jsonify({
            'success': True, 'file_id': file_id, 'filename': file.filename,
            'chunks': len(chunks), 'pages': page_count,
            'summary': summary, 'topics': topics
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def _prepare_chat(data):
    question = data['question']
    file_ids = data.get('file_ids')
    file_id = data.get('file_id')
    history = data.get('history')
    mode = data.get('mode', 'normal')

    if not file_ids and not file_id:
        return {'success': False, 'error': 'Missing file_id or file_ids'}

    lang = detect_language(question)

    if detect_greeting(question):
        if lang == 'english':
            system_prompt = (
                "You are a warm, friendly document assistant. The user is greeting you "
                "or making casual conversation. Respond with genuine warmth and enthusiasm. "
                "Politely mention you're here to help with their document and invite them "
                "to ask any questions about it. Keep it concise and natural."
            )
        else:
            system_prompt = (
                "You are a warm, friendly document assistant. The user is greeting you "
                "or making casual conversation in Roman Urdu / Urdu. Respond with natural "
                "warmth using respectful Urdu terms like 'aap' and 'ji'. Acknowledge their "
                "greeting warmly (e.g. 'Assalamualaikum!', 'Janaab!', 'Ji'). Invite them "
                "to ask questions about their document. Keep it friendly and natural."
            )
        return {
            'success': True, 'type': 'greeting', 'lang': lang,
            'system_prompt': system_prompt,
            'user_prompt': f"User said: {question}",
            'filename': '', 'history': history
        }

    get_or_create_collection()
    target_page = extract_page_query(question)

    if file_ids:
        if len(file_ids) == 1:
            where_clause = {'file_id': file_ids[0]}
        else:
            where_clause = {'$or': [{'file_id': fid} for fid in file_ids]}
    else:
        where_clause = {'file_id': file_id}

    all_queries = [question]
    alternatives = generate_search_queries(question)
    all_queries.extend(alternatives)
    results_per_query = 10 if alternatives else 20

    seen = set()
    all_docs = []
    all_dists = []
    all_metas = []

    for q in all_queries:
        try:
            q_results = collection.query(
                query_texts=[q],
                n_results=results_per_query,
                where=where_clause
            )
            if q_results and q_results['documents'] and q_results['documents'][0]:
                for d, dist, m in zip(q_results['documents'][0], q_results['distances'][0], q_results['metadatas'][0]):
                    cid = m.get('chunk_index')
                    if cid is not None and cid not in seen:
                        seen.add(cid)
                        all_docs.append(d)
                        all_dists.append(dist)
                        all_metas.append(m)
        except:
            continue

    results = {
        'documents': [all_docs],
        'distances': [all_dists],
        'metadatas': [all_metas]
    }

    if target_page and all_docs:
        filtered_docs = []
        filtered_dists = []
        filtered_metas = []
        for d, dist, m in zip(all_docs, all_dists, all_metas):
            if m.get('page') == target_page:
                filtered_docs.append(d)
                filtered_dists.append(dist)
                filtered_metas.append(m)
        if filtered_docs:
            results['documents'][0] = filtered_docs
            results['distances'][0] = filtered_dists
            results['metadatas'][0] = filtered_metas

    if not results or not results['documents'] or not results['documents'][0]:
        if target_page and file_id:
            total_pages = None
            all_meta = collection.get(where={'file_id': file_id})
            if all_meta and all_meta['metadatas']:
                existing_pages = sorted(set(m.get('page', 0) for m in all_meta['metadatas']))
                total_pages = max(existing_pages) if existing_pages else None
            system_prompt = (
                f"The user asked about page {target_page}, but that page was not found "
                f"in the document." +
                (f" The document has pages {existing_pages[0]} to {total_pages}."
                 if total_pages else "") +
                " Politely let them know and offer to help with an available page."
                + (' Respond in Roman Urdu with warmth and respect, using words like aap and ji.'
                   if lang != 'english' else '')
            )
        else:
            if lang == 'english':
                system_prompt = (
                    "You are a friendly document assistant. The user asked a question, but no "
                    "relevant information was found in their uploaded document. Respond politely "
                    "and warmly, letting them know you couldn't find an answer in the document "
                    "and suggest they try rephrasing or asking something else. Be empathetic, "
                    "not robotic."
                )
            else:
                system_prompt = (
                    "You are a friendly document assistant. The user asked a question in Roman Urdu, "
                    "but no relevant information was found in their uploaded document. Respond with "
                    "genuine warmth and empathy in Roman Urdu. Apologize politely (e.g. 'Mujhe maaf "
                    "karein'), explain you searched but couldn't find the answer, and gently suggest "
                    "they rephrase or ask something else. Use respectful words like aap and ji."
                )
        return {
            'success': True, 'type': 'no_answer', 'lang': lang,
            'system_prompt': system_prompt,
            'user_prompt': f"User asked: {question}",
            'filename': '', 'history': history
        }

    docs = results['documents'][0]
    dists = results['distances'][0]
    metas = results['metadatas'][0]
    query_terms = extract_query_terms(question)
    best_indices = mmr_rerank(docs, dists, query_terms=query_terms, top_k=12)

    context_parts = []
    for i in best_indices:
        page = metas[i].get('page', '?')
        context_parts.append(f"[Page {page}]\n{docs[i]}")
    context = '\n\n'.join(context_parts)

    filenames = sorted(set(m.get('filename', '') for m in metas))
    if mode == 'hr':
        filename = f"CVs ({len(filenames)} files)" if len(filenames) > 1 else (filenames[0] if filenames else 'CV')
    else:
        filename = filenames[0] if filenames else 'document'

    if lang == 'english':
        lang_instruction = "Answer in English. Be clear and natural."
        hr_lang = ""
    else:
        lang_instruction = (
            "Answer in Roman Urdu using friendly, natural language. "
            "Use respectful words like 'aap' instead of 'tum', and add 'ji' for warmth. "
            "Write conversationally as Urdu speakers naturally chat online."
        )
        hr_lang = " Provide candidate details in Roman Urdu with respectful language."

    page_prompt = ""
    if target_page:
        page_prompt = f"The user asked specifically about page {target_page}. "

    response_length_rule = (
        "Match your response length to the user's question. Short questions get short answers. "
        "Only give detailed answers when explicitly asked. "
        "Be conversational and natural, not robotic."
    )

    if mode == 'hr':
        system_prompt = (
            "You are an expert HR assistant analyzing candidate CVs. "
            "Answer the recruiter's query by referencing specific CV content. "
            "For each matching candidate mention their name (if visible), "
            "relevant skills, experience, and why they match. "
            "Be thorough and specific." + hr_lang + " " + lang_instruction + " "
            + response_length_rule
        )
    else:
        if lang == 'english':
            system_prompt = (
                "Synthesize the provided document context into a clear natural answer. "
                "Explain in your own words not copy paste. Be friendly and warm. "
                + page_prompt +
                "Each chunk is tagged with its source page number. Reference the page "
                "numbers naturally in your answer (e.g. 'As stated on page 3...'). "
                "If the context lacks enough information, summarize what you found "
                "and note what is missing. "
                + lang_instruction + " "
                + response_length_rule
            )
        else:
            system_prompt = (
                "Synthesize the provided document context into a helpful answer. "
                "Explain in your own words, don't copy paste. Be warm and respectful. "
                + page_prompt +
                "Each chunk has its page number. Mention the page when helpful "
                "(e.g. 'Page 3 par likha hai...'). "
                "If the context lacks enough information, summarize what you found "
                "and politely note what is missing. "
                + lang_instruction + " "
                + response_length_rule
            )

    user_prompt = f"Document: {filename}\n\nContext from document:\n{context}\n\nQuestion: {question}"

    return {
        'success': True, 'type': 'rag', 'lang': lang,
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'filename': filename, 'history': history
    }

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'success': False, 'error': 'Missing question'}), 400

    token = data.get('token')
    is_auth = False
    if token:
        with sqlite3.connect(DB_PATH) as conn:
            if conn.execute('SELECT id FROM users WHERE token = ?', (token,)).fetchone():
                is_auth = True
    if not is_auth:
        qc = data.get('question_count', 0)
        if qc > FREE_QUESTION_LIMIT:
            return jsonify({'success': False, 'error': 'Free question limit reached. Sign up to continue.'}), 403

    try:
        prep = _prepare_chat(data)
        if not prep['success']:
            return jsonify({'success': False, 'error': prep['error']}), 400

        answer = query_groq(prep['system_prompt'], prep['user_prompt'], prep['history'])
        return jsonify({
            'success': True,
            'answer': answer,
            'filename': prep['filename'],
            'lang': prep['lang']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'success': False, 'error': 'Missing question'}), 400

    token = data.get('token')
    is_auth = False
    if token:
        with sqlite3.connect(DB_PATH) as conn:
            if conn.execute('SELECT id FROM users WHERE token = ?', (token,)).fetchone():
                is_auth = True
    if not is_auth:
        qc = data.get('question_count', 0)
        if qc > FREE_QUESTION_LIMIT:
            def limit_error():
                yield f"data: {json.dumps({'type': 'error', 'text': 'Free question limit reached. Sign up to continue.'})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(limit_error(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    def generate():
        try:
            prep = _prepare_chat(data)
            if not prep['success']:
                yield f"data: {json.dumps({'type': 'error', 'text': prep['error']})}\n\n"
                yield "data: [DONE]\n\n"
                return

            yield f"data: {json.dumps({'type': 'lang', 'lang': prep['lang']})}\n\n"

            for token in query_groq_stream(prep['system_prompt'], prep['user_prompt'], prep['history']):
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'success': False, 'error': 'Email and password required'}), 400
    email = data['email'].strip().lower()
    password = data['password']
    if len(password) < 8:
        return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
    try:
        with sqlite3.connect(DB_PATH) as conn:
            existing = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
            if existing:
                return jsonify({'success': False, 'error': 'Email already registered'}), 409
            user_id = str(uuid.uuid4())
            password_hash = generate_password_hash(password)
            token = str(uuid.uuid4())
            conn.execute(
                'INSERT INTO users (id, email, password_hash, token, created_at) VALUES (?, ?, ?, ?, ?)',
                (user_id, email, password_hash, token, datetime.utcnow().isoformat())
            )
            conn.commit()
        return jsonify({'success': True, 'token': token, 'email': email})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'success': False, 'error': 'Email and password required'}), 400
    email = data['email'].strip().lower()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            user = conn.execute('SELECT id, password_hash, token FROM users WHERE email = ?', (email,)).fetchone()
            if not user:
                return jsonify({'success': False, 'error': 'No account found with this email'}), 401
            if not check_password_hash(user[1], data['password']):
                return jsonify({'success': False, 'error': 'Incorrect password'}), 401
            token = str(uuid.uuid4())
            conn.execute('UPDATE users SET token = ? WHERE id = ?', (token, user[0]))
            conn.commit()
        return jsonify({'success': True, 'token': token, 'email': email})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/verify', methods=['POST'])
def verify_token():
    data = request.get_json()
    if not data or not data.get('token'):
        return jsonify({'success': False, 'error': 'Token required'}), 400
    try:
        with sqlite3.connect(DB_PATH) as conn:
            user = conn.execute('SELECT email FROM users WHERE token = ?', (data['token'],)).fetchone()
            if not user:
                return jsonify({'success': False, 'error': 'Invalid token'}), 401
        return jsonify({'success': True, 'email': user[0]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    if not data or not data.get('email'):
        return jsonify({'success': False, 'error': 'Email is required'}), 400
    email = data['email'].strip().lower()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            user = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
            if not user:
                return jsonify({'success': False, 'error': 'No account found with this email'}), 404
            reset_token = str(uuid.uuid4())
            expiry = (datetime.utcnow().timestamp() + RESET_TOKEN_EXPIRY_HOURS * 3600)
            conn.execute('UPDATE users SET reset_token = ?, reset_token_expiry = ? WHERE id = ?',
                         (reset_token, str(expiry), user[0]))
            conn.commit()
        return jsonify({'success': True, 'reset_token': reset_token, 'email': email,
                        'message': 'Password reset link generated.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    if not data or not data.get('token') or not data.get('password'):
        return jsonify({'success': False, 'error': 'Token and new password required'}), 400
    password = data['password']
    if len(password) < 8:
        return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
    try:
        with sqlite3.connect(DB_PATH) as conn:
            user = conn.execute(
                'SELECT id, reset_token_expiry, email FROM users WHERE reset_token = ?',
                (data['token'],)
            ).fetchone()
            if not user:
                return jsonify({'success': False, 'error': 'Invalid or expired reset link'}), 400
            expiry = float(user[1])
            user_email = user[2]
            if datetime.utcnow().timestamp() > expiry:
                conn.execute('UPDATE users SET reset_token = NULL, reset_token_expiry = NULL WHERE id = ?',
                             (user[0],))
                conn.commit()
                return jsonify({'success': False, 'error': 'Reset link has expired. Please request a new one.'}), 400
            password_hash = generate_password_hash(password)
            new_token = str(uuid.uuid4())
            conn.execute(
                'UPDATE users SET password_hash = ?, token = ?, reset_token = NULL, reset_token_expiry = NULL WHERE id = ?',
                (password_hash, new_token, user[0])
            )
            conn.commit()
        return jsonify({'success': True, 'token': new_token, 'email': user_email, 'message': 'Password reset successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1', 'yes')
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=debug_mode)
