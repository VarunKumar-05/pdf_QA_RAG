QDRANT_COLLECTION = "all_pdf_chunks"

from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, Query, UploadFile, File, HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
import hashlib
import logging
from datetime import datetime, timezone
import numpy as np
import base64

# Import all the processing modules
from PyPDF2 import PdfReader
import re
from langchain_groq import ChatGroq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from qdrant_client.http import models as qmodels
import concurrent.futures
import spacy

# Audio processing
try:
    import soundfile as sf
    import speech_recognition as sr
    import pyttsx3
except ImportError:
    sf = None
    sr = None
    pyttsx3 = None

# Spell correction
from symspellpy import SymSpell, Verbosity

# Configuration
collection_name_combine = ''
# MONGODB config
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://varun:Passcode#123@cluster0.j76oos8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
DB_NAME = "new_api"
COLLECTION_NAME = "qa_pairs"

#QDRANT config
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DpbLy6l-AoQbSML4DUUZlgxATFKoC9PScoblOOMeL-0"
QDRANT_URL = "https://e4019dac-dae5-4390-ad95-183d3539e368.europe-west3-0.gcp.cloud.qdrant.io"
collection_name_combine=''
NLP = spacy.load("en_core_web_sm")
logging.basicConfig(level=logging.WARNING)

app = FastAPI(title="PDF AI Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    top_k_results: Optional[int] = 3
    pdf_hash: Optional[str] = None
    collection_name: Optional[str] = None

class QuestionGenerationRequest(BaseModel):
    context_topic: str
    num_questions: Optional[int] = 5
    pdf_hash: Optional[str] = None
    collection_name: Optional[str] = None

class TTSRequest(BaseModel):
    text: str

class PDFProcessResponse(BaseModel):
    success: bool
    message: str
    pdf_hash: str
    chunks_count: int
    avg_chunk_length: float

class QAResponse(BaseModel):
    question: str
    answer: str
    similar_found: bool
    sources: List[Dict[str, Any]]
    corrected_question: Optional[str] = None

class PDFProcessBatchResponse(BaseModel):
    results: List[PDFProcessResponse]

class namerequest(BaseModel):
    name:str


# Global variables to store models and data
embedding_model1 = None
embedding_model = None
llm_model = None
qg_tokenizer = None
qg_model = None
sym_spell = None
vector_stores: Dict[str, "QdrantVectorStore"] = {}
pdf_chunks_store: Dict[str, list] = {}
collection_name= None

def init_models():

    global embedding_model, llm_model, qg_tokenizer, qg_model, sym_spell, embedding_model1
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_model1 = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    if GROQ_API_KEY:
        llm_model = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0.2)
    try:
        qg_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        qg_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    except Exception as e:
        print(f"Could not load question generation model: {e}")
    max_edit_distance_dictionary = 2
    prefix_length = 7
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    dictionary_path = "frequency_dictionary_en_82_765.txt"
    term_index = 0
    count_index = 1
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("SymSpell dictionary loading failed.")

@app.post("/process-name")
async def collection(data: namerequest):
    global collection_name_combine
    collection_name_combine = data.name
    return {"success": True, "collection_name": collection_name_combine}

def symspell_correct_sentence(sentence):
    if not sym_spell:
        return sentence
    tokens = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
    corrected_tokens = []
    for token in tokens:
        if token.isalpha():
            suggestions = sym_spell.lookup(token, Verbosity.TOP, max_edit_distance=2)
            if suggestions and suggestions[0].term.lower() != token.lower():
                corrected_tokens.append(suggestions[0].term)
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)
    corrected_sentence = ""
    for i, token in enumerate(corrected_tokens):
        if i > 0 and (token.isalnum() and corrected_tokens[i-1].isalnum()):
            corrected_sentence += " "
        corrected_sentence += token
    return corrected_sentence.strip()

def get_mongo_collection(collection_name=None):
    try:
        import certifi
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=220000, tlsCAFile=certifi.where())
        client.admin.command('ping')
        db = client[DB_NAME]
        coll_name = collection_name or COLLECTION_NAME
        return db[coll_name]
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None

def create_timeseries_collection(collection_name):
    import certifi
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=220000, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    if collection_name not in db.list_collection_names():
        db.create_collection(
            collection_name,
            timeseries={
                'timeField': 'timestamp',
                'metaField': 'meta',
                "granularity": "seconds",
            },
            expireAfterSeconds=3600
        )
def normalize_question(q):
    return " ".join(q.lower().split())

def find_similar_questions(query, collection_name=None, top_k=3):
    try:
        collection = get_mongo_collection(collection_name)
        if collection is None:
            return []
        norm_query = normalize_question(query)
        # Exact duplicate check
        exact = list(collection.find({"normalized_question": norm_query}))
        if exact:
            return exact
        # Fallback to semantic similarity for near-duplicates
        query_emb = embedding_model.encode(query)
        docs = list(collection.find({}, {"question": 1, "embedding": 1, "answer": 1, "tags": 1, "normalized_question": 1}).sort("timestamp", -1).limit(500))
        if not docs:
            return []
        emb_matrix = np.array([doc["embedding"] for doc in docs if "embedding" in doc and doc["embedding"] is not None])
        if emb_matrix.ndim == 1:
            if emb_matrix.shape[0] == query_emb.shape[0]:
                emb_matrix = emb_matrix.reshape(1, -1)
            else:
                return []
        elif emb_matrix.shape[0] == 0:
            return []
        emb_matrix = emb_matrix.astype(np.float32)
        query_emb = query_emb.astype(np.float32)
        sims = util.pytorch_cos_sim(query_emb, emb_matrix)[0].cpu().numpy()
        actual_top_k = min(top_k, len(sims))
        if actual_top_k == 0:
            return []
        top_indices = sims.argsort()[-actual_top_k:][::-1]
        return [docs[i] for i in top_indices if sims[i] > 0.85]
    except Exception as e:
        print(f"Could not search MongoDB for similar questions: {e}")
        return []

def store_qa_pair(question, answer, tags=None, collection_name=None):
    try:
        create_timeseries_collection(collection_name)
        collection = get_mongo_collection(collection_name)
        if collection is None:
            return
        norm_q = normalize_question(question)
  
        if collection.find_one({"normalized_question": norm_q}):
            print("Duplicate question, not storing.")
            return
        embedding = embedding_model1.encode(question).tolist()
        doc = {
            "question": question,
            "answer": answer,
            "normalized_question": norm_q,
            "embedding": embedding,
            "timestamp": datetime.now(timezone.utc),
            "tags": tags or [],
            "meta": {"source": "pdf"}
        }
        collection.insert_one(doc)
    except Exception as e:
        print(f"Could not store Q&A: {e}")

def extract_sentences_from_pdf(pdf_file, filename, max_workers=4):
    import os
    base_filename = os.path.basename(filename)
    reader = PdfReader(pdf_file)
    chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                futures.append(
                    executor.submit(extract_sentences_from_page, page_text, page_idx, base_filename)
                )
        for fut in concurrent.futures.as_completed(futures):
            try:
                page_chunks = fut.result()
                chunks.extend(page_chunks)
            except Exception as e:
                print(f"Error in multithreaded extraction: {e}")
    return chunks
def extract_sentences_from_page(page_text, page_idx, filename):
    nlp = spacy.load("en_core_web_sm") 
    doc = NLP(page_text)
    return [
        {
            "text": sent.text.strip(),
            "source_pdf": filename,
            "page_number": page_idx + 1
        }
        for sent in doc.sents if sent.text.strip()
    ]
class QdrantVectorStore:
    def __init__(self, embeddings_model, collection_name):
        self.embeddings_model = embeddings_model
        self.collection_name = collection_name
        self.qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        recreate = False
        if not self.qdrant.collection_exists(self.collection_name):
            size = embeddings_model.get_sentence_embedding_dimension()
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=size, distance=qmodels.Distance.COSINE),
            )
        try:
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_pdf",
                field_schema="keyword"
            )
        except Exception as e:
            print(f"Index error (may already exist): {e}")
    def clear_collection(self):
        self.qdrant.delete(collection_name=self.collection_name, points_selector=Filter(must=[]))

    def upload_chunks(self, chunks):
        texts = [c['text'] for c in chunks]
        embeddings = self.embeddings_model.encode(texts, show_progress_bar=True)
        payloads = chunks
        points = []
        for i, (vec, payload) in enumerate(zip(embeddings, payloads)):
            points.append(
                qmodels.PointStruct(
                    id=i,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def similarity_search_top_k(self, query: str, k: int = 5) -> list:
        query_vec = self.embeddings_model.encode([query])[0]
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vec.tolist(),
            limit=k,
            with_payload=True
        )
        return [
            {
                'text': r.payload['text'],
                'meta': r.payload,
                'score': r.score,
                'index': r.id
            } for r in results
        ]

def create_qa_chain(doc_results: list, question: str):
    if not llm_model:
        return "LLM not loaded. Cannot generate answer."
    if not doc_results:
        return "No relevant document passages found to answer the question."
    sorted_results = sorted(doc_results, key=lambda x: x.get('score', 0), reverse=True)
    context_chunks = [chunk['text'] for chunk in sorted_results[:3]]
    context = "\n\n---\n\n".join(context_chunks)
    MAX_CONTEXT_CHARS = 15000
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...(context truncated)"
    prompt = f"""You are a helpful AI assistant. Answer the question based *only* on the context provided below.
Be concise and accurate and fast. If the answer is not found in the context, say "The answer is not found in the provided document context."

Context:
---
{context}
---

Question: {question}

Answer:"""
    try:
        response = llm_model.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Error generating answer: {e}"

def generate_question_for_sentence(sentence, context, tokenizer, model):
    input_text = f"generate different question based on  : context: {context} sentence: {sentence}  "
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

def list_collections():
    try:
        import certifi
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=220000, tlsCAFile=certifi.where())
        db = client[DB_NAME]
        collections = [c for c in db.list_collection_names() if not c.startswith("system.")]
        return collections
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return []
def fetch_qa_pairs(collection_name):
    try:
        collection = get_mongo_collection(collection_name)
        if collection is None:
            return []
        # Use a dict for projection to exclude _id
        docs = list(collection.find({}, {"_id": 0, "question": 1, "answer": 1, "tags": 1, "timestamp": 1}))
        return docs
    except Exception as e:
        print(f"Failed to fetch Q&A pairs: {e}")
        return []

def speech_to_text(audio_data):
    if sr is None:
        return ""
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_file.flush()
            with sr.AudioFile(tmp_file.name) as source:
                audio = recognizer.record(source)
            query = recognizer.recognize_google(audio)
            return query
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return ""

@app.on_event("startup")
async def startup_event():
    init_models()

@app.get("/")
async def root():
    return {"message": "PDF AI Assistant API is running!"}


@app.post("/upload-pdfs", response_model=PDFProcessBatchResponse)
async def upload_pdfs(    
    files: List[UploadFile] = File(...),
    mode: str = Query("combine"),
    collection_name: Optional[str] = Query(None)
):
    results = []
    all_chunks = []
    for file in files:
        try:
            if file.content_type != "application/pdf":
                results.append(PDFProcessResponse(
                    success=False,
                    message=f"File {file.filename} must be a PDF",
                    pdf_hash="",
                    chunks_count=0,
                    avg_chunk_length=0.0
                ))
                continue
            content = await file.read()
            pdf_hash = hashlib.md5(content).hexdigest()
            if mode == "combine":
                used_collection_name = collection_name or collection_name_combine
                if not used_collection_name or not used_collection_name.strip():
                    results.append(PDFProcessResponse(
                        success=False, 
                        message="Combined collection name not provided.",
                        pdf_hash=pdf_hash, chunks_count=0, avg_chunk_length=0.0
                    ))
                    continue
            else:
                used_collection_name = f"{pdf_hash}_{file.filename}"
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
                sentence_chunks = extract_sentences_from_pdf(tmp_file, file.filename)
                if not sentence_chunks:
                    results.append(PDFProcessResponse(
                        success=False,
                        message=f"No sentences were extracted from {file.filename}",
                        pdf_hash=pdf_hash, chunks_count=0, avg_chunk_length=0.0
                    ))
                    continue
                if mode == "combine":
                    all_chunks.extend(sentence_chunks)
                else:
                    if used_collection_name not in vector_stores:
                        qdrant_store = QdrantVectorStore(embedding_model, used_collection_name)
                        vector_stores[used_collection_name] = qdrant_store
                    else:
                        qdrant_store = vector_stores[used_collection_name]
                    qdrant_store.upload_chunks(sentence_chunks)
                avg_chunk_len = sum(len(c["text"]) for c in sentence_chunks) / len(sentence_chunks)
                results.append(PDFProcessResponse(
                    success=True,
                    message=f"PDF processed successfully! Extracted {len(sentence_chunks)} sentence-level chunks.",
                    pdf_hash=pdf_hash,
                    chunks_count=len(sentence_chunks),
                    avg_chunk_length=avg_chunk_len
                ))
        except Exception as e:
            results.append(PDFProcessResponse(
                success=False,
                message=f"Error processing PDF {file.filename}: {str(e)}",
                pdf_hash="",
                chunks_count=0,
                avg_chunk_length=0.0
            ))
    if mode == "combine" and all_chunks:
        final_collection_name = collection_name or collection_name_combine
        if not final_collection_name or not final_collection_name.strip():
            results.append(PDFProcessResponse(
                success=False, 
                message="Combined collection name not provided.",
                pdf_hash="", chunks_count=0, avg_chunk_length=0.0
            ))
        else:
            if final_collection_name not in vector_stores:
                qdrant_store = QdrantVectorStore(embedding_model, final_collection_name)
                vector_stores[final_collection_name] = qdrant_store
            else:
                qdrant_store = vector_stores[final_collection_name]
            qdrant_store.clear_collection()
            qdrant_store.upload_chunks(all_chunks)
    return PDFProcessBatchResponse(results=results)

@app.post("/remove-pdf")
async def remove_pdf(request: Request):
    try:
        data = await request.json()
        pdf_hash = data["pdf_hash"]
        filename = data["filename"]
        mode = data["mode"]
        collection_name = data.get("collection_name")  
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    try:
        filename=os.path.basename(filename)  
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        if mode == "combine":
            used_collection_name = collection_name or collection_name_combine
            if not used_collection_name:
                raise HTTPException(status_code=400, detail="No combined collection name provided.")
            filter_obj = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_pdf",
                        match=qmodels.MatchValue(value=filename)
                    )])
            qdrant.delete(collection_name=used_collection_name, points_selector=filter_obj)
            return {"success": True}
        else:
            # Drop the collection
            used_collection_name = f"{pdf_hash}_{filename}"
            qdrant.delete_collection(collection_name=used_collection_name)
            if used_collection_name in vector_stores:
                del vector_stores[used_collection_name]
            return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove PDF: {e}")
@app.post("/clear-collections")
def clear_all_qdrant_collections():
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    collections = qdrant.get_collections().collections
    for collection in collections:
        name = collection.name
        qdrant.delete_collection(collection_name=name)
        print(f"Deleted: {name}")


#new as of 23-06-25
@app.post("/clear-collection")
def clear_all_mongo_collections():
    import certifi
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=220000, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    for name in db.list_collection_names():
        db.drop_collection(name)
        print(f"✅ Dropped: {name}")



def clear_collection(self):
    self.qdrant.delete(collection_name=self.collection_name, points_selector=Filter(must=[]))
@app.post("/upload-pdf", response_model=PDFProcessResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")
        content = await file.read()
        pdf_hash = hashlib.md5(content).hexdigest()
        collection_name = QDRANT_COLLECTION
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            sentence_chunks = extract_sentences_from_pdf(tmp_file, file.filename)
            if not sentence_chunks:
                raise HTTPException(status_code=400, detail="No sentences were extracted from the PDF")
            if collection_name not in pdf_chunks_store:
                pdf_chunks_store[collection_name] = []
            pdf_chunks_store[collection_name].extend(sentence_chunks)
            if collection_name not in vector_stores:
                qdrant_store = QdrantVectorStore(embedding_model, collection_name)
                vector_stores[collection_name] = qdrant_store
            else:
                qdrant_store = vector_stores[collection_name]
            qdrant_store.upload_chunks(sentence_chunks)
            avg_chunk_len = sum(len(c["text"]) for c in sentence_chunks) / len(sentence_chunks)
            return PDFProcessResponse(
                success=True,
                message=f"PDF processed successfully! Uploaded {len(sentence_chunks)} sentence-level chunks.",
                pdf_hash=pdf_hash,
                chunks_count=len(sentence_chunks),
                avg_chunk_length=avg_chunk_len
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask-question", response_model=QAResponse)
async def ask_question(request: QuestionRequest):
    try:
        collection_name = request.collection_name#QDRANT collection  
        if collection_name not in vector_stores:
            raise HTTPException(status_code=400, detail="No data uploaded yet.PDF/Collection not found. Please upload a PDF first.")
       
        corrected_question = symspell_correct_sentence(request.question)
        vector_store = vector_stores[collection_name]
        similar_qa_pairs = find_similar_questions(corrected_question, collection_name, top_k=3)
        if similar_qa_pairs:
            qa_pair = similar_qa_pairs[0]
            return QAResponse(
                question=request.question,
                answer=qa_pair['answer'],
                similar_found=True,
                sources=[],
                corrected_question=corrected_question if corrected_question != request.question else None
            )
        doc_results = vector_store.similarity_search_top_k(corrected_question, k=request.top_k_results)
        if doc_results:
            answer = create_qa_chain(doc_results, corrected_question)
            if answer != "The answer is not found in the provided document context." :
                tags = [f"From: {r['meta']['source_pdf']} | Page {r['meta']['page_number']}" for r in doc_results if r.get('meta')]
                store_qa_pair(corrected_question, answer, tags=tags, collection_name=collection_name)
            sources = [
                {
                    "text": r['text'],
                    "score": r['score'],
                    "source_pdf": r['meta'].get('source_pdf', ''),
                    "page_number": r['meta'].get('page_number', 0)
                }
                for r in doc_results
            ]
            return QAResponse(
                question=request.question,
                answer=answer,
                similar_found=False,
                sources=sources,
                corrected_question=corrected_question if corrected_question != request.question else None
            )
        else:
            raise HTTPException(status_code=404, detail="No relevant information found in the PDF")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/generate-questions")
async def generate_questions(request: QuestionGenerationRequest):
    try:
        collection_name = request.collection_name
        if not collection_name or collection_name not in vector_stores:
            raise HTTPException(status_code=404, detail="PDF/Collection not found. Please upload a PDF first.")
        vector_store = vector_stores[collection_name]
        top_sentences = vector_store.similarity_search_top_k(request.context_topic, k=request.num_questions)
        similarity_threshold= 0.3
        if not top_sentences or all(passage['score']< similarity_threshold for passage in top_sentences):
            return{"message": "No context from the PDF"}
        else:
            generated_qa = []
            for passage in top_sentences:
                sentence = passage['text']
                question = generate_question_for_sentence(sentence, request.context_topic, qg_tokenizer, qg_model)
                doc_results = vector_store.similarity_search_top_k(question, k=3)
                answer = create_qa_chain(doc_results, question)
                generated_qa.append({
                    "sentence": sentence,
                    "generated_question": question,
                    "answer": answer,
                    "source_pdf": passage['meta'].get('source_pdf', ''),
                    "page_number": passage['meta'].get('page_number', 0),
                    "score": passage['score']
                })
            return {"generated_qa": generated_qa}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@app.post("/speech-to-text")
async def speech_to_text_endpoint(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        content = await file.read()
        text = speech_to_text(content)
        return {"transcribed_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/text-to-speech")
async def text_to_speech_endpoint(request: TTSRequest):
    try:
        if pyttsx3 is None:
            raise HTTPException(status_code=500, detail="Text-to-speech not available")
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wav_path = tmp_wav.name
        engine.save_to_file(request.text, wav_path)
        engine.runAndWait()
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        os.unlink(wav_path)
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return {"audio_data": audio_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/collections")
async def get_collections():
    try:
        collections = list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching collections: {str(e)}")

@app.get("/qa-pairs/{collection_name}")
async def get_qa_pairs(collection_name: str):
    try:
        qa_pairs = fetch_qa_pairs(collection_name)
        return {"qa_pairs": qa_pairs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching Q&A pairs: {str(e)}")
@app.post("/process-name")
async def collection(data: namerequest):
    global collection_name_combine
    collection_name_combine = data.name
    return {"success": True, "collection_name": collection_name_combine}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






