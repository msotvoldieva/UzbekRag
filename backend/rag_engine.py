import os
import uuid
import time
import re
from typing import List, Dict

import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import html
from openai import OpenAI
import json as json_lib

openai_client = OpenAI(api_key="API_KEY")

# # Load the model and tokenizer once (outside the function so it's not reloaded every call)
# MODEL_NAME = "ai-forever/mGPT"  
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# # Create a text-generation pipeline
# generator = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer
# )

# === CONFIG ===
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
COLLECTION_NAME = "HR_uzb_jina"
QDRANT_URL = "http://qdrant:6333" 
OLLAMA_URL = "http://host.docker.internal:11434"
DOCUMENT_METADATA_FILE = "/app/arizalar.json"

# Translations 
def translate_text(text: str, target_language: str) -> str:
    url = "https://translation.googleapis.com/language/translate/v2"
    
    params = {
        "q": text,
        "target": target_language,
        "key": API_KEY
    }
    
    response = requests.post(url, data=params)
    response.raise_for_status()
    
    translation = response.json()["data"]["translations"][0]["translatedText"]
    translation = html.unescape(translation)
    return translation

# === Init Embedding Model and Qdrant ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
client = QdrantClient(url=QDRANT_URL)

def wait_for_qdrant(timeout=30):
    print("⏳ Waiting for Qdrant to be ready...")
    for _ in range(timeout):
        try:
            client.get_collections()
            print("✅ Qdrant is ready.")
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("❌ Qdrant did not start in time.")


def ensure_collection_ready(folder_path="GGTexts", collection_name=COLLECTION_NAME):
    wait_for_qdrant()
    if client.collection_exists(collection_name=collection_name):
        print(f"✅ Collection '{collection_name}' already exists.")
        return
    
    print(f"📂 Collection '{collection_name}' not found. Creating and populating it...")

    # 1. Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print("✅ Created collection.")

    # 2. Load and split documents
    chunks = create_article_chunks(folder_path)
    print(f"✂️ Created {len(chunks)} chunks.")

    # 3. Store in Qdrant
    store_chunks(chunks)
    print(f"📦 Stored chunks into Qdrant collection '{collection_name}'.")


# Document Loading + Chunking
def load_documents(folder_path: str):
    print("Loading .txt files...")
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            path = os.path.join(folder_path, file_name)
            loader = TextLoader(path, encoding='utf-8')
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["title"] = file_name.replace(".txt", "")
                doc.metadata["source"] = file_name
            documents.extend(loaded_docs)
    return documents


def extract_articles(text: str) -> List[Dict[str, str]]:
    # Pattern to find Uzbek article headers: "1-modda. Title"
    pattern = r'(\d+)-modda\.\s+(.+)'
    matches = list(re.finditer(pattern, text))
    
    articles = []
    for i, match in enumerate(matches):
        article_number = match.group(1)
        article_title = match.group(2).strip()
        
        content_start = match.end()
        
        if i + 1 < len(matches):
            content_end = matches[i + 1].start()
        else:
            content_end = len(text)
        
        # Extract the actual article content
        article_content = text[content_start:content_end].strip()
        
        articles.append({
            'number': article_number,
            'title': article_title,
            'content': article_content,
            'full_text': f"{article_number}-modda. {article_title}\n{article_content}"
        })
    
    return articles

def split_long_article(article: Dict[str, str]) -> List[Dict[str, str]]:
    
    full_text = article['full_text']
    
    if len(full_text) <= 1500:
        return [article]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=['\n\n', '\n', ';', '.', ',', ' ']
    )
    
    chunks = splitter.split_text(full_text)
    result = []
    
    for i, chunk in enumerate(chunks):
        chunk_article = article.copy()
        chunk_article['full_text'] = chunk
        chunk_article['chunk_index'] = i
        chunk_article['total_chunks'] = len(chunks)
        chunk_article['title'] = f"{article['title']} (qism {i+1}/{len(chunks)})"
        result.append(chunk_article)
    
    return result

def create_article_chunks(folder: str):
    docs = load_documents(folder)
    all_chunks = []
    
    for doc in docs:
        print(f"📄 Processing document: {doc.metadata.get('title', 'Unknown')}")
        
        articles = extract_articles(doc.page_content)
        
        if not articles:
            print(f"⚠️ No articles found in {doc.metadata.get('title', 'Unknown')}. Using fallback chunking.")
            # Fallback to original chunking method
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            fallback_chunks = splitter.split_documents([doc])
            for chunk in fallback_chunks:
                chunk.metadata.update(doc.metadata)
            all_chunks.extend(fallback_chunks)
            continue
        
        print(f"📑 Found {len(articles)} articles")
        
        # Process each article
        for article in articles:
            article_chunks = split_long_article(article)

            for chunk_data in article_chunks:
                # Create a document-like object for each article chunk
                class ArticleChunk:
                    def __init__(self, content, metadata):
                        self.page_content = content
                        self.metadata = metadata
                
                metadata = {
                    **doc.metadata,
                    'article_number': chunk_data['number'],
                    'article_title': chunk_data['title'],
                    'chunk_type': 'article',
                }
                
                if 'chunk_index' in chunk_data:
                    metadata['chunk_index'] = chunk_data['chunk_index']
                    metadata['total_chunks'] = chunk_data['total_chunks']
                
                chunk = ArticleChunk(chunk_data['full_text'], metadata)
                all_chunks.append(chunk)
    
    return all_chunks

# Embedding and Qdrant Storage
def generate_embeddings(text: str) -> List[float]:
    return embedding_model.encode(
        text,
        convert_to_tensor=True,
        normalize_embeddings=True
    ).tolist()

def store_chunks(chunks):
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        payload = {
            "content": chunk.page_content,
            **chunk.metadata
        }
        embedding = generate_embeddings(chunk.page_content)
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=[PointStruct(id=chunk_id, vector=embedding, payload=payload)]
        )

# Querying for document download
def load_document_metadata() -> Dict:
    try:
        if os.path.exists(DOCUMENT_METADATA_FILE):
            with open(DOCUMENT_METADATA_FILE, 'r', encoding='utf-8') as f:
                return json_lib.load(f)
        else:
            print(f"⚠️  Document metadata file not found: {DOCUMENT_METADATA_FILE}")
            return {"documents": []}
    except Exception as e:
        print(f"❌ Error loading document metadata: {e}")
        return {"documents": []}

def find_form_by_ai_query(query: str) -> Dict[str, any]:
    try:
        print(f"AI Form Search for: '{query}'")
        
        metadata = load_document_metadata()
        documents = metadata.get("documents", [])
        
        forms_info = []
        for doc in documents:
            form_description = f"""
            Title: {doc.get('title', '')}
            Description: {doc.get('description', '')}
            """
            forms_info.append({
                "filename": doc.get("filename"),
                "info": form_description.strip()
            })
        
        forms_list = "\n---\n".join([f"Form {i+1}: {form['filename']}\n{form['info']}" 
                                   for i, form in enumerate(forms_info)])
        
        analysis_prompt = f"""
        Foyadalonovchi saoli: "{query}"
        
        Mavjud namunalar:
        {forms_list}
        
        Foydalanuvchi so'rovini tahlil qil va qaysi fayl eng mos kelishini aniqla. 
        Maqsad, Kontekst va ma'noga e'tibor ber.
        
        JSON formati bilan javob ber:
        {{
            "best_match": "filename.docx",
            "confidence": 0.85
            "alternative_matches": ["other_file.docx"]
        }}
        
        Agar yaxshi moslik bo'lmasa, best_match ni null qilib belgilang.
        """
        
        # Get AI response
        response = openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are an expert at matching user requests to document forms. Always respond with valid JSON."},
                {"role": "user", "content": analysis_prompt}
            ]
        )
        
        # Parse AI response
        try:
            ai_result = json_lib.loads(response.choices[0].message.content)
            
            best_match = ai_result.get("best_match")
            confidence = ai_result.get("confidence", 0)
            
            if best_match and confidence > 0.5:
                matching_doc = None
                for doc in documents:
                    if doc.get("filename") == best_match:
                        matching_doc = doc
                        break
                
                if matching_doc:
                    return {
                        "filename": best_match,
                        "document_info": matching_doc,
                        "confidence": confidence,
                        "alternatives": ai_result.get("alternative_matches", [])
                    }
            
            return {
                "error": "No suitable form found",
                "confidence": confidence
            }
            
        except json_lib.JSONDecodeError:
            print(f"❌ Failed to parse AI response: {response.choices[0].message.content}")
            return {"error": "AI analysis failed", "confidence": 0}
        
    except Exception as e:
        print(f"❌ Error in AI form search: {e}")
        return {"error": str(e), "confidence": 0}


# Detailed instruct necessary for E-5 model
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def query_relevant_chunks(user_prompt: str) -> str:
    task = 'Foydalanuvchi so\'rovini hisobga olgan holda, so\'rovga javob beradigan tegishli parchalarni toping.'
    querie = get_detailed_instruct(task, user_prompt)
    query_embedding = generate_embeddings(querie)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        with_payload=True,
        limit=12
    )

    print(f"✅ Found {len(results.points)} initial relevant chunks.")
    
    # Step 2: Identify the unique articles from the retrieved chunks
    unique_articles = set()
    for point in results.points:
        article_info = (
            point.payload.get('source', 'unknown_source'),
            point.payload.get('article_number', 'unknown_article')
        )
        unique_articles.add(article_info)
    
    print(f"📑 Identified {len(unique_articles)} unique articles to reconstruct.")
    
    # Step 3: Fetch all chunks for each unique article and reconstruct the full text
    reconstructed_articles = []
    for source, article_number in unique_articles:

        article_chunks = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding, 
            with_payload=True,
            limit=100,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="source", match=models.MatchText(text=source)),
                    models.FieldCondition(key="article_number", match=models.MatchValue(value=article_number))
                ]
            )
        )
        
        # Sort chunks by their index to ensure correct order
        sorted_chunks = sorted(article_chunks.points, key=lambda p: p.payload.get('chunk_index', 0))
        
        # Combine the sorted chunks into a single, full article text
        full_article_content = "".join([p.payload.get('content', '') for p in sorted_chunks])
        
        # Get the title from the first chunk
        article_title = sorted_chunks[0].payload.get('article_title', 'unknown_title')
        
        reconstructed_articles.append(f"### Article: {article_title} ({article_number})\n{full_article_content}\n")

    return "\n---\n".join(reconstructed_articles)

# def generate_response(prompt: str, max_new_tokens: int = 256) -> str:
#     try:
#         outputs = generator(
#             prompt,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9
#         )

#         return outputs[0]["generated_text"]

#     except Exception as e:
#         print("❌ Failed to get response from Transformers model:", e)
#         return f"[❌ Transformers failure] {str(e)}"


def generate_response(prompt: str) -> str:
    try:
        # response = requests.post(
        #     f"{OLLAMA_URL}/api/generate",
        #     json={"model": "gemma3:4b-it-qat", "prompt": prompt, "stream": False}
        # )
        # print("📥 Ollama raw response:", response.text)

        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",  # or gpt-4o, gpt-4.1, etc.
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        text = response.choices[0].message.content
        return text
    
    except Exception as e:
        print("❌ Failed to get response from Ollama:", e)
        return f"[❌ Ollama failure] {str(e)}"
