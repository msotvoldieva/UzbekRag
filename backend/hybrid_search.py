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
from fastembed.rerank.cross_encoder import TextCrossEncoder

# === CONFIG ===
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
COLLECTION_NAME = "HR_uzb_jina"
QDRANT_URL = "http://localhost:6335"  # updated to Docker service name
OLLAMA_URL = "http://localhost:11436"

# === Init Embedding Model and Qdrant ===
dense_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
reranker = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')

#embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
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


# === Document Loading + Chunking ===
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
        
        # Find content start (after the title line)
        content_start = match.end()
        
        # Find content end (start of next article or end of text)
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
    """
    Split articles that are too long into smaller chunks while preserving context.
    """
    full_text = article['full_text']
    
    if len(full_text) <= 1500:
        return [article]
    
    # Use RecursiveCharacterTextSplitter for long articles
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
    """
    Create chunks based on articles instead of arbitrary character limits.
    """
    docs = load_documents(folder)
    all_chunks = []
    
    for doc in docs:
        print(f"📄 Processing document: {doc.metadata.get('title', 'Unknown')}")
        
        # Extract articles from document
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
            # Split long articles if necessary
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
                
                # Add chunk information for split articles
                if 'chunk_index' in chunk_data:
                    metadata['chunk_index'] = chunk_data['chunk_index']
                    metadata['total_chunks'] = chunk_data['total_chunks']
                
                chunk = ArticleChunk(chunk_data['full_text'], metadata)
                all_chunks.append(chunk)
    
    return all_chunks

# def create_chunks(folder: str):
#     docs = load_documents(folder)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return splitter.split_documents(docs)

# === Embedding and Qdrant Storage ===
def generate_embeddings(text: str) -> List[float]:
    return dense_embedding_model.encode(
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

# === Querying & Generation ===
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def query_relevant_chunks(user_prompt: str) -> str:
    task = 'Foydalanuvchi so\'rovini hisobga olgan holda, so\'rovga javob beradigan tegishli parchalarni toping.'
    querie = get_detailed_instruct(task, user_prompt)
    query_embedding = generate_embeddings(querie)

    # Step 1: Perform the initial query to find the top N most relevant chunks
    initial_retrieval = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        with_payload=True,
        limit=20 
    )

    print(f"✅ Found {len(initial_retrieval.points)} initial relevant chunks.")

    chunk_texts = []
    for hit in initial_retrieval.points:
        chunk_texts.append(hit.payload.get("content", ""))  # content field from your payload

    # Step 3: Apply reranker
    new_scores = list(reranker.rerank(querie, chunk_texts))
    ranking = [(i, score) for i, score in enumerate(new_scores)]
    ranking.sort(key=lambda x: x[1], reverse=True)

    print(f"📊 Reranked top {len(ranking)} chunks.")
    top_hits = [initial_retrieval.points[idx] for idx, _ in ranking[:10]]

    # Step 2: Identify the unique articles from the retrieved chunks
    unique_articles = set()
    for point in top_hits:
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
        
        sorted_chunks = sorted(article_chunks.points, key=lambda p: p.payload.get('chunk_index', 0))
        full_article_content = "".join([p.payload.get('content', '') for p in sorted_chunks])
        article_title = sorted_chunks[0].payload.get('article_title', 'unknown_title')
        
        reconstructed_articles.append(f"### Article: {article_title} ({article_number})\n{full_article_content}\n")

    return "\n---\n".join(reconstructed_articles)

# def query_relevant_chunks(user_prompt: str) -> str:
#     task = 'Foydalanuvchi so\'rovini hisobga olgan holda, so\'rovga javob beradigan tegishli parchalarni toping.'
#     querie = get_detailed_instruct(task, user_prompt)
#     query_embedding = generate_embeddings(querie)

#     results = client.query_points(
#         collection_name=COLLECTION_NAME,
#         query=query_embedding,
#         with_payload=True,
#         limit=7
#     )

#     return "\n".join([
#         f"- Из: {pt.payload.get('title', 'Unknown')} \n{pt.payload.get('content', '')}\n"
#         for pt in results.points
#     ])

def generate_response(prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "gemma3:4b-it-qat", "prompt": prompt, "stream": False}
        )
        print("📥 Ollama raw response:", response.text)

        data = response.json()
        if "response" not in data:
            return f"[⚠️ Ollama error] Unexpected format: {data}"
        return data['response']

    except Exception as e:
        print("❌ Failed to get response from Ollama:", e)
        return f"[❌ Ollama failure] {str(e)}"
    
if __name__=='__main__':
    print("Ensuring collection: ")
    ensure_collection_ready()

    while True:
        question = input("Enter your question: ")
        passages = query_relevant_chunks(question)
        print("✅ Retrieved passages:\n", passages)
        
    # augmented_prompt = f"""
    # Quyida O‘zbekiston Mehnat kodeksidan tegishli moddalari keltirilgan.
    # <retrieved-data>
    # {passages}
    # </retrieved-data>

    # Bu foydalanuvchining asl so'rovi. Olingan parchalar yordamida javob bering. Agar tegishli modda bo'lmasa, ayting:
    # <user-prompt>
    # {question}
    # </user-prompt>
    # """
    # print("📦 Sending prompt to Ollama...")

    # answer = generate_response(augmented_prompt)
    # print("✅ Got response from Ollama:", answer)
