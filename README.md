# 🇺🇿 RAG Interface for Legal Docuemt Querying and Downloading

Build the container in docker:

```
docker compose build

```

After the container is built, run:

```
docker compose up
```

Open http://localhost:8501/ to see the app.

**To use local models instead of OpenAI API:**

*For response generation in rag_engine.py, uncomment the code:*
```
response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "gemma3:4b-it-qat", "prompt": prompt, "stream": False}
        )
        print("📥 Ollama raw response:", response.text)

```
