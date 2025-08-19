# 🇺🇿 RAG Interface for Legal Document Querying and Downloading

<img width="1116" height="800" alt="Image" src="https://github.com/user-attachments/assets/41302761-efec-4e5b-9716-f98a98618e34" />
*Querying the Labor Code*

<img width="1101" height="654" alt="Image" src="https://github.com/user-attachments/assets/2c1b38e5-d029-47b3-b1e4-e41e41539063" />
*Retrieving necessery documents for downlload*

Build the container in docker:

```
docker compose build

```

After the container is built, run:

```
docker compose up
```

Open http://localhost:8501/ to see the app.

<img width="983" height="438" alt="Image" src="https://github.com/user-attachments/assets/2033edd6-b95d-4251-8730-ce69a59105c3" />
*Logic behind the ReAct*

**To use local models instead of OpenAI API:**

*For response generation in rag_engine.py, uncomment the code:*
```
response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "gemma3:4b-it-qat", "prompt": prompt, "stream": False}
        )
        print("📥 Ollama raw response:", response.text)

```
