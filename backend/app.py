# from fastapi import FastAPI, HTTPException
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from rag_engine import query_relevant_chunks, generate_response
# from rag_engine import ensure_collection_ready, find_form_by_ai_query
# import os
# import glob
# from pathlib import Path

# app = FastAPI()

# class Query(BaseModel):
#     question: str

# class DocumentRequest(BaseModel):
#     document_name: str

# # Configuration
# RAG_TEXTS_PATH = "/app/GGTexts"        
# DOCUMENTS_PATH = "/app/arizalar"      

# def list_available_documents() -> list:
#     print(f"📂 Listing documents in: {DOCUMENTS_PATH}")
#     docx_files = glob.glob(os.path.join(DOCUMENTS_PATH, "*.docx"), recursive=True)
#     print(f"📄 Found {len(docx_files)} DOCX files: {docx_files}")
#     return [os.path.basename(f) for f in docx_files]

# @app.post("/ask")
# def ask_question(query: Query):
#     print("Ensuring collection: ")
#     ensure_collection_ready(RAG_TEXTS_PATH)

#     print("✅ Received question:", query.question)
    
#     try:
#         passages = query_relevant_chunks(query.question)
#         print("✅ Retrieved passages:\n", passages)
        
#         augmented_prompt = f"""
#         Siz O'zbekiston Mehnat kodeksi bo'yicha savollarga javob beruvchi yordamchisiz.
#         Kontekst: 
#         {passages} 

#         Foydalanuvchi savoli: 
#         {query.question}

#         Ko'rsatmalar:

#         * Faqat taqdim etilgan ma'lumotlardan foydalanib javob bering.
#         * Javob berishda qaysi moddalardan foydalanganingizni ko'rsating.
#         * Qisqa, aniq va tushunarli javob bering.
#         * Agar javob kontekstda topilmasa, shuni ayting.
#         * Qo'shimcha tafsilotlar yoki taxminlar kiritmang.
#         """
#         print("📦 Sending prompt to a model...")

#         answer = generate_response(augmented_prompt)
#         print("✅ Got response:", answer)

#         return {"answer": answer}

#     except Exception as e:
#         print("❌ ERROR:", str(e))
#         return {"error": str(e)}

# @app.post("/get_document")
# def get_document(request: DocumentRequest):
#     print(f"🔍 Searching for document: {request.document_name}")
    
#     try:        
        
#         ai_result = find_form_by_ai_query(request.document_name)
        
#         if ai_result.get("error"):
#             available_docs = list_available_documents()
#             raise HTTPException(
#                 status_code=404, 
#                 detail={
#                     "error": f"Form '{request.document_name}' not found.",
#                     "ai_search_error": ai_result.get("error"),
#                     "available_documents": available_docs[:10],
#                 }
#             )
        
#         filename = ai_result.get("filename")
#         confidence = ai_result.get("confidence", 0)
#         document_info = ai_result.get("document_info", {})
        
#         print(f"🎯 AI found form: {filename} (confidence: {confidence:.2f})")
        
#         if confidence < 0.6:  
#             available_docs = list_available_documents()
#             raise HTTPException(
#                 status_code=404,
#                 detail={
#                     "error": f"Low confidence match for '{request.document_name}'",
#                     "ai_suggestion": {
#                         "filename": filename,
#                         "title": document_info.get("title"),
#                         "description": document_info.get("description"),
#                         "confidence": confidence,
#                     },
#                     "alternatives": ai_result.get("alternatives", []),
#                     "available_documents": available_docs[:10],
#                 }
#             )
        
#         document_path = os.path.join(DOCUMENTS_PATH, filename)
        
#         if not os.path.exists(document_path):
#             raise HTTPException(
#                 status_code=404,
#                 detail={
#                     "error": f"Form '{filename}' found in metadata but file doesn't exist on disk.",
#                     "ai_result": {
#                         "filename": filename,
#                         "title": document_info.get("title"),
#                         "confidence": confidence
#                     }
#                 }
#             )
                    
#         if not os.path.exists(document_path):
#             raise HTTPException(status_code=404, detail={"error": "Document file not found on disk."})
        
#         filename = os.path.basename(document_path)
#         print(f"✅ Returning document: {filename}")
        
#         return FileResponse(
#             path=document_path,
#             filename=filename,
#             media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
#             headers={"Content-Disposition": f"attachment; filename={filename}"}
#         )
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"❌ ERROR retrieving document: {str(e)}")
#         raise HTTPException(status_code=500, detail={"error": f"Internal server error: {str(e)}"})


from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag_engine import query_relevant_chunks, generate_response
from rag_engine import ensure_collection_ready, find_form_by_ai_query
import os
import glob
from pathlib import Path
from openai import OpenAI

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key="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")

class IntelligentQuery(BaseModel):
    query: str

# Configuration
RAG_TEXTS_PATH = "/app/GGTexts"        
DOCUMENTS_PATH = "/app/arizalar"      

def list_available_documents() -> list:
    """List all available documents"""
    print(f"📂 Listing documents in: {DOCUMENTS_PATH}")
    docx_files = glob.glob(os.path.join(DOCUMENTS_PATH, "*.docx"), recursive=True)
    print(f"📄 Found {len(docx_files)} DOCX files: {docx_files}")
    return [os.path.basename(f) for f in docx_files]

def route_query(user_input: str) -> str:
    routing_prompt = f"""
    Quyidagi foydalanuvchi so'rovini tahlil qiling va eng mos keladigan amalni tanlang:
    
    So'rov: "{user_input}"
    
    Quyidagi amallardan BIRINI tanlang:
    1. RAG_QUERY - Foydalanuvchi Mehnat kodeksidan ma'lumot yoki javob so'ramoqda
    2. DOCUMENT_RETRIEVAL - Foydalanuvchi aniq hujat, ariza yoki forma topmoqchi
    
    Misollar:
    - "Mehnat shartnomasi qanday tuziladi?" → RAG_QUERY
    - "Ta'til uchun ariza topish kerak" → DOCUMENT_RETRIEVAL  
    - "Mehnat kodeksida dam olish kunlari haqida nima deyiladi?" → RAG_QUERY
    - "Kasallik varag'i uchun ariza" → DOCUMENT_RETRIEVAL
   
    Faqat shu so'zlardan javob berib, qo'shimcha yozmang: RAG_QUERY yoki DOCUMENT_RETRIEVAL
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": routing_prompt}]
        )
        
        route_decision = response.choices[0].message.content.strip()
        print(f"🧠 Routing decision: {route_decision}")
        
        # Clean up the response to ensure it's one of our expected values
        if "RAG_QUERY" in route_decision.upper():
            return "RAG_QUERY"
        elif "DOCUMENT_RETRIEVAL" in route_decision.upper():
            return "DOCUMENT_RETRIEVAL"
        else:
            # Default fallback - if unclear, assume RAG query
            print(f"⚠️ Unclear routing decision: {route_decision}, defaulting to RAG_QUERY")
            return "RAG_QUERY"
            
    except Exception as e:
        print(f"❌ Routing error: {str(e)}")
        return "RAG_QUERY"

def execute_rag_search(query: str):
    """Execute RAG search and return answer"""
    print(f"🔍 Executing RAG search for: {query}")
    
    try:
        # Ensure RAG collection is ready
        ensure_collection_ready(RAG_TEXTS_PATH)
        
        # Get relevant passages
        passages = query_relevant_chunks(query)
        print("✅ Retrieved RAG passages")
        
        # Create augmented prompt
        augmented_prompt = f"""
        Siz O'zbekiston Mehnat kodeksi bo'yicha savollarga javob beruvchi yordamchisiz.
        Kontekst: 
        {passages} 

        Foydalanuvchi savoli: 
        {query}

        Ko'rsatmalar:
        * Faqat taqdim etilgan ma'lumotlardan foydalanib javob bering.
        * Javob berishda qaysi moddalardan foydalanganingizni ko'rsating.
        * Qisqa, aniq va tushunarli javob bering.
        * Agar javob kontekstda topilmasa, shuni ayting.
        * Qo'shimcha tafsilotlar yoki taxminlar kiritmang.
        """
        
        # Generate response
        answer = generate_response(augmented_prompt)
        print("✅ Generated RAG response")
        
        return {"success": True, "answer": answer}
        
    except Exception as e:
        print(f"❌ RAG search error: {str(e)}")
        return {"success": False, "error": str(e)}

def execute_document_search(query: str):
    """Execute document search and return document info"""
    print(f"📄 Executing document search for: {query}")
    
    try:
        # Use existing AI document search
        ai_result = find_form_by_ai_query(query)
        
        if ai_result.get("error"):
            return {"success": False, "error": ai_result.get("error")}
        
        filename = ai_result.get("filename")
        confidence = ai_result.get("confidence", 0)
        document_info = ai_result.get("document_info", {})
        
        print(f"🎯 AI found document: {filename} (confidence: {confidence:.2f})")
        
        # Check confidence threshold
        if confidence < 0.6:
            return {
                "success": False, 
                "error": f"Aniq mos keluvchi hujat topilmadi (ishonch: {confidence:.2f})",
                "suggestion": {
                    "filename": filename,
                    "title": document_info.get("title"),
                    "confidence": confidence
                },
                "alternatives": ai_result.get("alternatives", [])
            }
        
        # Check if file exists
        document_path = os.path.join(DOCUMENTS_PATH, filename)
        if not os.path.exists(document_path):
            return {"success": False, "error": f"Hujat '{filename}' diskda topilmadi"}
        
        return {
            "success": True,
            "filename": filename,
            "path": document_path,
            "title": document_info.get("title", ""),
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"❌ Document search error: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/intelligent_query")
def intelligent_query(request: IntelligentQuery):
    print(f"🧠 Intelligent query received: {request.query}")
    
    try:
        mode = route_query(request.query)
        print(f"📍 Routed to mode: {mode}")
        
        if mode == "RAG_QUERY":
            result = execute_rag_search(request.query)
            
            if not result["success"]:
                raise HTTPException(status_code=500, detail={"error": result["error"]})
            
            return {
                "answer": result["answer"],
                "mode": "rag_query",
                "query": request.query
            }
            
        elif mode == "DOCUMENT_RETRIEVAL":
            result = execute_document_search(request.query)
            
            if not result["success"]:
                error_detail = {"error": result["error"]}
                raise HTTPException(status_code=404, detail=error_detail)
            
            # Return file download
            return FileResponse(
                path=result["path"],
                filename=result["filename"],
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f"attachment; filename={result['filename']}"}
            )
            
        else:
            raise HTTPException(status_code=500, detail={"error": "Noma'lum rejim"})
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Intelligent query error: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": f"Server xatosi: {str(e)}"})

