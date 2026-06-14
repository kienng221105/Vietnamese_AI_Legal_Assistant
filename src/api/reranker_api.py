import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sentence_transformers import CrossEncoder

# Khởi tạo FastAPI app
app = FastAPI(
    title="Legal Reranker API",
    description="API cho mô hình Cross-Encoder Reranker đã fine-tune trên dữ liệu pháp lý Việt Nam",
    version="1.0.0"
)

# Đường dẫn model (local path hoặc Hugging Face repo id)
# Mặc định ưu tiên repo fine-tuned đã push lên Hugging Face Hub.
MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH",
    "kiencnt2205/vietnamese-legal-reranker-bge-base",
)
MODEL_DEVICE = os.environ.get("RERANKER_DEVICE", "cpu")
MODEL_MAX_LENGTH = int(os.environ.get("RERANKER_MAX_LENGTH", "512"))
MODEL_BATCH_SIZE = int(os.environ.get("RERANKER_BATCH_SIZE", "16"))

print(f"Loading Reranker Model from: {MODEL_PATH} on device={MODEL_DEVICE} ...")
try:
    # Load model lên GPU (nếu có) hoặc CPU
    # bge-reranker output 1 score nên num_labels=1
    model = CrossEncoder(
        MODEL_PATH,
        num_labels=1,
        max_length=MODEL_MAX_LENGTH,
        device=MODEL_DEVICE,
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Định nghĩa các schema (Pydantic Models) cho API
class Document(BaseModel):
    id: str
    text: str # Gộp law_title và content vào đây

class RerankRequest(BaseModel):
    query: str
    documents: List[Document]
    top_k: int = 10 # Số lượng trả về tối đa (Optional)

class RerankResponse(BaseModel):
    id: str
    score: float

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "status": "healthy",
        "model_path": MODEL_PATH,
        "device": MODEL_DEVICE,
        "max_length": MODEL_MAX_LENGTH,
        "batch_size": MODEL_BATCH_SIZE,
    }

@app.post("/rerank", response_model=List[RerankResponse])
def rerank_documents(request: RerankRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    query = request.query
    documents = request.documents
    
    if not documents:
        return []

    # Format input cho CrossEncoder: List of pairs [ [query, doc_text], [query, doc_text] ]
    model_inputs = [[query, doc.text] for doc in documents]
    
    # Dự đoán (Predict scores)
    try:
        # CrossEncoder trả về list các score (float)
        scores = model.predict(model_inputs, batch_size=MODEL_BATCH_SIZE, show_progress_bar=False)
        
        # Gắn score vào ID
        results = []
        for doc, score in zip(documents, scores):
            results.append({
                "id": doc.id,
                "score": float(score)
            })
            
        # Sắp xếp giảm dần theo điểm
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Cắt lấy top_k
        if request.top_k > 0:
            results = results[:request.top_k]
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Cách chạy file này:
# Chạy terminal: uvicorn src.api.reranker_api:app --host 0.0.0.0 --port 8000 --workers 1
if __name__ == "__main__":
    uvicorn.run("src.api.reranker_api:app", host="0.0.0.0", port=8000, reload=True)
