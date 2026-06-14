from fastapi import APIRouter, Depends
from src.api.schemas import ChatRequest, ChatResponse
from src.pipeline.rag_pipeline import RAGPipeline

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # TODO: Implement RAG pipeline call
    return ChatResponse(
        answer="Hệ thống trợ lý pháp luật đang được khởi tạo.",
        sources=[]
    )