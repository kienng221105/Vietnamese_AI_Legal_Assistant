import uvicorn
from fastapi import FastAPI
from src.api.routers import chat, admin, kg
from src.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="Vietnamese Legal Assistant Graph RAG API",
    version="1.0.0"
)

# Include API routers
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
app.include_router(kg.router, prefix="/api/v1/kg", tags=["Knowledge Graph"])

@app.get("/")
async def root():
    return {"message": "Welcome to Vietnamese AI Legal Assistant API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)