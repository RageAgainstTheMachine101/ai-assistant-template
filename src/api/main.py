from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from ..agents.base_agent import ConversationManager
from ..api.security import SecurityConfig

app = FastAPI(title="LLM Agent API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    context: Optional[List[str]] = None

class Response(BaseModel):
    answer: str
    sources: Optional[List[str]] = None

# Initialize conversation manager
conversation_manager = ConversationManager()

@app.get("/")
async def root():
    return {"message": "Welcome to LLM Agent API"}

@app.post("/query", response_model=Response)
async def query(
    query: Query,
    api_key: str = Depends(SecurityConfig.validate_api_key)
):
    try:
        # Sanitize the query to prevent prompt injection
        sanitized_question = SecurityConfig.sanitize_query(query.question)
        
        # Process the query through the conversation manager
        result = await conversation_manager.process_query(
            question=sanitized_question,
            context=query.context
        )
        
        return Response(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
