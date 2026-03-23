import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.query import run_bot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",              
        "https://clutchquery.vercel.app",          
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Define shape of request body
class QuestionRequest(BaseModel):
    question: str
    history: list = []

@app.post("/ask")
async def ask(request: QuestionRequest):
    answer = await run_bot(request.question, request.history)
    return {"answer": answer}