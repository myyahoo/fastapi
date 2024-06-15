from fastapi import FastAPI
from pydantic import BaseModel
from service.rag_service import RagService
import uvicorn


app = FastAPI()

@app.get("/")
def root():
    return "true"


class Question(BaseModel):
    question: str

@app.post("/rag_chain")
def rag_chain(question:Question):
    rag_service = RagService()
    response = rag_service.get_chain(question.question)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)