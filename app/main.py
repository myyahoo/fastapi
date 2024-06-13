from fastapi import FastAPI
import RagService

app = FastAPI()

@app.get("/")
def root():
    return "test"


@app.get("/rag_chain")
def rag_chaing():
    rag_service = RagService()
