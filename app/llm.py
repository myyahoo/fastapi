llm = OpenAI()
retriever = FAISS.as_retriever()

rag_chain = {} | prompt | llm | StrOutputParser()

