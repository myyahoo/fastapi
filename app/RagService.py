from dotenv import load_dotenv
import os
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RagService:

    def __init__(self):
        load_dotenv()
        return

    def get_retriever(self):
        vectorstore = FAISS.load_local(os.getenv("DB_DIR"))
        retriever = vectorstore.as_retriver()
        return retriever

    def get_chain(self,question):

        retriever = self.get_retriever()

        prompt = hub.pull("rag/rag-prompt")
        api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                         openai_api_key=api_key)

        entry_point_chain = RunnableParallel({"context": retriever | self.format_docs, "question": RunnablePassthrough()})
        chain = entry_point_chain | prompt | llm | StrOutputParser()
        response = chain.invoke(question)

    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)


