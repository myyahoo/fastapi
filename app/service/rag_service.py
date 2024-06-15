from dotenv import load_dotenv
import os
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate


class RagService:

    def __init__(self):
        load_dotenv()

    def get_retriever(self):

        embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.load_local(os.getenv("DB_DIR"), embeddings=embedding,allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        return retriever

    def get_chain(self,question):

        retriever = self.get_retriever()

        template = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        Question: {question} 
        
        Context: {context} 
        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        print(prompt)
        #prompt = hub.pull("rlm/rag-prompt")

        api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                         openai_api_key=api_key)

        # 아래2개는 동일 RunnablePassthrough  사용할경우 RunnableParallel 로 자동 랩핑
        #entry_point_chain = RunnableParallel({"context": retriever | self.format_docs, "question": RunnablePassthrough()})
        entry_point_chain = {"context":retriever | self.format_docs, "question": RunnablePassthrough()}
        #chain = {"context":retriever | self.format_docs, "question":RunnablePassthrough() } | prompt | llm | StrOutputParser()
        chain = entry_point_chain | prompt | llm | StrOutputParser()

        #xx = retriever.invoke(question)
        #print(xx)
        response = chain.invoke(question)

        return response

    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)


