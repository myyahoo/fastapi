from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community import document_loaders
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

load_dotenv()

#loader = document_loaders.text.TextLoader("../source_data/smile.txt",encoding='UTF8')
loader = PyPDFLoader(file_path="../source_data/test.pdf")
doc = loader.load()
print(doc[10])

text_split = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)

split = text_split.split_documents(doc)

open_api_key = os.getenv("OPENAI_API_KEY")
embading = OpenAIEmbeddings(openai_api_key=open_api_key)

vectorstore = FAISS.from_documents(split,embading)

vectorstore.save_local(os.getenv("DB_DIR"))

