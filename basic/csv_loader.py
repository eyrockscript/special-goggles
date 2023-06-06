# from langchain.document_loaders import WebBaseLoader

from langchain.document_loaders import CSVLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAWithSourcesChain
from langchain.memory import ConversationBufferMemory
import json


loader = CSVLoader(file_path="./source/Temario.csv", encoding="utf-8", csv_args={'delimiter': ','})
pages = loader.load()
text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(pages)
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index_csv")