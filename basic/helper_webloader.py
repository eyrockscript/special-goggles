# from langchain.document_loaders import WebBaseLoader

from langchain.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAWithSourcesChain
from langchain.memory import ConversationBufferMemory
import json

# loader = WebBaseLoader("https://www.pinecone.io/learn/langchain-prompt-templates/")

urls = []

print(len(urls))
# create urls array
# Opening JSON file
f_docs = open('./source/docs_url.json')
  
# returns JSON object as 
# a dictionary
data_docs = json.load(f_docs)
  
# Iterating through the json
# list
for i in data_docs['urls']:
    urls.append(i)

print(len(urls))

# Closing file
f_docs.close()

# Opening JSON file
f_zndsk = open('./source/zendesk_url.json')
  
# returns JSON object as 
# a dictionary
data_zndsk = json.load(f_zndsk)
  
# Iterating through the json
# list
for i in data_zndsk['urls']:
    urls.append(i)

print(len(urls))

# Closing file
f_zndsk.close()

# Opening JSON file
f_others = open('./source/others.json')
  
# returns JSON object as 
# a dictionary
data_others = json.load(f_others)
  
# Iterating through the json
# list
for i in data_others['urls']:
    urls.append(i)

print(len(urls))

# Closing file
f_others.close()

loader = PlaywrightURLLoader(urls)
pages = loader.load()
text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(pages)
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index_unstructured")