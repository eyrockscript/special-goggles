from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import json

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index_csv", embeddings)

# chain
chat_history = []
llm = OpenAI(temperature=0)

memory = ConversationBufferMemory(input_key='question', return_messages=True)

retriever = vectorstore.as_retriever()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    max_tokens_limit=None,
    return_source_documents=True,
    combine_docs_chain_kwargs={
        'memory': memory
    }
)

while True:
    print(">")
    question = input()
    result = qa_chain({"question": question, "chat_history": chat_history})
    
    # print("------")
    # print(result)
    # print("------")
    
    chat_history.append((question, result["answer"]))
    print("IA:")
    print(result["answer"])
