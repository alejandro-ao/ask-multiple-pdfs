from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from src.SwitchLLM import switchLLM as llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nomic
from nomic import atlas
import numpy as numpy
import os.path


def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(
        separators=[" ", ",", "\n"],
        chunk_size=800, #the size of the chunk itself. 1000 characters 
        chunk_overlap=200, #you may lose some context if you end in the middle. starts the next chunk a few characters before. 
        length_function=len
    )
        chunks = text_splitter.split_text(text)
        return chunks
    
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="intfloat/e5-large-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore

def get_embeddings(text_chunks):
    embeddings =HuggingFaceInstructEmbeddings(model_name="intfloat/e5-large-v2")
    return embeddings.embed_documents(text_chunks)
    

def get_conversation_chain(vectorstore,llm):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def visualize(text,projectName):
    ids=[]
    for id in text:
        ids.append(id)
    nomic.login(os.getenv("ATLAS_TEST_API_KEY"))
    embeddings=get_embeddings(text)
    numpy_embeddings= numpy.array(embeddings)
                    
    onlineMap= atlas.map_embeddings(name='projectName',
    description= "",
    is_public = True,
    reset_project_if_exists=True,
    embeddings= numpy_embeddings,
    data=[{'id': id} for id in ids])
    print(onlineMap.maps)