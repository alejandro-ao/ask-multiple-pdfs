from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from src.SwitchLLM import switchLLM as llm
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

def get_conversation_chain(vectorstore,llm):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

