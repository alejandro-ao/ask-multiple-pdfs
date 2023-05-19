import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv


def get_pdf_text(pdf_documents):
  text = ""
  for pdf in pdf_documents:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
  
  
def get_vectorstore(text_chunks):
  embeddings = OpenAIEmbeddings()
  
  vector_store = FAISS.from_texts(text_chunks, embeddings)
  
  return vector_store


def main():
  load_dotenv()
  st.set_page_config(page_title='Chat Multiple PDFs',page_icon=':books:')
  
  
  with st.sidebar:
    pdf_documents = st.file_uploader('Upload your PDF file', type=['pdf'], accept_multiple_files=True)
  
    if st.button('Process'):
      with st.spinner('Processing...'):
        pdf_text = get_pdf_text(pdf_documents)
        
        text_chunks = get_chunks(pdf_text)
        
        vectorstore = get_vectorstore(text_chunks)
        
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
          OpenAI(temperature=0), 
          vectorstore.as_retriever(), 
          memory=memory
        )
        
        st.success('Done! You can now ask questions to your PDFs')


  st.title('Chat with Multiple PDFs :books:')
  user_question = st.text_input('Ask a question about the PDFs')
  if user_question is not None and user_question != '':
    with st.spinner('Searching...'):
      response = st.session_state.conversation({'question': user_question})
      st.write(response)
  else:
    st.write('Waiting for your question...')

if __name__ == '__main__':
    main()