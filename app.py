import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

bot_msg_container_html_template = '''
<div style='background-color: #475063; padding: 1.5rem; border-radius: 5px; margin-bottom: 10px; display: flex'>
    <div style="width: 20%; display: flex; justify-content: center">
        <img src="https://i.ibb.co/yVFMnvj/Photo-logo-4.png" style="max-height: 50px; max-width: 50px; border-radius: 50%;">
    </div>
    <div style="width: 80%;">
        $MSG
    </div>
</div>
'''

user_msg_container_html_template = '''
<div style='background-color: #2b313e; padding: 1.5rem; border-radius: 5px; margin-bottom: 10px; display: flex'>
    <div style="width: 78%">
        $MSG
    </div>
    <div style="width: 20%; margin-left: auto; display: flex; justify-content: center;">
        <img src="https://i.ibb.co/dpRdwK0/Photo-logo.jpg" style="max-width: 50px; max-height: 50px; float: right; border-radius: 50%;">
    </div>    
</div>
'''

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
  if "conversation" not in st.session_state:
    st.session_state.conversation = None
  if "history" not in st.session_state:
    st.session_state.history = []
  
  
  with st.sidebar:
    st.subheader('Your documents')
    pdf_documents = st.file_uploader('Upload your PDF files and click on "Process"', type=['pdf'], accept_multiple_files=True)
  
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
      st.session_state.history = response['chat_history']
      
      for i, message in enumerate(st.session_state.history):
        if i % 2 == 0:
            st.write(user_msg_container_html_template.replace("$MSG", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_msg_container_html_template.replace("$MSG", message.content), unsafe_allow_html=True)
      
      
  else:
    st.write('Waiting for your question...')
  
  # st.write(user_msg_container_html_template.replace("$MSG", "hello there"), unsafe_allow_html=True)

if __name__ == '__main__':
    main()