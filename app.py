"""
Configuration in Pycharm that allows for both Run & Debug:
module (instead of script -> python.exe): streamlit
script parameters: run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=4000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # Check leaderboard here: https://huggingface.co/spaces/mteb/leaderboard
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # Check leaderboard here: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha", model_kwargs={"temperature": 0.5,
                                                                                "max_length": 512,
                                                                                'max_new_tokens': 512})

    st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriever=vectorstore.as_retriever(),
                                                               memory=st.session_state.memory)
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with the FEED Phase reports", page_icon=":bridge_at_night:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with the FEED Phase reports :bridge_at_night:")

    # Initializing session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "prompt_bar" not in st.session_state:
        st.session_state.prompt_bar = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = None
    if "doc_process_status" not in st.session_state:
        st.session_state.doc_process_status = None

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)  # get pdf text
                text_chunks = get_text_chunks(raw_text)  # get the text chunks
                vectorstore = get_vectorstore(text_chunks)  # create vector store
                st.session_state.conversation = get_conversation_chain(vectorstore)  # create conversation chain
            st.write('Documents processed')

    def save_question_clear_prompt(ss):  # clearing the prompt bar after clicking enter
        ss.user_question = st.session_state.prompt_bar
        ss.prompt_bar = None

    st.text_input("Ask a question here:", key='prompt_bar', on_change=save_question_clear_prompt(st.session_state))

    if st.session_state.user_question:
        response = st.session_state.conversation({'question': st.session_state.user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    if st.button("Clear & forget conversation"):
        st.session_state.memory.clear()

    print(st.session_state.conversation.memory)

if __name__ == '__main__':
    main()
