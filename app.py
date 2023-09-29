import os

import openai
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
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS

from langchain.llms import HuggingFaceHub


def create_vectorstore(pdf):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    # docs = TextLoader(pdf).load_and_split(text_splitter)
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    return FAISS.from_texts(pdf, embeddings).as_retriever()


def get_retriever_list(pdf_docs):
    text = ""
    retriever_infos = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        store = create_vectorstore(text)
        retriever_infos.append({"name": "Leistungsbeschreibung", "description": "Beschreibung der Leistungen", "retriever": store})
    return retriever_infos


# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


def get_conversation_chain(retriever_infos):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = MultiRetrievalQAChain.from_retrievers(
        llm=llm,
        retriever_infos=retriever_infos,
     #   memory=memory,
     #   return_source_documents=True
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process the PDFs first.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i % 2 == 1:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv(override=True)
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                retriever_list = get_retriever_list(pdf_docs)

                # # get the text chunks
                # text_chunks = get_text_chunks(raw_text)
                #
                # # create vector store
                # vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    retriever_list)


if __name__ == '__main__':
    main()
