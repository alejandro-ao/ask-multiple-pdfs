import os
from typing import List, Dict

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
from typing.io import IO

from htmlTemplates import css, bot_template, user_template
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub


class PDFProcessor:
    @staticmethod
    def create_vectorstore(text: str) -> FAISS:
        """
        This function takes a text string and generates a FAISS retriever object.
        :param text: str: Input text
        :return: FAISS: A FAISS retriever object.
        """
        try:
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,  # Consider making these values configurable through parameters or config
                chunk_overlap=200,
                length_function=len
            )

            embeddings = OpenAIEmbeddings()
            # embeddings = HuggingFaceInstructEmbeddings(model_name="model_name")  # This line is commented out; adjust as per requirement

            # Creating a FAISS retriever
            retriever = FAISS.from_texts(text, embeddings).as_retriever()

            return retriever

        except Exception as e:
            # Properly log the exception, or re-raise it if it can't be handled here
            raise ValueError(f"Failed to create vector store: {str(e)}")

    @staticmethod
    def get_retriever_list(pdf_docs: List[IO]) -> List[Dict]:
        """
        Processes a list of PDF documents and returns a list of retriever information dictionaries.

        :param pdf_docs: List of PDF documents.
        :return: List of dictionaries containing retriever information.
        """
        retriever_infos = []

        try:
            for pdf in pdf_docs:
                text = ""
                pdf_reader = PdfReader(pdf)

                for page in pdf_reader.pages:
                    text += page.extract_text()

                # Assuming create_vectorstore is a static method in the same class
                retriever = PDFProcessor.create_vectorstore(text)

                retriever_infos.append({
                    "name": "Leistungsbeschreibung",
                    "description": "studie der it trends im jahr 2022",
                    "retriever": retriever
                })

            return retriever_infos

        except Exception as e:
            # Log the error if any occurred during the processing of PDFs.
            print(f"Error occurred while processing PDFs: {e}")


class ConversationManager:
    @staticmethod
    def get_conversation_chain(retriever_infos: List[Dict]) -> MultiRetrievalQAChain:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = MultiRetrievalQAChain.from_retrievers(
            llm=llm,
            retriever_infos=retriever_infos,
            verbose=True,
            #   memory=memory,
            #   return_source_documents=True
        )
        return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process the PDFs first.")
        return

    response = st.session_state.conversation.run(user_question)
    print(response)
    st.session_state.chat_history.append((user_question, response))

    for user_question, response in reversed(st.session_state.chat_history):
        st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
        st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)


def main():
    load_dotenv(override=True)
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question and user_question != st.session_state.chat_history[-1][0]:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                retriever_list = PDFProcessor.get_retriever_list(pdf_docs)
                # create conversation chain
                st.session_state.conversation = ConversationManager.get_conversation_chain(
                    retriever_list)


if __name__ == '__main__':
    main()
