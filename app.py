import streamlit as st  #streamlit is the GUI 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
#langchain is used to interact with the model
from huggingface_endpoint import HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from htmlTemplates import css, bot_template, user_template
from typing import Any, Dict, List, Mapping, Optional
import requests
from pydantic import Extra, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
from langchain.llms import HuggingFaceHub # hugging face can replace the openAI model
import os
from wz13 import wizardVicuna13
from getpass import getpass
from langchain import PromptTemplate, LLMChain
from repli import Replicate

from langchain.llms.octoai_endpoint import OctoAIEndpoint
from octoAICloud import OctoAiCloudLLM
#############################################################################################

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800, #the size of the chunk itself. 1000 characters 
        chunk_overlap=0, #you may lose some context if you end in the middle. starts the next chunk a few characters before. 
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print("chunk length:", len(chunks))
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = HuggingFaceInstructEmbeddings(model_name="intfloat/e5-large-v2")

    #Faiss is a database that allows you to store these embeddings
    #More importantly, Faiss runs locally 
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
#     llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs=
#                          {"temperature":0.5, "max_length":1024,
#                            "top_k": 30,
#   "top_p": 0.9,
#   "repetition_penalty": 1.02})

#     llm= HuggingFaceEndpoint(endpoint_url=os.getenv('ENDPOINT_URL'),task="text-generation",
#                               model_kwargs={"max_new_tokens": 512,
#   "top_k": 30,
#   "top_p": 0.9,
#  "temperature": 0.2,
#   "repetition_penalty": 1.02,})

    # llm=wizardVicuna13()
    
    llm = Replicate(
   model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
    # model="joehoover/falcon-40b-instruct:7eb0f4b1ff770ab4f68c3a309dd4984469749b7323a3d47fd2d5e09d58836d3c",
input= {"max_length":8000,"max_new_tokens": 8000})
  
    # llm = OctoAIEndpoint(
    #     model_kwargs={
    #         "max_new_tokens": 200,
    #         "temperature": 0.75,
    #         "top_p": 0.95,
    #         "repetition_penalty": 1,
    #         "seed": None,
    #         "stop": [],
    #     },
    # )

    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):

    #this is the response from the llm 
    response = st.session_state.conversation({'question': user_question})
    # `st.write(response)`
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
 
                "{{MSG}}", message.content), unsafe_allow_html=True)
#############################################################################################
# 메인 function
def main():
    load_dotenv()
    st.set_page_config(page_title="CIXD Senior Bot",
                       page_icon=":books:")
    #apply the css and html 
    st.write(css, unsafe_allow_html=True)

    #if we are using a session_state object, it is a good idea to initialize it before
    #initialize the object if it does not exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    #initialize the object if it does not exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("CIXD Senior Bot :books:")
    #store the value of the text input. just string 
    user_question = st.text_input("Ask a question about to your senior:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True) #this just lets us enable this function 
        if st.button("Process"):
            #all the process is going to be processed while the user sees a spinning wheel. The program is running and not frozen
            with st.spinner("Processing"): 
                # get pdf text
                # single string of the entire text 
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                # a list of the chucks of text 
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                #everytime someone clicks on a button, streamlit tries to reload the code. so if in order for the variables to not disappear, we 
                #want to save this to the streamlit session state 
                #also, the session state is a global variable that can be accessed elswhere 
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

#if the app is being executed directly and not imported, then run the main func 
if __name__ == '__main__': 
    main()
