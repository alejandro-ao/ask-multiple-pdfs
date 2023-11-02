"""
For PyCharm users:
    The "configuration" in Pycharm that allows for both Run & Debug:
        use the module: streamlit (instead of script -> python.exe)
        script parameters: run app.py
"""

import streamlit as st
from dotenv import load_dotenv
import pypdfium2 as pdfium  # Check leaderboard here: https://github.com/py-pdf/benchmarks  # yiwei-ang:feature/pdfium
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
        pdf_reader = pdfium.PdfDocument(pdf)
        for i in range(len(pdf_reader)):
            page = pdf_reader.get_page(i)
            textpage = page.get_textpage()
            text += textpage.get_text_range() + "\n"
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=5000, chunk_overlap=500, length_function=len)
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
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",
                         model_kwargs={"temperature": 0.1,
                                       "max_new_tokens": 1000})   # ,"max_length": 1000})
    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriever=vectorstore.as_retriever(),
                                                               memory=memory)
    return conversation_chain


def save_question_and_clear_prompt(ss):
    ss.user_question = ss.prompt_bar
    ss.prompt_bar = ""  # clearing the prompt bar after clicking enter to prevent automatic re-submissions


def write_chat(msgs):  # Write the Q&A in a pretty chat format
    for i, msg in enumerate(msgs):
        if i % 2 == 0:  # it's a question
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:  # it's an answer
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


def main():
    load_dotenv()  # loads api keys
    ss = st.session_state  # https://docs.streamlit.io/library/api-reference/session-state

    # Page design
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    # Initializing session state variables
    if "conversation_chain" not in ss:
        ss.conversation_chain = None  # the main variable storing the llm, retriever and memory
    if "prompt_bar" not in ss:
        ss.prompt_bar = ""
    if "user_question" not in ss:
        ss.user_question = ""
    if "docs_are_processed" not in ss:
        ss.docs_are_processed = False

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)  # get pdf text
                text_chunks = get_text_chunks(raw_text)  # get the text chunks
                vectorstore = get_vectorstore(text_chunks)  # create vector store
                ss.conversation_chain = get_conversation_chain(vectorstore)  # create conversation chain
                ss.docs_are_processed = True
        if ss.docs_are_processed:
            st.text('Documents processed')

    st.text_input("Ask a question here:", key='prompt_bar', on_change=save_question_and_clear_prompt(ss))

    if ss.user_question:
        ss.conversation_chain({'question': ss.user_question})  # This is what gets the response from the LLM!
        if hasattr(ss.conversation_chain.memory, 'chat_memory'):
            chat = ss.conversation_chain.memory.chat_memory.messages
            write_chat(chat)

    if hasattr(ss.conversation_chain, 'memory'):  # There is memory if the documents have been processed
        if hasattr(ss.conversation_chain.memory, 'chat_memory'):  # There is chat_memory if questions have been asked
            if st.button("Forget conversation"):  # adding a button
                ss.conversation_chain.memory.chat_memory.clear()  # clears the ConversationBufferMemory

    # st.write(ss)  # use this when debugging for visualizing the session_state variables


if __name__ == '__main__':
    main()
