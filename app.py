import streamlit as st  #streamlit is the GUI 
from dotenv import load_dotenv
from src.htmlTemplates import css, bot_template, user_template
from src.PDFHandler import PDFHandler
from src.chain import get_conversation_chain, get_text_chunks, get_vectorstore
#############################################################################################

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
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

    st.header("CIxD Senior Bot :books:")
    ## ---- Mode Selection
    modeselect = st.selectbox('Select the Mode: ', ('Summary','Section Q&A', 'Free Talking','Topic Recommendation' ))
    st.write('Selected Mode: ', modeselect)

    #store the value of the text input. just string 
    user_question = st.text_input("Ask a question about to your senior:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        pdf_docs = st.file_uploader(
           "Upload your own PDF here and click on 'Process'", accept_multiple_files=False) #this just lets us enable this function
        st.subheader("CIxD Papers")
        st.caption("Select one or more papers to learn.")
        pdf_list= ["Design Opportunities in Three Stages", "Teaching-Learning Interaction","Co-Performing Agent","Non-finito","Ten-Minute Silence"]
        pdf_file_names=["pdf/file_1.pdf","pdf/file_2.pdf","pdf/file_3.pdf","pdf/file_4.pdf","pdf/file_5.pdf"]
        pdf_checkbox=[]
        for pdf in pdf_list:
            pdf_checkbox.append(st.checkbox(pdf))
        ## ---- file uploader
       
        if st.button("Process"): 
            raw_text=''
            #all the process is going to be processed while the user sees a spinning wheel. The program is running and not frozen
            with st.spinner("Processing"): 
                for index, checkedpdf in enumerate(pdf_checkbox):
                    if(checkedpdf):
                        PDFHandler.get_pdf_text(pdf_file_names[index])
                        raw_text+=PDFHandler.get_pdf_text(pdf_file_names[index])
                
                if(pdf_docs):
                   raw_text=PDFHandler.get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
       
if __name__ == '__main__': 
    main()
