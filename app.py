import streamlit  #streamlit is the GUI 
from src.htmlTemplates import css, bot_template, user_template
from src.chain import get_conversation_chain, get_text_chunks, get_vectorstore
from dotenv import load_dotenv
from src.PDFHandler import PDFHandler


def handle_userInput(user_question):
    response = streamlit.session_state.conversation({'question': user_question})
    streamlit.session_state.chat_history = response['chat_history']
    for i, message in enumerate(streamlit.session_state.chat_history):
        if i % 2 == 0:
            streamlit.write(user_template.replace(
            "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            streamlit.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    #########variables##############
    pageTitle= "CIXD Senior Bot"
    pageIcon=':books'
    modeOptions=['Summary','Section Q&A', 'Free Talking','Topic Recommendation' ]
    textInputQuestion="Ask a question about to your senior:"
    header= "CIxD Senior Bot :books:"
    fileUploadText="Upload your own PDF here and click on 'Process'"
    accept_multiple_files=False
    subheaderText="CIxD Papers"
    captionText="Select one or more papers to learn."
    
    pdf_list=  ["Design Opportunities in Three Stages", "Teaching-Learning Interaction","Co-Performing Agent","Non-finito","Ten-Minute Silence"]
    pdf_file_names=["pdf/file_1.pdf","pdf/file_2.pdf","pdf/file_3.pdf","pdf/file_4.pdf","pdf/file_5.pdf"]
    pdf_checkbox=[]
        
    #############others#######################
    load_dotenv()
    
    ########initialize#######################
    
    streamlit.set_page_config(page_title=pageTitle,page_icon=pageIcon)
    streamlit.write(css, unsafe_allow_html=True)
    streamlit.header(header)
    modeselect = streamlit.selectbox('Select the Mode: ', options=[*modeOptions])
    streamlit.write('Selected Mode: ', modeselect)
    user_question = streamlit.text_input(textInputQuestion)
    
    if "conversation" not in streamlit.session_state:
        streamlit.session_state.conversation = None
    if  "chat_history" not in streamlit.session_state:
        streamlit.session_state.chat_history = None
        
    with streamlit.sidebar:
        pdf_docs = streamlit.file_uploader(
                fileUploadText, accept_multiple_files= accept_multiple_files) #this just lets us enable this function
        streamlit.subheader(subheaderText)
        streamlit.caption(captionText)
        for pdf in pdf_list:
                pdf_checkbox.append(streamlit.checkbox(pdf))
        
     ########interaction#######################
    if user_question:
        handle_userInput(user_question)
        
    with streamlit.sidebar:
            if streamlit.button("Process!"):
                raw_text=''
                with streamlit.spinner("Processing"): 
                    
                    ##SummaryMode
            
                    for index, checkedpdf in enumerate(pdf_checkbox):
                        if(checkedpdf):
                            raw_text+=PDFHandler.get_unfiltered_pdf_text(pdf_file_names[index])
                            
                    if(pdf_docs):
                        raw_text=PDFHandler.get_unfiltered_pdf_text(pdf_docs)
                
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    streamlit.session_state.conversation = get_conversation_chain(
                    vectorstore)

  #########################################
if __name__ == '__main__': 
    main()



        