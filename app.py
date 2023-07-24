from st_chat_message import message
import streamlit  #streamlit is the GUI 
from src.htmlTemplates import css, bot_template, user_template
from src.chain import get_conversation_chain, get_text_chunks, get_vectorstore
from dotenv import load_dotenv
from src.PDFHandler import PDFHandler
from os import listdir
from os.path import isfile, join
from src.SwitchLLM import switchLLM
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import  HumanMessage, SystemMessage

from src.config import LLMList


def handle_userInput(user_question,response_container):
    response = streamlit.session_state.conversation({'question': user_question})
    streamlit.session_state.chat_history = response['chat_history']
    

def main():
    #########variables##############
    pageTitle= "CIxD AIPowered Studying System"
    pageIcon=':robot_face:'
    modeOptions=['Summary','Question Generation','Answer Generation', 'Topic Recommendation']
    header= "CIxD AIPowered Studying System :robot_face:"
    fileUploadText="Upload a recent Cixd paper and click on 'Process'"
    accept_multiple_files=False
    subheaderText="CIxD Papers"
    captionText="Select one or more papers to learn."
    pdfFolderPath='./pdf/'
    pdfFiles=[]
    pdf_checkbox=[]
    llm=''
    #############others#######################
    load_dotenv()
    pdfHandler=PDFHandler('./ExtractTextInfoFromPDF.zip')

    ########initialize#######################
    
    streamlit.set_page_config(page_title=pageTitle,page_icon=pageIcon)
    streamlit.write(css, unsafe_allow_html=True)
    streamlit.header(header)
    modeselect = streamlit.selectbox('Select the Mode: ', options=[*modeOptions])
    streamlit.write('Selected Mode: ', modeselect)
    
    response_container = streamlit.container()
    container = streamlit.container()
    
    model_name = streamlit.sidebar.radio("Choose a model:", ([*LLMList]))
    llm=switchLLM(model_name)
    
    if "conversation" not in streamlit.session_state:
        streamlit.session_state.conversation = None
    if  "chat_history" not in streamlit.session_state:
        streamlit.session_state.chat_history = None
    if  "section_text" not in streamlit.session_state:
        streamlit.session_state.section_text = None

    
    with container:
        with streamlit.form(key='my_form', clear_on_submit=True):
            user_input = streamlit.text_area("You:", key='input', height=100)
            submit_button = streamlit.form_submit_button(label='Send')
            
        if submit_button and user_input:
            handle_userInput(user_input,response_container)
                
        if streamlit.session_state['chat_history']:
             with response_container:
                for i, conversation in enumerate(streamlit.session_state.chat_history):
                    if i % 2 == 0:
                    #     streamlit.write(user_template.replace(
                    # "{{MSG}}", conversation.content), unsafe_allow_html=True)
                        message(conversation.content, is_user=True, avatar_style= 'pixel-art',key=str(i) + '_user')
                    else:
                        # streamlit.write(bot_template.replace(
                        #     "{{MSG}}", conversation.content), unsafe_allow_html=True)
                        message(conversation.content, key=str(i))
        
    
    with streamlit.sidebar:
        pdf_docs = streamlit.file_uploader(
                fileUploadText, accept_multiple_files= accept_multiple_files,type="pdf") #this just lets us enable this function
        streamlit.subheader(subheaderText)
        streamlit.caption(captionText)
        
        pdfFiles = [f for f in listdir( pdfFolderPath) if isfile(join( pdfFolderPath, f))]
        
        for pdf in pdfFiles:
            pdf_checkbox.append(streamlit.checkbox(pdf))
        
     ########interaction#######################

        
    with streamlit.sidebar:
            if streamlit.button("Process"):
                with streamlit.spinner("Processing"): 
                    full_text=''
                    ##SummaryMode
                    #####Only one input should be posssible for summary mode
            
                    for index, checkedpdf in enumerate(pdf_checkbox):
                        if(checkedpdf):
                            pdfHandler.setPdfFile(pdfFolderPath+pdfFiles[index])
                            pdfHandler.structurePDF('local_file')
                            #full_text+=pdfHandler.getFilteredText()
                            #text_chunks = get_text_chunks(full_text)
                            #vectorstore = get_vectorstore(text_chunks)
                            #streamlit.session_state.conversation = get_conversation_chain(
                            #vectorstore,llm)
                            streamlit.session_state.section_text =pdfHandler.getFilteredTextBySection()
                            
                    if(pdf_docs):
                        pdfHandler.setStreamData(pdf_docs)
                        pdfHandler.structurePDF('stream')
                        #full_text+=pdfHandler.getFilteredText()
                        streamlit.session_state.section_text =pdfHandler.getFilteredTextBySection()
                 
                        
                        
    if streamlit.button('Summary By Section'):  
        with streamlit.spinner("Processing"): 
            template = """
            Provide a summary of the following text in structured bullet point lists.
            the text: {text}
            In addition, add the provided title at the top of your response:
            the title: {title}
            
            """
            prompt=PromptTemplate(
                input_variables=["text",'title'],
                template=template
                )             
            for sectionName in streamlit.session_state.section_text:
                summary_prompt=prompt.format(text=streamlit.session_state.section_text[sectionName],title=sectionName)
                num_tokens = llm.get_num_tokens(summary_prompt)
                print (f"This prompt + {sectionName} section has {num_tokens} tokens")
                if model_name=='ChatOpenAI':
                    summary = llm.predict_messages([HumanMessage(content= summary_prompt)])
                    with response_container:
                        message(summary.content)
                else:
                    summary = llm(summary_prompt)
                    with response_container:
                        message(summary)

  #########################################
if __name__ == '__main__': 
    main()
