import streamlit as st
from PyPDF2 import PdfReader


def get_pdf_text(pdf_documents):
  text = ""
  for pdf in pdf_documents:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text
  


def main():
  st.set_page_config(page_title='Chat Multiple PDFs',page_icon=':books:')
  
  if 'text' not in st.session_state:
    st.session_state.text = ""
  
  with st.sidebar:
    pdf_documents = st.file_uploader('Upload your PDF file', type=['pdf'], accept_multiple_files=True)
  
    if st.button('Process'):
      with st.spinner('Processing...'):
        pdf_text = get_pdf_text(pdf_documents)
        
        st.session_state.text = pdf_text
        
        # text_chunks = get_chunks(pdf_text)
        
        st.success('Done!')


  st.title('Chat with Multiple PDFs :books:')
  st.text_input('Ask a question about the PDFs')

  st.write(st.session_state.text)

if __name__ == '__main__':
    main()