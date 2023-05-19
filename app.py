import streamlit as st


def main():
  st.set_page_config(page_title='Chat Multiple PDFs',page_icon=':books:')
  
  with st.sidebar:
    pdf_documents = st.file_uploader('Upload your PDF file', type=['pdf'], accept_multiple_files=True)
  
    if st.button('Process'):
      with st.spinner('Processing...'):
        # pdf_text = get_pdf_text(pdf_documents)
        # text_chunks = get_chunks(pdf_text)
        
        st.success('Done!')

  st.title('Chat with Multiple PDFs :books:')
  st.text_input('Ask a question about the PDFs')

if __name__ == '__main__':
    main()