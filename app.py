import streamlit as st
import base64
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
import io
import json
import time
import re

# Must be the first Streamlit command
st.set_page_config(page_title="Chat with multiple PDFs",
                  page_icon=":books:",
                  layout="wide",
                  initial_sidebar_state="expanded"
                  )

def get_pdf_text(pdf_docs):
    text = ""
    chunk_locations = []
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        
        # Process each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            if page_text:
                # Store the location information for this chunk
                chunk_locations.append({
                    'file': pdf.name,
                    'page': page_num + 1,  # 1-based page numbers
                    'text': page_text,
                    'start_char': len(text),
                    'end_char': len(text) + len(page_text)
                })
                text += page_text
    
    return text, chunk_locations

def get_text_chunks(text):
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Calculate chunk locations
    chunk_locations = []
    
    for chunk in chunks:
        # Find which page this chunk belongs to
        chunk_start = text.find(chunk)
        chunk_end = chunk_start + len(chunk)
        
        # Find the page that contains this chunk
        containing_page = None
        for loc in st.session_state.chunk_locations:
            if (chunk_start >= loc['start_char'] and chunk_start < loc['end_char']) or \
               (chunk_end > loc['start_char'] and chunk_end <= loc['end_char']):
                containing_page = loc
                break
        
        if containing_page:
            chunk_locations.append({
                'file': containing_page['file'],
                'page': containing_page['page'],
                'text': chunk[:200] + "..." if len(chunk) > 200 else chunk
            })
        else:
            # Fallback if we can't find the page
            chunk_locations.append({
                'file': st.session_state.chunk_locations[0]['file'],
                'page': 1,
                'text': chunk[:200] + "..." if len(chunk) > 200 else chunk
            })
    
    return chunks, chunk_locations

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # Store chunk locations in metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={'chunk_index': i}
        ) for i, chunk in enumerate(text_chunks)
    ]
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key='answer'
    )
    return conversation_chain

def display_pdf(pdf_bytes, highlight_text=None, page_num=1):
    try:
        # Try to read with PyPDF2 to validate PDF
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(reader.pages)
            st.write(f"PDF loaded successfully ({num_pages} pages)")
        except Exception as e:
            st.error(f"PDF validation error: {str(e)}")
            return

        # Encode PDF bytes to base64
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Create viewer HTML
        pdf_display = f'''
            <div style="position: sticky; top: 0; background: white; z-index: 100; padding: 10px; border-bottom: 1px solid #ddd;">
                <div id="status" style="margin-bottom: 10px; padding: 10px; background: #f0f2f6; border-radius: 4px;">Loading viewer...</div>
                <div id="error" style="display: none; margin-bottom: 10px; padding: 10px; background: #ffe6e6; border-radius: 4px; color: red;"></div>
            </div>
            <div id="pdf-container" style="position: relative;"></div>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
            <script>
                pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

                function scrollToPage(pageNum) {{
                    const container = document.getElementById('pdf-container');
                    const pages = container.children;
                    if (pageNum > 0 && pageNum <= pages.length) {{
                        const targetPage = pages[pageNum - 1];
                        if (targetPage) {{
                            targetPage.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                        }}
                    }}
                }}

                // Listen for navigation messages
                window.addEventListener('message', function(e) {{
                    if (e.data.type === 'scrollToPage') {{
                        scrollToPage(e.data.pageNum);
                    }}
                }}, false);

                async function loadPDF() {{
                    try {{
                        const pdfData = atob('{base64_pdf}');
                        const loadingTask = pdfjsLib.getDocument({{data: pdfData}});
                        
                        loadingTask.promise.then(async function(pdf) {{
                            document.getElementById('status').textContent = 'Rendering PDF...';
                            const container = document.getElementById('pdf-container');
                            container.innerHTML = '';
                            
                            for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {{
                                const page = await pdf.getPage(pageNum);
                                const scale = 1.5;
                                const viewport = page.getViewport({{scale: scale}});

                                const pageDiv = document.createElement('div');
                                pageDiv.className = 'pdf-page';
                                pageDiv.setAttribute('data-page-number', pageNum);
                                pageDiv.style.margin = '20px auto';
                                pageDiv.style.position = 'relative';

                                const canvas = document.createElement('canvas');
                                canvas.style.display = 'block';
                                canvas.style.margin = '0 auto';
                                canvas.style.border = '1px solid #ddd';
                                canvas.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
                                
                                const context = canvas.getContext('2d');
                                canvas.height = viewport.height;
                                canvas.width = viewport.width;

                                pageDiv.appendChild(canvas);
                                container.appendChild(pageDiv);

                                await page.render({{
                                    canvasContext: context,
                                    viewport: viewport
                                }}).promise;

                                if ('{highlight_text}') {{
                                    const textContent = await page.getTextContent();
                                    const textItems = textContent.items;
                                    
                                    textItems.forEach(function(textItem) {{
                                        if (textItem.str.toLowerCase().includes('{highlight_text}'.toLowerCase())) {{
                                            const tx = pdfjsLib.Util.transform(
                                                viewport.transform,
                                                textItem.transform
                                            );
                                            
                                            context.beginPath();
                                            context.rect(tx[4], tx[5], textItem.width * scale, textItem.height * scale);
                                            context.fillStyle = 'yellow';
                                            context.globalAlpha = 0.3;
                                            context.fill();
                                            context.globalAlpha = 1.0;
                                        }}
                                    }});
                                }}
                            }}
                            
                            document.getElementById('status').textContent = 'PDF loaded successfully';
                            
                            // Initial scroll to page
                            scrollToPage({page_num});
                            
                        }}).catch(function(error) {{
                            document.getElementById('error').style.display = 'block';
                            document.getElementById('error').textContent = 'Error loading PDF: ' + error.message;
                            console.error('Error:', error);
                        }});
                    }} catch (error) {{
                        document.getElementById('error').style.display = 'block';
                        document.getElementById('error').textContent = 'Error: ' + error.message;
                        console.error('Error:', error);
                    }}
                }}

                loadPDF();
            </script>
        '''
        
        st.components.v1.html(pdf_display, height=800, scrolling=True)
        
        # Add a hidden component to handle page navigation
        st.markdown(f'''
            <script>
                // Function to navigate to a specific page
                window.navigateToPage = function(pageNum) {{
                    const viewer = document.querySelector('iframe');
                    if (viewer) {{
                        viewer.contentWindow.postMessage({{
                            type: 'scrollToPage',
                            pageNum: pageNum
                        }}, '*');
                    }}
                }};
            </script>
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def handle_userinput(user_question):
    if not user_question:
        return

    # Check if this is a new question
    if 'current_question' not in st.session_state or st.session_state.current_question != user_question:
        if st.session_state.conversation is None:
            st.error("Please process the documents first!")
            return
            
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        st.session_state.current_question = user_question
        if 'source_documents' in response:
            st.session_state.current_source_docs = response['source_documents']
        else:
            st.session_state.current_source_docs = []

    # Display chat messages
    source_counter = 0
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            bot_message = bot_template.replace("{{MSG}}", message.content)
            
            # Replace page references with links
            def replace_page(match):
                page_num = int(match.group(1))
                source_counter += 1
                return f'''<span 
                    class="page-ref" 
                    data-page="{page_num}" 
                    data-text="{message.content}" 
                    onclick="event.preventDefault(); window.navigateToPage({page_num}); return false;"
                    style="color: #1e88e5; text-decoration: underline; cursor: pointer;"
                >{match.group(0)}</span>'''
            
            bot_message = re.sub(r'page (\d+)', replace_page, bot_message, flags=re.IGNORECASE)
            st.write(bot_message, unsafe_allow_html=True)
            
            # Display source references
            if st.session_state.current_source_docs:
                st.markdown("### Sources:", unsafe_allow_html=True)
                for i, doc in enumerate(st.session_state.current_source_docs, 1):
                    chunk_index = doc.metadata.get('chunk_index')
                    if chunk_index is not None and chunk_index < len(st.session_state.chunk_locations):
                        location = st.session_state.chunk_locations[chunk_index]
                        page_num = location['page']
                        source_counter += 1
                        
                        if st.button(f"📄 Source {i}: Page {page_num}", 
                                   key=f"source_{source_counter}"):
                            st.session_state.target_page = page_num
                            st.rerun()
                        
                        st.markdown(f"""
                            <div style="margin-left: 20px; margin-top: 5px; font-size: 0.9em; color: #666;">
                                "{location['text']}"
                            </div>
                        """, unsafe_allow_html=True)

def main():
    load_dotenv()
    
    # Add custom CSS for layout
    st.markdown("""
        <style>
        .stApp {
            max-width: 100%;
            padding: 1rem;
        }
        .main > div {
            padding: 0;
        }
        .css-1d391kg {  /* PDF viewer container */
            width: 100%;
            height: calc(100vh - 200px);
            overflow-y: auto;
        }
        .block-container {
            padding: 1rem;
            max-width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = None
    if "chunk_locations" not in st.session_state:
        st.session_state.chunk_locations = None
    if "target_page" not in st.session_state:
        st.session_state.target_page = 1
    if "highlight_text" not in st.session_state:
        st.session_state.highlight_text = None
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "current_source_docs" not in st.session_state:
        st.session_state.current_source_docs = []

    st.header("Chat with multiple PDFs :books:")
    
    # Create the main layout with two columns
    left_col, right_col = st.columns([1, 1])
    
    # Left column for PDF upload and viewer
    with left_col:
        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", 
                accept_multiple_files=True,
                type=['pdf']
            )
            if st.button("Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text, chunk_locations = get_pdf_text(pdf_docs)
                    st.session_state.chunk_locations = chunk_locations
                    
                    # get the text chunks
                    text_chunks, chunk_locations = get_text_chunks(raw_text)
                    st.session_state.chunk_locations = chunk_locations
                    
                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    st.session_state.pdf_docs = pdf_docs
        
        st.subheader("PDF Viewer")
        if st.session_state.pdf_docs:
            pdf_doc = st.session_state.pdf_docs[0]  # Display the first PDF
            pdf_doc.seek(0)
            pdf_bytes = pdf_doc.read()
            display_pdf(pdf_bytes, highlight_text=st.session_state.highlight_text, page_num=st.session_state.target_page)
        else:
            st.image("https://via.placeholder.com/400x600.png?text=Upload+a+PDF",
                    caption="Upload a PDF to begin",
                    use_column_width=True)
    
    # Right column for chat interface
    with right_col:
        st.subheader("Chat Interface")
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()
