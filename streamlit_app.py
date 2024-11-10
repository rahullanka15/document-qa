# One
# import streamlit as st
# import google.generativeai as genai
# import PyPDF2
# import io
# from typing import List

# # Page config
# st.set_page_config(
#     page_title="Document Analysis & QA",
#     page_icon="📚",
#     layout="wide"
# )

# # Initialize Gemini
# if 'GOOGLE_API_KEY' not in st.secrets:
#     st.error("Please set GOOGLE_API_KEY in streamlit secrets")
#     st.stop()

# genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
# model = genai.GenerativeModel('gemini-1.5-flash')

# # Session state initialization
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'document_content' not in st.session_state:
#     st.session_state.document_content = None

# def chunk_text(text: str, chunk_size: int = 15000) -> List[str]:
#     """Split text into smaller chunks"""
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# def extract_text_from_pdf(pdf_file):
#     """Extract text from uploaded PDF"""
#     try:
#         pdf_reader = PyPDF2.PdfReader(pdf_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text() + "\n"
#         return text
#     except Exception as e:
#         st.error(f"Error reading PDF: {str(e)}")
#         return None

# def get_gemini_response(question: str, context: str) -> str:
#     """Get response from Gemini with error handling and chunking"""
#     try:
#         # Chunk the context if it's too large
#         context_chunks = chunk_text(context)
        
#         # Create a summarized context from chunks
#         summarized_context = ""
#         for chunk in context_chunks[:3]:  # Use first 3 chunks to stay within limits
#             prompt = f"Summarize this text briefly: {chunk}"
#             try:
#                 chunk_summary = model.generate_content(prompt).text
#                 summarized_context += chunk_summary + "\n"
#             except Exception as e:
#                 continue
        
#         # Final prompt with summarized context
#         final_prompt = f"""Context: {summarized_context}\n\nQuestion: {question}
#         Based on the provided context, please answer the question.
#         If the answer is not in the context, say "I cannot find this information in the document."
#         Keep the answer concise and relevant."""

#         response = model.generate_content(
#             final_prompt,
#             generation_config=genai.types.GenerationConfig(
#                 max_output_tokens=500,
#                 temperature=0.7
#             )
#         )
#         return response.text
#     except Exception as e:
#         return f"Error generating response: {str(e)}"

# # Main UI
# st.title("📚 Document Analysis & QA System")

# # Sidebar
# with st.sidebar:
#     st.header("Upload Documents")
#     uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
#     if uploaded_file is not None:
#         # Process the uploaded file
#         text_content = extract_text_from_pdf(uploaded_file)
#         if text_content:
#             st.session_state.document_content = text_content
#             st.success("Document processed successfully!")
            
#             # Show document preview
#             with st.expander("Document Preview"):
#                 st.text(text_content[:300] + "...")  # Show less preview text

# # Main content area
# if st.session_state.document_content:
#     # Chat interface
#     st.header("Chat with your Document")
    
#     # Display chat history
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
    
#     # Chat input
#     if question := st.chat_input("Ask a question about your document:"):
#         # Display user question
#         with st.chat_message("user"):
#             st.write(question)
        
#         # Get and display Gemini response
#         with st.chat_message("assistant"):
#             with st.spinner("Generating response..."):
#                 response = get_gemini_response(question, st.session_state.document_content)
#                 st.write(response)
        
#         # Update chat history
#         st.session_state.chat_history.extend([
#             {"role": "user", "content": question},
#             {"role": "assistant", "content": response}
#         ])
# else:
#     st.info("Please upload a document to start chatting!")

# # Footer
# st.markdown("---")
# st.markdown("Document Analysis & QA System - Powered by Gemini AI")
# streamlit_app.py


# Two
# import streamlit as st
# import google.generativeai as genai
# from src.document_processor import DocumentProcessor
# from typing import List, Dict
# import os

# # Page config
# st.set_page_config(
#     page_title="📚 Smart Document Assistant",
#     page_icon="📚",
#     layout="wide"
# )

# # Initialize components
# if 'GOOGLE_API_KEY' not in st.secrets:
#     st.error("Please set GOOGLE_API_KEY in streamlit secrets")
#     st.stop()

# genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
# model = genai.GenerativeModel('gemini-1.5-flash')
# doc_processor = DocumentProcessor()

# # Session state
# if 'documents' not in st.session_state:
#     st.session_state.documents = {}  # Store multiple documents
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'current_doc' not in st.session_state:
#     st.session_state.current_doc = None

# def get_friendly_response(question: str, context: Dict[str, str]) -> str:
#     """Get a friendly response from Gemini"""
#     try:
#         # Create a conversational prompt
#         prompt = f"""You are a friendly and helpful document assistant. 
#         Your task is to help the user understand their documents in a conversational way.
        
#         Current document: {context.get('name', 'document')}
#         Content: {context.get('content', '')}
        
#         User's question: {question}
        
#         Please provide a helpful, conversational response. If you can't find the information,
#         say something like "I don't see that information in this document, but I'd be happy to
#         help you find something else!" Always maintain a friendly, helpful tone.
#         """
        
#         response = model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0.7,
#                 top_k=40,
#                 top_p=0.8,
#                 max_output_tokens=1024,
#             )
#         )
#         return response.text
#     except Exception as e:
#         return f"I apologize, but I encountered an error: {str(e)}. How else can I help you?"

# # Main UI
# st.title("📚 Smart Document Assistant")
# st.markdown("Hi! I'm your friendly document assistant. Upload your documents, and I'll help you understand them! 😊")

# # Sidebar for document management
# with st.sidebar:
#     st.header("📎 Document Management")
    
#     # Multiple file upload
#     uploaded_files = st.file_uploader(
#         "Upload your documents (PDF, DOCX, TXT, TEX)",
#         type=['pdf', 'docx', 'doc', 'txt', 'tex'],
#         accept_multiple_files=True
#     )
    
#     # Process uploaded files
#     if uploaded_files:
#         for file in uploaded_files:
#             # Only process new files
#             if file.name not in st.session_state.documents:
#                 with st.spinner(f"Processing {file.name}..."):
#                     result = doc_processor.process_document(file)
#                     if result:
#                         st.session_state.documents[file.name] = result
#                         st.success(f"✅ {file.name} processed!")
    
#     # Document selection
#     if st.session_state.documents:
#         st.write("Select a document to chat about:")
#         for doc_name in st.session_state.documents:
#             if st.button(f"📄 {doc_name}", key=doc_name):
#                 st.session_state.current_doc = doc_name
#                 st.session_state.chat_history = []  # Reset chat for new document

# # Main chat area
# if st.session_state.current_doc:
#     current_doc = st.session_state.documents[st.session_state.current_doc]
    
#     # Document info
#     st.info(f"📄 Currently chatting about: {current_doc['name']}")
    
#     # Chat interface
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
    
#     # Chat input
#     if question := st.chat_input("Ask me anything about your document...!!!"):
#         # Show user message
#         with st.chat_message("user"):
#             st.write(question)
        
#         # Get and show response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking... 🤔"):
#                 response = get_friendly_response(question, current_doc)
#                 st.write(response)
        
#         # Update chat history
#         st.session_state.chat_history.extend([
#             {"role": "user", "content": question},
#             {"role": "assistant", "content": response}
#         ])
# else:
#     st.info("👈 Please upload and select a document to start chatting!")

# # Footer
# st.markdown("---")
# st.markdown("💡 Powered by Gemini AI - Here to help you understand your documents! 🚀")

# Three
# import streamlit as st
# import google.generativeai as genai
# from src.document_processor import DocumentProcessor
# from typing import List, Dict
# import os

# # Page config (same as before)
# st.set_page_config(
#     page_title="📚 Smart Document Assistant",
#     page_icon="📚",
#     layout="wide"
# )

# # Initialize Gemini with 1.5-flash
# if 'GOOGLE_API_KEY' not in st.secrets:
#     st.error("Please set GOOGLE_API_KEY in streamlit secrets")
#     st.stop()

# genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
# model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
# doc_processor = DocumentProcessor()

# # Enhanced session state
# if 'documents' not in st.session_state:
#     st.session_state.documents = {}  # Store all documents
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'active_docs' not in st.session_state:
#     st.session_state.active_docs = set()  # Store selected documents

# def get_combined_context(selected_docs: List[str]) -> str:
#     """Combine content from multiple selected documents"""
#     combined_text = ""
#     for doc_name in selected_docs:
#         doc = st.session_state.documents.get(doc_name)
#         if doc:
#             combined_text += f"\nDocument: {doc_name}\n{doc['content']}\n---\n"
#     return combined_text

# def get_friendly_response(question: str, active_docs: List[str]) -> str:
#     """Get response considering multiple documents"""
#     try:
#         # Get combined context from selected documents
#         context = get_combined_context(active_docs)
        
#         # Create a conversational prompt that handles multiple documents
#         prompt = f"""You are a friendly and helpful document assistant analyzing multiple documents.
        
#         Documents being analyzed:
#         {', '.join(active_docs)}
        
#         Combined content from selected documents:
#         {context}
        
#         User's question: {question}
        
#         Please provide a comprehensive answer based on ALL selected documents. 
#         If referring to specific information, mention which document it came from.
#         If you can't find the information in any document, let me know.
#         Maintain a friendly, conversational tone.
#         """
        
#         response = model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0.7,
#                 top_k=40,
#                 top_p=0.8,
#                 max_output_tokens=1024,
#             )
#         )
#         return response.text
#     except Exception as e:
#         return f"I apologize, but I encountered an error: {str(e)}. How else can I help you?"

# # Main UI
# st.title("📚 Smart Document Assistant")
# st.markdown("Hi! I'm your friendly document assistant. Upload your documents, and I'll help you understand them! 😊")

# # Enhanced sidebar for document management
# with st.sidebar:
#     st.header("📎 Document Management")
    
#     # Multiple file upload
#     uploaded_files = st.file_uploader(
#         "Upload your documents (PDF, DOCX, TXT, TEX)",
#         type=['pdf', 'docx', 'doc', 'txt', 'tex'],
#         accept_multiple_files=True
#     )
    
#     # Process uploaded files
#     if uploaded_files:
#         for file in uploaded_files:
#             if file.name not in st.session_state.documents:
#                 with st.spinner(f"Processing {file.name}..."):
#                     result = doc_processor.process_document(file)
#                     if result:
#                         st.session_state.documents[file.name] = result
#                         # Auto-add to active docs when successfully processed
#                         st.session_state.active_docs.add(file.name)
#                         st.success(f"✅ {file.name} processed!")
    
#     # Document selection with checkboxes
#     if st.session_state.documents:
#         st.write("### Select documents to analyze:")
        
#         # Select/Deselect All buttons
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("Select All"):
#                 st.session_state.active_docs = set(st.session_state.documents.keys())
#         with col2:
#             if st.button("Deselect All"):
#                 st.session_state.active_docs = set()
        
#         # Individual document checkboxes
#         for doc_name in st.session_state.documents:
#             if st.checkbox(
#                 f"📄 {doc_name}", 
#                 key=f"check_{doc_name}",
#                 value=doc_name in st.session_state.active_docs
#             ):
#                 st.session_state.active_docs.add(doc_name)
#             else:
#                 st.session_state.active_docs.discard(doc_name)
        
#         # Show active document count
#         st.write(f"🔍 Analyzing {len(st.session_state.active_docs)} documents")

# # Main chat area
# if st.session_state.active_docs:
#     # Show active documents
#     st.info(f"📚 Currently analyzing: {', '.join(st.session_state.active_docs)}")
    
#     # Chat interface
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
    
#     # Chat input
#     if question := st.chat_input("Ask me anything about your documents!"):
#         # Show user message
#         with st.chat_message("user"):
#             st.write(question)
        
#         # Get and show response
#         with st.chat_message("assistant"):
#             with st.spinner("Analyzing documents... 🤔"):
#                 response = get_friendly_response(
#                     question, 
#                     list(st.session_state.active_docs)
#                 )
#                 st.write(response)
        
#         # Update chat history
#         st.session_state.chat_history.extend([
#             {"role": "user", "content": question},
#             {"role": "assistant", "content": response}
#         ])
# else:
#     st.info("👈 Please upload and select documents to start chatting!")

# # Footer
# st.markdown("---")
# st.markdown("💡 Powered by Gemini 1.5 Flash - Here to help you understand your documents! 🚀")

import streamlit as st
import google.generativeai as genai
import PyPDF2
import docx
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
import os

# Page config
st.set_page_config(
    page_title="📚 Smart Document Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize Gemini
if 'GOOGLE_API_KEY' not in st.secrets:
    st.error("Please set GOOGLE_API_KEY in streamlit secrets")
    st.stop()

genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-pro')

# Session state initialization
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'active_docs' not in st.session_state:
    st.session_state.active_docs = set()
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}

class DocumentProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.supported_formats = {
            'pdf': self._process_pdf,
            'docx': self._process_docx,
            'doc': self._process_docx,
            'txt': self._process_txt,
            'tex': self._process_latex
        }
    
    def process_document(self, file) -> Dict[str, str]:
        """Process any supported document type"""
        file_ext = os.path.splitext(file.name)[1][1:].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return self.supported_formats[file_ext](file)
    
    def _process_pdf(self, file) -> Dict[str, str]:
        """Process PDF files"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return {
                "content": text,
                "type": "pdf",
                "name": file.name
            }
        except Exception as e:
            st.error(f"Error processing PDF {file.name}: {str(e)}")
            return None
    
    def _process_docx(self, file) -> Dict[str, str]:
        """Process DOCX files"""
        try:
            doc = docx.Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return {
                "content": text,
                "type": "docx",
                "name": file.name
            }
        except Exception as e:
            st.error(f"Error processing DOCX {file.name}: {str(e)}")
            return None
    
    def _process_txt(self, file) -> Dict[str, str]:
        """Process TXT files"""
        try:
            text = file.read().decode('utf-8')
            return {
                "content": text,
                "type": "txt",
                "name": file.name
            }
        except Exception as e:
            st.error(f"Error processing TXT {file.name}: {str(e)}")
            return None
    
    def _process_latex(self, file) -> Dict[str, str]:
        """Process LaTeX files"""
        try:
            text = file.read().decode('utf-8')
            import re
            text = re.sub(r'\\[a-zA-Z]+{', '', text)
            text = text.replace('}', '')
            return {
                "content": text,
                "type": "tex",
                "name": file.name
            }
        except Exception as e:
            st.error(f"Error processing LaTeX {file.name}: {str(e)}")
            return None
    
    async def process_files_parallel(self, files):
        """Process multiple files in parallel"""
        tasks = []
        for file in files:
            if file.name not in st.session_state.documents:
                st.session_state.processing_status[file.name] = "Processing..."
                task = asyncio.create_task(self.process_single_file(file))
                tasks.append(task)
        
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            if result:
                file_name = result['name']
                st.session_state.documents[file_name] = result
                st.session_state.active_docs.add(file_name)
                st.session_state.processing_status[file_name] = "Completed"
                st.success(f"✅ {file_name} processed and selected!")
    
    async def process_single_file(self, file):
        """Process a single file asynchronously"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.process_document,
                file
            )
            return result
        except Exception as e:
            st.session_state.processing_status[file.name] = f"Error: {str(e)}"
            return None

def get_combined_context(selected_docs: List[str]) -> str:
    """Combine content from multiple selected documents"""
    combined_text = ""
    for doc_name in selected_docs:
        doc = st.session_state.documents.get(doc_name)
        if doc:
            combined_text += f"\nDocument: {doc_name}\n{doc['content']}\n---\n"
    return combined_text

def get_friendly_response(question: str, active_docs: List[str]) -> str:
    """Get response considering multiple documents"""
    try:
        context = get_combined_context(active_docs)
        prompt = f"""You are a friendly and helpful document assistant analyzing multiple documents.
        
        Documents being analyzed:
        {', '.join(active_docs)}
        
        Combined content from selected documents:
        {context}
        
        User's question: {question}
        
        Please provide a comprehensive answer based on ALL selected documents. 
        If referring to specific information, mention which document it came from.
        If you can't find the information in any document, let me know.
        Maintain a friendly, conversational tone.
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_k=40,
                top_p=0.8,
                max_output_tokens=1024,
            )
        )
        return response.text
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. How else can I help you?"

# Main UI
st.title("📚 Smart Document Assistant")
st.markdown("Hi! I'm your friendly document assistant. Upload your documents, and I'll help you understand them! 😊")

# Sidebar for document management
with st.sidebar:
    st.header("📎 Document Management")
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, TXT, TEX)",
        type=['pdf', 'docx', 'doc', 'txt', 'tex'],
        accept_multiple_files=True
    )
    
    # Process files in parallel
    if uploaded_files:
        doc_processor = DocumentProcessor()
        asyncio.run(doc_processor.process_files_parallel(uploaded_files))
    
    # Display processing status
    if st.session_state.processing_status:
        st.write("### Processing Status:")
        for file_name, status in st.session_state.processing_status.items():
            st.text(f"{file_name}: {status}")
    
    # Document selection
    if st.session_state.documents:
        st.write("### Selected Documents:")
        
        # Deselect All button
        if st.button("Deselect All"):
            st.session_state.active_docs = set()
        
        # Individual document checkboxes
        for doc_name in st.session_state.documents:
            if st.checkbox(
                f"📄 {doc_name}",
                key=f"check_{doc_name}",
                value=doc_name in st.session_state.active_docs
            ):
                st.session_state.active_docs.add(doc_name)
            else:
                st.session_state.active_docs.discard(doc_name)

# Main chat area
if st.session_state.active_docs:
    st.info(f"📚 Currently analyzing: {', '.join(st.session_state.active_docs)}")
    
    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask me anything about your documents...!!!"):
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents... 🤔"):
                response = get_friendly_response(
                    question, 
                    list(st.session_state.active_docs)
                )
                st.write(response)
        
        # Update chat history
        st.session_state.chat_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ])
else:
    st.info("👈 Please upload and select documents to start chatting!")

# Footer
st.markdown("---")
st.markdown("💡 Powered by Gemini AI - Here to help you understand your documents! 🚀")

# Custom CSS to hide GitHub and Streamlit branding
hide_streamlit_style = """
    <style>
        footer {visibility: hidden;}
        .css-1d391kg {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)