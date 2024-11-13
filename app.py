# app.py
import streamlit as st
from document_assistant.core import init_session_state, logger
from document_assistant.document_processor import DocumentProcessor
from document_assistant.ui_components import DocumentContainer, ChatInterface
from pathlib import Path
import asyncio

class DocumentAssistant:
    """Main application class"""
    
    def __init__(self):
        init_session_state()
        self.setup_page()
        self.doc_processor = DocumentProcessor()

    def setup_page(self):
        """Configure page settings"""
        st.set_page_config(
            page_title="📚 Document Assistant",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Add custom CSS
        st.markdown("""
            <style>
            .main > div {
                padding-top: 2rem;
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            .element-container {
                margin-bottom: 1rem;
            }
            .stChatFloatingInputContainer {
                bottom: 1rem;
            }
            </style>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render sidebar with file upload and document management"""
        with st.sidebar:
            st.title("📎 Document Management")
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['pdf', 'docx', 'doc', 'txt'],
                accept_multiple_files=True,
                key="file_uploader",
                help="Supported formats: PDF, Word (DOCX/DOC), and Text files"
            )
            
            # Process uploaded files
            if uploaded_files:
                self.process_uploads(uploaded_files)
            
            # Document selection
            if st.session_state.documents:
                st.markdown("### 📑 Selected Documents")
                
                # Deselect all button
                if st.button("Deselect All", use_container_width=True):
                    st.session_state.active_docs.clear()
                    st.rerun()
                
                # Document checkboxes
                for doc_name in st.session_state.documents:
                    selected = st.checkbox(
                        f"📄 {Path(doc_name).stem}",
                        value=doc_name in st.session_state.active_docs,
                        key=f"select_{doc_name}"
                    )
                    
                    if selected:
                        st.session_state.active_docs.add(doc_name)
                    else:
                        st.session_state.active_docs.discard(doc_name)

    def process_uploads(self, files):
        """Process uploaded files"""
        current_files = {f.name for f in files}
        
        # Handle removed files
        removed_files = st.session_state.previous_files - current_files
        for file_name in removed_files:
            if file_name in st.session_state.documents:
                del st.session_state.documents[file_name]
                st.session_state.active_docs.discard(file_name)
        
        # Process new files
        new_files = [f for f in files if f.name not in st.session_state.documents]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} file(s)..."):
                asyncio.run(self.doc_processor.process_documents(new_files))
        
        # Update tracked files
        st.session_state.previous_files = current_files

        st.session_state.current_file_count = len(current_files)

    def render_main_content(self):
        """Render main content area with tabs"""
        st.title("📚 Document Assistant")
        
        if not st.session_state.documents:
            st.info("👈 Please upload documents to get started!")
            return
            
        # Create tabs
        tab_chat, tab_docs = st.tabs(["💭 Chat", "📑 Documents"])
        
        # Render chat tab
        with tab_chat:
            ChatInterface.render()
        
        # Render documents tab
        with tab_docs:
            if st.session_state.active_docs:
                for doc_name in st.session_state.active_docs:
                    doc_info = st.session_state.documents.get(doc_name)
                    if doc_info:
                        DocumentContainer.render(doc_name, doc_info)
            else:
                st.info("👈 Please select documents from the sidebar to view details")

    def run(self):
        """Run the application"""
        try:
            # Render sidebar
            self.render_sidebar()
            
            # Render main content
            self.render_main_content()
            
            # Footer
            st.markdown("---")
            st.markdown(
                "💡 Powered by Gemini & BART | Made with Streamlit",
                help="Using state-of-the-art AI models for document analysis"
            )
            
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error("An error occurred. Please try refreshing the page.")

def main():
    """Application entry point"""
    try:
        app = DocumentAssistant()
        app.run()
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        st.error("Failed to start the application. Please try again.")

if __name__ == "__main__":
    main()