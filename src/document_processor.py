# src/document_processor.py

import PyPDF2
import docx
import os
from typing import Dict, List
import streamlit as st

class DocumentProcessor:
    def __init__(self):
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
            # Remove LaTeX commands while keeping the content
            # This is a simple implementation - can be enhanced
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