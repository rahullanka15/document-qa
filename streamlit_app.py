# import time
# import pandas as pd
# import streamlit as st
# import google.generativeai as genai
# import PyPDF2
# import docx
# import asyncio
# import requests
# from concurrent.futures import ThreadPoolExecutor
# from typing import Dict, List, Optional, Tuple
# import os
# import tempfile
# import subprocess
# import logging
# import plotly.express as px
# from sentence_transformers import SentenceTransformer, util
# import torch
# import gc
# import io
# from PIL import Image
# import warnings
# import re
# from pathlib import Path

# # Transformers imports
# from transformers import (
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
#     pipeline
# )

# # Suppress warnings
# warnings.filterwarnings('ignore')

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Page configuration must come first
# st.set_page_config(
#     page_title="📚 Document Assistant",
#     page_icon="📚",
#     layout="wide"
# )

# # API Configuration
# def setup_apis():
#     """Setup and verify API configurations"""
#     if 'GOOGLE_API_KEY' not in st.secrets:
#         st.error("Please set GOOGLE_API_KEY in streamlit secrets")
#         st.stop()
#     if 'HF_API_KEY' not in st.secrets:
#         st.error("Please set HF_API_KEY in streamlit secrets")
#         st.stop()

#     # Configure APIs
#     genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
    
#     return {
#         'model': genai.GenerativeModel('gemini-1.5-pro'),
#         'hf_key': st.secrets["HF_API_KEY"],
#         'summary_url': "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
#         #'image_url': "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
#         'image_url': "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large",
#     }

# # Initialize APIs
# API_CONFIG = setup_apis()
# HEADERS = {
#     "Authorization": f"Bearer {API_CONFIG['hf_key']}",
#     "Content-Type": "application/json"
# }

# # Session state initialization
# def init_session_state():
#     """Initialize session state variables"""
#     if 'topic_classifier' not in st.session_state:
#         st.session_state.topic_classifier = None
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
#     if 'documents' not in st.session_state:
#         st.session_state.documents = {}
#     if 'active_docs' not in st.session_state:
#         st.session_state.active_docs = set()
#     if 'processing_status' not in st.session_state:
#         st.session_state.processing_status = {}
#     if 'previous_files' not in st.session_state:
#         st.session_state.previous_files = set()

# # Constants
# MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
# MAX_BATCH_SIZE = 5
# MAX_SUMMARY_LENGTH = 250
# MIN_SUMMARY_LENGTH = 50

# class TextProcessor:
#     """Utility class for text processing operations"""
    
#     @staticmethod
#     def clean_text(text: str) -> str:
#         """Clean and normalize text content"""
#         if not text:
#             return ""
        
#         # Basic cleaning
#         text = text.replace('\n', ' ')
#         text = text.replace('\t', ' ')
        
#         # Remove multiple spaces
#         while '  ' in text:
#             text = text.replace('  ', ' ')
            
#         # Remove special characters but keep essential punctuation
#         text = re.sub(r'[^\w\s.,!?-]', '', text)
        
#         return text.strip()
    
#     @staticmethod
#     def extract_title(text: str, filename: str) -> str:
#         """Extract title from text or use filename"""
#         if not text:
#             return filename
            
#         # Try to get first meaningful line
#         lines = text.split('\n')
#         for line in lines:
#             cleaned = line.strip()
#             if len(cleaned) > 5 and len(cleaned.split()) <= 20:
#                 return cleaned
                
#         return filename
    
#     @staticmethod
#     def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
#         """Split text into overlapping chunks"""
#         words = text.split()
#         chunks = []
        
#         for i in range(0, len(words), chunk_size - overlap):
#             chunk = ' '.join(words[i:i + chunk_size])
#             chunks.append(chunk)
            
#         return chunks
    
#     @staticmethod
#     def clean_table_text(text: str) -> str:
#         """Clean text extracted from tables"""
#         # Remove table formatting characters
#         text = re.sub(r'[\|\+\-\=]+', ' ', text)
#         # Normalize spaces
#         text = ' '.join(text.split())
#         return text.strip()
    
# class PDFProcessor:
#     """Handles PDF document processing"""
    
#     @staticmethod
#     def process(file) -> Dict[str, str]:
#         """Process PDF files with enhanced text extraction"""
#         try:
#             pdf_reader = PyPDF2.PdfReader(file)
#             text_content = []
            
#             # Process each page
#             for page in pdf_reader.pages:
#                 # Extract text
#                 page_text = page.extract_text()
#                 if page_text:
#                     # Clean and add to content
#                     cleaned_text = TextProcessor.clean_text(page_text)
#                     if cleaned_text:
#                         text_content.append(cleaned_text)
            
#             # Combine all text
#             full_text = '\n'.join(text_content)
            
#             if not full_text:
#                 raise ValueError("No text could be extracted from PDF")
                
#             return {
#                 "content": full_text,
#                 "type": "pdf",
#                 "name": file.name,
#                 "num_pages": len(pdf_reader.pages)
#             }
#         except Exception as e:
#             logger.error(f"PDF processing error: {str(e)}")
#             raise

# class DOCProcessor:
#     """Handles DOC document processing (older Word format)"""
    
#     @staticmethod
#     def process(file) -> Dict[str, str]:
#         """Process DOC files with formatting preservation"""
#         try:
#             # Create a temporary file to store the uploaded content
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as temp_file:
#                 temp_file.write(file.read())
#                 temp_path = temp_file.name
            
#             try:
#                 # Try using antiword first (more reliable for old .doc files)
#                 text = subprocess.check_output(['antiword', temp_path]).decode('utf-8')
#             except (subprocess.SubprocessError, FileNotFoundError):
#                 try:
#                     # Fallback to using python-docx
#                     doc = docx.Document(temp_path)
#                     text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
#                 except Exception as e:
#                     raise ValueError(f"Could not process DOC file: {str(e)}")
#             finally:
#                 # Clean up temporary file
#                 os.unlink(temp_path)
            
#             # Clean text
#             clean_text = TextProcessor.clean_text(text)
            
#             if not clean_text:
#                 raise ValueError("No valid text content in DOC file")
                
#             return {
#                 "content": clean_text,
#                 "type": "doc",
#                 "name": file.name
#             }
            
#         except Exception as e:
#             logger.error(f"DOC processing error: {str(e)}")
#             raise

# class DocxProcessor:
#     """Handles DOCX document processing"""
    
#     @staticmethod
#     def process(file) -> Dict[str, str]:
#         """Process DOCX files with formatting preservation"""
#         try:
#             doc = docx.Document(file)
#             text_content = []
            
#             # Process paragraphs
#             for paragraph in doc.paragraphs:
#                 if paragraph.text.strip():
#                     # Handle different formatting
#                     formatted_text = []
#                     for run in paragraph.runs:
#                         # Preserve text with its formatting
#                         formatted_text.append(run.text)
                    
#                     clean_text = TextProcessor.clean_text(' '.join(formatted_text))
#                     if clean_text:
#                         text_content.append(clean_text)
            
#             # Process tables
#             for table in doc.tables:
#                 table_content = []
#                 for row in table.rows:
#                     row_text = []
#                     for cell in row.cells:
#                         cell_text = TextProcessor.clean_text(cell.text)
#                         if cell_text:
#                             row_text.append(cell_text)
#                     if row_text:
#                         table_content.append(' | '.join(row_text))
#                 if table_content:
#                     text_content.append('\n'.join(table_content))
            
#             # Combine all content
#             full_text = '\n'.join(text_content)
            
#             if not full_text:
#                 raise ValueError("No text could be extracted from DOCX")
                
#             return {
#                 "content": full_text,
#                 "type": "docx",
#                 "name": file.name,
#                 "has_tables": len(doc.tables) > 0
#             }
#         except Exception as e:
#             logger.error(f"DOCX processing error: {str(e)}")
#             raise

# class TxtProcessor:
#     """Handles TXT document processing"""
    
#     @staticmethod
#     def process(file) -> Dict[str, str]:
#         """Process TXT files with encoding handling"""
#         try:
#             # Try different encodings
#             encodings = ['utf-8', 'latin-1', 'cp1252']
#             text = None
            
#             for encoding in encodings:
#                 try:
#                     text = file.read().decode(encoding)
#                     break
#                 except UnicodeDecodeError:
#                     continue
            
#             if text is None:
#                 raise ValueError("Could not decode text file with supported encodings")
            
#             # Clean text
#             clean_text = TextProcessor.clean_text(text)
            
#             if not clean_text:
#                 raise ValueError("No valid text content in file")
                
#             return {
#                 "content": clean_text,
#                 "type": "txt",
#                 "name": file.name
#             }
#         except Exception as e:
#             logger.error(f"TXT processing error: {str(e)}")
#             raise

# # After your imports and configurations, before DocumentProcessor class

# def get_combined_context(active_docs: List[str]) -> str:
#     """Combine content from multiple documents"""
#     combined_text = ""
#     for doc_name in active_docs:
#         doc = st.session_state.documents.get(doc_name)
#         if doc:
#             combined_text += f"\nDocument: {doc_name}\n"
#             if 'summary' in doc:
#                 combined_text += f"Summary: {doc['summary']}\n"
#             content_preview = doc['content'][:1000]
#             combined_text += f"Content Preview: {content_preview}\n"
#             combined_text += "---\n"
#     return combined_text

# def get_friendly_response(question: str, active_docs: List[str]) -> str:
#     """Generate response using Gemini model"""
#     try:
#         context = get_combined_context(active_docs)
#         prompt = f"""You are a helpful assistant analyzing these documents. 
#         Here are the documents and their summaries:
#         {context}

#         User question: {question}

#         Please provide a comprehensive answer based on ALL selected documents. 
#         If referring to specific information, mention which document it came from.
#         If you can't find the information in any document, say so clearly.
#         Maintain a friendly, professional, conversational tone and explain in simple easy terms.
        
#         Guidelines:
#         - Cite specific documents when referencing information
#         - Be clear about uncertain or missing information
#         - Use examples from the documents when relevant
#         - Keep the response focused and concise
#         """

#         response = API_CONFIG['model'].generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0.7,
#                 max_output_tokens=1024,
#             )
#         )
#         return response.text
        
#     except Exception as e:
#         logger.error(f"Response generation error: {str(e)}")
#         return f"I apologize, but I encountered an error: {str(e)}"

# def get_similarity_interpretation(score: float) -> Dict[str, str]:
#     """Get interpretation of similarity score"""
#     if score > 70:
#         return {
#             "level": "High",
#             "description": "Documents are very similar in content and context",
#             "color": "green"
#         }
#     elif score > 40:
#         return {
#             "level": "Moderate",
#             "description": "Documents share some common elements",
#             "color": "orange"
#         }
#     else:
#         return {
#             "level": "Low",
#             "description": "Documents are substantially different",
#             "color": "red"
#         }

# class DocumentProcessor:
#     """Main document processing class"""
    
#     def __init__(self):
#         self.executor = ThreadPoolExecutor(max_workers=5)
#         self.processors = {
#             'pdf': PDFProcessor.process,
#             'docx': DocxProcessor.process,
#             'doc': DOCProcessor.process,
#             'txt': TxtProcessor.process
#         }

#     async def process_single_file(self, file) -> Optional[Dict]:
#         """Process a single file with size and format validation"""
#         try:
#             # Check file size
#             if file.size > MAX_FILE_SIZE:
#                 raise ValueError(
#                     f"File too large: {file.name} "
#                     f"({file.size/1024/1024:.1f}MB). Maximum size is "
#                     f"{MAX_FILE_SIZE/1024/1024}MB"
#                 )

#             # Validate format
#             file_ext = Path(file.name).suffix[1:].lower()
#             if file_ext not in self.processors:
#                 raise ValueError(f"Unsupported file format: {file_ext}")

#             # Process document
#             try:
#                 # Create a new event loop for the executor
#                 loop = asyncio.new_event_loop()
#                 asyncio.set_event_loop(loop)
                
#                 # Process the file
#                 result = await loop.run_in_executor(
#                     self.executor,
#                     self.processors[file_ext],
#                     file
#                 )
                
#                 if result and result.get('content'):
#                     # Add metadata
#                     result['stats'] = {
#                         'word_count': len(result['content'].split()),
#                         'char_count': len(result['content']),
#                         'upload_time': time.time()
#                     }

#                     # Perform automatic classification
#                     if 'topic_classifier' in st.session_state:
#                         with st.spinner(f"Classifying {file.name}..."):
#                             classification_results = st.session_state.topic_classifier.classify_document(
#                                 result['content']
#                             )
#                             if classification_results:
#                                 result['default_classification'] = classification_results
#                                 st.success(f"✅ Classification completed for {file.name}")
                    
#                     return result
                
#                 return None

#             except Exception as processing_error:
#                 logger.error(f"Document processing error: {str(processing_error)}")
#                 raise ValueError(f"Error processing document: {str(processing_error)}")
            
#             finally:
#                 # Clean up the loop
#                 loop.close()

#         except Exception as e:
#             logger.error(f"File processing error: {str(e)}")
#             st.session_state.processing_status[file.name] = f"Error: {str(e)}"
#             return None

#     async def get_summary(self, text: str) -> str:
#         """Generate summary with chunking and retry logic"""
#         try:
#             text = TextProcessor.clean_text(text)
#             if not text:
#                 return "No valid text content to summarize"

#             # Handle different text lengths
#             chunks = TextProcessor.chunk_text(text)
#             summaries = []

#             for chunk in chunks:
#                 try:
#                     payload = {
#                         "inputs": chunk,
#                         "parameters": {
#                             "max_length": MAX_SUMMARY_LENGTH,
#                             "min_length": MIN_SUMMARY_LENGTH,
#                             "do_sample": False
#                         }
#                     }

#                     response = requests.post(
#                         API_CONFIG['summary_url'],
#                         headers=HEADERS,
#                         json=payload,
#                         timeout=30
#                     )

#                     if response.status_code == 200:
#                         summaries.append(response.json()[0]['summary_text'])
#                     elif response.status_code == 503:
#                         await asyncio.sleep(20)  # Wait for model to load
#                         response = requests.post(
#                             API_CONFIG['summary_url'],
#                             headers=HEADERS,
#                             json=payload
#                         )
#                         if response.status_code == 200:
#                             summaries.append(response.json()[0]['summary_text'])

#                     await asyncio.sleep(2)  # Rate limit handling

#                 except Exception as e:
#                     logger.error(f"Chunk summary error: {str(e)}")
#                     continue

#             if summaries:
#                 if len(summaries) == 1:
#                     return summaries[0]
#                 else:
#                     combined = " ".join(summaries)
#                     if len(combined.split()) > MAX_SUMMARY_LENGTH:
#                         # Recursively summarize the combined summaries
#                         return await self.get_summary(combined)
#                     return combined + "\n(Note: Combined from multiple sections)"

#             return "Could not generate summary. Please try with simpler content."

#         except Exception as e:
#             logger.error(f"Summarization error: {str(e)}")
#             return f"Error generating summary: {str(e)}"

#     # async def generate_image(self, text: str, title: str = "") -> Tuple[Optional[Image.Image], Optional[str]]:
#     #     """Generate image with enhanced prompt engineering"""
#     #     try:
#     #         if not text:
#     #             logger.error("No text provided for image generation")
#     #             return None, "No text provided for image generation"

#     #         # Create enhanced prompt
#     #         prompt = f"""Create a visualization about:
#     #         Title: {title}
#     #         """

#     #         payload = {"inputs": prompt}

#     #         try:
#     #             response = requests.post(
#     #                 API_CONFIG['image_url'],
#     #                 headers=HEADERS,
#     #                 json=payload,
#     #                 timeout=30
#     #             )

#     #             if response.status_code == 200:
#     #                 image = Image.open(io.BytesIO(response.content))
#     #                 return image, None
#     #             else:
#     #                 error_msg = response.json().get('error', 'Unknown error')
#     #                 if "Max requests total reached" in error_msg:
#     #                     return None, "⏳ Rate limit reached. Please wait 60 seconds..."
#     #                 else:
#     #                     logger.error(f"Image generation failed: {error_msg}")
#     #                     return None, f"Failed to generate image: {error_msg}"

#     #         except requests.exceptions.RequestException as e:
#     #             return None, f"Request failed: {str(e)}"

#     #     except Exception as e:
#     #         return None, f"Error generating image: {str(e)}"

#     async def generate_image(self, text: str, title: str = "") -> Tuple[Optional[Image.Image], Optional[str]]:
#         """Generate artistic visualization of document content"""
#         try:
#             if not text:
#                 return None, "No text provided for image generation"

#             # Extract main concept/theme from title and abstract
#             main_concept = f"{title}. {text[:200]}"  # Combine title with start of text

#             # Create a conceptual art prompt
#             prompt = f"""Create a single artistic concept visualization:
#             Main idea: {main_concept}
#             Style Requirements:
#             - Modern digital art style
#             - Professional futuristic design
#             - Abstract representation of the concept
#             - Rich symbolic visualization
#             - Vibrant colors and dynamic composition
#             - Highly detailed technological aesthetic
#             - Focus on the core idea, not technical details
#             - No text, charts, or diagrams
#             - Single cohesive image that captures the essence
#             - Professional sci-fi art quality
#             """

#             payload = {"inputs": prompt}

#             try:
#                 response = requests.post(
#                     API_CONFIG['image_url'],
#                     headers=HEADERS,
#                     json=payload,
#                     timeout=30
#                 )

#                 if response.status_code == 200:
#                     image = Image.open(io.BytesIO(response.content))
#                     return image, None
#                 else:
#                     error_msg = response.json().get('error', 'Unknown error')
#                     if "Max requests total reached" in error_msg:
#                         return None, "⏳ Rate limit reached. Please wait 60 seconds..."
#                     else:
#                         logger.error(f"Image generation failed: {error_msg}")
#                         return None, f"Failed to generate image: {error_msg}"

#             except requests.exceptions.RequestException as e:
#                 return None, f"Request failed: {str(e)}"

#         except Exception as e:
#             return None, f"Error generating image: {str(e)}"
            
#     def process_files_parallel(self, files: List):
#         """Process multiple files synchronously"""
#         start_time = time.time()
        
#         try:
#             if len(files) > MAX_BATCH_SIZE:
#                 st.warning(f"Processing files in batches of {MAX_BATCH_SIZE}...")

#             # Initialize topic classifier once for all files
#             if 'topic_classifier' not in st.session_state:
#                 st.session_state.topic_classifier = TopicClassifier()

#             for file in files:
#                 if file.name not in st.session_state.documents:
#                     st.session_state.processing_status[file.name] = "Processing..."
                    
#                     try:
#                         # Process single file
#                         file_ext = Path(file.name).suffix[1:].lower()
#                         if file_ext not in self.processors:
#                             raise ValueError(f"Unsupported file format: {file_ext}")
                        
#                         # Direct processing without async
#                         result = self.processors[file_ext](file)
                        
#                         if result and result.get('content'):
#                             # Add metadata
#                             result['stats'] = {
#                                 'word_count': len(result['content'].split()),
#                                 'char_count': len(result['content']),
#                                 'upload_time': time.time()
#                             }
                            
#                             # Perform automatic classification
#                             if st.session_state.topic_classifier:
#                                 with st.spinner(f"Classifying {file.name}..."):
#                                     classification_results = st.session_state.topic_classifier.classify_document(
#                                         result['content']
#                                     )
#                                     if classification_results:
#                                         result['default_classification'] = classification_results
#                                         st.success(f"✅ Classification completed for {file.name}")
#                                     else:
#                                         st.warning(f"⚠️ Could not classify {file.name}")
                            
#                             # Store result
#                             st.session_state.documents[file.name] = result
#                             st.session_state.active_docs.add(file.name)
#                             st.session_state.processing_status[file.name] = "Completed"
#                             st.success(f"✅ {file.name} processed successfully!")
                    
#                     except Exception as e:
#                         error_msg = str(e)
#                         logger.error(f"Error processing {file.name}: {error_msg}")
#                         st.error(f"Error processing {file.name}: {error_msg}")
#                         st.session_state.processing_status[file.name] = f"Error: {error_msg}"

#             processing_time = time.time() - start_time
#             logger.info(f"Total processing time: {processing_time:.2f} seconds")

#         except Exception as e:
#             logger.error(f"File processing error: {str(e)}")
#             st.error("An error occurred during file processing")
#         finally:
#             gc.collect()

# class DocumentSimilarity:
#     """Handles document similarity calculations"""
    
#     def __init__(self):
#         try:
#             with st.spinner("Loading similarity model..."):
#                 device = 'cuda' if torch.cuda.is_available() else 'cpu'
#                 self.model = SentenceTransformer(
#                     #'sentence-transformers/all-MiniLM-L6-v2',
#                     'sentence-transformers/all-mpnet-base-v2',
#                     device=device
#                 )
#         except Exception as e:
#             st.error(f"Error initializing similarity model: {str(e)}")
#             logger.error(f"Model initialization error: {str(e)}")
#             self.model = None

#     def calculate_similarity(self, source_text: str, comparison_texts: List[str]) -> Optional[List[float]]:
#         """Calculate similarity scores between documents"""
#         try:
#             if self.model is None:
#                 return None

#             # Convert texts to tensors
#             with torch.no_grad():
#                 source_embedding = self.model.encode(
#                     source_text,
#                     convert_to_tensor=True
#                 )
#                 comparison_embeddings = self.model.encode(
#                     comparison_texts,
#                     convert_to_tensor=True
#                 )
                
#                 # Calculate cosine similarity
#                 similarities = util.pytorch_cos_sim(
#                     source_embedding,
#                     comparison_embeddings
#                 )[0]
            
#             # Convert to percentages
#             return [float(score) * 100 for score in similarities]

#         except Exception as e:
#             logger.error(f"Similarity calculation error: {str(e)}")
#             st.error(f"Error calculating similarity: {str(e)}")
#             return None

#     @staticmethod
#     def get_similarity_color(score: float) -> str:
#         """Get color coding for similarity score"""
#         if score > 70:
#             return 'green'
#         elif score > 40:
#             return 'orange'
#         else:
#             return 'red'
        
# class UIComponents:
#     """Handles all UI components and layouts"""
    
#     # @staticmethod
#     # def render_chat_interface(container):
#     #     """Render chat interface with message history"""
#     #     for message in st.session_state.chat_history:
#     #         with container.chat_message(message["role"]):
#     #             st.markdown(message["content"])

#     @staticmethod
#     def render_document_stats(doc: Dict):
#         """Render document statistics"""
#         st.markdown(f"""
#         **Document Statistics:**
#         - Words: {len(doc['content'].split())}
#         - Characters: {len(doc['content'])}
#         - Type: {doc['type'].upper()}
#         {f"- Pages: {doc['num_pages']}" if 'num_pages' in doc else ""}
#         """)

#     @staticmethod
#     def render_document_content(doc: Dict, key_prefix: str):
#         """Render document content and summary"""
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("### Original Content")
#             st.write("")
#             st.text_area(
#                 label="",
#                 value=doc['content'][:1000] + "..." if len(doc['content']) > 1000 else doc['content'],
#                 height=300,
#                 disabled=True,
#                 key=f"content_{key_prefix}"
#             )
        
#         with col2:
#             st.markdown("### Summary")
#             st.write("")
#             if 'summary' in doc:
#                 st.text_area(
#                     label="",
#                     value=doc['summary'],
#                     height=300,
#                     disabled=True,
#                     key=f"summary_{key_prefix}"
#                 )
#             else:
#                 st.info("Click 'Generate Summaries' to create summary")

#     @staticmethod
#     async def handle_rate_limit():
#         """Handle API rate limit with countdown"""
#         time_left = 60
#         progress_text = "Please wait..."
#         progress_bar = st.progress(0)
        
#         for i in range(time_left):
#             progress_bar.progress((i + 1) / time_left)
#             st.text(f"Time remaining: {time_left - i} seconds")
#             await asyncio.sleep(1)
            
#         st.success("You can try again now!")
#         st.rerun()
    
# def render_similarity_tab():
#     """Render similarity analysis interface"""
#     st.markdown("### Document Similarity Analysis")
    
#     if len(st.session_state.active_docs) < 2:
#         st.warning("Please select at least 2 documents to use similarity analysis.")
#         return
    
#     available_docs = list(st.session_state.active_docs)
    
#     selected_doc = st.selectbox(
#         "Select a document to compare with others",
#         available_docs,
#         key="similarity_doc_selector"
#     )
    
#     if selected_doc:
#         source_doc = st.session_state.documents[selected_doc]
        
#         comparison_docs = {
#             name: doc for name, doc in st.session_state.documents.items() 
#             if name != selected_doc and name in st.session_state.active_docs
#         }
        
#         if comparison_docs:
#             with st.spinner("Calculating similarity scores..."):
#                 try:
#                     similarity_calc = DocumentSimilarity()
                    
#                     comparison_texts = [doc['content'] for doc in comparison_docs.values()]
#                     scores = similarity_calc.calculate_similarity(
#                         source_doc['content'],
#                         comparison_texts
#                     )
                    
#                     if scores:
#                         st.markdown("### Similarity Scores")
                        
#                         for (doc_name, doc), score in zip(comparison_docs.items(), scores):
#                             with st.expander(f"**{doc_name}**", expanded=True):
#                                 col1, col2 = st.columns([3, 1])
                                
#                                 with col1:
#                                     if 'summary' in doc:
#                                         st.markdown("**Summary:**")
#                                         st.info(doc['summary'][:200] + "...")
                                
#                                 with col2:
#                                     color = DocumentSimilarity.get_similarity_color(score)
#                                     st.markdown(
#                                         f"<h3 style='text-align: center; color: {color}'>"
#                                         f"{score:.1f}%</h3>",
#                                         unsafe_allow_html=True
#                                     )
                                    
#                                     interp = get_similarity_interpretation(score)
#                                     if score > 70:
#                                         st.success(interp["description"])
#                                     elif score > 40:
#                                         st.warning(interp["description"])
#                                     else:
#                                         st.error(interp["description"])
                                
#                                 st.markdown("---")
#                     else:
#                         st.error("Error calculating similarity scores")
                
#                 except Exception as e:
#                     st.error(f"Error in similarity analysis: {str(e)}")
#                     logger.error(f"Similarity error details: {str(e)}")


# class TopicClassifier:
#     def __init__(self):
#         self.model_id = "cross-encoder/nli-deberta-v3-large"  # Changed to recommended model
#         self.default_categories = [
#             "Artificial Intelligence", "Machine Learning", "Natural Language Processing",
#             "Computer Vision", "Robotics", "Data Science", "Physics", "Mathematics",
#             "Statistics", "Biology", "Chemistry", "Economics", "Finance", "Medicine",
#             "Engineering", "Space Science", "Earth Science", "Materials Science"
#         ]
#         self.classifier = self.initialize_classifier()

#     def initialize_classifier(self):
#         """Initialize the classifier with proper error handling"""
#         try:
#             st.info("Loading classification model...")
#             return pipeline(
#                 "zero-shot-classification",
#                 model=self.model_id,
#                 device=-1,  # Force CPU usage for stability
#                 hypothesis_template="This text is about {}."
#             )
#         except Exception as e:
#             st.error(f"Failed to initialize classifier: {str(e)}")
#             logger.error(f"Classifier initialization error: {str(e)}")
#             return None

#     def classify_document(self, text: str, categories: List[str] = None, multi_label: bool = True) -> Optional[Dict]:
#         """Classify document into topics"""
#         if self.classifier is None:
#             st.error("Classifier not properly initialized")
#             return None
            
#         try:
#             if categories is None:
#                 categories = self.default_categories
                
#             # Clean and truncate text if needed
#             text = TextProcessor.clean_text(text)
#             if len(text) > 1024:  # Add text length limit
#                 text = text[:1024]
            
#             if not text:
#                 return None
                
#             # Run classification
#             with st.spinner("Classifying document..."):
#                 result = self.classifier(
#                     text,
#                     candidate_labels=categories,
#                     multi_label=multi_label
#                 )
                
#                 # Process results
#                 topic_scores = list(zip(result['labels'], result['scores']))
#                 topic_scores.sort(key=lambda x: x[1], reverse=True)
                
#                 return {
#                     'topics': [t[0] for t in topic_scores],
#                     'scores': [t[1] for t in topic_scores]
#                 }
                
#         except Exception as e:
#             st.error(f"Classification failed: {str(e)}")
#             logger.error(f"Classification error: {str(e)}")
#             return None
# class DocumentTabs:
#     """Handles different tab views in the application"""

#     def __init__(self, doc_processor: DocumentProcessor):
#         self.doc_processor = doc_processor
#         # Initialize topic classifier properly
#         if 'topic_classifier' not in st.session_state:
#             st.session_state.topic_classifier = TopicClassifier()

#     def classify_uploaded_document(self, doc_name: str, doc_content: str):
#         """Classify newly uploaded document"""
#         if 'topic_classifier' in st.session_state:
#             results = st.session_state.topic_classifier.classify_document(doc_content)
#             if results:
#                 st.session_state.documents[doc_name]['default_classification'] = results


#     def render_chat_tab(self, tab):
#         """Render chat interface tab with fixed input and proper message flow"""
#         with tab:
#             st.markdown("### Chat with your Documents")
            
#             if st.session_state.active_docs:
#                 st.info(f"📚 Currently analyzing: {', '.join(st.session_state.active_docs)}")
                
#                 # Create containers for messages and input
#                 messages_container = st.container()
#                 input_container = st.container()
                
#                 # Handle input first (at the bottom)
#                 with input_container:
#                     # Add some spacing before the input
#                     st.markdown("<br>" * 2, unsafe_allow_html=True)
                    
#                     # Get user input
#                     prompt = st.chat_input("Ask me anything about your documents...")
                
#                 # Display messages in the messages container
#                 with messages_container:
#                     for message in st.session_state.chat_history:
#                         with st.chat_message(message["role"]):
#                             st.markdown(message["content"])
                    
#                     # Handle new message if there's input
#                     if prompt:
#                         # Immediately show user message
#                         with st.chat_message("user"):
#                             st.markdown(prompt)
                        
#                         # Show assistant response with loading indicator
#                         with st.chat_message("assistant"):
#                             with st.spinner("Thinking..."):
#                                 response = get_friendly_response(
#                                     prompt,
#                                     list(st.session_state.active_docs)
#                                 )
#                                 st.markdown(response)
                        
#                         # Update chat history
#                         st.session_state.chat_history.extend([
#                             {"role": "user", "content": prompt},
#                             {"role": "assistant", "content": response}
#                         ])
                        
#                         # Rerun to update the display properly
#                         st.rerun()
#             else:
#                 st.info("👈 Please upload and select documents to start chatting!")


#     def render_documents_tab(self, tab):
#         """Render documents analysis tab"""
#         with tab:
#             st.markdown("### Document Summaries")
            
#             # Document selector with default to all documents
#             available_docs = list(st.session_state.documents.keys())
#             selected_docs = st.multiselect(
#                 "Select documents to summarize",
#                 available_docs,
#                 default=available_docs,  # Default to all documents
#                 key="summary_doc_selector"
#             )
            
#             if selected_docs:
#                 if st.button("Generate Summaries", key="generate_summaries"):
#                     total_docs = len(selected_docs)
#                     progress_bar = st.progress(0)
                    
#                     for idx, doc_name in enumerate(selected_docs):
#                         if doc_name in st.session_state.documents:
#                             doc = st.session_state.documents[doc_name]
#                             status_text = st.empty()
                            
#                             if 'summary' not in doc:
#                                 try:
#                                     status_text.text(f"Processing {doc_name}...")
#                                     summary = asyncio.run(self.doc_processor.get_summary(doc['content']))
#                                     if summary:
#                                         st.session_state.documents[doc_name]['summary'] = summary
#                                         st.success(f"✅ Summary generated for {doc_name}")
#                                 except Exception as e:
#                                     st.error(f"Error generating summary for {doc_name}: {str(e)}")
#                             else:
#                                 st.info(f"Summary already exists for {doc_name}")
                            
#                             # Update progress
#                             progress_bar.progress((idx + 1) / total_docs)
                    
#                     st.success("All documents processed!")
            
#             # Display documents
#             for doc_name in selected_docs:
#                 if doc_name in st.session_state.documents:
#                     doc = st.session_state.documents[doc_name]
#                     with st.expander(f"📄 {doc_name}", expanded=True):
#                         UIComponents.render_document_content(doc, doc_name)
#                         UIComponents.render_document_stats(doc)

#     def render_image_tab(self, tab):
#         """Render image generation tab"""
#         with tab:
#             st.markdown("### Document Visualization")
            
#             # Document selector
#             available_docs = list(st.session_state.documents.keys())
#             selected_doc = st.selectbox(
#                 "Select a document to visualize",
#                 [""] + available_docs,
#                 key="image_doc_selector"
#             )
            
#             if selected_doc:
#                 doc = st.session_state.documents.get(selected_doc)
#                 if doc:
#                     col1, col2 = st.columns([2, 1])
#                     with col1:
#                         generate_btn = st.button(
#                             "Generate Image",
#                             key="generate_image",
#                             use_container_width=True
#                         )
                    
#                     if generate_btn:
#                         asyncio.run(self.handle_image_generation(doc, selected_doc))
                    
#                     # Display existing image
#                     if 'image' in doc and doc['image'] is not None:
#                         st.markdown("### Generated Image")
#                         st.image(doc['image'], use_container_width=True)
                        
#                         if st.button("🔄 Generate New Image", key="regenerate"):
#                             asyncio.run(self.handle_image_generation(doc, selected_doc))

#     def render_topic_tab(self, tab):
#         """Render topic classification interface"""
#         with tab:
#             st.markdown("### Document Topic Classification")
            
#             if not st.session_state.active_docs:
#                 st.info("👈 Please upload and select documents first")
#                 return

#             # Make sure topic classifier exists
#             if 'topic_classifier' not in st.session_state or st.session_state.topic_classifier is None:
#                 st.session_state.topic_classifier = TopicClassifier()

#             # Perform default classification if not already done
#             for doc_name in st.session_state.active_docs:
#                 doc = st.session_state.documents[doc_name]
#                 if 'default_classification' not in doc:
#                     with st.spinner(f"Classifying {doc_name}..."):
#                         results = st.session_state.topic_classifier.classify_document(
#                             doc['content']
#                         )
#                         if results:
#                             doc['default_classification'] = results

#             # Custom categories option
#             use_custom = st.checkbox("Use custom categories", value=False)
            
#             if use_custom:
#                 custom_input = st.text_area(
#                     "Enter custom categories (one per line)",
#                     help="Enter each category on a new line"
#                 )
#                 categories = [cat.strip() for cat in custom_input.split('\n') if cat.strip()]
                
#                 if categories:
#                     if st.button("Classify with Custom Categories", type="primary"):
#                         for doc_name in st.session_state.active_docs:
#                             doc = st.session_state.documents[doc_name]
#                             with st.spinner(f"Classifying {doc_name}..."):
#                                 results = st.session_state.topic_classifier.classify_document(
#                                     doc['content'],
#                                     categories=categories
#                                 )
#                                 if results:
#                                     doc['custom_classification'] = results
#                                 else:
#                                     st.error(f"Failed to classify {doc_name} with custom categories")
#                 else:
#                     st.warning("Please enter at least one category")
            
#             # Display results for all documents
#             st.markdown("### Classification Results")
#             for doc_name in st.session_state.active_docs:
#                 doc = st.session_state.documents[doc_name]
                
#                 with st.expander(f"📄 {doc_name}", expanded=True):
#                     # Show results based on classification type
#                     results = None
#                     if use_custom and 'custom_classification' in doc:
#                         results = doc['custom_classification']
#                         st.markdown("#### Custom Categories Classification")
#                     elif 'default_classification' in doc:
#                         results = doc['default_classification']
#                         st.markdown("#### Default Categories Classification")
                    
#                     if results:
#                         # Create DataFrame
#                         df = pd.DataFrame({
#                             'Topic': results['topics'],
#                             'Confidence': [f"{score*100:.2f}%" for score in results['scores']]
#                         })
                        
#                         col1, col2 = st.columns([2, 1])
                        
#                         with col1:
#                             st.dataframe(
#                                 df,
#                                 column_config={
#                                     "Topic": st.column_config.TextColumn("Topic"),
#                                     "Confidence": st.column_config.TextColumn("Confidence")
#                                 },
#                                 hide_index=True
#                             )
                        
#                         with col2:
#                             st.metric(
#                                 "Primary Classification",
#                                 results['topics'][0],
#                                 f"{results['scores'][0]*100:.1f}%"
#                             )
                        
#                         # Visualization
#                         fig = px.bar(
#                             df.head(8),
#                             x='Topic',
#                             y=[float(s.strip('%')) for s in df.head(8)['Confidence']],
#                             title='Topic Classification Results',
#                             labels={'y': 'Confidence (%)', 'x': 'Topic'}
#                         )
#                         fig.update_layout(xaxis_tickangle=-45, height=400)
#                         st.plotly_chart(fig, use_container_width=True)
#                     else:
#                         st.info("Performing classification...")
#                         # Try to classify document
#                         with st.spinner("Classifying document..."):
#                             results = st.session_state.topic_classifier.classify_document(
#                                 doc['content']
#                             )
#                             if results:
#                                 doc['default_classification'] = results
#                                 st.rerun()
#                             else:
#                                 st.error("Classification failed")

#     async def handle_image_generation(self, doc: Dict, doc_name: str):
#         """Handle image generation process"""
#         with st.spinner(f"Processing {doc_name}..."):
#             try:
#                 # Ensure summary exists
#                 if 'summary' not in doc or not doc['summary']:
#                     st.info("Generating summary first...")
#                     summary = await self.doc_processor.get_summary(doc['content'])
#                     if summary:
#                         st.session_state.documents[doc_name]['summary'] = summary
#                         st.success("Summary generated!")
#                     else:
#                         st.error("Failed to generate summary")
#                         return

#                 # Extract title
#                 title = TextProcessor.extract_title(doc['content'], doc_name)
                
#                 # Generate image
#                 st.info("Generating image...")
#                 image, error_msg = await self.doc_processor.generate_image(
#                     doc['summary'],
#                     title=title
#                 )
                
#                 if image:
#                     st.session_state.documents[doc_name]['image'] = image
#                     st.success("✅ Image generated successfully!")
#                     st.rerun()
#                 elif "Rate limit" in str(error_msg):
#                     await UIComponents.handle_rate_limit()
#                 else:
#                     st.error(error_msg)
                    
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")
#                 logger.error(f"Error details: {str(e)}")

# def main():
#     """Main application entry point"""
#     # Initialize session state
#     init_session_state()

#     # Initialize topic classifier
#     if 'topic_classifier' not in st.session_state or st.session_state.topic_classifier is None:
#         st.session_state.topic_classifier = TopicClassifier()
    
#     # Setup document processor
#     doc_processor = DocumentProcessor()
#     tabs_handler = DocumentTabs(doc_processor)
    
#     # Main UI
#     st.title("📚 Document Assistant")
#     st.markdown("Upload your documents for summaries and interactive chat!")
    
#     # Sidebar
#     with st.sidebar:
#         render_sidebar(doc_processor)
    
#     # Main content
#     if st.session_state.active_docs:
#         render_main_content(tabs_handler)
#     else:
#         st.info("👈 Please upload and select documents to get started!")
    
#     # Footer
#     st.markdown("---")
#     st.markdown("💡 Powered by Gemini & BART | Made with Streamlit")

# def render_sidebar(doc_processor: DocumentProcessor):
#     """Render sidebar content"""
#     st.header("📎 Document Management")
    
#     # File uploader
#     uploaded_files = st.file_uploader(
#         "Upload documents",
#         type=['pdf', 'docx', 'doc', 'txt'],
#         accept_multiple_files=True,
#         key="file_uploader"
#     )
    
#     # Handle file tracking and removal
#     current_files = set(f.name for f in uploaded_files) if uploaded_files else set()
#     deleted_files = st.session_state.previous_files - current_files
    
#     # Process deleted files
#     for deleted_file in deleted_files:
#         if deleted_file in st.session_state.documents:
#             del st.session_state.documents[deleted_file]
#             st.session_state.active_docs.discard(deleted_file)
#             if deleted_file in st.session_state.processing_status:
#                 del st.session_state.processing_status[deleted_file]
    
#     # Update tracked files
#     st.session_state.previous_files = current_files
    
#     # Process new files
#     if uploaded_files:
#         doc_processor.process_files_parallel(uploaded_files)  # Remove asyncio.run
    
#     # Document selection
#     if st.session_state.documents:
#         st.markdown("### Selected Documents")
        
#         if st.button("Deselect All"):
#             st.session_state.active_docs = set()
#             st.rerun()
        
#         for doc_name in st.session_state.documents:
#             checkbox = st.checkbox(
#                 f"📄 {doc_name}",
#                 key=f"check_{doc_name}",
#                 value=doc_name in st.session_state.active_docs
#             )
            
#             if checkbox:
#                 st.session_state.active_docs.add(doc_name)
#             else:
#                 st.session_state.active_docs.discard(doc_name)

# def render_main_content(tabs_handler: DocumentTabs):
#     """Render main content area with tabs"""
#     # Determine which tabs to show
#     show_similarity = len(st.session_state.active_docs) >= 2
#     tabs = ["💭 Chat", "📑 Documents", "🎨 Images", "📊 Topics"]
#     if show_similarity:
#         tabs.append("🔄 Similarity")
    
#     # Create tabs
#     selected_tabs = st.tabs(tabs)
    
#     # Render each tab
#     tabs_handler.render_chat_tab(selected_tabs[0])
#     tabs_handler.render_documents_tab(selected_tabs[1])
#     tabs_handler.render_image_tab(selected_tabs[2])
#     tabs_handler.render_topic_tab(selected_tabs[3])
    
#     if show_similarity:
#         with selected_tabs[4]:
#             render_similarity_tab()

# if __name__ == "__main__":
#     main()
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

        # Hide all Streamlit elements
        hide_streamlit_style = """
            <style>
            /* Hide standard Streamlit elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            footer:after {
                content:'goodbye'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }

            /* Hide deployment and GitHub buttons */
            .stDeployButton {display: none;}
            #stDecoration {display: none;}
            .viewerBadge_container__r5tak {display: none !important;}
            .viewerBadge_link__qRIco {display: none !important;}
            .stToolbar {display: none !important;}
            .stGitButton {display: none !important;}
            [data-testid="stGitButtonContainer"] {display: none !important;}
            [data-testid="StyledGitButton"] {display: none !important;}

            /* Hide Streamlit Badge */
            ._container_51w34_1 {
                display: none !important;
            }
            ._link_51w34_10 {
                display: none;
            }
            
            /* Hide Profile Container */
            ._profileContainer_51w34_53 {
                display: none !important;
            }
            ._profilePreview_51w34_63 {
                display: none !important;
            }
            ._profileImage_51w34_76 {
                display: none !important;
            }

            /* Additional backup selectors */
            [data-testid="appCreatorAvatar"] {
                display: none !important;
            }
            a[href*="streamlit.io/cloud"] {
                display: none !important;
            }
            a[href*="share.streamlit.io/user"] {
                display: none !important;
            }
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

        # Add custom CSS for layout
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