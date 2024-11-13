# document_processor.py
from document_assistant.core import logger, st, MAX_BATCH_SIZE
from document_assistant.processors import PDFProcessor, DocxProcessor, DocProcessor, TxtProcessor
from document_assistant.models import ModelManager
from typing import Dict, List, Optional
from pathlib import Path
import time
import asyncio

class DocumentProcessor:
    """Main document processing class"""
    
    def __init__(self):
        self.processors = {
            'pdf': PDFProcessor.process,
            'docx': DocxProcessor.process,
            'doc': DocProcessor.process,
            'txt': TxtProcessor.process
        }
        self.model_manager = ModelManager()
        
    async def process_documents(self, files: List) -> None:
        """Process multiple documents with parallel processing of all features"""
        try:
            # Initialize models if not already done
            if not self.model_manager.is_initialized:
                self.model_manager.initialize_models()
            
            # Process files in batches if needed
            if len(files) > MAX_BATCH_SIZE:
                st.warning(f"Processing files in batches of {MAX_BATCH_SIZE}...")
            
            for file in files:
                if file.name not in st.session_state.documents:
                    await self.process_single_document(file)
                    
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            st.error("An error occurred during document processing")

    async def process_single_document(self, file) -> None:
        """Process a single document with all features"""
        try:
            st.session_state.processing_status[file.name] = "Processing..."
            progress_text = st.empty()
            
            # 1. Process document text
            progress_text.text("Extracting text...")
            doc_info = await self._process_text(file)
            if not doc_info:
                return
            
            # 2. Generate summary
            progress_text.text("Generating summary...")
            summary = await self.model_manager.summary_model.generate_summary(doc_info['content'])
            doc_info['summary'] = summary
            
            # 3. Generate concept image - Added await
            progress_text.text("Creating visualization...")
            try:
                image, error = await self.model_manager.image_generator.generate_image(
                    summary,
                    Path(file.name).stem
                )
                if image:
                    doc_info['image'] = image
                elif error:
                    logger.warning(f"Image generation warning: {error}")
            except Exception as img_error:
                logger.error(f"Image generation error: {str(img_error)}")
            
            # 4. Classify document
            progress_text.text("Classifying document...")
            classification = self.model_manager.classifier.classify_document(doc_info['content'])
            if classification:
                doc_info['classification'] = classification
            
            # Store document first
            st.session_state.documents[file.name] = doc_info
            st.session_state.active_docs.add(file.name)
            
            # 5. Calculate similarities for all documents
            if len(st.session_state.documents) > 1:
                progress_text.text("Calculating similarities...")
                await self.update_all_similarities()
            
            # Update status
            st.session_state.processing_status[file.name] = "Completed"
            progress_text.empty()
            st.success(f"✅ {file.name} processed successfully!")
            
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            st.session_state.processing_status[file.name] = f"Error: {str(e)}"
            st.error(f"Error processing {file.name}: {str(e)}")

    async def update_all_similarities(self) -> None:
        """Update similarities for all document pairs"""
        try:
            docs = st.session_state.documents
            if len(docs) < 2:
                return
            
            # Get all document pairs
            doc_names = list(docs.keys())
            for i in range(len(doc_names)):
                for j in range(i + 1, len(doc_names)):
                    doc1_name = doc_names[i]
                    doc2_name = doc_names[j]
                    doc1 = docs[doc1_name]
                    doc2 = docs[doc2_name]
                    
                    # Initialize similarities dict if needed
                    if 'similarities' not in doc1:
                        doc1['similarities'] = {}
                    if 'similarities' not in doc2:
                        doc2['similarities'] = {}
                    
                    # Calculate similarity
                    scores = self.model_manager.similarity_model.calculate_similarity(
                        doc1['content'],
                        [doc2['content']]
                    )
                    
                    if scores and len(scores) > 0:
                        similarity_score = scores[0]
                        # Update both documents
                        doc1['similarities'][doc2_name] = similarity_score
                        doc2['similarities'][doc1_name] = similarity_score
            
        except Exception as e:
            logger.error(f"Similarity calculation error: {str(e)}")

            
    async def _process_text(self, file) -> Optional[Dict]:
        """Process document text based on file type"""
        try:
            file_ext = Path(file.name).suffix[1:].lower()
            if file_ext not in self.processors:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            result = self.processors[file_ext](file)
            if not result or not result.get('content'):
                raise ValueError("No content could be extracted")
            
            return result
            
        except Exception as e:
            logger.error(f"Text processing error: {str(e)}")
            return None
        
    async def _calculate_similarities(self, new_doc: Dict) -> None:
        """Calculate similarities between documents"""
        try:
            docs = st.session_state.documents
            if len(docs) < 2:  # Need at least 2 documents
                return
            
            # Calculate similarities for all pairs
            for doc1_name, doc1 in docs.items():
                if 'similarities' not in doc1:
                    doc1['similarities'] = {}
                
                for doc2_name, doc2 in docs.items():
                    if doc1_name != doc2_name:
                        if 'similarities' not in doc2:
                            doc2['similarities'] = {}
                        
                        # Calculate similarity
                        scores = self.model_manager.similarity_model.calculate_similarity(
                            doc1['content'],
                            [doc2['content']]
                        )
                        
                        if scores and len(scores) > 0:
                            similarity_score = scores[0]
                            # Update both documents
                            doc1['similarities'][doc2_name] = similarity_score
                            doc2['similarities'][doc1_name] = similarity_score
            
        except Exception as e:
            logger.error(f"Similarity calculation error: {str(e)}")

    # async def _calculate_similarities(self, new_doc: Dict) -> None:
    #     """Calculate similarities for all documents"""
    #     try:
    #         docs = st.session_state.documents
    #         if len(docs) < 2:  # Need at least 2 documents
    #             return
                
    #         # Calculate similarities for all document pairs
    #         for doc_name, doc in docs.items():
    #             if doc_name != new_doc['name']:
    #                 other_docs = {n: d for n, d in docs.items() if n != doc_name}
    #                 comparison_texts = [d['content'] for d in other_docs.values()]
                    
    #                 scores = self.model_manager.similarity_model.calculate_similarity(
    #                     doc['content'],
    #                     comparison_texts
    #                 )
                    
    #                 if scores:
    #                     # Initialize similarities dict if it doesn't exist
    #                     if 'similarities' not in doc:
    #                         doc['similarities'] = {}
                        
    #                     # Update similarities for this document
    #                     doc['similarities'].update({
    #                         list(other_docs.keys())[i]: score 
    #                         for i, score in enumerate(scores)
    #                     })
            
    #         # Calculate similarities for the new document
    #         other_docs = {n: d for n, d in docs.items() if n != new_doc['name']}
    #         if other_docs:
    #             comparison_texts = [d['content'] for d in other_docs.values()]
    #             scores = self.model_manager.similarity_model.calculate_similarity(
    #                 new_doc['content'],
    #                 comparison_texts
    #             )
                
    #             if scores:
    #                 new_doc['similarities'] = {
    #                     list(other_docs.keys())[i]: score 
    #                     for i, score in enumerate(scores)
    #                 }
                    
    #     except Exception as e:
    #         logger.error(f"Similarity calculation error: {str(e)}")
    def update_similarities(self) -> None:
        """Update similarities for all documents"""
        try:
            docs = st.session_state.documents
            if len(docs) < 2:
                return
            
            for doc_name, doc in docs.items():
                other_docs = {n: d for n, d in docs.items() if n != doc_name}
                comparison_texts = [d['content'] for d in other_docs.values()]
                
                scores = self.model_manager.similarity_model.calculate_similarity(
                    doc['content'],
                    comparison_texts
                )
                
                if scores:
                    doc['similarities'] = {
                        list(other_docs.keys())[i]: score 
                        for i, score in enumerate(scores)
                    }
                    
        except Exception as e:
            logger.error(f"Similarity update error: {str(e)}")