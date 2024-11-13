# ui_components.py
from document_assistant.core import logger
import streamlit as st
import plotly.express as px
from typing import Dict, List
import pandas as pd

class DocumentContainer:
    """Handles the display of individual document information"""
    
    #@staticmethod
    @staticmethod
    def render(doc_name: str, doc_info: Dict):
        """Render a complete document container with organized layout"""
        with st.container():
            # Document Name
            st.markdown(f"## {doc_name.split('.')[0]}")
            
            # Center Image with better quality
            if 'image' in doc_info:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("""
                        <style>
                        [data-testid="stImage"] {
                            border-radius: 15px;
                        }
                        [data-testid="stImage"] img {
                            border-radius: 15px;
                            max-width: 512px !important;
                            width: 100%;
                            margin: auto;
                            display: block;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    st.image(
                        doc_info['image'],
                        use_container_width=True,
                        output_format="PNG"
                    )

            # Summary Section
            if 'summary' in doc_info:
                st.markdown("### 📝 Summary")
                st.markdown("""
                    <style>
                    .summary-box {
                        height: 200px;
                        overflow-y: auto;
                        border: 1px solid #ddd;
                        padding: 10px;
                        border-radius: 5px;
                        text-align: justify;
                    }
                    </style>
                """, unsafe_allow_html=True)
                st.markdown(f'<div class="summary-box">{doc_info["summary"]}</div>', unsafe_allow_html=True)

            # Classification Section
            if 'classification' in doc_info:
                st.markdown("### 🏷️ Classification")
                
                # Display top topics
                df = pd.DataFrame({
                    'Topic': doc_info['classification']['topics'][:5],
                    'Confidence': [f"{score*100:.1f}%" for score in doc_info['classification']['scores'][:5]]
                })
                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                )
                
                # Classification graph
                fig = px.bar(
                    df,
                    x='Topic',
                    y=[float(s.strip('%')) for s in df['Confidence']],
                    labels={'y': 'Confidence (%)', 'x': 'Topic'},
                    height=300
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Statistics Section
            st.markdown("### 📊 Document Statistics")
            cols = st.columns(4)
            stats = doc_info['stats']
            
            with cols[0]:
                st.metric("Words", f"{stats['word_count']:,}")
            with cols[1]:
                st.metric("Characters", f"{stats['char_count']:,}")
            with cols[2]:
                st.metric("Size", f"{stats['file_size']/1024:.1f} KB")
            with cols[3]:
                if 'num_pages' in stats:
                    st.metric("Pages", stats['num_pages'])
                elif 'line_count' in stats:
                    st.metric("Lines", stats['line_count'])

            # Similarity Section (if available)
            if 'similarities' in doc_info and doc_info['similarities']:
                st.markdown("### 🔄 Similar Documents")
                similarities = doc_info['similarities']
                if similarities:  # Check if not empty
                    for other_doc, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
                        score_color = "green" if score > 70 else "orange" if score > 40 else "red"
                        st.markdown(f"""
                            <div style='padding: 10px; 
                                    border-left: 4px solid {score_color}; 
                                    margin: 5px 0;
                                    background-color: #f8f9fa;
                                    border-radius: 0 5px 5px 0;'>
                                <strong>{other_doc.split('.')[0]}</strong>: {score:.1f}% similar
                            </div>
                        """, unsafe_allow_html=True)

            st.markdown("---")  # Divider between documents


    # Two
    # def render(doc_name: str, doc_info: Dict):
    #     """Render a complete document container with organized layout"""
    #     with st.container():
    #         # Container styling
    #         st.markdown("""
    #             <style>
    #             .doc-container {
    #                 background-color: #f8f9fa;
    #                 border-radius: 1rem;
    #                 padding: 2rem;
    #                 margin: 1rem 0;
    #             }
    #             .image-container {
    #                 display: flex;
    #                 justify-content: center;
    #                 margin: 1rem 0;
    #             }
    #             .image-container img {
    #                 border-radius: 1rem;
    #                 max-width: 400px;
    #                 width: 100%;
    #             }
    #             .section-heading {
    #                 margin: 1.5rem 0 0.5rem 0;
    #                 color: #1f2937;
    #             }
    #             </style>
    #         """, unsafe_allow_html=True)

    #         st.markdown('<div class="doc-container">', unsafe_allow_html=True)
            
    #         # Document Image
    #         if 'image' in doc_info:
    #             st.markdown('<div class="image-container">', unsafe_allow_html=True)
    #             st.image(doc_info['image'], width=400)
    #             st.markdown('</div>', unsafe_allow_html=True)

    #         # Document Name and Title
    #         st.markdown(f"### 📄 {doc_name.split('.')[0]}")
    #         if 'title' in doc_info:
    #             st.markdown(f"#### {doc_info['title']}")

    #         # Summary Section
    #         if 'summary' in doc_info:
    #             st.markdown('<p class="section-heading">📝 Summary</p>', unsafe_allow_html=True)
    #             st.markdown('<div class="content-box">', unsafe_allow_html=True)
    #             st.write(doc_info['summary'])
    #             st.markdown('</div>', unsafe_allow_html=True)

    #         # Classification Section
    #         if 'classification' in doc_info:
    #             st.markdown('<p class="section-heading">🏷️ Classification</p>', unsafe_allow_html=True)
    #             st.markdown('<div class="content-box">', unsafe_allow_html=True)
                
    #             # Display top topics
    #             df = pd.DataFrame({
    #                 'Topic': doc_info['classification']['topics'][:5],
    #                 'Confidence': [f"{score*100:.1f}%" for score in doc_info['classification']['scores'][:5]]
    #             })
    #             st.dataframe(df, hide_index=True, use_container_width=True)
                
    #             # Classification graph
    #             fig = px.bar(
    #                 df,
    #                 x='Topic',
    #                 y=[float(s.strip('%')) for s in df['Confidence']],
    #                 title='Topic Distribution',
    #                 labels={'y': 'Confidence (%)', 'x': 'Topic'},
    #                 height=300
    #             )
    #             fig.update_layout(
    #                 margin=dict(l=20, r=20, t=40, b=20),
    #                 title_x=0.5,
    #                 xaxis_tickangle=-45,
    #             )
    #             st.plotly_chart(fig, use_container_width=True)
    #             st.markdown('</div>', unsafe_allow_html=True)

    #         # Statistics Section
    #         st.markdown('<p class="section-heading">📊 Document Statistics</p>', unsafe_allow_html=True)
    #         st.markdown('<div class="content-box">', unsafe_allow_html=True)
    #         cols = st.columns(4)
    #         stats = doc_info['stats']
            
    #         with cols[0]:
    #             st.metric("Words", f"{stats['word_count']:,}")
    #         with cols[1]:
    #             st.metric("Characters", f"{stats['char_count']:,}")
    #         with cols[2]:
    #             st.metric("Size", f"{stats['file_size']/1024:.1f} KB")
    #         with cols[3]:
    #             if 'num_pages' in stats:
    #                 st.metric("Pages", stats['num_pages'])
    #             elif 'line_count' in stats:
    #                 st.metric("Lines", stats['line_count'])
    #         st.markdown('</div>', unsafe_allow_html=True)

    #         # Similarity Section (if available)
    #         if 'similarities' in doc_info and doc_info['similarities']:
    #             st.markdown('<p class="section-heading">🔄 Similar Documents</p>', unsafe_allow_html=True)
    #             st.markdown('<div class="content-box">', unsafe_allow_html=True)
                
    #             for other_doc, score in doc_info['similarities'].items():
    #                 score_color = "green" if score > 70 else "orange" if score > 40 else "red"
    #                 st.markdown(f"""
    #                     <div style='padding: 0.5rem; 
    #                               border-left: 4px solid {score_color}; 
    #                               margin-bottom: 0.5rem;
    #                               background-color: #f8f9fa;
    #                               border-radius: 0 0.3rem 0.3rem 0;'>
    #                         <strong>{other_doc.split('.')[0]}</strong>: {score:.1f}% similar
    #                     </div>
    #                 """, unsafe_allow_html=True)
    #             st.markdown('</div>', unsafe_allow_html=True)

    #         st.markdown('</div>', unsafe_allow_html=True)  # Close main container
    #         st.markdown("<br>", unsafe_allow_html=True)  # Space between documents

    # One
    # def render(doc_name: str, doc_info: Dict):
    #     """Render a complete document container"""
    #     with st.container():
    #         st.markdown(f"## 📄 {doc_name.split('.')[0]}")  # Show name without extension
            
    #         # Create three columns for layout
    #         col1, col2 = st.columns([1, 1])
            
    #         with col1:
    #             # Document Statistics
    #             DocumentContainer._render_stats(doc_info['stats'])
                
    #             # Document Summary
    #             if 'summary' in doc_info:
    #                 st.markdown("### 📝 Summary")
    #                 st.info(doc_info['summary'])
                
    #             # Classification Results
    #             if 'classification' in doc_info:
    #                 DocumentContainer._render_classification(doc_info['classification'])
                
    #             # Similarity Scores (if available)
    #             if 'similarities' in doc_info:
    #                 DocumentContainer._render_similarities(doc_info['similarities'])
            
    #         with col2:
    #             # Concept Image
    #             if 'image' in doc_info:
    #                 st.markdown("### 🎨 Concept Visualization")
    #                 st.image(doc_info['image'], use_container_width=True)
            
    #         st.markdown("---")  # Divider between documents

    @staticmethod
    def _render_stats(stats: Dict):
        """Render document statistics"""
        st.markdown("### 📊 Document Statistics")
        
        # Create a clean statistics display
        stats_md = f"""
        - **Type:** {stats['type'].upper()}
        - **Size:** {stats['file_size'] / 1024:.1f} KB
        - **Words:** {stats['word_count']:,}
        - **Characters:** {stats['char_count']:,}
        """
        
        # Add specific stats based on file type
        if stats['type'] == 'pdf':
            stats_md += f"- **Pages:** {stats['num_pages']}"
        elif stats['type'] == 'docx':
            stats_md += f"""
            - **Paragraphs:** {stats['paragraph_count']}
            - **Tables:** {stats['table_count']}
            """
        elif stats['type'] == 'txt':
            stats_md += f"- **Lines:** {stats['line_count']}"
        
        st.markdown(stats_md)

    @staticmethod
    def _render_classification(classification: Dict):
        """Render classification results"""
        st.markdown("### 🏷️ Topic Classification")
        
        # Create DataFrame for display
        df = pd.DataFrame({
            'Topic': classification['topics'],
            'Confidence': [f"{score*100:.1f}%" for score in classification['scores']]
        }).head(5)  # Show top 5 topics
        
        # Display as table
        st.dataframe(
            df,
            column_config={
                "Topic": st.column_config.TextColumn("Topic"),
                "Confidence": st.column_config.TextColumn("Confidence")
            },
            hide_index=True
        )
        
        # Visualization
        fig = px.bar(
            df,
            x='Topic',
            y=[float(s.strip('%')) for s in df['Confidence']],
            title='Top Topics',
            labels={'y': 'Confidence (%)', 'x': 'Topic'}
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_similarities(similarities: Dict):
        """Render similarity scores"""
        if not similarities:
            return
            
        st.markdown("### 🔄 Document Similarities")
        
        # Create DataFrame for similarities
        df = pd.DataFrame([
            {"Document": doc, "Similarity": f"{score:.1f}%"}
            for doc, score in similarities.items()
        ]).sort_values("Similarity", ascending=False)
        
        # Display similarities
        for _, row in df.iterrows():
            score = float(row["Similarity"].strip('%'))
            if score > 70:
                st.success(f"📄 {row['Document']}: {row['Similarity']}")
            elif score > 40:
                st.warning(f"📄 {row['Document']}: {row['Similarity']}")
            else:
                st.info(f"📄 {row['Document']}: {row['Similarity']}")

class ChatInterface:
    """Handles the chat interface"""
    
    @staticmethod
    def render():
        """Render the chat interface"""
        st.markdown("### 💬 Chat with Documents")
        
        if not st.session_state.active_docs:
            st.info("Please upload and select documents to start chatting!")
            return
            
        st.info(f"📚 Currently analyzing: {', '.join(st.session_state.active_docs)}")
        
        # Create container
        chat_container = st.container()
        input_container = st.container()
        
        # Handle input first (at bottom)
        with input_container:
            # st.markdown("<br>" * 2, unsafe_allow_html=True)
            prompt = st.chat_input("Ask me anything about your documents...")
        
        # Display messages
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Handle new message
            if prompt:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_friendly_response(
                            prompt,
                            list(st.session_state.active_docs)
                        )
                        st.markdown(response)
                
                # Update chat history
                st.session_state.chat_history.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ])
                
                st.rerun()

def get_friendly_response(question: str, active_docs: List[str]) -> str:
    """Generate response using Gemini model"""
    try:
        context = get_combined_context(active_docs)
        prompt = f"""You are a helpful assistant analyzing these documents. 
        Here are the documents and their summaries:
        {context}

        User question: {question}

        Please provide a comprehensive answer based on ALL selected documents. 
        If referring to specific information, mention which document it came from.
        If you can't find the information in any document, say so clearly.
        Maintain a friendly, professional, conversational tone.
        
        Guidelines:
        - Cite specific documents when referencing information
        - Be clear about uncertain or missing information
        - Use examples from the documents when relevant
        - Keep the response focused and concise
        """

        response = st.session_state.llm_model.generate_content(
            prompt,
            generation_config=st.session_state.llm_config
        )
        return response.text
        
    except Exception as e:
        logger.error(f"Response generation error: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"

def get_combined_context(active_docs: List[str]) -> str:
    """Combine content from multiple documents"""
    combined_text = ""
    for doc_name in active_docs:
        doc = st.session_state.documents.get(doc_name)
        if doc:
            combined_text += f"\nDocument: {doc_name}\n"
            if 'summary' in doc:
                combined_text += f"Summary: {doc['summary']}\n"
            combined_text += f"Content Preview: {doc['content'][:1000]}\n"
            combined_text += "---\n"
    return combined_text