# models.py
import streamlit as st
from typing import Dict, List, Optional, Tuple
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import requests
from document_assistant.core import logger, API_CONFIG, HEADERS, MAX_SUMMARY_LENGTH, MIN_SUMMARY_LENGTH
from PIL import Image
import io
import asyncio

class ModelManager:
    """Manages initialization and access to all AI models"""
    
    def __init__(self):
        self.summary_model = None
        self.classifier = None
        self.similarity_model = None
        self.image_generator = None
        self.is_initialized = False

    def initialize_models(self):
        """Initialize all models in parallel"""
        if not self.is_initialized:
            with st.spinner("Initializing AI models..."):
                try:
                    # Initialize classifier
                    self.classifier = TopicClassifier()
                    
                    # Initialize similarity model
                    self.similarity_model = SimilarityCalculator()
                    
                    # Initialize summary model
                    self.summary_model = SummaryGenerator()

                    # Initialize image generator
                    self.image_generator = ImageGenerator()
                    
                    self.is_initialized = True
                    st.success("✅ AI models initialized successfully!")
                except Exception as e:
                    logger.error(f"Model initialization error: {str(e)}")
                    st.error("Error initializing AI models")

class SummaryGenerator:
    """Handles document summarization"""
    
    def __init__(self):
        self.url = API_CONFIG['summary_url']
        self.headers = HEADERS

    async def generate_summary(self, text: str) -> str:
        try:
            chunks = self._chunk_text(text)
            summaries = []

            for chunk in chunks:
                payload = {
                    "inputs": chunk,
                    "parameters": {
                        "max_length": MAX_SUMMARY_LENGTH,
                        "min_length": MIN_SUMMARY_LENGTH,
                        "do_sample": False
                    }
                }

                try:
                    response = requests.post(
                        self.url,
                        headers=self.headers,
                        json=payload,
                        timeout=30
                    )

                    if response.status_code == 200:
                        summaries.append(response.json()[0]['summary_text'])
                    elif response.status_code == 503:
                        await asyncio.sleep(20)
                        response = requests.post(
                            self.url,
                            headers=self.headers,
                            json=payload
                        )
                        if response.status_code == 200:
                            summaries.append(response.json()[0]['summary_text'])

                    await asyncio.sleep(2)

                except Exception as e:
                    logger.error(f"Chunk summary error: {str(e)}")
                    continue

            if summaries:
                return " ".join(summaries)
            return "Could not generate summary."

        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

class TopicClassifier:
    """Handles document classification"""
    
    def __init__(self):
        self.model_id = "cross-encoder/nli-deberta-v3-large"
        self.default_categories = [
            "Artificial Intelligence", "Machine Learning", "Natural Language Processing",
            "Computer Vision", "Robotics", "Data Science", "Physics", "Mathematics",
            "Statistics", "Biology", "Chemistry", "Economics", "Finance", "Medicine",
            "Engineering", "Space Science", "Earth Science", "Materials Science"
        ]
        self.classifier = self._initialize_classifier()

    def _initialize_classifier(self):
        try:
            return pipeline(
                "zero-shot-classification",
                model=self.model_id,
                device=-1,
                hypothesis_template="This text is about {}."
            )
        except Exception as e:
            logger.error(f"Classifier initialization error: {str(e)}")
            return None

    def classify_document(self, text: str, categories: List[str] = None) -> Optional[Dict]:
        try:
            if self.classifier is None:
                return None

            if categories is None:
                categories = self.default_categories

            # Truncate text if too long
            if len(text) > 1024:
                text = text[:1024]

            result = self.classifier(
                text,
                candidate_labels=categories,
                multi_label=True
            )

            topic_scores = list(zip(result['labels'], result['scores']))
            topic_scores.sort(key=lambda x: x[1], reverse=True)

            return {
                'topics': [t[0] for t in topic_scores],
                'scores': [t[1] for t in topic_scores]
            }

        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return None

class SimilarityCalculator:
    """Handles document similarity calculations"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._initialize_model()

    def _initialize_model(self):
        try:
            return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=self.device)
        except Exception as e:
            logger.error(f"Similarity model initialization error: {str(e)}")
            return None

    def calculate_similarity(self, source_text: str, comparison_texts: List[str]) -> Optional[List[float]]:
        try:
            if self.model is None:
                return None

            with torch.no_grad():
                source_embedding = self.model.encode(source_text, convert_to_tensor=True)
                comparison_embeddings = self.model.encode(comparison_texts, convert_to_tensor=True)
                similarities = torch.nn.functional.cosine_similarity(
                    source_embedding.unsqueeze(0),
                    comparison_embeddings
                )

            return [float(score) * 100 for score in similarities]

        except Exception as e:
            logger.error(f"Similarity calculation error: {str(e)}")
            return None

class ImageGenerator:
    """Handles concept image generation"""
    
    def __init__(self):
        self.url = API_CONFIG['image_url']
        self.headers = HEADERS

    async def generate_image(self, text: str, title: str = "") -> Tuple[Optional[Image.Image], Optional[str]]:
        try:
            # Create prompt for image generation
            prompt = self._create_prompt(text, title)
            payload = {"inputs": prompt}

            # Make API request
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    return image, None
                else:
                    error_msg = response.json().get('error', 'Unknown error')
                    if "Max requests total reached" in error_msg:
                        return None, "⏳ Rate limit reached. Please wait 60 seconds..."
                    return None, f"Image generation failed: {error_msg}"

            except requests.exceptions.RequestException as e:
                return None, f"Request failed: {str(e)}"

        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            return None, str(e)

    def _create_prompt(self, text: str, title: str) -> str:
        """Create a prompt for image generation"""
        return f"""Create a single artistic concept visualization:
        Main idea: {title}
        Content: {text[:200]}
        Style Requirements:
        - Modern digital art style
        - Professional futuristic design
        - Abstract representation of the concept
        - Rich symbolic visualization
        - Vibrant colors and dynamic composition
        - Highly detailed technological aesthetic
        - Focus on the core idea, not technical details
        - No text, charts, or diagrams
        - Single cohesive image that captures the essence
        - Professional sci-fi art quality
        """