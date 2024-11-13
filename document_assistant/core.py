# core.py
import time
import pandas as pd
import streamlit as st
import logging
import google.generativeai as genai
import PyPDF2
import docx
import asyncio
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import os
import tempfile
import subprocess
import plotly.express as px
from sentence_transformers import SentenceTransformer
import torch
import gc
import io
from PIL import Image
import warnings
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_BATCH_SIZE = 5
MAX_SUMMARY_LENGTH = 250
MIN_SUMMARY_LENGTH = 50

def setup_apis():
    """Setup and verify API configurations"""
    if 'GOOGLE_API_KEY' not in st.secrets:
        st.error("Please set GOOGLE_API_KEY in streamlit secrets")
        st.stop()
    if 'HF_API_KEY' not in st.secrets:
        st.error("Please set HF_API_KEY in streamlit secrets")
        st.stop()

    genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
    
    return {
        'model': genai.GenerativeModel('gemini-1.5-pro'),
        'hf_key': st.secrets["HF_API_KEY"],
        'summary_url': "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
        'image_url': "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large",
    }

# Initialize APIs
API_CONFIG = setup_apis()
HEADERS = {
    "Authorization": f"Bearer {API_CONFIG['hf_key']}",
    "Content-Type": "application/json"
}

def init_session_state():
    """Initialize session state variables"""
    # Document management
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'active_docs' not in st.session_state:
        st.session_state.active_docs = set()
    if 'previous_files' not in st.session_state:
        st.session_state.previous_files = set()
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}

    # Model states
    if 'models_initialized' not in st.session_state:
        st.session_state.models_initialized = False
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None

    # Chat functionality
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Additional states for tracking
    if 'current_file_count' not in st.session_state:
        st.session_state.current_file_count = 0
    if 'last_processed_file' not in st.session_state:
        st.session_state.last_processed_file = None
    if 'processing_errors' not in st.session_state:
        st.session_state.processing_errors = {}

    # LLM Model initialization
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = API_CONFIG['model']
    if 'llm_config' not in st.session_state:
        st.session_state.llm_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=1024,
        )