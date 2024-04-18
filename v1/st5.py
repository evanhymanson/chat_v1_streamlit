import streamlit as st
from streamlit import session_state as ss
import openai
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.node_parser import MarkdownElementNodeParser
import pybase64

try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

import os
from dotenv import load_dotenv
load_dotenv()

# env: /Users/evanhymanson/Desktop/LlamaIndex/venv
import nest_asyncio
nest_asyncio.apply()

# API access to llama-cloud
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
# Using Anthropic API for embeddings/LLMs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="CELESTRAL AI", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = OPENAI_API_KEY
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")

         
## Upload File 

import streamlit as st


uploaded_files = st.file_uploader(
    "Choose a PDF file",
    accept_multiple_files=True,
    type=['pdf']
)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()

    st.write("filename:", uploaded_file.name)
    base64_pdf = pybase64.b64encode(bytes_data).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)