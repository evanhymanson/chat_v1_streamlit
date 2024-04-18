import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.node_parser import MarkdownElementNodeParser

try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

import os
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
nest_asyncio.apply()

# API access to llama-cloud
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
# Using Anthropic API for embeddings/LLMs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Chat with the Manufactr=uring docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = OPENAI_API_KEY
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")

         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about your manufacturing needs!"}
    ]

##### Load Data, Parse 
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs hang tight! This should take 5 minutes."):
        # GPT 3.5
        embed_model=OpenAIEmbedding(model="text-embedding-3-small")
        llm = OpenAI(model="gpt-3.5-turbo-0125", temperature= 0.5, system_prompt='You are a manufacturing assistant. You recommend the best components and provide details about those components.')

        # Parsing 
        parser = LlamaParse(api_key= LLAMA_CLOUD_API_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY
                            result_type="markdown",  # "markdown" and "text" are available
                            parsing_instruction="""It contains tables and specifications for different components. 
                            Find information for each component and be ready to provide it.""",
                            verbose=True
        )

        file_extractor = {".pdf": parser}
        docs = SimpleDirectoryReader("./Al_rods", file_extractor=file_extractor).load_data()
        
        # Nodes 
        node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-3.5-turbo-0125"), num_workers=8)
        nodes = node_parser.get_nodes_from_documents(docs)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

        # Rerank
        reranker = FlagEmbeddingReranker(top_n=5, model="BAAI/bge-reranker-large")

        service_context = ServiceContext.from_defaults(llm=llm)
        
        # Recursive Index 
        recursive_index = VectorStoreIndex(nodes=base_nodes+objects, service_context=service_context)
        
        index = recursive_index
        return index

index = load_data()


if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
