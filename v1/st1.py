import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

st.title("hola")

