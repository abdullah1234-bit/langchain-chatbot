import os
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
import streamlit as st

# Set the Groq API Key as an environment variable
os.environ["GROQ_API_KEY"] = "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"

# Streamlit UI
st.title('LangChain Demo with Groq API')
input_text = st.text_input("Search the topic you want")

# Define LLM with Groq API
llm = ChatGroq(model_name="mixtral-8x7b-32768")  # Ensure the model name is valid

if input_text:
    try:
        # Use the invoke method instead of directly calling the LLM
        response = llm.invoke(input_text)
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
